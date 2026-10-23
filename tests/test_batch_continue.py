from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import types
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from claw_eval.cli import (
    _batch_score_counts, _load_completed_results, _merge_continue_results, cmd_batch,
)


def trial(score: float, trace: str | None = None) -> dict:
    row = {"task_score": score, "passed": score >= 0.75}
    if trace:
        row.update(trace=trace, model_input_tokens=10, model_output_tokens=5,
                   input_tokens=10, output_tokens=5, tokens=15,
                   model_time_s=2.0, tool_time_s=1.0, other_time_s=0.5, wall_time_s=3.5)
    return row


def write_trace(root: Path, task_id: str, suffix: str, score: float) -> dict:
    path = root / f"{task_id}_{suffix}.jsonl"
    path.write_text(json.dumps({
        "type": "trace_end", "model_input_tokens": 10, "model_output_tokens": 5,
        "total_tokens": 15, "model_time_s": 2.0, "tool_time_s": 1.0,
        "other_time_s": 0.5, "wall_time_s": 3.5,
    }) + "\n" + json.dumps({
        "type": "grading_result", "task_id": task_id, "task_score": score,
        "passed": score >= 0.75,
        "scores": {"completion": score, "robustness": 1, "safety": 1},
    }) + "\n")
    return trial(score, str(path))


class InlinePool:
    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def submit(self, function, **kwargs):
        future = Future()
        try:
            future.set_result(function(**kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


class BatchContinueTests(unittest.TestCase):
    def test_split_merge_matches_full_run_with_errors(self):
        error = {"trial": 1, "task_score": 0.0, "passed": False, "error": "API 403"}
        a = {"task_id": "A", "task_name": "Alpha", "trials": [trial(1, "a1"), trial(1, "a2"), trial(1, "a3")]}
        b = {"task_id": "B", "task_name": "Beta", "trials": [trial(1, "b1"), trial(1, "b2"), error]}
        full = _merge_continue_results([], [a, b], 3)
        prior = [a, {**b, "trials": b["trials"][:1]}]
        updates = [{**b, "trials": b["trials"][1:]}]
        resumed = _merge_continue_results(prior, updates, 3)
        self.assertEqual(resumed, full)
        # Final trace scan already contains the newly completed successful trial.
        final = _merge_continue_results([a, {**b, "trials": b["trials"][:2]}], updates, 3)
        self.assertEqual(final, full)
        avg, passed, any_pass, errors = _batch_score_counts(final, 3)
        self.assertAlmostEqual(avg, 5 / 6)
        self.assertEqual((passed, any_pass, errors), (1, 2, 1))

    def test_missing_trials_and_startup_errors_cannot_pass(self):
        rows = _merge_continue_results([], [
            {"task_id": "A", "trials": [trial(1), trial(1)]},
            {"task_id": "B", "trials": [], "error": "container failed"},
        ], 3)
        self.assertLess(rows[0]["pass_hat_k"], 1)
        self.assertEqual(_batch_score_counts(rows, 3), (1 / 3, 0, 1, 1))

    def test_retry_replaces_old_errors_but_keeps_new_errors(self):
        completed = [{"task_id": "A", "trials": [trial(1, "a0"), trial(1, "a1")]}]
        old = [{"task_id": "A", "task_name": "Alpha", "trials": [
            trial(1, "a0"), {"error": "old 403", "task_score": 0, "passed": False},
        ]}]
        current = [{"task_id": "A", "trials": [
            {"error": "new timeout", "task_score": 0, "passed": False},
        ], "error": "new timeout"}]
        merged = _merge_continue_results(completed, current, 3, old)
        self.assertEqual(len(merged[0]["trials"]), 3)
        self.assertEqual(merged[0]["task_name"], "Alpha")
        self.assertEqual(merged[0]["trials"][-1]["error"], "new timeout")
        self.assertNotIn("old 403", json.dumps(merged))
        self.assertEqual(_batch_score_counts(merged, 3), (2 / 3, 0, 1, 1))

    def test_continue_outputs_include_old_and_new_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traces = root / "traces"
            traces.mkdir()
            tasks = root / "tasks"
            for tid in ("A", "B"):
                (tasks / tid).mkdir(parents=True)
                (tasks / tid / "task.yaml").write_text("task_id: " + tid)
            a_trials = [write_trace(traces, "A", str(i), 1) for i in range(3)]
            b_first = write_trace(traces, "B", "0", 1)
            # Old failed/interrupted attempts must not be reused as new errors.
            (traces / "B_interrupted.jsonl").write_text('{"type":"trace_start"}\n{')
            metadata = [
                {"task_id": "A", "task_name": "Alpha", "trials": a_trials},
                {"task_id": "B", "task_name": "Beta", "trials": [b_first], "error": "old failure"},
            ]
            (traces / "batch_results.json").write_text(json.dumps(metadata))
            args = argparse.Namespace(
                tasks_dir=str(tasks), config=None, continue_dir=str(traces),
                filter=None, tag=None, range=None, parallel=1, trials=3,
                model=None, api_key=None, base_url=None, trace_dir=None,
                no_judge=False, sandbox=False,
            )
            cfg = types.SimpleNamespace(
                model=types.SimpleNamespace(model_id="test"),
                defaults=types.SimpleNamespace(trace_dir=str(traces)),
            )
            snapshots = []
            real_dump = json.dump

            def capture_dump(value, handle, **kwargs):
                if isinstance(value, list):
                    snapshots.append(json.loads(json.dumps(value)))
                return real_dump(value, handle, **kwargs)

            def worker(**kwargs):
                self.assertEqual(Path(kwargs["task_dir"]).name, "B")
                self.assertEqual(kwargs["trials"], 2)
                return {"task_id": "B", "task_name": "Beta", "error": None,
                        "trials": [write_trace(traces, "B", str(i), 1) for i in (1, 2)]}

            with patch("claw_eval.config.load_config", return_value=cfg), \
                 patch("claw_eval.cli.ProcessPoolExecutor", InlinePool), \
                 patch("claw_eval.cli._run_single_task", side_effect=worker) as run, \
                 patch("claw_eval.cli.json.dump", side_effect=capture_dump), \
                 contextlib.redirect_stdout(io.StringIO()):
                cmd_batch(args)
                run.assert_called_once()
                summary = json.loads((traces / "batch_summary.json").read_text())
                results = json.loads((traces / "batch_results.json").read_text())
                self.assertTrue(all(len(snapshot) == 2 for snapshot in snapshots))
                self.assertEqual([len(row["trials"]) for row in results], [3, 3])
                self.assertEqual([row["task_name"] for row in results], ["Alpha", "Beta"])
                self.assertEqual(summary["tasks"], 2)
                self.assertEqual(summary["tasks_finished_this_run"], 1)
                self.assertEqual(summary["pass_hat_3"], 2)
                self.assertEqual(summary["pass_at_3"], 2)
                self.assertEqual(summary["avg_score"], 1)
                self.assertEqual(summary["errored"], 0)
                self.assertEqual(summary["total_tokens"], 90)
                self.assertEqual(summary["total_wall_time_s"], 21)
                # Already-complete continuation rebuilds output without any calls.
                cmd_batch(args)
                run.assert_called_once()
                rebuilt = json.loads((traces / "batch_summary.json").read_text())
                self.assertEqual(rebuilt.pop("tasks_finished_this_run"), 0)
                summary.pop("tasks_finished_this_run")
                self.assertEqual(rebuilt, summary)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULES = (
    "mock_services.web_real.search_serp",
    "mock_services.web_real_injection.search_serp",
)


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        data: dict | None = None,
        text: str = "",
        headers: dict[str, str] | None = None,
        json_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._data = data or {}
        self.text = text
        self.headers = headers or {}
        self._json_error = json_error

    def json(self) -> dict:
        if self._json_error:
            raise self._json_error
        return self._data


class SerperRetryTests(unittest.TestCase):
    def test_transient_failures_retry_then_succeed(self) -> None:
        for module_name in MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                outcomes = [
                    TimeoutError("temporary timeout"),
                    FakeResponse(503, text="temporarily unavailable"),
                    FakeResponse(
                        200,
                        data={
                            "organic": [
                                {"title": "ok", "link": "https://example.com"}
                            ]
                        },
                        text='{"organic": []}',
                    ),
                ]
                calls: list[dict] = []
                records: list[tuple] = []
                sleeps: list[float] = []

                def fake_post(*args, **kwargs):
                    calls.append(kwargs)
                    outcome = outcomes.pop(0)
                    if isinstance(outcome, Exception):
                        raise outcome
                    return outcome

                def record(query, status, attempt, error=None):
                    records.append((query, status, attempt, error))

                with (
                    patch.object(module.requests, "post", side_effect=fake_post),
                    patch.object(module.time, "sleep", side_effect=sleeps.append),
                    patch.object(module.random, "uniform", return_value=0.0),
                    patch.object(module, "_record_serp_request", side_effect=record),
                ):
                    result = module.search_serp("test query", timeout=20)

                self.assertEqual(result["status"], 200)
                self.assertEqual(result["output"][0]["title"], "ok")
                self.assertEqual(len(calls), 3)
                self.assertTrue(all(call["timeout"] == 20 for call in calls))
                self.assertEqual(sleeps, [1.0, 2.0])
                self.assertEqual(
                    [(r[1], r[2]) for r in records],
                    [(-1, 1), (503, 2), (200, 3)],
                )

    def test_non_transient_status_does_not_retry(self) -> None:
        for module_name in MODULES:
            for status in (400, 401, 403):
                with self.subTest(module=module_name, status=status):
                    module = importlib.import_module(module_name)
                    calls = 0

                    def fake_post(*args, **kwargs):
                        nonlocal calls
                        calls += 1
                        return FakeResponse(status, text="rejected")

                    with (
                        patch.object(module.requests, "post", side_effect=fake_post),
                        patch.object(
                            module.time,
                            "sleep",
                            side_effect=AssertionError("must not retry"),
                        ),
                    ):
                        result = module.search_serp("bad query")

                    self.assertEqual(
                        result,
                        {"status": status, "output": [], "error": "rejected"},
                    )
                    self.assertEqual(calls, 1)

    def test_429_honors_retry_after(self) -> None:
        for module_name in MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                outcomes = [
                    FakeResponse(
                        429,
                        text="rate limited",
                        headers={"Retry-After": "2.5"},
                    ),
                    FakeResponse(
                        200, data={"organic": []}, text='{"organic": []}'
                    ),
                ]
                sleeps: list[float] = []
                with (
                    patch.object(
                        module.requests,
                        "post",
                        side_effect=lambda *a, **kw: outcomes.pop(0),
                    ),
                    patch.object(module.time, "sleep", side_effect=sleeps.append),
                ):
                    result = module.search_serp("rate limited query")

                self.assertEqual(result["status"], 200)
                self.assertEqual(sleeps, [2.5])

    def test_exhausted_quota_does_not_retry(self) -> None:
        for module_name in MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                calls = 0

                def fake_post(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    return FakeResponse(429, text="quota exceeded: no remaining credit")

                with (
                    patch.object(module.requests, "post", side_effect=fake_post),
                    patch.object(
                        module.time,
                        "sleep",
                        side_effect=AssertionError("permanent failure must not retry"),
                    ),
                ):
                    result = module.search_serp("quota test")

                self.assertEqual(result["status"], 429)
                self.assertEqual(calls, 1)

    def test_network_failure_exhausts_three_attempts(self) -> None:
        for module_name in MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                calls = 0
                records: list[tuple] = []

                def fail(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    raise TimeoutError("still unavailable")

                def record(query, status, attempt, error=None):
                    records.append((query, status, attempt, error))

                with (
                    patch.object(module.requests, "post", side_effect=fail),
                    patch.object(module.time, "sleep", return_value=None),
                    patch.object(module, "_record_serp_request", side_effect=record),
                ):
                    result = module.search_serp("timeout query")

                self.assertEqual(result["status"], -1)
                self.assertIn("still unavailable", result["error"])
                self.assertEqual(calls, 3)
                self.assertEqual(
                    [(r[1], r[2]) for r in records],
                    [(-1, 1), (-1, 2), (-1, 3)],
                )

    def test_invalid_json_is_retried_and_logged(self) -> None:
        for module_name in MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                outcomes = [
                    FakeResponse(
                        200,
                        text="not json",
                        json_error=ValueError("invalid json"),
                    ),
                    FakeResponse(
                        200, data={"organic": []}, text='{"organic": []}'
                    ),
                ]
                records: list[tuple] = []

                def record(query, status, attempt, error=None):
                    records.append((query, status, attempt, error))

                with (
                    patch.object(
                        module.requests,
                        "post",
                        side_effect=lambda *a, **kw: outcomes.pop(0),
                    ),
                    patch.object(module.time, "sleep", return_value=None),
                    patch.object(module, "_record_serp_request", side_effect=record),
                ):
                    result = module.search_serp("invalid json query")

                self.assertEqual(result["status"], 200)
                self.assertEqual([(r[1], r[2]) for r in records], [(200, 1), (200, 2)])
                self.assertIn("Invalid JSON response", records[0][3])

    def test_serp_log_records_attempt_and_error(self) -> None:
        from claw_eval import serp_log

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                "os.environ",
                {"SERP_LOG_DIR": temp_dir, "CLAW_TASK_ID": "T_TEST"},
            ):
                serp_log._path_cache.clear()
                serp_log._turn_counter.clear()
                serp_log.log_serp_request(
                    query="logged query",
                    status=-1,
                    attempt=2,
                    error="Read timed out after 20 seconds",
                )

            files = list(Path(temp_dir).rglob("*.jsonl"))
            self.assertEqual(len(files), 1)
            record = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(record["task_id"], "T_TEST")
            self.assertEqual(record["query"], "logged query")
            self.assertEqual(record["http_status"], -1)
            self.assertEqual(record["attempt"], 2)
            self.assertEqual(record["error"], "Read timed out after 20 seconds")


if __name__ == "__main__":
    unittest.main()

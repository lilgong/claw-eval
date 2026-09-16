from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from claw_eval.graders.llm_judge import LLMJudge
from claw_eval.runner.providers.openai_compat import OpenAICompatProvider
from claw_eval.runner.user_agent import UserAgent


class PermanentAPIError(Exception):
    status_code = 402


class TransientAPIError(Exception):
    status_code = 429


def failing_client(counter: list[int]):
    def create(**_kwargs):
        counter.append(1)
        raise PermanentAPIError("insufficient balance")

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create),
        )
    )


def transient_then_success_client(content: str, counter: list[int]):
    def create(**_kwargs):
        counter.append(1)
        if len(counter) == 1:
            raise TransientAPIError("rate limited")
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=content),
            )]
        )

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create),
        )
    )


class ExtraHeadersTests(unittest.TestCase):
    def test_omitted_headers_preserve_default_behavior(self) -> None:
        with patch("claw_eval.runner.providers.openai_compat.OpenAI") as client:
            OpenAICompatProvider(
                model_id="test", api_key="key", base_url="https://example.test/v1"
            )
        self.assertIsNone(client.call_args.kwargs["default_headers"])

    def test_configured_headers_are_passed_to_client(self) -> None:
        headers = {"X-Real-Base-URL": "https://model.test/v1"}
        with patch("claw_eval.runner.providers.openai_compat.OpenAI") as client:
            provider = OpenAICompatProvider(
                model_id="test",
                api_key="key",
                base_url="https://proxy.test/v1",
                extra_headers=headers,
            )
        self.assertEqual(client.call_args.kwargs["default_headers"], headers)
        self.assertEqual(provider.extra_headers, headers)


class PermanentFailureTests(unittest.TestCase):
    def test_user_agent_stops_after_first_permanent_failure(self) -> None:
        calls: list[int] = []
        agent = UserAgent("test", "key", "https://example.test/v1")
        agent.client = failing_client(calls)
        with (
            patch("claw_eval.runner.user_agent.time.sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "kind=permanent"),
        ):
            agent.generate_response("persona", [])
        self.assertEqual(len(calls), 1)
        sleep.assert_not_called()

    def test_all_judge_entrypoints_stop_after_first_permanent_failure(self) -> None:
        invocations = (
            lambda judge: judge.evaluate("task", "conversation", "actions", "rubric"),
            lambda judge: judge.evaluate_actions("task", "artifacts", "rubric"),
            lambda judge: judge.evaluate_visual("rubric", [], []),
        )
        for invoke in invocations:
            with self.subTest(entrypoint=invoke):
                calls: list[int] = []
                judge = LLMJudge(
                    model_id="not-gemini",
                    api_key="key",
                    base_url="https://example.test/v1",
                    openai_compatible=True,
                )
                judge.client = failing_client(calls)
                with (
                    patch("claw_eval.graders.llm_judge.time.sleep") as sleep,
                    self.assertRaisesRegex(RuntimeError, "kind=permanent"),
                ):
                    invoke(judge)
                self.assertEqual(len(calls), 1)
                sleep.assert_not_called()

    def test_transient_user_and_judge_failures_still_retry(self) -> None:
        user_calls: list[int] = []
        agent = UserAgent("test", "key", "https://example.test/v1")
        agent.client = transient_then_success_client("[DONE]", user_calls)
        with patch("claw_eval.runner.user_agent.time.sleep") as user_sleep:
            self.assertIsNone(agent.generate_response("persona", []))
        self.assertEqual(len(user_calls), 2)
        user_sleep.assert_called_once()

        judge_calls: list[int] = []
        judge = LLMJudge(
            model_id="not-gemini",
            api_key="key",
            base_url="https://example.test/v1",
            openai_compatible=True,
        )
        judge.client = transient_then_success_client(
            '{"score": 0.5, "reasoning": "ok"}', judge_calls,
        )
        with patch("claw_eval.graders.llm_judge.time.sleep") as judge_sleep:
            result = judge.evaluate("task", "conversation", "actions", "rubric")
        self.assertEqual(result.score, 0.5)
        self.assertEqual(len(judge_calls), 2)
        judge_sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()

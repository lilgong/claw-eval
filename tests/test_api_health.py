from __future__ import annotations

import unittest

from claw_eval.api_health import classify_failure


class FailureClassificationTests(unittest.TestCase):
    def test_http_transient_status_wins_over_broad_auth_text(self) -> None:
        self.assertEqual(
            classify_failure(503, "authentication service unavailable"),
            "transient",
        )

    def test_statusless_network_failure_wins_over_auth_service_name(self) -> None:
        self.assertEqual(
            classify_failure(
                None, "connection error: authentication service unavailable"
            ),
            "transient",
        )
        self.assertEqual(
            classify_failure(None, "timeout contacting authentication backend"),
            "transient",
        )

    def test_rate_limit_language_is_not_treated_as_permanent(self) -> None:
        self.assertEqual(classify_failure(429, "quota exceeded"), "transient")
        self.assertEqual(classify_failure(429, "rate limit exceeded"), "transient")

    def test_explicit_billing_exhaustion_is_permanent(self) -> None:
        self.assertEqual(classify_failure(429, "insufficient balance"), "permanent")
        self.assertEqual(classify_failure(503, "insufficient credit"), "permanent")

    def test_auth_and_payment_statuses_are_permanent(self) -> None:
        self.assertEqual(classify_failure(401, "unauthorized"), "permanent")
        self.assertEqual(classify_failure(402, "payment required"), "permanent")
        self.assertEqual(classify_failure(403, "invalid api key"), "permanent")

    def test_generic_policy_forbidden_does_not_trip_api_breaker(self) -> None:
        self.assertIsNone(classify_failure(403, "request blocked by safety policy"))


if __name__ == "__main__":
    unittest.main()

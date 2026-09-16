from __future__ import annotations

import json
import sys
import types
import unittest
from unittest.mock import patch

from claw_eval.api_health import MARKER
from mock_services.web_real import server as web_server
from mock_services.web_real_injection import server as injection_server


class SerperErrorHandlingTests(unittest.TestCase):
    def _call_search(self, server, status: int, error: str):
        fake_module = types.ModuleType("search_serp")
        fake_module.search_serp = lambda **_kwargs: {
            "status": status,
            "output": [],
            "error": error,
        }
        request = server.SearchRequest(
            query=f"error-handling-test-{status}", max_results=1,
        )
        with (
            patch.dict(sys.modules, {"search_serp": fake_module}),
            patch.object(server, "_cache_get", return_value=None),
            patch.object(server, "_log_call"),
        ):
            response = server.web_search(request)
        return response, json.loads(response.body)

    def test_transient_failure_is_an_agent_visible_tool_error(self) -> None:
        for server in (web_server, injection_server):
            with self.subTest(server=server.__name__):
                response, body = self._call_search(server, 429, "rate limited")
                self.assertEqual(response.status_code, 502)
                self.assertNotIn(MARKER, body["error"])
                self.assertIn("temporarily unavailable", body["error"])

    def test_permanent_failure_keeps_internal_circuit_marker(self) -> None:
        for server in (web_server, injection_server):
            with self.subTest(server=server.__name__):
                response, body = self._call_search(server, 401, "invalid API key")
                self.assertEqual(response.status_code, 502)
                self.assertIn(MARKER, body["error"])
                self.assertIn("kind=permanent", body["error"])


if __name__ == "__main__":
    unittest.main()

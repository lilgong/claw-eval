"""Tool dispatcher: routes tool_use blocks to mock HTTP services."""

from __future__ import annotations

import json
import time

import httpx

from ..api_health import MARKER
from ..models.content import TextBlock, ToolResultBlock, ToolUseBlock
from ..models.tool import ToolEndpoint
from ..models.trace import ToolDispatch


WEB_SEARCH_TIMEOUT_SECONDS = 150.0


class ToolDispatcher:
    """Dispatches tool_use blocks to mock service endpoints via HTTP."""

    def __init__(self, endpoints: dict[str, ToolEndpoint]) -> None:
        self._endpoints = endpoints
        self._client = httpx.Client(timeout=30.0)

    def dispatch(
        self, tool_use: ToolUseBlock, trace_id: str
    ) -> tuple[ToolResultBlock, ToolDispatch]:
        """Execute HTTP request to mock service.

        Returns:
            - ToolResultBlock for the model conversation
            - ToolDispatch event for the trace log
        """
        endpoint = self._endpoints.get(tool_use.name)
        if endpoint is None:
            result = ToolResultBlock(
                tool_use_id=tool_use.id,
                content=[TextBlock(text=f"Error: unknown tool '{tool_use.name}'")],
                is_error=True,
            )
            dispatch_event = ToolDispatch(
                trace_id=trace_id,
                tool_use_id=tool_use.id,
                tool_name=tool_use.name,
                endpoint_url="",
                request_body=tool_use.input,
                response_status=404,
                response_body={"error": f"unknown tool '{tool_use.name}'"},
            )
            return result, dispatch_event

        t0 = time.monotonic()
        try:
            resp = self._client.request(
                method=endpoint.method,
                url=endpoint.url,
                json=tool_use.input,
                # Serper may make three 45-second attempts with backoff. Keep
                # the normal 30-second limit for every other tool.
                timeout=(
                    WEB_SEARCH_TIMEOUT_SECONDS
                    if tool_use.name == "web_search"
                    else 30.0
                ),
            )
            latency_ms = (time.monotonic() - t0) * 1000
            resp_body = resp.json()
            if isinstance(resp_body, dict) and MARKER in str(resp_body.get("error", "")):
                raise RuntimeError(resp_body["error"])

            is_error = resp.status_code >= 400
            content_text = json.dumps(resp_body, ensure_ascii=False)

            result = ToolResultBlock(
                tool_use_id=tool_use.id,
                content=[TextBlock(text=content_text)],
                is_error=is_error,
            )
            dispatch_event = ToolDispatch(
                trace_id=trace_id,
                tool_use_id=tool_use.id,
                tool_name=tool_use.name,
                endpoint_url=endpoint.url,
                request_body=tool_use.input,
                response_status=resp.status_code,
                response_body=resp_body,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            if MARKER in str(exc):
                raise
            latency_ms = (time.monotonic() - t0) * 1000
            result = ToolResultBlock(
                tool_use_id=tool_use.id,
                content=[TextBlock(text=f"Error: {exc}")],
                is_error=True,
            )
            dispatch_event = ToolDispatch(
                trace_id=trace_id,
                tool_use_id=tool_use.id,
                tool_name=tool_use.name,
                endpoint_url=endpoint.url,
                request_body=tool_use.input,
                response_status=500,
                response_body={"error": str(exc)},
                latency_ms=latency_ms,
            )

        return result, dispatch_event

    def close(self) -> None:
        self._client.close()

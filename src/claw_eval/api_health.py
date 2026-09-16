"""Identify credible upstream API-wide failures for batch circuit breaking."""

from __future__ import annotations

import re


MARKER = "CLAW_API_FAILURE"
_AUTH_HINTS = (
    "invalid api key", "incorrect api key", "unauthorized", "authentication",
    "鉴权失败", "密钥无效",
)
_BILLING_HINTS = (
    "insufficient balance", "insufficient credit", "insufficient_quota",
    "not enough credit", "no remaining credit", "credits exhausted",
    "quota exhausted", "payment required", "余额不足", "额度不足",
)
_TRANSIENT_HINTS = (
    "timeout", "timed out", "connection", "server disconnected",
    "peer closed", "remoteprotocol", "service unavailable", "bad gateway",
)


def classify_failure(status: object, detail: object) -> str | None:
    """Return permanent/transient only for credible API-wide failures."""
    try:
        code = int(status) if status is not None else None
    except (TypeError, ValueError):
        code = None
    text = str(detail).lower()

    if code in (401, 402):
        return "permanent"
    # Explicit billing exhaustion is permanent even when a gateway wraps it
    # in 429/5xx.  Generic phrases such as "quota exceeded" are intentionally
    # excluded because they are also used for temporary rate limits.
    if any(hint in text for hint in _BILLING_HINTS):
        return "permanent"
    # HTTP status takes precedence over broad text such as "authentication
    # service unavailable" so a temporary upstream outage cannot stop a batch.
    if code == 429 or (code is not None and 500 <= code <= 599):
        return "transient"
    # Transport exceptions often have no HTTP status.  In that case network
    # wording must win over incidental service names such as
    # "authentication backend".
    if code is None and any(hint in text for hint in _TRANSIENT_HINTS):
        return "transient"
    if any(hint in text for hint in _AUTH_HINTS):
        return "permanent"
    # Generic 403 may be prompt safety policy; do not treat it as API-wide.
    if code == 403:
        return None
    if any(hint in text for hint in _TRANSIENT_HINTS):
        return "transient"
    return None


def failure_marker(source: str, status: object, detail: object) -> str | None:
    kind = classify_failure(status, detail)
    if kind is None:
        return None
    code = "none" if status is None else str(status)
    summary = " ".join(str(detail).split())[:300]
    return f"{MARKER} source={source} kind={kind} status={code}: {summary}"


def marker_from_exception(source: str, exc: Exception) -> str | None:
    status = (
        getattr(exc, "status_code", None)
        or getattr(exc, "code", None)
        or getattr(getattr(exc, "response", None), "status_code", None)
    )
    return failure_marker(source, status, exc)


def parse_marker(text: object) -> tuple[str, str] | None:
    match = re.search(
        rf"{MARKER} source=([a-z_]+) kind=(permanent|transient)", str(text)
    )
    return (match.group(1), match.group(2)) if match else None

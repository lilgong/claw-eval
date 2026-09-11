"""Identify credible upstream API-wide failures for batch circuit breaking."""

from __future__ import annotations

import re


MARKER = "CLAW_API_FAILURE"
_PERMANENT_HINTS = (
    "invalid api key", "incorrect api key", "unauthorized", "authentication",
    "insufficient balance", "insufficient credit", "insufficient_quota",
    "not enough credit", "no remaining credit", "credits exhausted",
    "quota exhausted", "quota exceeded", "payment required",
    "余额不足", "额度不足", "鉴权失败", "密钥无效",
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

    if code in (401, 402) or any(hint in text for hint in _PERMANENT_HINTS):
        return "permanent"
    # Generic 403 may be prompt safety policy; do not treat it as API-wide.
    if code == 403:
        return None
    if code == 429 or (code is not None and 500 <= code <= 599):
        return "transient"
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


def record_failure(
    failure_times: dict[str, list[float]],
    source: str,
    kind: str,
    now: float,
    *,
    window_s: float = 60.0,
    threshold: int = 3,
) -> bool:
    """Record one task-level API failure and return whether to open the circuit."""
    if kind == "permanent":
        return True
    recent = [
        seen for seen in failure_times.get(source, []) if now - seen <= window_s
    ]
    recent.append(now)
    failure_times[source] = recent
    return len(recent) >= threshold

"""Best-effort request logging for billed Serper searches only."""

from __future__ import annotations

import datetime
import json
import os
import re
import threading
import uuid


_lock = threading.Lock()
_path_cache: dict[tuple[str, str], str] = {}
_turn_counter: dict[str, int] = {}


def _enabled() -> bool:
    return os.getenv("CLAW_SERP_LOG", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _monthly_root(base: str) -> str:
    leaf = os.path.basename(os.path.normpath(base))
    try:
        datetime.datetime.strptime(leaf, "%Y-%m")
        return base
    except ValueError:
        return os.path.join(base, datetime.date.today().strftime("%Y-%m"))


def _log_path(key_suffix: str) -> str:
    today = str(datetime.date.today())
    cache_key = (key_suffix, today)
    cached = _path_cache.get(cache_key)
    if cached:
        return cached

    base = os.path.expanduser(os.getenv("SERP_LOG_DIR") or "~/api_usage/claweval")
    root = _monthly_root(base)
    os.makedirs(root, exist_ok=True)
    safe_suffix = re.sub(r"[^A-Za-z0-9_-]", "_", key_suffix)
    path = os.path.join(root, f"serp_requests_{safe_suffix}_{today.replace('-', '')}.jsonl")
    _path_cache[cache_key] = path
    return path


def log_serp_request(
    *,
    query: str,
    status: int,
    api_key: str | None,
    attempt: int = 1,
    error: str | None = None,
    task_id: str | None = None,
) -> None:
    """Append one Serper attempt record; logging never affects evaluation."""
    if not _enabled():
        return
    try:
        key_suffix = api_key[-8:] if api_key else "no-key"
        resolved_task = task_id or os.getenv("CLAW_TASK_ID") or None
        counter_key = resolved_task or "-"
        with _lock:
            turn = _turn_counter.get(counter_key, 0) + 1
            _turn_counter[counter_key] = turn

        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "key_suffix": key_suffix,
            "task_id": resolved_task,
            "turn": turn,
            "call_id": uuid.uuid4().hex[:12],
            "query": query,
            "http_status": int(status),
            "attempt": int(attempt),
        }
        if error:
            record["error"] = " ".join(str(error).split())[:300]
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _lock, open(_log_path(key_suffix), "a", encoding="utf-8") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.write(line)
                handle.flush()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:
                handle.write(line)
                handle.flush()
    except Exception:
        return

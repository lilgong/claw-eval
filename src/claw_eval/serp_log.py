"""Best-effort request logging for billed Serper searches only."""

from __future__ import annotations

import datetime
import json
import os
import threading
import uuid


_lock = threading.Lock()
_path_cache: dict[str, str] = {}
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


def _log_path() -> str:
    today = str(datetime.date.today())
    cached = _path_cache.get(today)
    if cached:
        return cached

    base = os.getenv("SERP_LOG_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "serp_request_log",
    )
    root = _monthly_root(base)
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"serp_requests_{today.replace('-', '')}.jsonl")
    _path_cache[today] = path
    return path


def log_serp_request(
    *, query: str, status: int, task_id: str | None = None,
) -> None:
    """Append one Serper request record; logging never affects evaluation."""
    if not _enabled():
        return
    try:
        resolved_task = task_id or os.getenv("CLAW_TASK_ID") or None
        counter_key = resolved_task or "-"
        with _lock:
            turn = _turn_counter.get(counter_key, 0) + 1
            _turn_counter[counter_key] = turn

        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "task_id": resolved_task,
            "turn": turn,
            "call_id": uuid.uuid4().hex[:12],
            "query": query,
            "http_status": int(status),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _lock, open(_log_path(), "a", encoding="utf-8") as handle:
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

"""Search SERP through yibuapi's Serper-compatible endpoint.

The public ``search_serp`` return shape stays compatible with the web_real
service while the upstream request uses POST JSON plus Bearer authentication.
"""

from __future__ import annotations

import os
import random
import re
import time

import requests

SERP_API_URL = os.getenv("SERP_API_URL", "https://yibuapi.com/serper/search")
SERP_DEV_KEY = (
    os.getenv("SERP_API_KEY")
    or os.getenv("SERP_DEV_KEY")
    or os.getenv("YIBUAPI_KEY", "")
)
SERP_TIMEOUT_SECONDS = 45
SERP_MAX_ATTEMPTS = 3  # Initial request plus at most two retries.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _detect_language(query: str) -> tuple[str, str]:
    if re.search(r"[\u4e00-\u9fff]", query):
        return "zh", "cn"
    return "en", "us"


def _record_serp_request(query: str, status: int) -> None:
    """Record a Serper request without making logging affect search."""
    try:
        from claw_eval.serp_log import log_serp_request

        log_serp_request(query=query, status=status)
    except Exception:
        pass


def _retry_delay(attempt: int) -> float:
    """Return a short exponential-backoff delay with jitter."""
    return (2 ** attempt) + random.uniform(0, 1)


def search_serp(
    query: str,
    timeout: int = SERP_TIMEOUT_SECONDS,
    num: int = 10,
    start: int = 1,
    raw_save_path: str | None = None,
) -> dict:
    """Search Google via SERP API and return extracted results.

    Every actual upstream attempt is audited. Transient upstream failures are
    retried twice, while authentication and request errors return immediately.
    """
    hl, gl = _detect_language(query)
    n = min(max(num, 1), 10)
    page = max(1, ((max(start, 1) - 1) // n) + 1)
    body = {
        "q": query,
        "num": n,
        "hl": hl,
        "gl": gl,
        "page": page,
    }
    headers = {
        "Authorization": f"Bearer {SERP_DEV_KEY}",
        "Content-Type": "application/json",
    }
    for attempt in range(SERP_MAX_ATTEMPTS):
        try:
            resp = requests.post(
                SERP_API_URL, headers=headers, json=body, timeout=timeout
            )
            # Audit every upstream attempt, including failures that are retried.
            _record_serp_request(query, resp.status_code)
            if resp.status_code == 200:
                if raw_save_path:
                    os.makedirs(os.path.dirname(raw_save_path) or ".", exist_ok=True)
                    with open(raw_save_path, "w", encoding="utf-8") as f:
                        f.write(resp.text)
                data = resp.json()
                results = [
                    {
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "date": item.get("date", ""),
                        "query": query,
                    }
                    for item in data.get("organic", [])
                ]
                return {"status": resp.status_code, "output": results}
            if (
                resp.status_code not in RETRYABLE_STATUS_CODES
                or attempt == SERP_MAX_ATTEMPTS - 1
            ):
                return {"status": resp.status_code, "output": [], "error": resp.text[:300]}
        except (requests.Timeout, requests.ConnectionError) as exc:
            # A transport failure is still a billable/search attempt.
            _record_serp_request(query, -1)
            if attempt == SERP_MAX_ATTEMPTS - 1:
                return {"status": -1, "output": [], "error": str(exc)[:300]}

        time.sleep(_retry_delay(attempt))

    raise AssertionError("unreachable")


if __name__ == "__main__":
    import json

    result = search_serp("Python web scraping", num=3)
    print(f"status={result['status']}  count={len(result['output'])}")
    print(json.dumps(result["output"], indent=2, ensure_ascii=False)[:1000])

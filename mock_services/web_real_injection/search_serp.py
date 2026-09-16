"""Search SERP through yibuapi's Serper-compatible endpoint.

The public ``search_serp`` return shape stays compatible with the injected
web service while the upstream request uses POST JSON plus Bearer auth.
"""

from __future__ import annotations

import os
import random
import re
import time
import requests

from claw_eval.api_health import classify_failure

SERP_API_URL = os.getenv("SERP_API_URL", "https://yibuapi.com/serper/search")
SERP_DEV_KEY = (
    os.getenv("SERP_API_KEY")
    or os.getenv("SERP_DEV_KEY")
    or os.getenv("YIBUAPI_KEY", "")
)
SERP_MAX_ATTEMPTS = 3


def _detect_language(query: str) -> tuple[str, str]:
    if re.search(r"[\u4e00-\u9fff]", query):
        return "zh", "cn"
    return "en", "us"


def _record_serp_request(
    query: str, status: int, attempt: int, error: str | None = None,
) -> None:
    """Record one Serper attempt without making logging affect search."""
    try:
        from claw_eval.serp_log import log_serp_request

        log_serp_request(
            query=query, status=status, attempt=attempt, error=error,
        )
    except Exception:
        pass


def _retry_delay(attempt: int, response: requests.Response | None = None) -> float:
    """Return a bounded Retry-After or exponential delay with jitter."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), 30.0)
            except ValueError:
                pass
    return min(2 ** (attempt - 1), 8) + random.uniform(0.0, 0.5)


def _retryable_status(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def search_serp(
    query: str,
    timeout: int = 20,
    num: int = 10,
    start: int = 1,
    raw_save_path: str | None = None,
) -> dict:
    """Search Google via SERP API and return extracted results.

    Args:
        query: Search query string.
        timeout: Request timeout in seconds.
        num: Number of results (1-10).
        start: 1-based result offset.

    Returns:
        dict with keys:
            status (int): HTTP status code, or -1 on error.
            output (list[dict]): List of result dicts with keys:
                title, link, snippet, date, query.
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
    last_error = ""
    last_status = -1
    for attempt in range(1, SERP_MAX_ATTEMPTS + 1):
        resp: requests.Response | None = None
        try:
            resp = requests.post(
                SERP_API_URL, headers=headers, json=body, timeout=timeout,
            )
            last_status = resp.status_code
            if resp.status_code != 200:
                last_error = resp.text[:300]
                _record_serp_request(query, resp.status_code, attempt, last_error)
                if classify_failure(resp.status_code, last_error) == "permanent":
                    return {
                        "status": resp.status_code,
                        "output": [],
                        "error": last_error,
                    }
                if (
                    _retryable_status(resp.status_code)
                    and attempt < SERP_MAX_ATTEMPTS
                ):
                    time.sleep(_retry_delay(attempt, resp))
                    continue
                return {
                    "status": resp.status_code,
                    "output": [],
                    "error": last_error,
                }

            try:
                data = resp.json()
            except Exception as exc:
                last_status = -1
                last_error = f"Invalid JSON response: {exc}"[:300]
                _record_serp_request(query, resp.status_code, attempt, last_error)
                if attempt < SERP_MAX_ATTEMPTS:
                    time.sleep(_retry_delay(attempt, resp))
                    continue
                return {"status": -1, "output": [], "error": last_error}

            _record_serp_request(query, resp.status_code, attempt)
            if raw_save_path:
                os.makedirs(os.path.dirname(raw_save_path) or ".", exist_ok=True)
                with open(raw_save_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
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
        except Exception as exc:
            last_status = -1
            last_error = str(exc)[:300]
            _record_serp_request(query, -1, attempt, last_error)
            if attempt < SERP_MAX_ATTEMPTS:
                time.sleep(_retry_delay(attempt, resp))

    return {"status": last_status, "output": [], "error": last_error}


if __name__ == "__main__":
    import json

    result = search_serp("Python web scraping", num=3)
    print(f"status={result['status']}  count={len(result['output'])}")
    print(json.dumps(result["output"], indent=2, ensure_ascii=False)[:1000])

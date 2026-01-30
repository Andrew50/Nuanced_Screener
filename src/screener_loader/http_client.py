from __future__ import annotations

import threading
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .rate_limit import FixedIntervalRateLimiter

_local = threading.local()

_rate_lock = threading.Lock()
_host_limiters: dict[str, FixedIntervalRateLimiter] = {}


def configure_host_rate_limit(host: str, calls_per_minute: int) -> None:
    """
    Configure a shared per-host rate limiter used by all HTTP sessions.

    This enforces limits across different code paths/stages and also across
    retries performed inside the HTTP adapter.
    """
    h = str(host).strip().lower()
    if not h:
        raise ValueError("host must be non-empty")
    cpm = int(calls_per_minute)
    if cpm <= 0:
        raise ValueError("calls_per_minute must be > 0")
    with _rate_lock:
        limiter = _host_limiters.get(h)
        if limiter is None or limiter.calls_per_minute != cpm:
            _host_limiters[h] = FixedIntervalRateLimiter(calls_per_minute=cpm)


def _limiter_for_url(url: str) -> FixedIntervalRateLimiter | None:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return None
    if not host:
        return None
    with _rate_lock:
        return _host_limiters.get(host)


class _RateLimitedHTTPAdapter(HTTPAdapter):
    def send(self, request, **kwargs):  # type: ignore[override]
        limiter = _limiter_for_url(getattr(request, "url", "") or "")
        if limiter is not None:
            limiter.wait()
        return super().send(request, **kwargs)


def _build_session() -> requests.Session:
    # Retries here are small; the outer update loop also retries per ticker.
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = _RateLimitedHTTPAdapter(
        pool_connections=32,
        pool_maxsize=32,
        max_retries=retry,
    )

    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def get_thread_local_session() -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = _build_session()
        _local.session = s
    return s


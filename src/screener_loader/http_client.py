from __future__ import annotations

import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_local = threading.local()


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
    adapter = HTTPAdapter(
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


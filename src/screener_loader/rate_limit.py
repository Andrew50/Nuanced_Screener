from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field


@dataclass
class FixedIntervalRateLimiter:
    """
    Enforces a minimum interval between calls (good for simple per-minute limits).
    """

    calls_per_minute: int
    _next_allowed_ts: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be > 0")

    @property
    def interval_seconds(self) -> float:
        return 60.0 / float(self.calls_per_minute)

    def wait(self) -> None:
        # Thread-safe: multiple concurrent callers share the same schedule.
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed_ts:
                time.sleep(self._next_allowed_ts - now)
            self._next_allowed_ts = time.monotonic() + self.interval_seconds


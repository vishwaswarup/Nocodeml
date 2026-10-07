"""Request rate limiting (sliding window, in memory).

Counts requests per key (a user id, or a client IP before sign-in) inside a rolling time window.
In-process by design: correct for a single API process. With several API processes each keeps its own
counts, so the effective limit is multiplied; move the counters to Redis at that point.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from typing import Callable

MAX_KEYS = 10_000


class RateLimited(Exception):
    def __init__(self, retry_after: int):
        super().__init__(f"Too many requests. Try again in {retry_after}s.")
        self.retry_after = retry_after


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._hits: OrderedDict[tuple[str, str], deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, bucket: str, key: str, limit: int, window: float) -> None:
        """Record one request; raise RateLimited if `key` already made `limit` requests in `window` seconds."""
        now = self._clock()
        k = (bucket, key)
        with self._lock:
            q = self._hits.pop(k, None)
            if q is None:
                q = deque()
            while q and q[0] <= now - window:
                q.popleft()
            if len(q) >= limit:
                self._hits[k] = q
                raise RateLimited(max(1, math.ceil(q[0] + window - now)))
            q.append(now)
            self._hits[k] = q                      # most recently used goes last
            while len(self._hits) > MAX_KEYS:      # bound memory: forget the least recently seen keys
                self._hits.popitem(last=False)

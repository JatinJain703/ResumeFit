"""In-memory store of scored batches, so /rerank can apply new weights and filters without re-scoring.

Kept per server process, newest first; the oldest are dropped beyond `max_items` or after `ttl` seconds.
Nothing is written to disk.
"""
from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from typing import Any


class SessionStore:
    def __init__(self, max_items: int = 20, ttl: float = 3600.0):
        self.max_items, self.ttl = max(1, max_items), ttl
        self._items: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def _expire(self, now: float):
        for key in [k for k, (t, _) in self._items.items() if now - t > self.ttl]:
            del self._items[key]
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def put(self, value: Any) -> str:
        key = secrets.token_urlsafe(12)
        with self._lock:
            now = time.time()
            self._items[key] = (now, value)
            self._expire(now)
        return key

    def get(self, key: str) -> Any | None:
        """The stored value (and refresh its lifetime), or None if unknown or expired."""
        with self._lock:
            now = time.time()
            self._expire(now)
            item = self._items.get(key)
            if item is None:
                return None
            self._items[key] = (now, item[1])
            self._items.move_to_end(key)
            return item[1]

    def __len__(self) -> int:
        return len(self._items)

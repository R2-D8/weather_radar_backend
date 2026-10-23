"""A small in-memory cache of rendered tiles, evicting the least recently used when full."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Hashable


class TileCache:
    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._items: OrderedDict[Hashable, bytes] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> bytes | None:
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, key: Hashable, value: bytes) -> None:
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._size -= len(old)
            self._items[key] = value
            self._size += len(value)
            while self._size > self._max_bytes and self._items:
                _, evicted = self._items.popitem(last=False)
                self._size -= len(evicted)

    @property
    def size_bytes(self) -> int:
        return self._size

    def __len__(self) -> int:
        return len(self._items)

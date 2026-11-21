"""Per-operation LRU for decoded detector frames and empirical templates."""
from collections import OrderedDict


class ByteLRU:
    def __init__(self, max_entries, max_bytes):
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("Cache limits must be positive")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = 0
        self.hits = self.loads = self.evictions = self.oversize = 0

    def get(self, key, load, size_bytes):
        if key in self.entries:
            self.entries.move_to_end(key)
            self.hits += 1
            return self.entries[key][0]
        value = load()
        self.loads += 1
        size = size_bytes(value)
        if size > self.max_bytes:
            self.oversize += 1
            return value
        while len(self.entries) >= self.max_entries or self.bytes + size > self.max_bytes:
            _, (_, removed_bytes) = self.entries.popitem(last=False)
            self.evictions += 1
            self.bytes -= removed_bytes
        self.entries[key] = (value, size)
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        return value

    def statistics(self):
        return {"hits": self.hits, "loads": self.loads, "evictions": self.evictions,
                "oversize_loads": self.oversize, "current_bytes": self.bytes,
                "peak_cached_bytes": self.peak_bytes, "max_bytes": self.max_bytes}

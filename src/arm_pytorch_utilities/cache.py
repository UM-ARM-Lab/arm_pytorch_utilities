import os
import torch


class LocalCache:
    def __init__(self, filename, directory="."):
        self._cache = None
        self._dir = directory
        # cache the results and load it if we've cached them before
        self._cache_path = os.path.join(self._dir, filename)
        if os.path.exists(self._cache_path):
            self._cache = torch.load(self._cache_path)
        else:
            self._cache = {}

    def save(self):
        os.makedirs(self._dir, exist_ok=True)
        torch.save(self._cache, os.path.join(self._dir, self._cache_path))

    def items(self):
        return self._cache.items()

    def __contains__(self, item):
        return item in self._cache

    def __getitem__(self, item):
        return self._cache[item]

    def __setitem__(self, key, value):
        self._cache[key] = value

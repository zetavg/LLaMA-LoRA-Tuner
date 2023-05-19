from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity=5):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key in self.cache:
            # Move the accessed item to the end of the OrderedDict
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key, value):
        if key in self.cache:
            # If the key already exists, update its value
            self.cache[key] = value
        else:
            # If the cache has reached its capacity, remove the least recently used item
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = value

    def clear(self):
        self.cache.clear()

    def prepare_to_set(self):
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)

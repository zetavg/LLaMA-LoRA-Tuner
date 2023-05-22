import pdb
from collections import OrderedDict
import gc

from ..lazy_import import get_torch
# from ..lib.get_device import get_device

from .lru_cache import LRUCache

# device_type = get_device()


class ModelLRUCache(LRUCache):
    # def __init__(self, capacity=5):
    #     self.cache = OrderedDict()
    #     self.capacity = capacity

    def get(self, key):
        if key in self.cache:
            # Move the accessed item to the end of the OrderedDict
            self.cache.move_to_end(key)

            models_did_move = False
            for k, v in self.cache.items():
                m, d = v
                if key != k and m.device.type != 'cpu':
                    models_did_move = True
                    self.cache[k] = (m.to('cpu'), d)

            if models_did_move:
                clear_cache()

            model, data = self.cache[key]

            if (
                model.device.type != data['device'].type or
                    hasattr(model, "model") and
                    model.model.device.type != data['device'].type):
                model = model.to(data['device'].type)

            return model
        return None

    def set(self, key, value):
        if key in self.cache:
            # If the key already exists, update its value
            self.cache[key] = (value, {'device': value.device})
        else:
            # If the cache has reached its capacity, remove the least recently used item
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = (value, {'device': value.device})

    def clear(self):
        self.cache.clear()
        clear_cache()

    def make_space(self):
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)

        models_did_move = False
        for k, v in self.cache.items():
            m, d = v
            if m.device.type != 'cpu':
                models_did_move = True
                self.cache[k] = (m.to('cpu'), d)

        if models_did_move:
            clear_cache()


def clear_cache():
    gc.collect()
    torch = get_torch()
    with torch.no_grad():
        torch.cuda.empty_cache()

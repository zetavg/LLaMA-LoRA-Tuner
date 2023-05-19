from collections import OrderedDict
import gc
import torch
from ..lib.get_device import get_device

device_type = get_device()


class ModelLRUCache:
    def __init__(self, capacity=5):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key in self.cache:
            # Move the accessed item to the end of the OrderedDict
            self.cache.move_to_end(key)

            models_did_move = False
            for k, m in self.cache.items():
                if key != k and m.device.type != 'cpu':
                    models_did_move = True
                    self.cache[k] = m.to('cpu')

            if models_did_move:
                gc.collect()
                # if not shared.args.cpu: # will not be running on CPUs anyway
                with torch.no_grad():
                    torch.cuda.empty_cache()

            model = self.cache[key]

            if (model.device.type != device_type or
                    hasattr(model, "model") and
                    model.model.device.type != device_type):
                model = model.to(device_type)

            return model
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

        models_did_move = False
        for k, m in self.cache.items():
            if m.device.type != 'cpu':
                models_did_move = True
                self.cache[k] = m.to('cpu')

        if models_did_move:
            gc.collect()
            # if not shared.args.cpu: # will not be running on CPUs anyway
            with torch.no_grad():
                torch.cuda.empty_cache()

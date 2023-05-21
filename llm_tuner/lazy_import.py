import importlib

torch = None


def get_torch():
    global torch
    if torch:
        return torch
    touch = importlib.import_module('torch')
    return touch

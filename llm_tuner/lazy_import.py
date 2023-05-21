import importlib

torch = None


def get_torch():
    global torch
    if torch:
        return torch
    torch = importlib.import_module('torch')
    return torch


peft = None


def get_peft():
    global peft
    if peft:
        return peft
    peft = importlib.import_module('peft')
    return peft

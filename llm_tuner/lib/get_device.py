import importlib


def get_device():
    torch = importlib.import_module('torch')
    device ="cpu"
    if torch.cuda.is_available():
        device = "cuda"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass

    return device

import torch


def get_device():
    device ="cpu"
    if torch.cuda.is_available():
        device = "cuda"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass

    return device

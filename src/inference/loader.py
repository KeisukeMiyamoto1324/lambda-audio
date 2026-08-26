import torch


def select_device() -> torch.device:
    # ---------------------------------------------------------
    # Select the fastest available device in the requested order.
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

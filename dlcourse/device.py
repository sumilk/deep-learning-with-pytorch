"""Device selection used by every notebook.

Notebooks never assume CUDA. They call ``get_device()`` once, print the
result, and pass the device explicitly to tensors and modules.
"""

import os
import platform
import sys

import torch

_VALID = ("cpu", "cuda", "mps")


def get_device(prefer=None):
    """Return the best available device, or the one requested.

    Resolution order:
    1. ``prefer`` argument ("cpu", "cuda", or "mps"), if given.
    2. ``DLCOURSE_DEVICE`` environment variable, if set.
    3. CUDA if available, then MPS if available, then CPU.

    A requested device that is unavailable raises an error instead of
    silently falling back, so a "GPU run" never secretly runs on the CPU.
    """
    requested = prefer or os.environ.get("DLCOURSE_DEVICE")
    if requested:
        requested = requested.lower()
        if requested not in _VALID:
            raise ValueError(f"Unknown device {requested!r}; expected one of {_VALID}")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def has_gpu():
    """True if a CUDA or MPS device is available."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def describe_environment(device=None):
    """Print versions and hardware so results can be reproduced and compared."""
    device = device or get_device()
    print(f"Python  {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f" — {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        print()
    print(f"MPS available:  {torch.backends.mps.is_available()}")
    print(f"Selected device: {device}")
    return device

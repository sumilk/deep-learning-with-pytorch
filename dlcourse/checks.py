"""Checks used by exercises.

Each check raises ``AssertionError`` with a message that says what was
expected, so a learner can tell what to fix. ``passed`` prints a short
confirmation once all checks in a cell succeed.
"""

import math

import torch


def passed(name="Check"):
    print(f"✅ {name} passed")


def check(condition, message):
    """Assert ``condition`` with a learner-facing failure message."""
    if not condition:
        raise AssertionError(message)


def check_shape(tensor, expected, name="tensor"):
    """Assert a tensor has exactly the expected shape."""
    check(isinstance(tensor, torch.Tensor), f"{name} should be a torch.Tensor, got {type(tensor).__name__}")
    actual = tuple(tensor.shape)
    check(actual == tuple(expected), f"{name} has shape {actual}, expected {tuple(expected)}")


def check_close(actual, expected, name="value", rtol=1e-5, atol=1e-6):
    """Assert numbers or tensors are close, reporting the largest difference."""
    if isinstance(expected, torch.Tensor):
        check(isinstance(actual, torch.Tensor), f"{name} should be a torch.Tensor, got {type(actual).__name__}")
        check(
            tuple(actual.shape) == tuple(expected.shape),
            f"{name} has shape {tuple(actual.shape)}, expected {tuple(expected.shape)}",
        )
        a = actual.detach().to("cpu", torch.float64)
        e = expected.detach().to("cpu", torch.float64)
        if not torch.allclose(a, e, rtol=rtol, atol=atol, equal_nan=False):
            diff = (a - e).abs().max().item()
            raise AssertionError(f"{name} differs from the expected value (max abs difference {diff:.3g})")
        return
    a, e = float(actual), float(expected)
    if not math.isclose(a, e, rel_tol=rtol, abs_tol=atol):
        raise AssertionError(f"{name} = {a:.6g}, expected {e:.6g}")

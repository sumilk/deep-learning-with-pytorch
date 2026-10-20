"""Synthetic datasets used throughout the course.

Every generator takes a ``seed`` and uses its own ``torch.Generator``, so a
dataset never depends on, or disturbs, the global random state. All return
``X`` with shape (N, D) as float32 and ``y`` with shape (N,) as int64.
"""

import math

import torch


def _generator(seed):
    return torch.Generator().manual_seed(seed)


def make_blobs(n=200, centers=((-1.0, -1.0), (1.0, 1.0)), std=0.5, seed=0):
    """Gaussian clusters, one class per center. ``n`` points per class."""
    gen = _generator(seed)
    centers = torch.tensor(centers, dtype=torch.float32)  # (K, D)
    k, d = centers.shape
    X = centers.repeat_interleave(n, dim=0) + std * torch.randn(k * n, d, generator=gen)
    y = torch.arange(k).repeat_interleave(n)
    return X, y


def make_xor(n=400, noise=0.0, seed=0):
    """Points uniform on [-1, 1]²; class 1 where x₁·x₂ > 0 (first and third quadrants).

    ``noise`` adds Gaussian jitter to the inputs after labelling, so points
    near the axes can land on the "wrong" side.
    """
    gen = _generator(seed)
    X = 2 * torch.rand(n, 2, generator=gen) - 1
    y = (X[:, 0] * X[:, 1] > 0).long()
    X = X + noise * torch.randn(n, 2, generator=gen)
    return X, y


def make_rings(n=400, radii=(0.5, 1.0), noise=0.1, seed=0):
    """Concentric rings, one class per radius. ``n`` points per ring."""
    gen = _generator(seed)
    xs, ys = [], []
    for label, radius in enumerate(radii):
        angle = 2 * math.pi * torch.rand(n, generator=gen)
        r = radius + noise * torch.randn(n, generator=gen)
        xs.append(torch.stack([r * torch.cos(angle), r * torch.sin(angle)], dim=1))
        ys.append(torch.full((n,), label))
    return torch.cat(xs), torch.cat(ys)


def make_spirals(n=300, classes=2, turns=1.5, noise=0.05, seed=0):
    """Interleaved spiral arms, one class per arm. ``n`` points per arm."""
    gen = _generator(seed)
    xs, ys = [], []
    for label in range(classes):
        t = torch.rand(n, generator=gen)  # position along the arm, 0 = center
        angle = 2 * math.pi * (turns * t + label / classes)
        r = 0.1 + 0.9 * t
        arm = torch.stack([r * torch.cos(angle), r * torch.sin(angle)], dim=1)
        xs.append(arm + noise * torch.randn(n, 2, generator=gen))
        ys.append(torch.full((n,), label))
    return torch.cat(xs), torch.cat(ys)


def make_checkerboard(n=800, tiles=4, seed=0):
    """Points uniform on [-1, 1]², labelled by the color of a tiles×tiles checkerboard."""
    gen = _generator(seed)
    X = 2 * torch.rand(n, 2, generator=gen) - 1
    cells = ((X + 1) / 2 * tiles).floor().clamp(0, tiles - 1).long()
    y = (cells[:, 0] + cells[:, 1]) % 2
    return X, y

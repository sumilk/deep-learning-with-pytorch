"""Consistent plot styling for notebooks."""

import matplotlib.pyplot as plt

# Color-blind-safe palette (Okabe–Ito), used in a fixed order across the course.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def setup_style():
    """Apply the course's matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.figsize": (6, 4),
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=COLORS),
            "font.size": 10,
        }
    )


def plot_dataset(X, y, ax=None, title=None, alpha=0.8, s=12):
    """Scatter 2-D points colored by integer class label."""
    import torch

    ax = ax or plt.gca()
    X, y = X.detach().cpu(), y.detach().cpu()
    for label in torch.unique(y).tolist():
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], s=s, alpha=alpha, color=COLORS[label % len(COLORS)],
                   edgecolors="white", linewidths=0.3, label=f"class {label}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return ax


def plot_decision_boundary(predict, X, y, ax=None, title=None, resolution=200, padding=0.3, kind="label"):
    """Shade the plane by a classifier's output and overlay the data.

    ``predict`` maps a (P, 2) float tensor to a (P,) tensor:
    - ``kind="label"``: integer class labels; regions are shaded by class.
    - ``kind="prob"``: the probability of class 1; shaded continuously, with
      the 0.5 contour drawn as the decision boundary.
    """
    import torch
    from matplotlib.colors import ListedColormap

    ax = ax or plt.gca()
    Xc = X.detach().cpu()
    lo = Xc.min(dim=0).values - padding
    hi = Xc.max(dim=0).values + padding
    g1 = torch.linspace(lo[0].item(), hi[0].item(), resolution)
    g2 = torch.linspace(lo[1].item(), hi[1].item(), resolution)
    G1, G2 = torch.meshgrid(g1, g2, indexing="xy")
    grid = torch.stack([G1.reshape(-1), G2.reshape(-1)], dim=1)
    with torch.no_grad():
        Z = predict(grid).detach().cpu().float().reshape(resolution, resolution)
    if kind == "prob":
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("class_prob", [COLORS[0], "#ffffff", COLORS[1]])
        mesh = ax.contourf(G1, G2, Z, levels=torch.linspace(0, 1, 11).tolist(), cmap=cmap, alpha=0.45, vmin=0, vmax=1)
        ax.contour(G1, G2, Z, levels=[0.5], colors="black", linewidths=1.2)
        plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="P(class 1)")
    else:
        n_classes = int(max(Z.max().item(), y.max().item())) + 1
        cmap = ListedColormap(COLORS[:n_classes])
        ax.contourf(G1, G2, Z, levels=[i - 0.5 for i in range(n_classes + 1)], cmap=cmap, alpha=0.25)
        # Outline each class region separately, so boundaries between non-adjacent labels are drawn once.
        for k in range(n_classes):
            inside = (Z == k).float()
            if 0 < inside.mean() < 1:
                ax.contour(G1, G2, inside, levels=[0.5], colors="black", linewidths=1.0)
    plot_dataset(X, y, ax=ax, title=title)
    ax.set_xlim(lo[0].item(), hi[0].item())
    ax.set_ylim(lo[1].item(), hi[1].item())
    return ax

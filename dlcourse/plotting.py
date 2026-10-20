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

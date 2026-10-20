# %% [markdown]
# # 08 — Linear separability
#
# | | |
# | --- | --- |
# | **Prerequisites** | Notebooks 00–07: tensors and broadcasting, dot products and norms, the perceptron and its learning rule |
# | **You will learn** | What it means for a dataset to be linearly separable; how to find weights for AND, OR, and NAND by hand; a proof that XOR is not linearly separable; how many Boolean functions a single threshold unit can compute; the geometric margin; the perceptron convergence theorem with a full proof and a numerical check; what goes wrong on non-separable data and the pocket algorithm; how a feature map can make data separable |
# | **Estimated time** | About 2.5 hours, plus 1–2 hours for the exercises |
# | **Compute** | CPU, well under a minute. No GPU section: every computation here is a few thousand tiny dot products. |

# %% [markdown]
# ## Setup

# %%
# Setup: make the course helpers importable and record the environment.
import sys
from pathlib import Path

try:
    import dlcourse
except ImportError:  # running without `uv sync`, e.g. on Colab after cloning the repo
    root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "dlcourse").is_dir())
    sys.path.insert(0, str(root))
    import dlcourse

import matplotlib.pyplot as plt
import torch

from dlcourse import check, check_close, check_shape, passed, set_seed
from dlcourse.plotting import setup_style

setup_style()
device = dlcourse.describe_environment()
set_seed(0)

# %%
import itertools
import math

from dlcourse.data import make_blobs, make_rings, make_xor
from dlcourse.plotting import COLORS, plot_dataset, plot_decision_boundary

# %% [markdown]
# ## The motivating problem
#
# In notebook 07 the perceptron sometimes stopped after a few epochs with every training point classified correctly, and sometimes kept making mistakes for as long as we let it run. This notebook answers three questions about that behavior:
#
# 1. **When** does the perceptron succeed? (Exactly when a separating line exists.)
# 2. **How fast?** (At most $(R/\gamma)^2$ updates, where $R$ measures the size of the data and $\gamma$ how much room there is between the classes. We prove it.)
# 3. **Which problems can no single linear classifier solve?** (XOR is the classic example, and most Boolean functions of many inputs are like it.)
#
# All teaching code runs on the CPU. Most of it uses `float64` so that counts of mistakes and updates are exact and reproducible.

# %% [markdown]
# ## 1. Linearly separable datasets
#
# Throughout this chapter a dataset is a matrix $X$ of shape $(N, D)$, one example $\mathbf{x}_i \in \mathbb{R}^D$ per row, with labels $y_i \in \{0, 1\}$. The perceptron works with **signed labels** $t_i = 2y_i - 1 \in \{-1, +1\}$, and classifies with the sign of the score
#
# $$z_i = \mathbf{w} \cdot \mathbf{x}_i + b.$$
#
# The example is classified correctly exactly when $z_i$ has the sign of $t_i$, that is when $t_i z_i > 0$.
#
# **Definition.** A dataset is **linearly separable** if there is some weight vector $\mathbf{w} \in \mathbb{R}^D$ and bias $b$ with
#
# $$t_i\,(\mathbf{w} \cdot \mathbf{x}_i + b) > 0 \quad \text{for every } i = 1, \dots, N.$$
#
# Geometrically: some line (in 2-D), plane (in 3-D) or hyperplane (in general) has all class-1 points strictly on one side and all class-0 points strictly on the other. The inequality is **strict**: a point exactly on the boundary does not count as correctly classified.

# %%
def to_signed(y):
    """Labels {0, 1} (int64, shape (N,)) -> signed labels {-1, +1} as float64, shape (N,)."""
    return 2 * y.double() - 1


def separates(w, b, X, t):
    """True if (w, b) classifies every row of X correctly: t_i (w·x_i + b) > 0 for all i."""
    z = X.double() @ w.double() + b      # (N,) scores
    return bool((t * z > 0).all())


# %% [markdown]
# ### Logical functions as datasets
#
# A Boolean function of two inputs is a tiny dataset: four points, the corners of the unit square $\{0,1\}^2$, each with a label 0 or 1. We store the four inputs once, in the order $(0,0), (0,1), (1,0), (1,1)$, and one label vector per function — its **truth table**.

# %%
X_bool = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)  # (4, 2)
truth = {
    "AND":  torch.tensor([0, 0, 0, 1]),
    "OR":   torch.tensor([0, 1, 1, 1]),
    "NAND": torch.tensor([1, 1, 1, 0]),
    "XOR":  torch.tensor([0, 1, 1, 0]),
}
print(" x1 x2 | AND OR NAND XOR")
for i in range(4):
    x1, x2 = X_bool[i].int().tolist()
    print(f"  {x1}  {x2} |  {truth['AND'][i]}   {truth['OR'][i]}   {truth['NAND'][i]}    {truth['XOR'][i]}")

# %% [markdown]
# ### Weights for AND and OR by hand
#
# Take $\mathbf{w} = (1, 1)$, so the score is $x_1 + x_2 + b$. On the four corners $x_1 + x_2$ takes the values $0, 1, 1, 2$.
#
# - **AND** is 1 only when $x_1 + x_2 = 2$. Any threshold strictly between 1 and 2 works; the midpoint gives $b = -1.5$, and the scores are $-1.5, -0.5, -0.5, +0.5$.
# - **OR** is 1 when $x_1 + x_2 \ge 1$. A threshold between 0 and 1: $b = -0.5$, scores $-0.5, +0.5, +0.5, +1.5$.
#
# Many other choices work: any positive multiple of a solution is again a solution, and the threshold can move anywhere inside its gap.

# %%
hand_weights = {
    "AND": (torch.tensor([1.0, 1.0], dtype=torch.float64), -1.5),
    "OR":  (torch.tensor([1.0, 1.0], dtype=torch.float64), -0.5),
}
for name, (w, b) in hand_weights.items():
    t = to_signed(truth[name])
    z = X_bool @ w + b                                   # (4,)
    print(f"{name:4s} scores z = {z.tolist()}   t·z = {(t * z).tolist()}   separates: {separates(w, b, X_bool, t)}")

# %% [markdown]
# ### Checkpoint 1.1
#
# Find weights for **NAND** ("not AND": 0 only when both inputs are 1). Store a weight tensor `w_nand` of shape `(2,)` (dtype `float64`) and a number `b_nand`. Reason it out first: NAND is the opposite of AND on every corner.

# %% [markdown] tags=["hint"]
# If $(\mathbf{w}, b)$ separates AND, then $(-\mathbf{w}, -b)$ flips the sign of every score, so it gives the opposite label everywhere.

# %%
# BEGIN SOLUTION
w_nand = torch.tensor([-1.0, -1.0], dtype=torch.float64)
b_nand = 1.5
# END SOLUTION
print("NAND scores:", (X_bool @ w_nand + b_nand).tolist())

# %% tags=["check"]
check_shape(w_nand, (2,), "w_nand")
check(separates(w_nand, b_nand, X_bool, to_signed(truth["NAND"])),
      "(w_nand, b_nand) must give t·z > 0 on all four NAND examples")
passed("Checkpoint 1.1")

# %% [markdown]
# Now we draw each function with a separating line. We shade the plane by the classifier's decision with `plot_decision_boundary`; the black line is $\mathbf{w}\cdot\mathbf{x} + b = 0$. For XOR we have no weights, so we draw the OR line as a (failed) attempt.

# %%
plot_weights = {
    "AND": hand_weights["AND"],
    "OR": hand_weights["OR"],
    "NAND": (torch.tensor([-1.0, -1.0], dtype=torch.float64), 1.5),
    "XOR (OR's line: a failed attempt)": hand_weights["OR"],
}
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, (name, (w, b)) in zip(axes, plot_weights.items()):
    y = truth[name.split()[0]]
    plot_decision_boundary(lambda G, w=w, b=b: (G.double() @ w + b > 0).long(), X_bool, y, ax=ax, padding=0.5)
    ax.scatter(X_bool[:, 0], X_bool[:, 1], s=140, c=[COLORS[k] for k in y.tolist()], edgecolors="k", zorder=3)
    ax.set_title(f"{name}\nw = {w.tolist()}, b = {b}", fontsize=9)
    ax.get_legend() and ax.get_legend().remove()
fig.suptitle("Two-input logical functions: blue = class 0, orange = class 1; shading = predicted class")
plt.tight_layout()
plt.show()

# %% [markdown]
# AND, OR, and NAND each have a line with the two colors on opposite sides. For XOR the class-1 corners $(0,1)$ and $(1,0)$ sit on one diagonal and the class-0 corners on the other; OR's line gets three corners right but puts $(1,1)$ on the wrong side. No amount of sliding or rotating seems to help. The next section proves that it can't.

# %% [markdown]
# ## 2. XOR is not linearly separable
#
# ### The proof
#
# Suppose some $(w_1, w_2, b)$ did separate XOR. With $t = 2y - 1$, the four conditions $t_i z_i > 0$ read:
#
# | corner | label $y$ | $t$ | condition $t\,z > 0$ |
# | --- | --- | --- | --- |
# | $(0, 0)$ | 0 | $-1$ | $b < 0$ |
# | $(0, 1)$ | 1 | $+1$ | $w_2 + b > 0$ |
# | $(1, 0)$ | 1 | $+1$ | $w_1 + b > 0$ |
# | $(1, 1)$ | 0 | $-1$ | $w_1 + w_2 + b < 0$ |
#
# Add the two middle inequalities (both "$> 0$"): $w_1 + w_2 + 2b > 0$.
#
# Add the first and the last (both "$< 0$"): $w_1 + w_2 + 2b < 0$.
#
# The same number $w_1 + w_2 + 2b$ cannot be both positive and negative. So no $(w_1, w_2, b)$ satisfies all four conditions: **XOR is not linearly separable.** $\blacksquare$
#
# The heart of the argument is that the two diagonals have the same "sum": $(0,1) + (1,0) = (0,0) + (1,1)$. Any affine score $z(\mathbf{x}) = \mathbf{w}\cdot\mathbf{x} + b$ therefore has $z(0,1) + z(1,0) = z(0,0) + z(1,1)$, so it cannot be positive on one diagonal and negative on the other. We check that identity numerically for a few random $(\mathbf{w}, b)$:

# %%
gen = torch.Generator().manual_seed(0)
for _ in range(3):
    w = torch.randn(2, generator=gen, dtype=torch.float64)
    b = torch.randn((), generator=gen, dtype=torch.float64)
    z = X_bool @ w + b                                     # (4,): scores at 00, 01, 10, 11
    print(f"z(01) + z(10) = {(z[1] + z[2]).item():+.6f}    z(00) + z(11) = {(z[0] + z[3]).item():+.6f}")

# %% [markdown]
# ### Checkpoint 2.1
#
# Evaluate the four XOR conditions for OR's line, $\mathbf{w} = (1, 1)$, $b = -0.5$. Store the four values $t_i z_i$ in `xor_tz` (shape `(4,)`, same corner order as `X_bool`) and the number of satisfied conditions ($t_i z_i > 0$) as a Python int `n_satisfied`. Predict both on paper first.

# %% [markdown] tags=["hint"]
# `t_xor = to_signed(truth["XOR"])`, scores `X_bool @ w + b`, then multiply elementwise. `(xor_tz > 0).sum().item()` counts the satisfied ones.

# %%
w_or, b_or = hand_weights["OR"]
# BEGIN SOLUTION
t_xor = to_signed(truth["XOR"])
xor_tz = t_xor * (X_bool @ w_or + b_or)
n_satisfied = int((xor_tz > 0).sum().item())
# END SOLUTION
print("t·z =", xor_tz.tolist(), "  satisfied:", n_satisfied)

# %% tags=["check"]
check_shape(xor_tz, (4,), "xor_tz")
check_close(xor_tz, torch.tensor([0.5, 0.5, 0.5, -1.5], dtype=torch.float64), "xor_tz")
check(n_satisfied == 3, f"n_satisfied should be 3, got {n_satisfied}")
passed("Checkpoint 2.1")

# %% [markdown]
# ### A reference perceptron
#
# To watch the perceptron on XOR we need a training routine. Here is the one from notebook 07, written compactly. It uses **augmented inputs**: append a constant 1 to every example, $\tilde{\mathbf{x}}_i = (\mathbf{x}_i, 1) \in \mathbb{R}^{D+1}$, and fold the bias into the weights, $\tilde{\mathbf{w}} = (\mathbf{w}, b)$, so that $\tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}}_i = \mathbf{w}\cdot\mathbf{x}_i + b$.
#
# Starting from $\tilde{\mathbf{w}} = \mathbf{0}$, it visits the examples one at a time. Whenever an example is misclassified, $t_i\,\tilde{\mathbf{w}}\cdot\tilde{\mathbf{x}}_i \le 0$ (note "$\le$": a score of exactly 0 counts as a mistake), it updates
#
# $$\tilde{\mathbf{w}} \leftarrow \tilde{\mathbf{w}} + \eta\, t_i\, \tilde{\mathbf{x}}_i.$$
#
# One pass over the data is an **epoch**. Training stops after the first epoch with no updates — at that point every example is classified correctly — or after `epochs` epochs. By default the examples are visited in their stored order; with a `seed`, each epoch uses a fresh random order.

# %%
def augment(X):
    """(N, D) -> (N, D + 1) float64: append a column of ones, so the bias becomes a weight."""
    X = X.double()
    return torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64)], dim=1)


def perceptron_train(X, t, epochs=100, eta=1.0, seed=None):
    """Perceptron on augmented inputs, from w = 0.

    X: (N, D), t: (N,) in {-1, +1}. Returns a dict with
      w         (D + 1,) final augmented weights (w_1..w_D, b)
      converged True if some epoch made no update (all examples correct)
      updates   total number of updates
      mistakes  list: number of updates made in each epoch
      history   (updates + 1, D + 1) the weights after every update (row 0 is the start)
    """
    Xa = augment(X)                                    # (N, D + 1)
    t = t.double()
    w = torch.zeros(Xa.shape[1], dtype=torch.float64)  # (D + 1,)
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    history, mistakes = [w.clone()], []
    for _ in range(epochs):
        order = torch.randperm(len(t), generator=gen).tolist() if gen is not None else range(len(t))
        m = 0
        for i in order:
            if t[i] * (w @ Xa[i]) <= 0:                # mistake (or exactly on the boundary)
                w = w + eta * t[i] * Xa[i]
                history.append(w.clone())
                m += 1
        mistakes.append(m)
        if m == 0:
            break
    return {"w": w, "converged": mistakes[-1] == 0, "updates": len(history) - 1,
            "mistakes": mistakes, "history": torch.stack(history)}


for name in ["AND", "OR", "NAND"]:
    res = perceptron_train(X_bool, to_signed(truth[name]))
    print(f"{name:4s}: converged={res['converged']}  updates={res['updates']:2d}  epochs={len(res['mistakes'])}  w̃ = {res['w'].tolist()}")

# %% [markdown]
# On AND, OR, and NAND the perceptron finds a separator within 6–9 epochs — with different weights from ours, but any separator will do.
#
# ### The perceptron on XOR cycles forever
#
# Now XOR, in the fixed order, for 8 epochs. We print every weight vector it visits.

# %%
res_xor = perceptron_train(X_bool, to_signed(truth["XOR"]), epochs=8)
print("converged:", res_xor["converged"], "   updates per epoch:", res_xor["mistakes"])
for k, w in enumerate(res_xor["history"][:9]):
    print(f"after update {k}: w̃ = {w.tolist()}")
period = 4
H = res_xor["history"]
print(f"history[k] == history[k + {period}] for every k:", torch.equal(H[:-period], H[period:]))

# %% [markdown]
# Every example is a mistake in every epoch, and after each epoch the weights are back at exactly $\tilde{\mathbf{w}} = (0, 0, 0)$: the four updates $-\tilde{\mathbf{x}}_{00} + \tilde{\mathbf{x}}_{01} + \tilde{\mathbf{x}}_{10} - \tilde{\mathbf{x}}_{11}$ sum to zero — the same diagonal identity as in the proof. The sequence repeats with period 4 for as long as we run it.
#
# Does a random visiting order escape? We run 1000 epochs with a fresh random order each epoch.

# %%
res_xor_rand = perceptron_train(X_bool, to_signed(truth["XOR"]), epochs=1000, seed=0)
H = res_xor_rand["history"]
distinct = torch.unique(H, dim=0)
print("converged:", res_xor_rand["converged"])
print("fewest updates in any epoch:", min(res_xor_rand["mistakes"]))
print("largest ‖w̃‖ over all", res_xor_rand["updates"], "updates:", H.norm(dim=1).max().item())
print("number of distinct weight vectors visited:", len(distinct))

# %% [markdown]
# Some epochs now make only one update, but none makes zero — an epoch without updates would mean all four conditions hold, which we proved impossible. The weights stay small: with $\eta = 1$ and integer inputs they are integer vectors, and here they never get longer than $\sqrt 6 \approx 2.45$, so over 3333 updates the perceptron just revisits the same 47 vectors again and again. That the weights stay bounded on non-separable data is a general fact (the *perceptron cycling theorem*, stated here without proof): the perceptron neither converges nor blows up; it keeps moving.
#
# ### Brute force over a grid of weights
#
# As a final sanity check (not a proof — a grid can only ever test finitely many weights), we try every $(w_1, w_2, b)$ on a grid from $-3$ to $3$ in steps of $0.1$ — about 227 thousand combinations — and count how many of the four XOR corners each gets right. Broadcasting does all of them at once: weights `(M, 2)` times inputs `(4, 2)` gives scores `(M, 4)`.

# %%
grid = torch.linspace(-3, 3, 61, dtype=torch.float64)
W1, W2, B = torch.meshgrid(grid, grid, grid, indexing="ij")
W_grid = torch.stack([W1.reshape(-1), W2.reshape(-1)], dim=1)   # (M, 2)
B_grid = B.reshape(-1)                                            # (M,)
Z = W_grid @ X_bool.T + B_grid[:, None]                           # (M, 4) scores at the 4 corners
n_correct = (to_signed(truth["XOR"]) * Z > 0).sum(dim=1)          # (M,)
counts = torch.bincount(n_correct, minlength=5)
print("weight combinations tried:", len(B_grid))
for k in range(5):
    print(f"  {k} of 4 correct: {counts[k].item():7d}")

# %%
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(range(5), counts.tolist(), color=COLORS[0])
ax.set_xticks(range(5))
ax.set_xlabel("number of XOR corners classified correctly")
ax.set_ylabel("number of grid weights $(w_1, w_2, b)$")
ax.set_title("XOR: brute-force search over 61³ weight combinations")
plt.show()

# %% [markdown]
# About 31 thousand weight combinations get 3 of the 4 corners right, but the bar at 4 is empty, exactly as the proof says. (The bar at 0 is empty too. That is no accident: if $(\mathbf{w}, b)$ got all four corners strictly wrong, then $(-\mathbf{w}, -b)$ would get all four right.)

# %% [markdown]
# ## 3. Counting: how many Boolean functions are linearly separable?
#
# With two inputs there are 4 corners, each labelled 0 or 1, so there are $2^4 = 16$ Boolean functions. We number them $k = 0, \dots, 15$ by reading the truth table as a binary number: bit $i$ of $k$ is the label of corner $i$. AND has labels $(0,0,0,1)$ — only bit 3 is set — so it is $k = 2^3 = 8$; XOR has labels $(0,1,1,0)$, so $k = 2^1 + 2^2 = 6$.
#
# Which ones are separable? For each function we run the perceptron with a cap of 100 epochs.

# %%
def labels_of(k, n_corners):
    """Truth table of Boolean function number k: bit i of k is the label of corner i. Returns (n_corners,) int64."""
    return torch.tensor([(k >> i) & 1 for i in range(n_corners)])


two_input = []
for k in range(16):
    y = labels_of(k, 4)
    res = perceptron_train(X_bool, to_signed(y), epochs=100)
    two_input.append((k, y, res))
n_conv = sum(res["converged"] for _, _, res in two_input)
print(f"perceptron converged on {n_conv} of 16 functions")
print("did not converge:", [(k, y.tolist()) for k, y, res in two_input if not res["converged"]])

# %% [markdown]
# Fourteen converge. The two that do not are $k = 6$, XOR $(0,1,1,0)$, and $k = 9$, its negation XNOR $(1,0,0,1)$ ("equal"). Is the count **exactly** 14? Both directions are now proved:
#
# - **At least 14 are separable**: for each of those 14 the perceptron *found* a separator, and a found separator is a certificate (we can check it with `separates`).
# - **At most 14**: XOR is not separable (Section 2). XNOR is not either — if $(\mathbf{w}, b)$ separated XNOR, then $(-\mathbf{w}, -b)$ would separate XOR.
#
# Note the asymmetry: when the perceptron *does not* converge within a cap, that alone proves nothing — it might just need more epochs. Non-separability needs an argument.

# %%
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
xs_line = torch.tensor([-0.5, 1.5], dtype=torch.float64)
for ax, (k, y, res) in zip(axes.flat, two_input):
    w1, w2, b = res["w"].tolist()
    ax.scatter(X_bool[:, 0], X_bool[:, 1], s=110, c=[COLORS[v] for v in y.tolist()], edgecolors="k", zorder=3)
    if not res["converged"]:
        ax.set_title(f"k={k} {''.join(map(str, y.tolist()))}: not separable", fontsize=9, color=COLORS[1])
    else:
        assert separates(res["w"][:2], b, X_bool, to_signed(y))
        if abs(w2) > 1e-12:
            ax.plot(xs_line, -(w1 * xs_line + b) / w2, color="k")
        elif abs(w1) > 1e-12:
            ax.axvline(-b / w1, color="k")
        ax.set_title(f"k={k} {''.join(map(str, y.tolist()))}", fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
for ax in axes[-1]:
    ax.set_xlabel("$x_1$")
for ax in axes[:, 0]:
    ax.set_ylabel("$x_2$")
fig.suptitle("All 16 Boolean functions of two inputs, with the perceptron's separating line\n"
             "(title: k and its labels at corners 00, 01, 10, 11; orange = 1)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Every panel except XOR and XNOR has a black line with the orange corners on one side. Two separable functions have no line in the plot: $k = 0$ (always 0) and $k = 15$ (always 1) are separated by $\mathbf{w} = \mathbf{0}$ and a bias of the right sign — the "line" is at infinity.
#
# ### More inputs
#
# With $n$ inputs there are $2^n$ corners of the cube $\{0,1\}^n$ and $2^{2^n}$ Boolean functions. How many are linearly separable (also called **threshold functions**)? Searching a grid of weights gives a **lower bound**: every distinct labelling a grid weight produces is a separable function. We use integer weights $w_j \in \{-m, \dots, m\}$ and half-integer biases, so a score is never exactly 0.

# %%
def threshold_functions_on_grid(n, m):
    """Distinct labellings of {0,1}^n produced by integer weights in [-m, m]^n and half-integer biases."""
    corners = torch.tensor(list(itertools.product([0.0, 1.0], repeat=n)))           # (2^n, n)
    ws = torch.tensor(list(itertools.product(range(-m, m + 1), repeat=n)), dtype=torch.float32)  # (W, n)
    bs = torch.arange(-n * m - 0.5, n * m + 1.0, 1.0)                                 # (Bn,) half-integers
    W = ws.repeat_interleave(len(bs), dim=0)                                          # (W·Bn, n)
    Bv = bs.repeat(len(ws))                                                           # (W·Bn,)
    labels = (corners @ W.T + Bv > 0).long()                                          # (2^n, W·Bn)
    codes = (labels * 2 ** torch.arange(2**n)[:, None]).sum(dim=0)                    # function number k
    return torch.unique(codes)


for n, m in [(2, 2), (3, 2), (4, 3)]:
    found = threshold_functions_on_grid(n, m)
    print(f"n = {n}: grid with weights in [-{m}, {m}] finds {len(found):5d} separable functions out of 2^{2**n} = {2 ** 2**n}")

# %% [markdown]
# The grid finds 14 for $n = 2$, which we just proved is exact. For $n = 3$ and $n = 4$ it finds 104 and 1882. These numbers are lower bounds from our computation; that they are also the exact counts is a **known result** (the counts of threshold functions, sequence A000609 in the OEIS), stated here without proof. The known counts continue: 94,572 for $n = 5$ and 15,028,134 for $n = 6$.

# %%
known = {1: 4, 2: 14, 3: 104, 4: 1882, 5: 94572, 6: 15028134}
ns = list(known)
frac = [known[n] / 2 ** (2**n) for n in ns]
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.semilogy(ns, frac, "o-", color=COLORS[0])
for n, f in zip(ns, frac):
    ax.annotate(f"{known[n]:,} / 2^{2**n}", (n, f), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("number of inputs $n$")
ax.set_ylabel("fraction of Boolean functions that are separable (log scale)")
ax.set_title("Linearly separable Boolean functions (n ≤ 4 verified above, n = 5, 6 quoted)")
ax.set_xlim(0.8, 7.2)
plt.show()

# %% [markdown]
# All 4 functions of one input are separable, 87.5% of two-input functions, 41% for three inputs, under 3% for four, and about $10^{-12}$ for six. A single threshold unit can represent only a vanishing fraction of what is possible.
#
# ### Checkpoint 3.1
#
# Two famous three-input functions: **majority** (1 when at least two inputs are 1) and **parity** (1 when an odd number of inputs are 1, a three-input XOR). Using `perceptron_train` with `epochs=100` on the 8 corners `X3`, store whether it converged on each in `majority_converged` and `parity_converged` (Python bools).

# %% [markdown] tags=["hint"]
# The labels are `(X3.sum(dim=1) >= 2).long()` and `(X3.sum(dim=1) % 2).long()`. Convert with `to_signed` and read `["converged"]` from the result.

# %%
X3 = torch.tensor(list(itertools.product([0.0, 1.0], repeat=3)), dtype=torch.float64)  # (8, 3)
# BEGIN SOLUTION
y_maj = (X3.sum(dim=1) >= 2).long()
y_par = (X3.sum(dim=1) % 2).long()
majority_converged = perceptron_train(X3, to_signed(y_maj), epochs=100)["converged"]
parity_converged = perceptron_train(X3, to_signed(y_par), epochs=100)["converged"]
# END SOLUTION
print("majority:", majority_converged, "   parity:", parity_converged)

# %% tags=["check"]
check(majority_converged is True, "majority is separable (w = (1, 1, 1), b = -1.5), so the perceptron must converge")
check(parity_converged is False, "parity should not converge: think about what it does on the face x3 = 0")
passed("Checkpoint 3.1")

# %% [markdown]
# Majority is separated by $\mathbf{w} = (1,1,1)$, $b = -1.5$. For parity, non-convergence is again not a proof by itself, but here is one: on the four corners with $x_3 = 0$, three-input parity *is* two-input XOR of $x_1, x_2$. If $(w_1, w_2, w_3, b)$ separated parity, then $(w_1, w_2, b)$ would separate XOR on that face, since $w_3 x_3 = 0$ there. Impossible. Parity of any number $n \ge 2$ of inputs fails for the same reason.

# %% [markdown]
# ## 4. The margin
#
# A separable dataset usually has infinitely many separators. Are some better than others? Intuitively, a line that passes close to a training point is fragile: a slightly perturbed point could land on the wrong side.
#
# Notebook 07 showed that the **signed distance** from a point $\mathbf{x}$ to the line $\mathbf{w}\cdot\mathbf{x} + b = 0$ is $(\mathbf{w}\cdot\mathbf{x} + b)/\|\mathbf{w}\|$. Multiplying by $t_i$ makes it positive on the correct side. The **geometric margin** of $(\mathbf{w}, b)$ on the dataset is the smallest of these:
#
# $$\gamma(\mathbf{w}, b) = \min_i \frac{t_i\,(\mathbf{w}\cdot\mathbf{x}_i + b)}{\|\mathbf{w}\|}.$$
#
# It is positive exactly when $(\mathbf{w}, b)$ separates the data, and then it is the width of the empty strip on each side of the boundary. Dividing by $\|\mathbf{w}\|$ makes it a true distance: scaling $(\mathbf{w}, b)$ by any $c > 0$ leaves it unchanged.
#
# ### Checkpoint 4.1
#
# A small dataset: class $+1$ at $(3, 1)$ and $(1, 2)$, class $-1$ at $(0, 0)$ and $(1, 0)$. The separator is $\mathbf{w} = (1, 1)$, $b = -2$. Compute its geometric margin `gamma_small` (a float or 0-d tensor). By hand first: the scores are $2, 1, -2, -1$.

# %% [markdown] tags=["hint"]
# $t_i z_i = 2, 1, 2, 1$; the smallest is 1, and $\|\mathbf{w}\| = \sqrt 2$. In code: `(t_small * (X_small @ w_small + b_small)).min() / w_small.norm()`.

# %%
X_small = torch.tensor([[3.0, 1.0], [1.0, 2.0], [0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)  # (4, 2)
t_small = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64)
w_small, b_small = torch.tensor([1.0, 1.0], dtype=torch.float64), -2.0
# BEGIN SOLUTION
gamma_small = (t_small * (X_small @ w_small + b_small)).min() / w_small.norm()
# END SOLUTION
print(f"geometric margin: {float(gamma_small):.4f}")

# %% tags=["check"]
check_close(gamma_small, 1 / math.sqrt(2), "gamma_small")
passed("Checkpoint 4.1")

# %% [markdown]
# ### Different separators, different margins
#
# Different visiting orders lead the perceptron to different separators. We take two well-separated Gaussian blobs (30 points per class) and train with ten random orders.

# %%
def geometric_margin(w, b, X, t):
    """min_i t_i (w·x_i + b) / ‖w‖ for w (D,), scalar b, X (N, D), t (N,). Negative if any point is misclassified."""
    return ((t * (X.double() @ w.double() + b)).min() / w.double().norm()).item()


X_bl, y_bl = make_blobs(n=30, std=0.4, seed=3)   # (60, 2), (60,)
t_bl = to_signed(y_bl)
runs = []
for seed in range(10):
    res = perceptron_train(X_bl, t_bl, epochs=1000, seed=seed)
    w, b = res["w"][:2], res["w"][2].item()
    runs.append((seed, res, geometric_margin(w, b, X_bl, t_bl)))
    print(f"order seed {seed}: converged={res['converged']}  updates={res['updates']:2d}  "
          f"w = ({w[0].item():6.3f}, {w[1].item():6.3f}), b = {b:5.2f}   margin γ = {runs[-1][2]:.4f}")

# %%
best = max(runs, key=lambda r: r[2])
worst = min(runs, key=lambda r: r[2])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, (seed, res, gamma), label in [(axes[0], best, "largest"), (axes[1], worst, "smallest")]:
    w, b = res["w"][:2], res["w"][2].item()
    plot_decision_boundary(lambda G, w=w, b=b: (G.double() @ w + b > 0).long(), X_bl, y_bl, ax=ax)
    # the margin strip: the lines w·x + b = ±γ‖w‖, parallel to the boundary at distance γ
    g1 = torch.linspace(-3, 3, 2, dtype=torch.float64)
    for s in (-1, 1):
        ax.plot(g1, -(w[0] * g1 + b - s * gamma * w.norm()) / w[1], "--", color="gray")
    d = t_bl * (X_bl.double() @ w + b) / w.norm()       # (N,) distances of every point
    closest = d.argmin()
    ax.scatter(X_bl[closest, 0], X_bl[closest, 1], s=160, facecolors="none", edgecolors="k", linewidths=1.5,
               label="closest point")
    ax.set_title(f"Order seed {seed} ({label} margin): γ = {gamma:.3f}")
    ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# Both lines separate the training data perfectly, so by training accuracy they are equally good. But on the right, the boundary nearly touches one point (circled): the dashed strip is narrow, and a small shift of that point would cross it. On the left, the nearest point is about 40 times farther away (γ = 0.647 against 0.016). The two smallest margins came from runs that made a single update: the perceptron stops at the first separator it reaches, however close to a point it lies.
#
# The perceptron stops as soon as there are no mistakes; it does not try to make the margin large, so which separator you get is an accident of the visiting order. The separator with the **largest** possible margin is the idea behind support vector machines (not covered further in this course). Exercise 2 finds it by brute force in 2-D.

# %% [markdown]
# ## 5. The perceptron convergence theorem
#
# Margins also control **how fast** the perceptron converges. The theorem is stated for augmented inputs $\tilde{\mathbf{x}}_i = (\mathbf{x}_i, 1)$ and augmented weights, because that is what the algorithm updates.
#
# **Theorem (Novikoff, 1962).** Suppose
#
# 1. every augmented input is bounded: $\|\tilde{\mathbf{x}}_i\| \le R$ for all $i$, and
# 2. the data are separable with margin $\gamma > 0$: there is a **unit** vector $\mathbf{u} \in \mathbb{R}^{D+1}$ ($\|\mathbf{u}\| = 1$) with $t_i\,\mathbf{u}\cdot\tilde{\mathbf{x}}_i \ge \gamma$ for all $i$.
#
# Then the perceptron started from $\tilde{\mathbf{w}}_0 = \mathbf{0}$, with any learning rate $\eta > 0$ and **any** order of visiting examples, makes at most
#
# $$k \le \left(\frac{R}{\gamma}\right)^2$$
#
# updates in total. After that, it makes no more mistakes, so the next full epoch has none and training stops.
#
# Notice what the bound does **not** depend on: the number of examples $N$, the dimension $D$, the learning rate, or the order.

# %% [markdown]
# ### The proof, in three steps
#
# Let $\tilde{\mathbf{w}}_k$ be the weights after $k$ updates, with $\tilde{\mathbf{w}}_0 = \mathbf{0}$. Update $k+1$ happens on some example $i$ that was misclassified, $t_i\,\tilde{\mathbf{w}}_k\cdot\tilde{\mathbf{x}}_i \le 0$, and sets $\tilde{\mathbf{w}}_{k+1} = \tilde{\mathbf{w}}_k + \eta\,t_i\,\tilde{\mathbf{x}}_i$.
#
# **Step 1: the alignment with $\mathbf{u}$ grows at least linearly.** Take the dot product of the update with $\mathbf{u}$:
#
# $$\tilde{\mathbf{w}}_{k+1}\cdot\mathbf{u} = \tilde{\mathbf{w}}_k\cdot\mathbf{u} + \eta\,t_i\,\tilde{\mathbf{x}}_i\cdot\mathbf{u} \;\ge\; \tilde{\mathbf{w}}_k\cdot\mathbf{u} + \eta\gamma,$$
#
# by assumption 2. Each update adds at least $\eta\gamma$, and we started from 0, so $\tilde{\mathbf{w}}_k\cdot\mathbf{u} \ge k\,\eta\gamma$.
#
# **Step 2: the squared length grows at most linearly.** Expand the squared norm of the update:
#
# $$\|\tilde{\mathbf{w}}_{k+1}\|^2 = \|\tilde{\mathbf{w}}_k\|^2 + 2\eta\,\underbrace{t_i\,\tilde{\mathbf{w}}_k\cdot\tilde{\mathbf{x}}_i}_{\le 0\ \text{(a mistake)}} + \eta^2 \underbrace{t_i^2}_{=1} \|\tilde{\mathbf{x}}_i\|^2 \;\le\; \|\tilde{\mathbf{w}}_k\|^2 + \eta^2 R^2.$$
#
# The middle term is where "we only update on mistakes" is used. Starting from 0: $\|\tilde{\mathbf{w}}_k\|^2 \le k\,\eta^2 R^2$, i.e. $\|\tilde{\mathbf{w}}_k\| \le \sqrt{k}\,\eta R$.
#
# **Step 3: combine with Cauchy–Schwarz.** From notebook 02, $\mathbf{a}\cdot\mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta \le \|\mathbf{a}\|\|\mathbf{b}\|$ (the **Cauchy–Schwarz inequality**). With $\|\mathbf{u}\| = 1$:
#
# $$k\,\eta\gamma \;\le\; \tilde{\mathbf{w}}_k\cdot\mathbf{u} \;\le\; \|\tilde{\mathbf{w}}_k\| \;\le\; \sqrt{k}\,\eta R.$$
#
# Divide by $\sqrt{k}\,\eta\gamma$ (positive for $k \ge 1$): $\sqrt{k} \le R/\gamma$, so $k \le (R/\gamma)^2$. $\blacksquare$
#
# The picture: every update pushes $\tilde{\mathbf{w}}$ a fixed amount further along $\mathbf{u}$, but its length grows only like $\sqrt{k}$. A vector whose component along $\mathbf{u}$ grows faster than its length must stop growing — the component can never exceed the length. $\eta$ cancels, which is why it does not appear in the bound.

# %% [markdown]
# ### Two kinds of margin
#
# The $\gamma$ in the theorem is measured in the **augmented** space, $\min_i t_i\,\mathbf{u}\cdot\tilde{\mathbf{x}}_i$ with $\|\mathbf{u}\| = 1$ in $\mathbb{R}^{D+1}$. If $\mathbf{u}$ comes from a separator $(\mathbf{w}, b)$, then $\mathbf{u} = (\mathbf{w}, b)/\|(\mathbf{w}, b)\|$ and
#
# $$\gamma_{\text{aug}} = \min_i \frac{t_i(\mathbf{w}\cdot\mathbf{x}_i + b)}{\sqrt{\|\mathbf{w}\|^2 + b^2}} = \gamma_{\text{geo}} \cdot \frac{\|\mathbf{w}\|}{\sqrt{\|\mathbf{w}\|^2 + b^2}} \;\le\; \gamma_{\text{geo}}.$$
#
# They agree when $b = 0$. Likewise $R$ is the largest $\|\tilde{\mathbf{x}}_i\| = \sqrt{\|\mathbf{x}_i\|^2 + 1}$, not the largest $\|\mathbf{x}_i\|$. Mixing the two conventions breaks the bound — Exercise 3 is about exactly that.

# %% [markdown]
# ### Checkpoint 5.1
#
# For the small dataset of Checkpoint 4.1 (`X_small`, `t_small`) and its separator $\tilde{\mathbf{w}} = (1, 1, -2)$, compute: `R_small` (the largest norm of an augmented input), `gamma_aug_small` (the augmented margin of $\mathbf{u} = \tilde{\mathbf{w}}/\|\tilde{\mathbf{w}}\|$), and `bound_small` $= (R/\gamma)^2$. By hand: $R = \|(3,1,1)\| = \sqrt{11}$ and $\gamma = 1/\sqrt 6$.

# %% [markdown] tags=["hint"]
# `Xa = augment(X_small)`; `R_small = Xa.norm(dim=1).max()`; `u = w_aug / w_aug.norm()`; `gamma_aug_small = (t_small * (Xa @ u)).min()`.

# %%
w_aug_small = torch.tensor([1.0, 1.0, -2.0], dtype=torch.float64)
# BEGIN SOLUTION
Xa_small = augment(X_small)                       # (4, 3)
R_small = Xa_small.norm(dim=1).max()
u_small = w_aug_small / w_aug_small.norm()
gamma_aug_small = (t_small * (Xa_small @ u_small)).min()
bound_small = (R_small / gamma_aug_small) ** 2
# END SOLUTION
res_small = perceptron_train(X_small, t_small, epochs=1000)
print(f"R = {float(R_small):.4f}   γ = {float(gamma_aug_small):.4f}   bound = {float(bound_small):.1f}   "
      f"actual updates = {res_small['updates']}")

# %% tags=["check"]
check_close(R_small, math.sqrt(11), "R_small")
check_close(gamma_aug_small, 1 / math.sqrt(6), "gamma_aug_small")
check_close(bound_small, 66.0, "bound_small")
check(res_small["updates"] <= float(bound_small), "the theorem guarantees updates <= bound")
passed("Checkpoint 5.1")

# %% [markdown]
# The theorem allows up to 66 updates; the perceptron needs only a fraction of that. The bound uses one particular $\mathbf{u}$ — any separating unit vector gives a valid bound, and the best possible bound uses the $\mathbf{u}$ with the largest augmented margin.
#
# ### Checking the theorem numerically
#
# We make datasets with a **controlled margin**: take the uniform points on $[-1, 1]^2$ from `make_xor` (ignoring its labels), fix a unit vector $\mathbf{u}^*$ in augmented space, label each point by the sign of $\mathbf{u}^*\cdot\tilde{\mathbf{x}}$, and delete every point with $|\mathbf{u}^*\cdot\tilde{\mathbf{x}}| < \gamma_0$. What remains is separated by $\mathbf{u}^*$ with margin at least $\gamma_0$.

# %%
U_STAR = torch.tensor([1.0, -2.0, 0.3], dtype=torch.float64)
U_STAR = U_STAR / U_STAR.norm()                       # (3,) unit vector in augmented space


def make_margin_data(gamma0, n=400, seed=1):
    """Points uniform on [-1, 1]^2, labelled by sign(u*·x̃), keeping only those with |u*·x̃| >= gamma0."""
    X, _ = make_xor(n=n, seed=seed)                  # only the inputs are used
    s = augment(X) @ U_STAR                          # (n,) signed scores under the planted separator
    keep = s.abs() >= gamma0
    return X[keep].double(), torch.sign(s[keep])


margin_rows = []
for gamma0 in [0.3, 0.1, 0.03, 0.01]:
    Xm, tm = make_margin_data(gamma0)
    Xa = augment(Xm)
    R = Xa.norm(dim=1).max().item()
    gamma = (tm * (Xa @ U_STAR)).min().item()         # actual margin of u* on these points
    res = perceptron_train(Xm, tm, epochs=100_000)
    margin_rows.append((gamma0, len(tm), R, gamma, res))
    print(f"γ0 = {gamma0:5.2f}: N = {len(tm)}  R = {R:.3f}  γ = {gamma:.4f}  bound (R/γ)² = {(R / gamma) ** 2:8.0f}"
          f"   actual updates = {res['updates']:3d}  (converged: {res['converged']}, {len(res['mistakes'])} epochs)")

# %% [markdown]
# On every dataset the actual number of updates is below the bound, as it must be; as the margin shrinks the bound grows like $1/\gamma^2$, and the actual count grows far more slowly — 59 updates against a bound of over 28,000 for the smallest margin. The bound is a worst-case guarantee, over every possible dataset with this $R$ and $\gamma$ and every visiting order; typical data are much easier.
#
# Let us watch the three steps of the proof happen, on the smallest-margin dataset. We plot $\tilde{\mathbf{w}}_k\cdot\mathbf{u}^*$ with its lower bound $k\eta\gamma$, and $\|\tilde{\mathbf{w}}_k\|$ with its upper bound $\sqrt{k}\,\eta R$.

# %%
gamma0, n_m, R, gamma, res = margin_rows[-1]
Hm = res["history"]                                   # (updates + 1, 3)
k = torch.arange(len(Hm), dtype=torch.float64)
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.plot(k, Hm.norm(dim=1), color=COLORS[1], label=r"$\|\tilde{\mathbf{w}}_k\|$")
ax.plot(k, k.sqrt() * R, "--", color=COLORS[1], label=r"upper bound $\sqrt{k}\,\eta R$ (step 2)")
ax.plot(k, Hm @ U_STAR, color=COLORS[0], label=r"$\tilde{\mathbf{w}}_k \cdot \mathbf{u}^*$")
ax.plot(k, k * gamma, "--", color=COLORS[0], label=r"lower bound $k\,\eta\gamma$ (step 1)")
ax.set_xlabel("number of updates $k$")
ax.set_ylabel("value (η = 1)")
ax.set_title(f"The proof's inequalities along a real run (γ = {gamma:.4f}, R = {R:.2f})")
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# Each solid curve stays on the right side of its dashed bound at every step, and the blue curve never rises above the orange one (Cauchy–Schwarz). After the first few updates the two solid curves almost coincide: $\tilde{\mathbf{w}}_k\cdot\mathbf{u}^* \approx \|\tilde{\mathbf{w}}_k\|$ means the weights point almost exactly along $\mathbf{u}^*$. Both dashed bounds are loose: the alignment grows far faster than $k\gamma$, because most points lie much farther than $\gamma$ from the planted line, while the proof must assume that every update is made on a worst-case point.
#
# ### Where the data sit matters
#
# $R$ includes the constant feature, so the bound changes if we simply **translate** the data. Shifting all points by a vector $\mathbf{c}$ keeps them separable with the same geometric margin, but it increases $\|\tilde{\mathbf{x}}_i\|$ and the bias needed, so the augmented margin shrinks. We shift the blobs of Section 4 away from the origin and count updates (averaged over 10 orders).

# %%
for shift in [0.0, 2.0, 5.0, 10.0]:
    Xs = X_bl.double() + shift                       # every point moved by (shift, shift)
    ups = [perceptron_train(Xs, t_bl, epochs=100_000, seed=s)["updates"] for s in range(10)]
    print(f"shift ({shift:4.1f}, {shift:4.1f}):  max ‖x̃‖ = {augment(Xs).norm(dim=1).max().item():6.2f}   "
          f"mean updates = {sum(ups) / len(ups):7.1f}")

# %% [markdown]
# The geometry is identical in every row, but the perceptron needs far more updates when the data are far from the origin: the bias has to be learned through a feature (the constant 1) that is tiny compared with the others. This is one reason to **center** inputs before training, a habit we will keep for every model.

# %% [markdown]
# ## 6. Non-separable data
#
# The theorem assumes a separating $\mathbf{u}$ exists. Real data rarely oblige: classes overlap because of noise. Here are two Gaussian blobs with standard deviation 1, wide enough to overlap.

# %%
X_ov, y_ov = make_blobs(n=100, std=1.0, seed=0)       # (200, 2), (200,)
t_ov = to_signed(y_ov)
EPOCHS_OV = 60
res_ov = perceptron_train(X_ov, t_ov, epochs=EPOCHS_OV, seed=0)
Xa_ov = augment(X_ov)


def error_rate(w_aug, Xa, t):
    """Fraction of rows of the augmented inputs Xa (N, D + 1) with t_i (w̃·x̃_i) <= 0."""
    return (t * (Xa @ w_aug) <= 0).double().mean().item()


# training error of the weights after every update
err_path = torch.tensor([error_rate(w, Xa_ov, t_ov) for w in res_ov["history"]])
ends = torch.tensor(res_ov["mistakes"]).cumsum(0)    # index in history of the weights at the end of each epoch
print("converged:", res_ov["converged"], "   total updates:", res_ov["updates"])
print("updates in the last 10 epochs:", res_ov["mistakes"][-10:])
print("training error at the end of the last 10 epochs:", [round(e, 3) for e in err_path[ends[-10:]].tolist()])

# %% [markdown]
# The perceptron makes mistakes in every epoch and never stops. The weights it holds when we stop are whatever the last few updates made them, and their training error jumps around from epoch to epoch. For comparison: the two blobs have the same spherical spread, so by symmetry the best line is the perpendicular bisector of the two centers, $x_1 + x_2 = 0$ (stated without proof).

# %%
w_bisector = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
print(f"training error of x1 + x2 = 0: {error_rate(w_bisector, Xa_ov, t_ov):.3f}")

# %% [markdown]
# ### The pocket algorithm
#
# A simple practical patch (Gallant, 1990): run the perceptron as usual, but keep "in your pocket" the best weights seen so far, measured by training error, and return those instead of the last ones.

# %%
def pocket_train(X, t, epochs=100, seed=None):
    """Perceptron that also returns the weights with the lowest training error seen after any update.

    Returns (pocket_w (D + 1,), pocket_error, perceptron result dict).
    """
    res = perceptron_train(X, t, epochs=epochs, seed=seed)
    Xa = augment(X)
    errors = torch.tensor([error_rate(w, Xa, t.double()) for w in res["history"]])  # (updates + 1,)
    best = errors.argmin()
    return res["history"][best], errors[best].item(), res


w_pocket, err_pocket, _ = pocket_train(X_ov, t_ov, epochs=EPOCHS_OV, seed=0)
print(f"last perceptron weights: error {err_path[-1].item():.3f}")
print(f"pocket weights:          error {err_pocket:.3f}   w̃ = {[round(v, 3) for v in w_pocket.tolist()]}")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
ax1.plot(err_path, color=COLORS[0], lw=0.8, label="current perceptron weights")
ax1.plot(err_path.cummin(0).values, color=COLORS[2], lw=2, label="pocket (best so far)")
ax1.axhline(error_rate(w_bisector, Xa_ov, t_ov), color="gray", ls="--", label="bisector $x_1 + x_2 = 0$")
ax1.set_xlabel("update number")
ax1.set_ylabel("training error rate")
ax1.set_ylim(0, 0.5)
ax1.set_title("Overlapping blobs: training error along the run")
ax1.legend(fontsize=8)

stops = [ends[9].item(), ends[29].item(), ends[-1].item()]
plot_dataset(X_ov, y_ov, ax=ax2, title="Where the perceptron is when we stop")
g1 = torch.linspace(-4, 4, 2, dtype=torch.float64)
for (idx, label), c in zip([(stops[0], "after 10 epochs"), (stops[1], "after 30 epochs"),
                            (stops[2], f"after {EPOCHS_OV} epochs")], [COLORS[3], COLORS[4], COLORS[7]]):
    w = res_ov["history"][idx]
    ax2.plot(g1, -(w[0] * g1 + w[2]) / w[1], color=c, label=f"{label} (error {err_path[idx].item():.2f})")
ax2.plot(g1, -(w_pocket[0] * g1 + w_pocket[2]) / w_pocket[1], color=COLORS[2], lw=2.5,
         label=f"pocket (error {err_pocket:.2f})")
ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4)
ax2.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# Left: the current perceptron's error (blue) never settles; it keeps jumping, often to 10% or more and occasionally to 50%, long after the pocket (green) has found good weights. The pocket's 4.0% is even below the bisector's 5.5%: the bisector is the best line for new data drawn from these two Gaussians, not necessarily for this particular sample of 200 points. Right: three snapshots of the same run give three noticeably different lines, and the pocket line is tilted well away from the bisector — it is tuned to a few specific training points.
#
# The pocket algorithm is a patch, not a principle. It costs a full pass over the data after every update to measure the error. Gallant showed that a variant ("pocket with ratchet") approaches a minimum-error separator with probability approaching 1 as training runs longer (stated without proof) — but with no useful bound on how long. And "minimum number of errors" is itself a hard objective: it is piecewise constant in the weights, so its gradient is zero almost everywhere and tells an optimizer nothing.
#
# The principled alternative is to replace the hard threshold by a smooth function that outputs a **probability**, and minimize a smooth loss with gradient descent, as in notebook 06. That is **logistic regression**, notebook 09.

# %% [markdown]
# ## 7. Feature maps can make data separable
#
# Some data are not separable in their original coordinates but become separable after adding the right features. Take two concentric rings (`make_rings`, inner radius 0.5 is class 0, outer radius 1.0 is class 1). No line separates a disk from the ring around it, and the perceptron never converges:

# %%
X_rg, y_rg = make_rings(n=150, noise=0.05, seed=0)   # (300, 2), (300,)
t_rg = to_signed(y_rg)
res_raw = perceptron_train(X_rg, t_rg, epochs=50, seed=0)
print("raw inputs: converged =", res_raw["converged"], "  updates in the last 5 epochs:", res_raw["mistakes"][-5:])

# %% [markdown]
# But the classes differ in **radius**. Add the squared radius as a third feature,
#
# $$\phi(\mathbf{x}) = (x_1,\; x_2,\; x_1^2 + x_2^2) \in \mathbb{R}^3,$$
#
# and the inner ring has small third coordinate, the outer ring large. In the new 3-D space a plane can separate them, so the perceptron (unchanged — it never knew what its inputs mean) can succeed.

# %%
def lift(X):
    """Feature map (N, 2) -> (N, 3): (x1, x2, x1² + x2²)."""
    X = X.double()
    return torch.cat([X, (X**2).sum(dim=1, keepdim=True)], dim=1)


res_lift = perceptron_train(lift(X_rg), t_rg, epochs=1000, seed=0)
w_lift = res_lift["w"]                               # (4,): weights for x1, x2, r², and the bias
print("lifted inputs: converged =", res_lift["converged"], "  updates =", res_lift["updates"])
print("w̃ =", [round(v, 4) for v in w_lift.tolist()])

# %% [markdown]
# ### Checkpoint 7.1
#
# The boundary in the lifted space is the plane $w_1 x_1 + w_2 x_2 + w_3 (x_1^2 + x_2^2) + b = 0$. Back in the original plane this is a **circle** (when $w_3 \ne 0$). Complete the square to find its center and radius:
#
# $$\Big(x_1 + \frac{w_1}{2w_3}\Big)^2 + \Big(x_2 + \frac{w_2}{2w_3}\Big)^2 = \frac{w_1^2 + w_2^2}{4w_3^2} - \frac{b}{w_3}.$$
#
# From `w_lift`, compute `center` (shape `(2,)`) and `radius` (a float or 0-d tensor).

# %% [markdown] tags=["hint"]
# `w1, w2, w3, b = w_lift`; the center is `torch.stack([-w1 / (2 * w3), -w2 / (2 * w3)])`, and the radius is the square root of the right-hand side.

# %%
# BEGIN SOLUTION
w1, w2, w3, b = w_lift
center = torch.stack([-w1 / (2 * w3), -w2 / (2 * w3)])
radius = torch.sqrt((w1**2 + w2**2) / (4 * w3**2) - b / w3)
# END SOLUTION
print(f"center = {[round(v, 4) for v in center.tolist()]}   radius = {float(radius):.4f}")

# %% tags=["check"]
check_shape(center, (2,), "center")
angles = torch.linspace(0, 2 * math.pi, 50, dtype=torch.float64)
on_circle = center + float(radius) * torch.stack([angles.cos(), angles.sin()], dim=1)   # (50, 2)
check_close(augment(lift(on_circle)) @ w_lift, torch.zeros(50, dtype=torch.float64), "score on your circle",
            atol=1e-9)
passed("Checkpoint 7.1")

# %%
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
P = lift(X_rg)
for label in (0, 1):
    m = y_rg == label
    ax1.scatter(P[m, 0], P[m, 1], P[m, 2], s=6, color=COLORS[label], label=f"class {label}")
g = torch.linspace(-1.1, 1.1, 10, dtype=torch.float64)
G1, G2 = torch.meshgrid(g, g, indexing="xy")
ax1.plot_surface(G1, G2, -(w_lift[0] * G1 + w_lift[1] * G2 + w_lift[3]) / w_lift[2], alpha=0.25, color="gray")
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_zlabel("$x_1^2 + x_2^2$")
ax1.set_title("Lifted space: a plane separates the classes")
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(1, 2, 2)
plot_decision_boundary(lambda G: (augment(lift(G)) @ w_lift > 0).long(), X_rg, y_rg, ax=ax2,
                       title="Original space: the boundary is a circle")
plt.tight_layout()
plt.show()

# %% [markdown]
# Left: lifting puts every point on the bowl $z = x_1^2 + x_2^2$; the inner ring sits low, the outer ring high, and the perceptron's plane slices between them. Right: the same classifier drawn in the original plane — a linear boundary in feature space becomes a circle here, with the radius of about 0.82 found in Checkpoint 7.1, between the two rings.
#
# This is the general lesson. A linear classifier on **features** $\phi(\mathbf{x})$ can draw curved boundaries in the input space. Everything in this notebook — separability, margins, the convergence theorem — applies unchanged, with $\phi(\mathbf{x}_i)$ in place of $\mathbf{x}_i$. The catch is choosing $\phi$: here we knew the answer depended on the radius. For images or text nobody knows the right features. The alternative is to **learn** them: a hidden layer of units computes features, and the output unit is a linear classifier on top. Notebooks 11–12 and Chapter 3 build exactly that.

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1 (modify): which features help XOR?
#
# The cell below repeats Section 7's experiment with a configurable dataset and feature map. As given, it runs the rings with $\phi(\mathbf{x}) = (x_1, x_2, x_1^2 + x_2^2)$.
#
# 1. Change the dataset to `make_xor(n=200, seed=0)` and the feature map to $(x_1, x_2, x_1 x_2)$. Run it. Does the perceptron converge? How many epochs and updates does it need, compared with the rings? Look at the plot.
# 2. Now try the feature map $(x_1, x_2, x_1 + x_2)$ on XOR. What happens, and why could you have predicted it without running anything?
# 3. Answer in the markdown cell below.

# %% [markdown] tags=["hint"]
# `make_xor` labels a point 1 exactly when $x_1 x_2 > 0$. For part 2: is $w_1 x_1 + w_2 x_2 + w_3(x_1 + x_2) + b$ any more flexible than $w_1' x_1 + w_2' x_2 + b$? For part 1, think about the margin: how close to the axes are the closest points?

# %%
X_mod, y_mod = make_rings(n=150, noise=0.05, seed=0)


def phi_mod(X):
    """Feature map (N, 2) -> (N, 3)."""
    X = X.double()
    return torch.cat([X, (X[:, 0] ** 2 + X[:, 1] ** 2)[:, None]], dim=1)


t_mod = to_signed(y_mod)
res_mod = perceptron_train(phi_mod(X_mod), t_mod, epochs=300)
print(f"converged: {res_mod['converged']}   epochs: {len(res_mod['mistakes'])}   updates: {res_mod['updates']}")
print("updates in the last 5 epochs:", res_mod["mistakes"][-5:])
print("w̃ =", [round(v, 4) for v in res_mod["w"].tolist()])
fig, ax = plt.subplots(figsize=(5, 5))
plot_decision_boundary(lambda G: (augment(phi_mod(G)) @ res_mod["w"] > 0).long(), X_mod, y_mod, ax=ax,
                       title="Perceptron on features φ(x), drawn in the input plane")
plt.show()

# %% [markdown]
# *Your answer:*

# %% [markdown] tags=["solution"]
# **XOR with $(x_1, x_2, x_1x_2)$.** The perceptron converges: 46 epochs and 496 updates, against 4 epochs and 7 updates for the rings. The boundary in the input plane is the curve $w_1x_1 + w_2x_2 + w_3x_1x_2 + b = 0$, and the learned weights are dominated by $w_3 \approx 15.2$ with $w_1 \approx 0.02$, $w_2 \approx -0.16$, $b = 0$: very close to $x_1x_2 = 0$, i.e. the two axes, which is exactly how `make_xor` labels points. Why so many updates? The data are separable in feature space by $\mathbf{u} \propto (0, 0, 1, 0)$, but the margin is $\min_i |x_{i1}x_{i2}|$ divided by the normalization, and points near an axis have tiny $|x_1 x_2|$. The convergence theorem's $(R/\gamma)^2$ is huge, and the perceptron needs many passes to tilt the boundary just right near the axes. (With `make_xor(noise=...)` > 0 the data stop being separable even in feature space, and the perceptron never converges.)
#
# **XOR with $(x_1, x_2, x_1 + x_2)$.** It never converges: about 100 of the 200 points are mistakes in each epoch, no better than chance. The new feature is a **linear** function of the old ones, so $w_1x_1 + w_2x_2 + w_3(x_1+x_2) + b = (w_1 + w_3)x_1 + (w_2+w_3)x_2 + b$ is still an ordinary line in the input plane. Linear features add nothing; a feature map helps only if it is **nonlinear**. The same fact, for layers, is why notebook 02 found that stacking affine maps without a nonlinearity gains nothing.

# %% [markdown]
# ### Exercise 2 (implement): the maximum-margin separator in 2-D, by brute force
#
# Section 4 showed that separators differ in their margin. Find the one with the **largest** margin, for 2-D data, by searching over directions.
#
# Fix a unit direction $\mathbf{w} = (\cos\theta, \sin\theta)$. The positive class must lie on the positive side, so the best bias for this direction puts the boundary halfway between the two classes' projections. With $p = \min_{t_i = +1} \mathbf{w}\cdot\mathbf{x}_i$ (the lowest positive projection) and $q = \max_{t_i = -1} \mathbf{w}\cdot\mathbf{x}_i$ (the highest negative projection):
#
# $$b(\theta) = -\frac{p + q}{2}, \qquad \gamma(\theta) = \frac{p - q}{2}.$$
#
# (Positives have distance $\ge p + b = (p - q)/2$, negatives $\ge -q - b = (p-q)/2$; moving $b$ either way makes one of them smaller.) If $p \le q$ the direction cannot separate and $\gamma(\theta) \le 0$.
#
# Write `max_margin_2d(X, t, n_angles=3600)` for $X$ of shape `(N, 2)` and $t$ of shape `(N,)` in $\{-1, +1\}$. Evaluate $\gamma(\theta)$ at `n_angles` equally spaced angles $\theta_j = 2\pi j / n_{\text{angles}}$, $j = 0, \dots, n_{\text{angles}} - 1$, and return `(w, b, gamma)` for the best angle: `w` of shape `(2,)` with unit norm, `b` and `gamma` as floats. Use `float64` and no Python loop over angles (broadcasting: projections of shape `(n_angles, N)`).

# %% [markdown] tags=["hint"]
# `theta = 2 * math.pi * torch.arange(n_angles, dtype=torch.float64) / n_angles`; `W = torch.stack([theta.cos(), theta.sin()], dim=1)` is `(n_angles, 2)`; `proj = W @ X.T` is `(n_angles, N)`. For $p$, replace the negatives' projections by `+inf` before `.min(dim=1)`: `torch.where(t > 0, proj, torch.inf)`. Then `argmax` over $\gamma$.

# %%
def max_margin_2d(X, t, n_angles=3600):
    # BEGIN SOLUTION
    X, t = X.double(), t.double()
    theta = 2 * math.pi * torch.arange(n_angles, dtype=torch.float64) / n_angles   # (A,)
    W = torch.stack([theta.cos(), theta.sin()], dim=1)                             # (A, 2) unit directions
    proj = W @ X.T                                                                 # (A, N)
    p = torch.where(t > 0, proj, torch.inf).min(dim=1).values                      # (A,) lowest positive
    q = torch.where(t < 0, proj, -torch.inf).max(dim=1).values                     # (A,) highest negative
    gammas = (p - q) / 2                                                           # (A,)
    j = gammas.argmax()
    return W[j], (-(p[j] + q[j]) / 2).item(), gammas[j].item()
    # END SOLUTION

# %% tags=["check"]
# A hand-checkable case: negatives at x1 = 0, positives at x1 = 2 -> boundary x1 = 1, margin 1.
X_h = torch.tensor([[0.0, 0.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]], dtype=torch.float64)
t_h = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float64)
w_h, b_h, g_h = max_margin_2d(X_h, t_h)
check_shape(w_h, (2,), "w")
check_close(w_h.norm(), 1.0, "‖w‖")
check_close(w_h, torch.tensor([1.0, 0.0], dtype=torch.float64), "w for the hand case", atol=1e-9)
check_close(b_h, -1.0, "b for the hand case", atol=1e-9)
check_close(g_h, 1.0, "gamma for the hand case", atol=1e-9)
# Checkpoint 4.1's data: the margin you return must be the true margin of (w, b), and at least 1/√2.
w_s, b_s, g_s = max_margin_2d(X_small, t_small)
check_close(g_s, geometric_margin(w_s, b_s, X_small, t_small), "gamma vs geometric_margin(w, b)", atol=1e-9)
check(g_s >= 1 / math.sqrt(2) - 1e-9, "the maximum margin cannot be smaller than that of (1, 1), b = -2")
# The blobs of Section 4: at least as large as every perceptron margin.
w_b, b_b, g_b = max_margin_2d(X_bl, t_bl)
check(g_b >= max(r[2] for r in runs) - 1e-9, "the max-margin separator must beat every perceptron separator")
passed("Exercise 2")
print(f"Checkpoint 4.1 data: maximum margin ≈ {g_s:.4f} with w = {[round(v, 4) for v in w_s.tolist()]}, b = {b_s:.4f}")
print(f"blobs: maximum margin ≈ {g_b:.4f} (perceptron margins ranged {min(r[2] for r in runs):.4f}–{max(r[2] for r in runs):.4f})")

# %% [markdown] tags=["solution"]
# On the hand case the search returns exactly $\mathbf{w} = (1, 0)$, $b = -1$, $\gamma = 1$ (angle 0 is on the grid). On Checkpoint 4.1's data the maximum margin is about 0.894, with $\mathbf{w} \approx (0.448, 0.894)$ (direction close to $(1, 2)/\sqrt5$) and $b \approx -1.342$ — clearly more than the $1/\sqrt2 \approx 0.707$ of the separator we were given, so that line was not the best one. On the blobs, the best margin is about 0.663; the best perceptron run (0.647) came close, the worst (0.016) was some 40 times smaller. The result is exact only up to the angular grid (0.1° here); because $\gamma(\theta)$ is continuous, a finer grid converges to the true maximum. This brute force works only in 2-D — the number of directions to try grows exponentially with $D$. Support vector machines find the maximum-margin separator in any dimension by solving a convex optimization problem.

# %% [markdown]
# ### Exercise 3 (debug): a bound that is violated
#
# `novikoff_bound(X, t, u)` should return $(R/\gamma)^2$ from the convergence theorem, for inputs $X$ `(N, D)`, signed labels $t$, and a separating direction $\mathbf{u}$ in **augmented** space `(D + 1,)` that need not have unit length. The version in the comments runs without error, but on the dataset below its "bound" is smaller than the number of updates the perceptron actually makes — impossible if it computed the theorem's quantity. It has **two** bugs. Fix both.

# %% [markdown] tags=["hint"]
# Re-read the theorem's assumptions: what vector's norm does $R$ bound, and what must be true of $\mathbf{u}$? Section 5 ("Two kinds of margin") lists both conventions.

# %%
def novikoff_bound(X, t, u):
    # Buggy version:
    #     R = X.double().norm(dim=1).max()
    #     gamma = (t * (augment(X) @ u)).min()
    #     return ((R / gamma) ** 2).item()
    # BEGIN SOLUTION
    Xa = augment(X)                         # (N, D + 1): the theorem is about augmented inputs
    u = u.double() / u.double().norm()      # the theorem needs a unit vector
    R = Xa.norm(dim=1).max()
    gamma = (t * (Xa @ u)).min()
    return ((R / gamma) ** 2).item()
    # END SOLUTION


X_dbg, t_dbg = make_margin_data(0.3)
u_dbg = torch.tensor([1.0, -2.0, 0.3], dtype=torch.float64)    # the planted direction, not normalized
updates_dbg = perceptron_train(X_dbg, t_dbg, epochs=1000)["updates"]
print(f"actual updates: {updates_dbg}")

# %% tags=["check"]
bound_dbg = novikoff_bound(X_dbg, t_dbg, u_dbg)
print(f"your bound: {bound_dbg:.2f}")
check(bound_dbg >= updates_dbg, f"bound {bound_dbg:.2f} is below the actual {updates_dbg} updates: the theorem is violated")
_Xa = augment(X_dbg)
check_close(bound_dbg, ((_Xa.norm(dim=1).max() / (t_dbg * (_Xa @ U_STAR)).min()) ** 2).item(), "bound")
check_close(novikoff_bound(X_dbg, t_dbg, 7 * u_dbg), bound_dbg, "bound for a rescaled u (must not change)")
passed("Exercise 3")

# %% [markdown]
# Here is what each bug does on its own:

# %%
def novikoff_bound_variants(X, t, u, augment_R, normalize_u):
    Xa = augment(X)
    R = (Xa if augment_R else X.double()).norm(dim=1).max()
    if normalize_u:
        u = u / u.norm()
    return ((R / (t * (Xa @ u)).min()) ** 2).item()


for aug_R, norm_u, label in [(False, False, "both bugs"), (True, False, "u not normalized"),
                             (False, True, "R without the 1"), (True, True, "correct")]:
    print(f"{label:18s}: bound = {novikoff_bound_variants(X_dbg, t_dbg, u_dbg, aug_R, norm_u):6.2f}"
          f"   (actual updates {updates_dbg})")

# %% [markdown] tags=["solution"]
# **Bug 1: $\mathbf{u}$ is not normalized.** The margin $\min_i t_i\,\mathbf{u}\cdot\tilde{\mathbf{x}}_i$ scales with $\|\mathbf{u}\|$, so any separator can be made to look like it has a huge margin just by multiplying it by a big number. The theorem's Step 3 used $\tilde{\mathbf{w}}_k\cdot\mathbf{u} \le \|\tilde{\mathbf{w}}_k\|\,\|\mathbf{u}\|$ with $\|\mathbf{u}\| = 1$; for a longer $\mathbf{u}$ the inequality picks up a factor $\|\mathbf{u}\|$ that the buggy code ignores. Here $\|\mathbf{u}\| = \|(1, -2, 0.3)\| \approx 2.26$, so $\gamma$ is inflated by 2.26 and the bound shrinks by $2.26^2 \approx 5.1$. The third check, rescaling $\mathbf{u}$ by 7, catches this bug on its own.
#
# **Bug 2: $R$ is computed on non-augmented inputs.** The perceptron updates with $\tilde{\mathbf{x}}_i = (\mathbf{x}_i, 1)$, and Step 2 needs $\|\tilde{\mathbf{x}}_i\|^2 = \|\mathbf{x}_i\|^2 + 1 \le R^2$. Leaving out the 1 underestimates $R$; for points in $[-1,1]^2$ that turns $R^2 \le 3$ into $R^2 \le 2$.
#
# With both bugs the "bound" is 3.93, below the 6 updates actually made. Each bug alone also makes the bound too small (6.10 and 20.02 instead of 31.03), yet each alone still stays at or above 6 here. So a numerical check of the form "bound ≥ actual" is a weak test: it catches a wrong formula only when the error happens to be large enough. Comparing with an independently computed value, and testing an invariance (rescaling $\mathbf{u}$ must not change the answer), catches it reliably.

# %% [markdown]
# ### Exercise 4 (investigate): how does the number of updates scale with $(R/\gamma)^2$?
#
# The theorem gives an upper bound. How close do real runs come to it, and does the actual count grow like $(R/\gamma)^2$, or more slowly? The scaffold generates separable datasets in dimension $D \in \{2, 10, 50\}$ with planted margins from 0.3 down to 0.003 (points uniform in $[-1, 1]^D$, a random planted unit $\mathbf{u}^*$ in augmented space, points closer than $\gamma_0$ to its boundary removed), runs the perceptron in 3 random orders on each, and plots the mean number of updates against $(R/\gamma)^2$ on log–log axes, with the line $y = x$ (the bound).
#
# Run it, then answer in the markdown cell:
#
# 1. Is any point above the line $y = x$? Should one be?
# 2. Roughly what slope do the points follow on the log–log plot for each $D$? A slope of 1 would mean updates grow proportionally to $(R/\gamma)^2$.
# 3. How does the dimension $D$ affect the number of updates at a given $(R/\gamma)^2$? Is that consistent with the theorem, which does not mention $D$?
#
# Optionally change `N_POINTS` or add $\gamma_0$ values.

# %% [markdown] tags=["hint"]
# On a log–log plot, a power law $y = c\,x^a$ is a straight line with slope $a$. The scaffold prints a fitted slope per dimension. For question 3, ask what $R$ is for points in $[-1,1]^D$, and whether a random dataset is a worst case.

# %%
N_POINTS = 300
GAMMAS0 = [0.3, 0.1, 0.03, 0.01, 0.003]
DIMS = [2, 10, 50]


def planted_data(D, gamma0, seed):
    """Uniform points in [-1, 1]^D, labelled by a random unit u* in R^{D+1}, with margin >= gamma0."""
    gen = torch.Generator().manual_seed(seed)
    u = torch.randn(D + 1, generator=gen, dtype=torch.float64)
    u = u / u.norm()
    X = 2 * torch.rand(4 * N_POINTS, D, generator=gen, dtype=torch.float64) - 1
    s = augment(X) @ u
    keep = (s.abs() >= gamma0).nonzero().squeeze(1)[:N_POINTS]
    X, s = X[keep], s[keep]
    return X, torch.sign(s), u


inv_rows = []   # (D, gamma0, (R/γ)², mean updates)
for D in DIMS:
    for gamma0 in GAMMAS0:
        X_inv, t_inv, u_inv = planted_data(D, gamma0, seed=D)
        Xa = augment(X_inv)
        R = Xa.norm(dim=1).max().item()
        gamma = (t_inv * (Xa @ u_inv)).min().item()
        ups = [perceptron_train(X_inv, t_inv, epochs=100_000, seed=s)["updates"] for s in range(3)]
        inv_rows.append((D, gamma0, (R / gamma) ** 2, sum(ups) / len(ups)))
        print(f"D = {D:2d}  γ0 = {gamma0:5.3f}  N = {len(t_inv)}  (R/γ)² = {(R / gamma) ** 2:12.0f}   mean updates = {sum(ups) / 3:8.1f}")

fig, ax = plt.subplots(figsize=(6.5, 4.8))
for c, D in zip(COLORS, DIMS):
    rows = [r for r in inv_rows if r[0] == D]
    xb = torch.tensor([r[2] for r in rows])
    yu = torch.tensor([r[3] for r in rows])
    slope = torch.linalg.lstsq(torch.stack([xb.log(), torch.ones_like(xb)], 1), yu.log()[:, None]).solution[0].item()
    ax.loglog(xb, yu, "o-", color=c, label=f"D = {D} (fitted slope {slope:.2f})")
    print(f"D = {D:2d}: log–log slope of updates vs (R/γ)² ≈ {slope:.2f}")
lims = torch.tensor([1.0, max(r[2] for r in inv_rows) * 2])
ax.loglog(lims, lims, "k--", label="bound: updates = $(R/\\gamma)^2$")
ax.set_xlabel("$(R/\\gamma)^2$")
ax.set_ylabel("perceptron updates (mean of 3 orders)")
ax.set_title("Actual updates versus the Novikoff bound")
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# *Your conclusion:*

# %% [markdown] tags=["solution"]
# **1. Nothing is above the line**, and nothing can be: the theorem holds for every separable dataset, every visiting order, and every separating unit vector — including the planted $\mathbf{u}^*$ we used to compute $\gamma$. The gap is enormous: at the smallest margins the perceptron needs 96 (D = 2), 756 (D = 10), and 279 (D = 50) updates against bounds of $2 \times 10^5$ to $8 \times 10^5$.
#
# **2. The slopes are far below 1**: about 0.40 for D = 2, 0.47 for D = 10, and 0.20 for D = 50. On these random datasets the number of updates grows roughly like the square root of $(R/\gamma)^2$ or slower, and it flattens out at the smallest margins (D = 2: 80 → 96 updates while the bound grows elevenfold). Two reasons. First, the bound is a worst case over all datasets and orders, and the proof assumes every update is made on a point at distance exactly $\gamma$; in random data almost all points are much farther away. Second, $\gamma$ here is the margin of the *planted* $\mathbf{u}^*$, but the theorem holds for *any* separating unit vector, so the true guarantee uses the largest margin of any separator. Once $\gamma_0$ is smaller than the typical gap between the closest sampled points, shrinking it further removes almost no points, and other separators keep a margin much larger than $\gamma_0$ — so the actual difficulty stops increasing even though our computed bound keeps growing.
#
# **3. Dimension matters even though the bound does not mention it.** At the same planted margin, larger $D$ means larger $R$ ($R^2 \approx D + 1$ for points in $[-1, 1]^D$), which moves the points to the right; and at roughly the same $(R/\gamma)^2 \approx 260$–$290$, the perceptron needs about 13 updates in D = 2 but about 50 in D = 50. More dimensions give the weights more directions to explore. But at small margins D = 50 needs *fewer* updates than D = 10: 300 points in 50 dimensions are far from "full", so many separators with large margins exist and the perceptron finds one quickly. None of this contradicts the theorem, which is only an upper limit: it says that $N$, $D$, and the order cannot make things worse than $(R/\gamma)^2$, not that they have no effect below it.

# %% [markdown]
# ## Summary and what's next
#
# - A dataset is **linearly separable** if some $(\mathbf{w}, b)$ gives $t_i(\mathbf{w}\cdot\mathbf{x}_i + b) > 0$ for every example. AND, OR, and NAND are; we found their weights by hand.
# - **XOR is not**: adding its four inequalities in pairs gives $w_1 + w_2 + 2b$ both positive and negative. On XOR the perceptron cycles forever, and no weights on a grid get more than 3 of 4 corners right. Of the 16 two-input Boolean functions exactly 14 are separable; the fraction collapses as the number of inputs grows (104 of 256 for three, 1882 of 65,536 for four).
# - The **geometric margin** $\min_i t_i(\mathbf{w}\cdot\mathbf{x}_i + b)/\|\mathbf{w}\|$ measures how much room a separator leaves. The perceptron returns *some* separator, with a margin that depends on the visiting order.
# - **Novikoff's theorem**: with $\|\tilde{\mathbf{x}}_i\| \le R$ and a unit $\mathbf{u}$ achieving margin $\gamma$, the perceptron makes at most $(R/\gamma)^2$ updates — whatever $N$, $D$, $\eta$, or the order. The proof plays a linear lower bound on $\tilde{\mathbf{w}}_k\cdot\mathbf{u}$ against a square-root upper bound on $\|\tilde{\mathbf{w}}_k\|$. Real runs need far fewer updates than the bound.
# - On **non-separable** data the theorem promises nothing: the perceptron never stops, and its weights depend on when you stop. The pocket algorithm keeps the best weights seen, but it is a heuristic for a hard, gradient-free objective.
# - A nonlinear **feature map** can make data separable: rings become separable with $x_1^2 + x_2^2$, XOR with $x_1x_2$. Linear features never help.
#
# What remains unsolved: the perceptron's hard threshold gives no notion of **confidence** — a point barely on the right side counts the same as one far away — and the number of mistakes, as a function of the weights, is flat almost everywhere, so it has **no useful gradient**. On overlapping data we want the "best" line in a principled sense. Notebook 09 replaces the threshold with a smooth sigmoid, reads its output as a probability, and derives a smooth loss from maximum likelihood: **logistic regression**, trained by the same gradient descent as notebook 06.

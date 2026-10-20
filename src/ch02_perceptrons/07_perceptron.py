# %% [markdown]
# # 07 — The perceptron
#
# | | |
# | --- | --- |
# | **Prerequisites** | Notebooks 00–06: tensors and broadcasting, dot products and orthogonality, gradients and gradient descent, likelihood, autograd, the linear-regression training loop |
# | **You will learn** | What a classifier is; how a perceptron decides with a weighted sum and a threshold; the geometry of its decision boundary (normal vector, bias, signed distance); the perceptron learning rule and why it works; training in epochs; folding the bias into the weights; the perceptron as minimizing a loss; what goes wrong when the classes overlap |
# | **Estimated time** | About 2.5 hours, plus 1–2 hours for the exercises |
# | **Compute** | CPU, under a minute. No GPU section: the perceptron updates one example at a time and gains nothing from a GPU at this size. |

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
import math

from dlcourse.data import make_blobs
from dlcourse.plotting import COLORS, plot_dataset, plot_decision_boundary

# %% [markdown]
# ## The motivating problem
#
# Notebook 06 predicted a *number*. Many questions instead ask for a *decision*: given some measurements of an item, is it a yes or a no? Is this email spam, is this tumour malignant, will this part fail inspection? This is **classification**.
#
# In 1958 Frank Rosenblatt proposed the **perceptron**, a machine that makes such a decision by weighing its inputs and comparing the total with a threshold — and that **learns the weights from its own mistakes**. It was built on the threshold neuron of McCulloch and Pitts (1943), and every unit in a modern neural network is its descendant.
#
# This notebook answers two questions: **how does a perceptron decide, and how does it learn?** Along the way we will see exactly what it can do, and the first signs of what it cannot.

# %% [markdown]
# ## 1. Classification
#
# In **regression** (notebook 06) the target $y$ is a real number and the loss measures how far the prediction is from it. In **classification** the target is one of a finite set of classes. With two classes we have **binary classification**, and we label the classes $0$ and $1$.
#
# A classifier is a **decision rule**: a function that maps an input $\mathbf{x}$ to a class. Here is our data: $N = 100$ points in the plane, $50$ per class, each with $D = 2$ measurements $(x_1, x_2)$. They come from `make_blobs`, which draws Gaussian clouds around given centers.

# %%
X, y = make_blobs(n=50, centers=((1.0, 1.0), (3.0, 3.0)), std=0.4, seed=0)
print("X:", tuple(X.shape), X.dtype, "   y:", tuple(y.shape), y.dtype)
print("first labels as generated:", y[:8].tolist(), "...", y[-8:].tolist())

# %% [markdown]
# `make_blobs` returns all class-0 points first, then all class-1 points. The perceptron will visit the examples in order, and an order that shows 50 examples of one class in a row is a strange curriculum, so we shuffle the rows once with a fixed permutation. Rows of `X` and entries of `y` must be permuted *together*, with the same index tensor.

# %%
perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
X, y = X[perm], y[perm]       # (N, 2), (N,)
N, D = X.shape
print("first labels after shuffling:", y[:12].tolist())
print("class counts:", torch.bincount(y).tolist())

# %%
fig, ax = plt.subplots(figsize=(5, 5))
plot_dataset(X, y, ax=ax, title="Two classes of measurements (N = 100)")
ax.legend()
plt.show()

# %% [markdown]
# The two clouds are clearly apart: a straight line drawn between them, from top-left to bottom-right, would separate them. Looking at the picture, the rule "**class 1 if $x_1 + x_2 > 4$**" seems reasonable — the line $x_1 + x_2 = 4$ passes through $(2, 2)$, halfway between the centers $(1, 1)$ and $(3, 3)$. Let us test it.

# %%
rule = (X[:, 0] + X[:, 1] > 4).long()        # (N,) predicted class, 0 or 1
print("predictions of the hand-made rule:", rule[:12].tolist())
print("true labels:                      ", y[:12].tolist())
print(f"accuracy: {(rule == y).float().mean().item():.2f}")

# %% [markdown]
# The hand-made rule gets every point right. But we chose it by looking at a picture, which is impossible with 100 measurements per example instead of 2. We want an algorithm that **finds such a rule from the data**. First we need a precise description of this kind of rule.
#
# ### Labels $\pm 1$
#
# The datasets in this course give $y \in \{0, 1\}$. The perceptron is much tidier with the labels $t \in \{-1, +1\}$, because then "the sign of a score agrees with the label" becomes a single product, as we will see. We convert explicitly:
#
# $$t = 2y - 1, \qquad y = 0 \mapsto t = -1, \qquad y = 1 \mapsto t = +1.$$
#
# We keep both: `y` for plotting functions that expect class indices, `t` for the perceptron's arithmetic.

# %% [markdown]
# ### Checkpoint 1.1
#
# Compute `t` from `y` as a `float32` tensor of shape `(N,)`. (Floating point, because `t` will be multiplied with the float scores and inputs.)

# %% [markdown] tags=["hint"]
# `2 * y - 1` is still `int64`; convert with `.float()`.

# %%
# BEGIN SOLUTION
t = (2 * y - 1).float()
# END SOLUTION
print("y:", y[:8].tolist(), "  t:", t[:8].tolist())

# %% tags=["check"]
check_shape(t, (N,), "t")
check(t.dtype == torch.float32, f"t should be float32, got {t.dtype}")
check(set(t.unique().tolist()) == {-1.0, 1.0}, "t should contain only -1 and +1")
check(torch.equal(t == 1, y == 1), "t must be +1 exactly where y is 1")
passed("Checkpoint 1.1")

# %% [markdown]
# The rest of the notebook uses this reference conversion, so that later cells do not depend on your answer:

# %%
t = (2 * y - 1).float()   # (N,) labels in {-1, +1}

# %% [markdown]
# ## 2. How a perceptron decides
#
# A perceptron computes a **weighted sum** of its inputs plus a constant, the **score**
#
# $$z = \mathbf{w} \cdot \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_D x_D + b,$$
#
# with a **weight vector** $\mathbf{w} \in \mathbb{R}^D$ and a **bias** $b \in \mathbb{R}$. Then it applies a **threshold**: if the score is positive it answers $+1$, otherwise $-1$:
#
# $$\hat{t} = \operatorname{step}(z) = \begin{cases} +1 & z > 0, \\ -1 & z \le 0. \end{cases}$$
#
# This is the same affine map $\mathbf{w}\cdot\mathbf{x} + b$ as linear regression, followed by a hard yes/no. Our hand-made rule is a perceptron with $\mathbf{w} = (1, 1)$ and $b = -4$: $z = x_1 + x_2 - 4 > 0$ exactly when $x_1 + x_2 > 4$.
#
# (Exactly $z = 0$ is a tie; we send it to $-1$. With real-valued data it is rare, but it happens at the start of training, when $\mathbf{w} = \mathbf{0}$ and $b = 0$ give $z = 0$ for every input. Section 4 deals with it.)

# %%
w_hand = torch.tensor([1.0, 1.0])   # (D,)
b_hand = -4.0
pts = torch.tensor([[1.0, 1.0], [3.0, 3.0], [2.5, 1.0]])    # (3, D): three test points
z_pts = pts @ w_hand + b_hand                                # (3,)
pred_pts = torch.where(z_pts > 0, 1.0, -1.0)                 # (3,)
for p, z_, tp in zip(pts.tolist(), z_pts.tolist(), pred_pts.tolist()):
    print(f"x = {p}:  z = {z_:+.1f}  ->  prediction {tp:+.0f}")

# %% [markdown]
# Check these by hand: $(1, 1)$ gives $1 + 1 - 4 = -2$; $(3, 3)$ gives $+2$; $(2.5, 1)$ gives $-0.5$, a negative score close to zero.
#
# ### The decision boundary is a line, and $\mathbf{w}$ is perpendicular to it
#
# The perceptron changes its answer where $z$ changes sign, on the set
#
# $$\{\mathbf{x} : \mathbf{w}\cdot\mathbf{x} + b = 0\},$$
#
# the **decision boundary**. In 2-D, $w_1 x_1 + w_2 x_2 + b = 0$ is the equation of a straight line; in $D$ dimensions it is a flat $(D-1)$-dimensional surface called a **hyperplane**. The boundary splits space into two **half-spaces**, $z > 0$ and $z \le 0$.
#
# **Claim:** $\mathbf{w}$ is perpendicular to the boundary. Take any two points $\mathbf{p}, \mathbf{q}$ on it. Then $\mathbf{w}\cdot\mathbf{p} + b = 0$ and $\mathbf{w}\cdot\mathbf{q} + b = 0$; subtracting, $\mathbf{w}\cdot(\mathbf{p} - \mathbf{q}) = 0$. Every direction $\mathbf{p} - \mathbf{q}$ *along* the boundary is orthogonal to $\mathbf{w}$ (notebook 02: dot product zero means perpendicular). So $\mathbf{w}$ is a **normal vector** of the boundary.
#
# **Which side is positive?** Moving from a boundary point $\mathbf{p}$ a step $s$ along $\mathbf{w}$ gives $z = \mathbf{w}\cdot(\mathbf{p} + s\mathbf{w}) + b = s\|\mathbf{w}\|^2$, positive for $s > 0$. So **$\mathbf{w}$ points into the $+1$ half-space**.

# %%
p_on, q_on = torch.tensor([4.0, 0.0]), torch.tensor([0.5, 3.5])    # both satisfy x1 + x2 = 4
print("z at p and q:", (p_on @ w_hand + b_hand).item(), (q_on @ w_hand + b_hand).item())
print("w · (p - q) =", (w_hand @ (p_on - q_on)).item())
for s in [-0.5, 0.5]:
    print(f"z at p + {s:+.1f} w = {((p_on + s * w_hand) @ w_hand + b_hand).item():+.2f}   (s·|w|² = {s * (w_hand @ w_hand).item():+.2f})")

# %% [markdown]
# ### Signed distance to the boundary
#
# The score is not just a sign; its size means something. Let $\mathbf{p}$ be any point on the boundary and $\hat{\mathbf{w}} = \mathbf{w}/\|\mathbf{w}\|$ the unit normal. The distance from $\mathbf{x}$ to the boundary, measured along the normal, is the projection (notebook 02) of $\mathbf{x} - \mathbf{p}$ onto $\hat{\mathbf{w}}$:
#
# $$d(\mathbf{x}) = \hat{\mathbf{w}}\cdot(\mathbf{x} - \mathbf{p}) = \frac{\mathbf{w}\cdot\mathbf{x} - \mathbf{w}\cdot\mathbf{p}}{\|\mathbf{w}\|} = \frac{\mathbf{w}\cdot\mathbf{x} + b}{\|\mathbf{w}\|} = \frac{z}{\|\mathbf{w}\|},$$
#
# using $\mathbf{w}\cdot\mathbf{p} = -b$. It is **signed**: positive on the $+1$ side, negative on the other. So **the score is the signed distance times $\|\mathbf{w}\|$**. To check: stepping from $\mathbf{x}$ back by $d(\mathbf{x})$ along $\hat{\mathbf{w}}$ must land exactly on the boundary.

# %%
w_norm = torch.linalg.vector_norm(w_hand)
d_pts = z_pts / w_norm                                       # (3,) signed distances
foot = pts - d_pts[:, None] * (w_hand / w_norm)              # (3, D): closest boundary points
print("signed distances:", [round(v, 4) for v in d_pts.tolist()])
print("z at the foot points (should be 0):", (foot @ w_hand + b_hand).tolist())
print("|x - foot| (should equal |d|):     ", [round(v, 4) for v in torch.linalg.vector_norm(pts - foot, dim=1).tolist()])

# %%
def draw_line(ax, w, b, xlim, **kwargs):
    """Draw the line w·x + b = 0 for 2-D w across the horizontal range xlim."""
    w1, w2 = float(w[0]), float(w[1])
    if abs(w2) > 1e-12 * max(abs(w1), 1e-30):
        xs = torch.tensor(xlim)
        ax.plot(xs, -(w1 * xs + float(b)) / w2, **kwargs)
    else:                                       # w2 = 0: the line is vertical, x1 = -b / w1
        ax.axvline(-float(b) / w1, **kwargs)


fig, ax = plt.subplots(figsize=(6, 6))
plot_dataset(X, y, ax=ax, alpha=0.35)
draw_line(ax, w_hand, b_hand, (-0.5, 4.5), color="k", lw=1.5, label="boundary $x_1 + x_2 - 4 = 0$")
ax.annotate("", xy=(2 + w_hand[0].item(), 2 + w_hand[1].item()), xytext=(2, 2),
            arrowprops=dict(arrowstyle="->", lw=2, color=COLORS[2]))
ax.text(2.55, 3.2, r"$\mathbf{w} = (1, 1)$", color=COLORS[2], fontsize=11)
for p, f, z_ in zip(pts, foot, z_pts):
    ax.plot([p[0], f[0]], [p[1], f[1]], ls=":", color="gray")
    ax.plot(*p.tolist(), "s", color="k", ms=6)
    ax.text(p[0].item() + 0.1, p[1].item() - 0.25, f"z = {z_.item():+.1f}", fontsize=10)
ax.text(0.0, 4.0, "$z > 0$: predict +1", color=COLORS[1], fontsize=10)
ax.text(2.6, -0.2, "$z \\leq 0$: predict −1", color=COLORS[0], fontsize=10)
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 4.5)
ax.set_title("A perceptron's boundary, normal vector, and scores")
ax.legend(loc="upper right", fontsize=8)
plt.show()

# %% [markdown]
# The boundary is the black line; $\mathbf{w}$ (green) is drawn from the boundary point $(2, 2)$ and is perpendicular to it, pointing toward the orange class-1 cloud, which is the positive side. The dotted segments are the shortest paths from the three test points to the line, and their lengths are $|z|/\|\mathbf{w}\| = |z|/\sqrt 2$. The point $(2.5, 1)$ has the smallest $|z|$ and is the closest.
#
# ### What the bias does, and a hidden redundancy
#
# The point of the boundary closest to the origin is at signed distance $-b/\|\mathbf{w}\|$ from it (put $\mathbf{x} = \mathbf{0}$ in $d(\mathbf{x})$ and flip the sign). So **$\mathbf{w}$ sets the boundary's direction and $b$ shifts it** along $\mathbf{w}$: without $b$, every boundary would pass through the origin.
#
# Now multiply both by any $c > 0$. The score becomes $c\,z$, which has the same sign as $z$, so every prediction stays the same: **$(\mathbf{w}, b)$ and $(c\mathbf{w}, cb)$ are the same classifier**. The signed distance $cz/\|c\mathbf{w}\| = z/\|\mathbf{w}\|$ does not change either. Only the raw score scales.

# %%
for c in [0.1, 1.0, 37.0]:
    z_c = X @ (c * w_hand) + c * b_hand                       # (N,)
    same = torch.equal(z_c > 0, X @ w_hand + b_hand > 0)
    dist_c = z_c / torch.linalg.vector_norm(c * w_hand)
    print(f"c = {c:5.1f}: same predictions on all {N} points: {same}   "
          f"max |score| = {z_c.abs().max().item():8.3f}   signed distance of point 0 = {dist_c[0].item():+.4f}")

# %% [markdown]
# So the pair $(\mathbf{w}, b)$ has one redundant degree of freedom: its overall scale. Keep this in mind; it explains a surprising fact about the learning rate in Section 5. (A negative $c$ is different: it flips every prediction.)

# %% [markdown]
# ### Checkpoint 2.1
#
# By hand first: a perceptron with $\mathbf{w} = (2, -1)$ and $b = 1$, and the three points $(1, 1)$, $(0, 3)$, $(-1, 0)$ (the rows of `P3`). Compute
#
# - `z3`: the three scores, shape `(3,)`;
# - `pred3`: the predictions in $\{-1, +1\}$, shape `(3,)`, float;
# - `dist3`: the signed distances to the boundary, shape `(3,)`.

# %% [markdown] tags=["hint"]
# The scores are $2 - 1 + 1 = 2$, $0 - 3 + 1 = -2$, and $-2 + 0 + 1 = -1$. Use `torch.where(z3 > 0, 1.0, -1.0)` for the predictions, and divide by $\|\mathbf{w}\| = \sqrt 5$ for the distances.

# %%
w3 = torch.tensor([2.0, -1.0])
b3 = 1.0
P3 = torch.tensor([[1.0, 1.0], [0.0, 3.0], [-1.0, 0.0]])   # (3, 2)
# BEGIN SOLUTION
z3 = P3 @ w3 + b3
pred3 = torch.where(z3 > 0, 1.0, -1.0)
dist3 = z3 / torch.linalg.vector_norm(w3)
# END SOLUTION
print("z3:", z3.tolist(), "  pred3:", pred3.tolist(), "  dist3:", [round(v, 4) for v in dist3.tolist()])

# %% tags=["check"]
check_shape(z3, (3,), "z3")
check_close(z3, torch.tensor([2.0, -2.0, -1.0]), "z3")
check_close(pred3, torch.tensor([1.0, -1.0, -1.0]), "pred3")
check_close(dist3, torch.tensor([2.0, -2.0, -1.0]) / math.sqrt(5), "dist3")
passed("Checkpoint 2.1")

# %% [markdown]
# ## 3. Deciding for a whole batch at once
#
# With the data stored one example per row, $X \in \mathbb{R}^{N \times D}$, all $N$ scores come from one matrix–vector product (notebook 02):
#
# $$\mathbf{z} = X\mathbf{w} + b, \qquad z_i = \mathbf{x}_i \cdot \mathbf{w} + b,$$
#
# with shapes `(N, D) @ (D,) -> (N,)`, and the scalar $b$ broadcast to every entry. No loop over examples. The **accuracy** is the fraction of examples whose prediction equals the label.
#
# We now define the helpers used for the rest of the notebook. `predict` returns labels in $\{-1, +1\}$, the perceptron's own convention; `predict01` returns class indices $\{0, 1\}$ for `plot_decision_boundary`, which shades the plane by class.

# %%
def scores(X, w, b):
    """Perceptron scores. X: (N, D), w: (D,), b: scalar -> (N,)."""
    return X @ w + b


def predict(X, w, b):
    """Predicted labels in {-1, +1}, shape (N,). A score of exactly 0 counts as -1."""
    return torch.where(scores(X, w, b) > 0, 1.0, -1.0)


def predict01(X, w, b):
    """Predicted class indices in {0, 1}, shape (N,)."""
    return (scores(X, w, b) > 0).long()


def accuracy(X, t, w, b):
    """Fraction of examples with predict == t (t in {-1, +1})."""
    return (predict(X, w, b) == t).float().mean().item()


print("scores:", tuple(scores(X, w_hand, b_hand).shape))
print(f"accuracy of the hand-made rule: {accuracy(X, t, w_hand, b_hand):.2f}")

# %%
fig, ax = plt.subplots(figsize=(5, 5))
plot_decision_boundary(lambda G: predict01(G, w_hand, b_hand), X, y, ax=ax,
                       title="Hand-made perceptron: regions by predicted class")
plt.show()

# %% [markdown]
# `plot_decision_boundary` evaluates the classifier on a $200 \times 200$ grid — a batch of 40 000 points in one call to `predict01` — and shades each region by its predicted class. The straight edge between the regions is the line $x_1 + x_2 = 4$, and every point sits in the region of its own color.

# %% [markdown]
# ### Checkpoint 3.1
#
# Using one vectorized expression each (no Python loops), compute for the hand-made perceptron (`w_hand`, `b_hand`) on the whole dataset:
#
# - `dist_all`: the signed distance of every point to the boundary, shape `(N,)`;
# - `n_close`: the number of points within distance $0.5$ of the boundary (on either side), as a Python `int`.

# %% [markdown] tags=["hint"]
# `scores(X, w_hand, b_hand) / torch.linalg.vector_norm(w_hand)`, then count with `(dist_all.abs() < 0.5).sum().item()`.

# %%
# BEGIN SOLUTION
dist_all = scores(X, w_hand, b_hand) / torch.linalg.vector_norm(w_hand)
n_close = int((dist_all.abs() < 0.5).sum().item())
# END SOLUTION
print(f"{n_close} points lie within 0.5 of the boundary; the closest is at distance {dist_all.abs().min().item():.3f}")

# %% tags=["check"]
check_shape(dist_all, (N,), "dist_all")
_d = (X @ w_hand + b_hand) / math.sqrt(2)
check_close(dist_all, _d, "dist_all")
check(n_close == int((_d.abs() < 0.5).sum()), f"n_close = {n_close}, expected {int((_d.abs() < 0.5).sum())}")
passed("Checkpoint 3.1")

# %% [markdown]
# ## 4. The perceptron learning rule
#
# We want an algorithm that finds $\mathbf{w}$ and $b$ by itself. Rosenblatt's idea: show the examples one at a time; **if the current perceptron gets an example right, do nothing; if it gets it wrong, nudge the weights toward the right answer**.
#
# With $\pm 1$ labels, "right" has a compact form. The prediction is correct when $z$ has the same sign as $t$, i.e. when $t\,z > 0$. So example $(\mathbf{x}, t)$ is a **mistake** when
#
# $$t\,z = t\,(\mathbf{w}\cdot\mathbf{x} + b) \le 0.$$
#
# (Ties $z = 0$ count as mistakes for *both* classes: a point on the boundary is not on either side.) On a mistake, update
#
# $$\mathbf{w} \leftarrow \mathbf{w} + \eta\, t\, \mathbf{x}, \qquad b \leftarrow b + \eta\, t,$$
#
# with a learning rate $\eta > 0$. That is the whole algorithm.
#
# ### Why this helps
#
# Compute the new value of $t\,z$ for the same example after the update:
#
# $$t\,z_{\text{new}} = t\big((\mathbf{w} + \eta t\mathbf{x})\cdot\mathbf{x} + b + \eta t\big) = t\,z_{\text{old}} + \eta\,t^2\big(\|\mathbf{x}\|^2 + 1\big) = t\,z_{\text{old}} + \eta\big(\|\mathbf{x}\|^2 + 1\big),$$
#
# because $t^2 = 1$. The update always increases $t\,z$ for that example by the positive amount $\eta(\|\mathbf{x}\|^2 + 1)$: it moves the score toward the correct sign. It does not promise to fix the example in one step (if $t\,z_{\text{old}}$ was very negative, it may still be negative), and it may break other examples that were right. Whether the process ever ends is the subject of notebook 08.
#
# **Geometrically**, for a positive example ($t = +1$) the update adds $\eta\mathbf{x}$ to $\mathbf{w}$, rotating $\mathbf{w}$ toward $\mathbf{x}$, so that $\mathbf{w}\cdot\mathbf{x}$ grows; for a negative example it subtracts $\eta\mathbf{x}$, rotating $\mathbf{w}$ away from $\mathbf{x}$. The bias moves the boundary one step away from the misclassified point's side, so that the point ends up further toward its correct side.

# %% [markdown]
# ### One update by hand
#
# Take $\mathbf{w} = (1, -1)$, $b = 0$, $\eta = 1$, and the example $\mathbf{x} = (2, 1)$ with label $t = -1$.
#
# 1. Score: $z = 1\cdot 2 + (-1)\cdot 1 + 0 = 1$. Then $t\,z = -1 \le 0$: a mistake (the perceptron says $+1$).
# 2. Update: $\mathbf{w} \leftarrow (1, -1) + 1\cdot(-1)\cdot(2, 1) = (-1, -2)$, and $b \leftarrow 0 + 1\cdot(-1) = -1$.
# 3. New score: $z = -2 - 2 - 1 = -5$, so $t\,z = 5$. The prediction is now correct.
#
# The formula predicts $t\,z_{\text{new}} = -1 + 1\cdot(\|\mathbf{x}\|^2 + 1) = -1 + 6 = 5$. ✓

# %%
w_ex, b_ex, eta_ex = torch.tensor([1.0, -1.0]), 0.0, 1.0
x_ex, t_ex = torch.tensor([2.0, 1.0]), -1.0

z_old = x_ex @ w_ex + b_ex
print(f"before: z = {z_old.item():+.1f},  t·z = {t_ex * z_old.item():+.1f}  ->  mistake: {t_ex * z_old.item() <= 0}")
w_new = w_ex + eta_ex * t_ex * x_ex
b_new = b_ex + eta_ex * t_ex
z_new = x_ex @ w_new + b_new
print(f"after:  w = {w_new.tolist()}, b = {b_new:+.1f}, z = {z_new.item():+.1f},  t·z = {t_ex * z_new.item():+.1f}")
print(f"t·z_old + η(|x|² + 1) = {t_ex * z_old.item() + eta_ex * ((x_ex @ x_ex).item() + 1):+.1f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
for ax, (w_, b_, label) in zip(axes, [(w_ex, b_ex, "before"), (w_new, b_new, "after")]):
    draw_line(ax, w_, b_, (-3, 3), color="k", lw=1.5)
    ax.annotate("", xy=(w_[0].item(), w_[1].item()), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2, color=COLORS[2]))
    ax.text(w_[0].item() + 0.1, w_[1].item() + 0.15, r"$\mathbf{w}$", color=COLORS[2], fontsize=12)
    ax.plot(*x_ex.tolist(), "o", color=COLORS[0], ms=10)
    ax.text(x_ex[0].item() + 0.15, x_ex[1].item() + 0.1, "x, t = −1", color=COLORS[0])
    z_ = (x_ex @ w_ + b_).item()
    ax.set_title(f"{label} the update: w = {[round(v) for v in w_.tolist()]}, b = {b_:+.0f}, z(x) = {z_:+.0f}")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
plt.tight_layout()
plt.show()

# %% [markdown]
# Before the update, $\mathbf{w}$ points to the side of the line that contains $\mathbf{x}$, so the negative example is classified $+1$. Subtracting $\mathbf{x}$ swings $\mathbf{w}$ around to point away from $\mathbf{x}$, and the new boundary leaves $\mathbf{x}$ on the negative side.

# %% [markdown]
# ### Checkpoint 4.1
#
# By hand first: $\mathbf{w} = (0.5, 1)$, $b = -1$, $\eta = 0.5$, example $\mathbf{x} = (-1, 2)$ with $t = -1$. Is it a mistake? Apply one perceptron update (only if it is a mistake) and store the results in `w_upd` (shape `(2,)`) and `b_upd` (a float). Then compute `tz_upd`, the new value of $t\,z$ for this example.

# %% [markdown] tags=["hint"]
# $z = -0.5 + 2 - 1 = 0.5$ and $t\,z = -0.5 \le 0$: a mistake. Then $\mathbf{w} + \eta t \mathbf{x} = (0.5, 1) + 0.5\cdot(-1)\cdot(-1, 2)$. The formula $t\,z_{\text{old}} + \eta(\|\mathbf{x}\|^2 + 1)$ lets you check `tz_upd`.

# %%
w_c4, b_c4, eta_c4 = torch.tensor([0.5, 1.0]), -1.0, 0.5
x_c4, t_c4 = torch.tensor([-1.0, 2.0]), -1.0
# BEGIN SOLUTION
if t_c4 * (x_c4 @ w_c4 + b_c4) <= 0:
    w_upd = w_c4 + eta_c4 * t_c4 * x_c4
    b_upd = b_c4 + eta_c4 * t_c4
else:
    w_upd, b_upd = w_c4, b_c4
tz_upd = t_c4 * (x_c4 @ w_upd + b_upd)
# END SOLUTION
print("w_upd:", w_upd.tolist(), " b_upd:", float(b_upd), " t·z after:", float(tz_upd))

# %% tags=["check"]
check_close(w_upd, torch.tensor([1.0, 0.0]), "w_upd")
check_close(float(b_upd), -1.5, "b_upd")
check_close(float(tz_upd), 2.5, "tz_upd")
passed("Checkpoint 4.1")

# %% [markdown]
# ## 5. Training: epochs and mistakes
#
# Training repeats the rule over the dataset. One pass over all $N$ examples, in order, is an **epoch**. We start from $\mathbf{w} = \mathbf{0}$, $b = 0$ (a perceptron that answers $-1$ to everything, since $z = 0$), count the mistakes in each epoch, and stop after the first epoch with **no** mistakes: then every training example is classified correctly, and since nothing changed during that epoch, nothing ever will.
#
# Note the structure: the *update* is per example, so this is a Python loop over examples, not a vectorized full-batch step as in notebook 06. We also record a snapshot of $(\mathbf{w}, b)$ after every epoch, and how many times each individual example caused an update.

# %%
ETA = 1.0
MAX_EPOCHS = 100

w = torch.zeros(D)            # (D,)
b = torch.tensor(0.0)         # ()
mistakes_per_epoch = []
snapshots = []                # (w, b) after each epoch
update_counts = torch.zeros(N)   # (N,): number of updates caused by each example
for epoch in range(MAX_EPOCHS):
    mistakes = 0
    for i in range(N):
        z_i = X[i] @ w + b
        if t[i] * z_i <= 0:                 # mistake (or tie)
            w = w + ETA * t[i] * X[i]
            b = b + ETA * t[i]
            mistakes += 1
            update_counts[i] += 1
    mistakes_per_epoch.append(mistakes)
    snapshots.append((w.clone(), b.clone()))
    if mistakes == 0:
        break

w_perc, b_perc = w, b
print("mistakes per epoch:", mistakes_per_epoch)
print(f"stopped after {len(mistakes_per_epoch)} epochs, {sum(mistakes_per_epoch)} updates in total")
print(f"learned w = {[round(v, 3) for v in w_perc.tolist()]}, b = {b_perc.item():.1f}")
print(f"training accuracy: {accuracy(X, t, w_perc, b_perc):.2f}")

# %% [markdown]
# The perceptron made 45 updates over 6 epochs and then completed a seventh with no mistakes. It reached 100% training accuracy without anyone choosing the line by eye.
#
# Its weights are not the hand-made $(1, 1, -4)$, but compare *directions*: dividing by $-b$ gives roughly $\mathbf{w}/(-b) \approx (0.26, 0.21)$, $b/(-b) = -1$, while the hand-made rule scaled the same way is $(0.25, 0.25, -1)$. Same kind of line, slightly tilted.

# %%
fig = plt.figure(figsize=(13, 6.5))
show = [0, 1, 2, 3, 4, len(snapshots) - 1]
for k, e in enumerate(show):
    ax = fig.add_subplot(2, 4, [1, 2, 3, 5, 6, 7][k])
    w_e, b_e = snapshots[e]
    plot_decision_boundary(lambda G: predict01(G, w_e, b_e), X, y, ax=ax, resolution=150)
    wrong = predict(X, w_e, b_e) != t
    ax.scatter(X[wrong, 0], X[wrong, 1], s=60, facecolors="none", edgecolors="k", lw=1.2)
    ax.set_title(f"after epoch {e + 1}: {int(wrong.sum())} wrong", fontsize=9)
ax_m = fig.add_subplot(1, 4, 4)
epochs_axis = list(range(1, len(mistakes_per_epoch) + 1))
ax_m.bar(epochs_axis, mistakes_per_epoch, color=COLORS[0])
ax_m.set_xlabel("epoch")
ax_m.set_ylabel("mistakes (= updates) during the epoch")
ax_m.set_title("Mistakes per epoch")
fig.suptitle("Perceptron training on the blobs (η = 1, start from w = 0, b = 0)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Left: the boundary after each epoch; circled points are misclassified by that epoch's final weights. Already after epoch 1 the line lies between the clouds, but one class-0 point near $(1.6, 2.4)$, the one closest to the orange cloud, is on the wrong side. The later epochs are a tug of war around it: fixing it in epoch 2 tilted the line so that five points were wrong at the end of that epoch, and it stays wrong after epochs 3–5 until the line finally becomes steep enough to pass between it and the orange points. Right: the number of updates per epoch falls from 16 to 0. The last epoch has zero mistakes (an empty bar), which is the stopping signal.
#
# Two details in the numbers. Mistakes during an epoch are counted *while* the weights change, so they differ from the number of points the end-of-epoch weights get wrong (the panel titles): the end-of-epoch error count went $1 \to 5 \to 1$, not steadily down, because an update that fixes one example can break others. And the bias changes by exactly $\pm\eta$ per update, so with $\eta = 1$ it is always an integer.
#
# ### The weights are a sum of training examples
#
# Every update adds $\eta\, t_i \mathbf{x}_i$ to $\mathbf{w}$, starting from $\mathbf{0}$. So if example $i$ caused $\alpha_i$ updates in total,
#
# $$\mathbf{w} = \eta\sum_{i=1}^{N} \alpha_i\, t_i\, \mathbf{x}_i, \qquad b = \eta\sum_{i=1}^{N}\alpha_i\, t_i.$$
#
# The learned weight vector is a weighted combination of the training inputs, and only the examples that were ever misclassified (those with $\alpha_i > 0$) contribute.

# %%
print(f"{int((update_counts > 0).sum())} of {N} examples ever caused an update; counts: {sorted(update_counts[update_counts > 0].int().tolist(), reverse=True)}")

# %% [markdown]
# ### Checkpoint 5.1
#
# Reconstruct the final weights from `update_counts` (the $\alpha_i$, shape `(N,)`), `t`, `X`, and `ETA` without any loop: store them in `w_from_counts` (shape `(D,)`) and `b_from_counts` (0-d tensor). They should equal `w_perc` and `b_perc`.

# %% [markdown] tags=["hint"]
# $\sum_i \alpha_i t_i \mathbf{x}_i$ is a matrix–vector product: the row vector of coefficients $\alpha_i t_i$ (shape `(N,)`) times $X$ (shape `(N, D)`), i.e. `(update_counts * t) @ X`.

# %%
# BEGIN SOLUTION
coef = ETA * update_counts * t          # (N,)
w_from_counts = coef @ X                # (D,)
b_from_counts = coef.sum()
# END SOLUTION
print("from counts:", [round(v, 4) for v in w_from_counts.tolist()], round(float(b_from_counts), 4))
print("trained:    ", [round(v, 4) for v in w_perc.tolist()], round(float(b_perc), 4))

# %% tags=["check"]
check_shape(w_from_counts, (D,), "w_from_counts")
check_close(w_from_counts, w_perc, "w_from_counts", rtol=1e-5, atol=1e-4)
check_close(float(b_from_counts), float(b_perc), "b_from_counts")
passed("Checkpoint 5.1")

# %% [markdown]
# ### A reusable training function
#
# We package the loop for the rest of the notebook. `perceptron_train` also accepts initial weights, and an `order_seed`: if given, each epoch visits the examples in a fresh random order drawn from a generator seeded with it; if `None`, it uses the stored order every epoch, exactly like the loop above.

# %%
def perceptron_train(X, t, epochs, eta=1.0, w_init=None, b_init=0.0, order_seed=None):
    """Train a perceptron one example at a time.

    X: (N, D) inputs, t: (N,) labels in {-1, +1}. Stops after the first epoch with no mistakes.
    Returns (w (D,), b 0-d tensor, mistakes per epoch (list), history: list of (w, b) after each epoch).
    """
    n, d = X.shape
    w = torch.zeros(d, dtype=X.dtype) if w_init is None else w_init.clone().to(X.dtype)
    b = torch.tensor(float(b_init), dtype=X.dtype)
    gen = torch.Generator().manual_seed(order_seed) if order_seed is not None else None
    mistakes, history = [], []
    for _ in range(epochs):
        order = torch.randperm(n, generator=gen).tolist() if gen is not None else range(n)
        m = 0
        for i in order:
            if t[i] * (X[i] @ w + b) <= 0:
                w = w + eta * t[i] * X[i]
                b = b + eta * t[i]
                m += 1
        mistakes.append(m)
        history.append((w.clone(), b.clone()))
        if m == 0:
            break
    return w, b, mistakes, history


w_f, b_f, m_f, _ = perceptron_train(X, t, MAX_EPOCHS)
check(m_f == mistakes_per_epoch and torch.equal(w_f, w_perc) and torch.equal(b_f, b_perc), "perceptron_train differs from the loop")
passed("perceptron_train reproduces the training loop exactly")

# %% [markdown]
# ### The learning rate does not matter (from a zero start)
#
# In notebook 06, the learning rate decided between slow convergence and divergence. Here it changes nothing. **Claim:** starting from $\mathbf{w} = \mathbf{0}$, $b = 0$ and visiting examples in the same order, training with rate $\eta$ produces exactly $\eta$ times the weights of training with rate $1$, and the same mistakes.
#
# **Proof by induction over updates.** Initially both runs have parameters $\mathbf{0} = \eta\cdot\mathbf{0}$. Suppose that before some example the $\eta$-run has $(\eta\mathbf{w}, \eta b)$ when the $1$-run has $(\mathbf{w}, b)$. The $\eta$-run's mistake test is $t(\eta\mathbf{w}\cdot\mathbf{x} + \eta b) \le 0$, which, since $\eta > 0$, holds exactly when the $1$-run's test $t(\mathbf{w}\cdot\mathbf{x} + b) \le 0$ does. If both update, the $\eta$-run gets $\eta\mathbf{w} + \eta t\mathbf{x} = \eta(\mathbf{w} + t\mathbf{x})$, again $\eta$ times the other. ∎
#
# Since $(\eta\mathbf{w}, \eta b)$ is the same classifier as $(\mathbf{w}, b)$ (Section 2), **$\eta$ only rescales the weights, which the step function ignores.** (From a non-zero start, $\eta$ does matter: it sets how big each update is compared with the initial weights.)

# %%
w_a, b_a, m_a, _ = perceptron_train(X, t, MAX_EPOCHS, eta=1.0)
w_b, b_b, m_b, _ = perceptron_train(X, t, MAX_EPOCHS, eta=0.01)
print("mistakes with η = 1:   ", m_a)
print("mistakes with η = 0.01:", m_b)
print("w(η=0.01) / 0.01 =", [round(v, 4) for v in (w_b / 0.01).tolist()], "  b / 0.01 =", round((b_b / 0.01).item(), 4))
print("w(η=1)           =", [round(v, 4) for v in w_a.tolist()], "  b =", round(b_a.item(), 4))

# %% [markdown]
# Identical mistakes, and the weights agree up to `float32` rounding (multiplying by $0.01$ at every update and dividing at the end is not exact in binary floating point, so the last digits may differ). The proof assumed exact arithmetic; with rounding, a score that lands extremely close to 0 could in principle tip the other way, but that did not happen here.

# %% [markdown]
# ## 6. Folding the bias into the weights
#
# The bias is a nuisance: it gets its own update line. A standard trick removes it. Append a constant feature $1$ to every input, and the bias to the weights:
#
# $$\tilde{\mathbf{x}} = (x_1, \dots, x_D, 1), \qquad \tilde{\mathbf{w}} = (w_1, \dots, w_D, b), \qquad \tilde{\mathbf{w}}\cdot\tilde{\mathbf{x}} = \mathbf{w}\cdot\mathbf{x} + b\cdot 1 = z.$$
#
# The **augmented** inputs form an $N \times (D+1)$ matrix $\tilde X = [X, \mathbf{1}]$, the same design matrix as least squares in notebook 06. The update $\tilde{\mathbf{w}} \leftarrow \tilde{\mathbf{w}} + \eta t\tilde{\mathbf{x}}$ updates the first $D$ entries by $\eta t\mathbf{x}$ and the last by $\eta t\cdot 1$: exactly the two lines of the rule, in one.
#
# Geometrically, a boundary with bias in $D$ dimensions becomes a boundary *through the origin* in $D+1$ dimensions, restricted to the slice where the last coordinate is 1. Proofs about perceptrons (notebook 08) usually assume this form, because it has one fewer symbol.

# %% [markdown]
# ### Checkpoint 6.1
#
# Build `X_aug` of shape `(N, D + 1)` by appending a column of ones to `X`, and `w_aug` of shape `(D + 1,)` from `w_perc` and `b_perc`. Check that `X_aug @ w_aug` reproduces the scores.

# %% [markdown] tags=["hint"]
# `torch.cat([X, torch.ones(N, 1)], dim=1)` for the matrix. For the vector, `torch.cat([w_perc, b_perc.reshape(1)])`.

# %%
# BEGIN SOLUTION
X_aug = torch.cat([X, torch.ones(N, 1)], dim=1)       # (N, D + 1)
w_aug = torch.cat([w_perc, b_perc.reshape(1)])        # (D + 1,)
# END SOLUTION
print("X_aug:", tuple(X_aug.shape), "  first row:", [round(v, 3) for v in X_aug[0].tolist()])

# %% tags=["check"]
check_shape(X_aug, (N, D + 1), "X_aug")
check_shape(w_aug, (D + 1,), "w_aug")
check(torch.equal(X_aug[:, -1], torch.ones(N)), "the last column of X_aug should be all ones")
check_close(X_aug @ w_aug, scores(X, w_perc, b_perc), "X_aug @ w_aug", atol=1e-4)
passed("Checkpoint 6.1")

# %% [markdown]
# Now train in augmented form, with one vector and one update line, and compare with the version with separate $\mathbf{w}$ and $b$. We build the augmented matrix again here so this teaching cell does not depend on your checkpoint answer.

# %%
Xa = torch.cat([X, torch.ones(N, 1)], dim=1)   # (N, D + 1)
theta = torch.zeros(D + 1)                     # (D + 1,): (w1, w2, b)
aug_mistakes = []
for epoch in range(MAX_EPOCHS):
    m = 0
    for i in range(N):
        if t[i] * (Xa[i] @ theta) <= 0:
            theta = theta + ETA * t[i] * Xa[i]
            m += 1
    aug_mistakes.append(m)
    if m == 0:
        break
print("augmented:  mistakes", aug_mistakes, "  theta =", [round(v, 4) for v in theta.tolist()])
print("separate b: mistakes", mistakes_per_epoch, "  (w, b) =", [round(v, 4) for v in w_perc.tolist()], round(b_perc.item(), 4))
check_close(theta[:D], w_perc, "augmented w", atol=1e-5)
check_close(theta[D], b_perc, "augmented b")
passed("Augmented training is the same algorithm")

# %% [markdown]
# ## 7. The perceptron as minimizing a loss
#
# Notebook 06 built learning from three pieces: a model, a loss, and gradient descent. The perceptron was invented as a rule, not as a loss minimizer — but it *is* one.
#
# ### The perceptron criterion
#
# Define the per-example loss
#
# $$\ell(\mathbf{w}, b; \mathbf{x}, t) = \max(0, -t\,z), \qquad L = \sum_{i=1}^{N} \max(0, -t_i z_i).$$
#
# It is $0$ for a correctly classified example ($t z > 0$), and for a misclassified one it equals $-t z = |z|$, which is the distance to the boundary times $\|\mathbf{w}\|$: how badly wrong it is. This is the **perceptron criterion**.
#
# **Its gradient.** Where $t z < 0$, $\ell = -t(\mathbf{w}\cdot\mathbf{x} + b)$, a linear function, so
#
# $$\nabla_{\mathbf{w}}\ell = -t\,\mathbf{x}, \qquad \frac{\partial \ell}{\partial b} = -t.$$
#
# Where $t z > 0$, $\ell = 0$ and both derivatives are $0$. A gradient-descent step with rate $\eta$ on one misclassified example is therefore
#
# $$\mathbf{w} \leftarrow \mathbf{w} - \eta(-t\mathbf{x}) = \mathbf{w} + \eta t\mathbf{x}, \qquad b \leftarrow b + \eta t,$$
#
# **exactly the perceptron rule**, and on a correct example it does nothing — also the rule. So the perceptron is gradient descent on $\ell$, one example at a time. (Taking a step on one example's loss rather than the full sum is **stochastic gradient descent**, the subject of notebook 19.)
#
# Let us check with autograd, at the misclassified example from Section 4:

# %%
w_g = torch.tensor([1.0, -1.0], requires_grad=True)
b_g = torch.tensor(0.0, requires_grad=True)
loss_g = torch.clamp(-t_ex * (x_ex @ w_g + b_g), min=0)   # max(0, -t z)
loss_g.backward()
print(f"loss = {loss_g.item():.1f}")
print("autograd: dℓ/dw =", w_g.grad.tolist(), "  dℓ/db =", b_g.grad.item())
print("formula:  -t x  =", (-t_ex * x_ex).tolist(), "  -t =", -t_ex)
with torch.no_grad():
    print("gradient step (η = 1):  w =", (w_g - 1.0 * w_g.grad).tolist(), " b =", (b_g - 1.0 * b_g.grad).item())
print("perceptron rule:        w =", w_new.tolist(), " b =", b_new)

# %% [markdown]
# ### The kink at $t z = 0$
#
# $\max(0, -tz)$ has a corner at $t z = 0$, exactly like ReLU at 0 (notebook 03; indeed $\ell = \mathrm{ReLU}(-tz)$). There it has no derivative. The perceptron rule treats a tie as a mistake and uses $-t\mathbf{x}$ there, one of the valid choices for the slope at a corner (a **subgradient**: any slope between the left and right slopes; we will not need the formal definition). PyTorch's ReLU makes the other choice, slope 0. That matters at the very start, where $\mathbf{w} = \mathbf{0}$, $b = 0$ makes *every* $z$ exactly zero:

# %%
w_0 = torch.zeros(D, requires_grad=True)
b_0 = torch.tensor(0.0, requires_grad=True)
L_0 = torch.clamp(-t * scores(X, w_0, b_0), min=0).sum()      # perceptron criterion at w = 0, b = 0
L_0.backward()
print(f"L at w = 0, b = 0: {L_0.item()}   autograd gradient: w {w_0.grad.tolist()}, b {b_0.grad.item()}")

# %% [markdown]
# The loss is exactly 0 — the all-zero perceptron is a "minimizer" of $L$! — and autograd's gradient is 0, so plain gradient descent on $L$ from zero would never move. This is a real defect of the criterion: $L = 0$ is achieved trivially by $\mathbf{w} = \mathbf{0}$, $b = 0$, which classifies nothing. The perceptron escapes it only because its rule counts ties as mistakes. (Notebook 09's loss has no such degenerate minimum.)
#
# ### Compare with MSE, and with counting mistakes
#
# Write each loss as a function of the **margin** $m = t z$ (positive means correct):
#
# - **Perceptron criterion:** $\max(0, -m)$.
# - **Squared error** on the $\pm 1$ targets, as if we did linear regression (notebook 06) and then thresholded: $(z - t)^2 = (t z - t^2)^2 = (m - 1)^2$, using $t^2 = 1$ to factor out $t$.
# - **0–1 loss:** $1$ if $m \le 0$ (a mistake), else $0$. Summed over examples, it is the number of mistakes — what we actually care about.

# %%
m_axis = torch.linspace(-3, 4, 701)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(m_axis, (m_axis <= 0).float(), color="k", lw=2, label="0–1 loss: mistake or not")
ax.plot(m_axis, torch.clamp(-m_axis, min=0), color=COLORS[0], lw=2, label="perceptron criterion max(0, −m)")
ax.plot(m_axis, (m_axis - 1) ** 2, color=COLORS[1], lw=2, label="squared error (m − 1)²")
ax.set_ylim(-0.3, 6)
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel("margin $m = t\\,z$ (positive = correctly classified)")
ax.set_ylabel("loss of one example")
ax.set_title("Three losses for a ±1 classifier, as functions of the margin")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.36, 0.98))
plt.show()

# %% [markdown]
# Three things to notice.
#
# - **Squared error punishes being *too* right.** An example with margin $m = 4$ is correctly and confidently classified, yet it costs $(4-1)^2 = 9$, more than a mistake at $m = -1$ (cost 4). Gradient descent on MSE would move the boundary to make such points *less* confident, possibly at the expense of misclassifying others. The perceptron criterion is exactly zero for every correct example: once a point is on the right side, it stops influencing the weights.
# - **The 0–1 loss is flat.** Its derivative is 0 everywhere except at $m = 0$, where it jumps. Gradient descent on it gets no signal at all — and the same is true of the perceptron's output itself: the step function's derivative is zero almost everywhere. We can see autograd agree:

# %%
w_s = torch.tensor([0.3, -0.2], requires_grad=True)
n_wrong = (torch.sign(scores(X, w_s, 0.1)) != t).float().sum()     # number of mistakes: not differentiable
out = torch.sign(scores(X, w_s, 0.1)).sum()                        # the ±1 outputs themselves
out.backward()
print(f"mistakes at this w: {int(n_wrong)}   (requires_grad: {n_wrong.requires_grad})")
print("gradient of the sum of sign(z) w.r.t. w:", w_s.grad.tolist())

# %% [markdown]
# Comparing with `!=` produces a boolean tensor, cut off from autograd entirely (`requires_grad` is `False`). Even the sign function, which autograd can differentiate, gives gradient exactly zero. This is why we **cannot train a network of step-function units by backpropagation**: no gradient flows through a step. The perceptron sidesteps the problem with a hand-made rule, which works for a single unit. Notebook 09 replaces the step with a smooth S-shaped curve, the **sigmoid**, whose derivative is nowhere zero, and derives a loss from probability; that is the route that scales to deep networks.

# %% [markdown]
# ### Checkpoint 7.1
#
# By hand, then in code: the perceptron of Checkpoint 2.1 ($\mathbf{w} = (2, -1)$, $b = 1$) on the points `P3` with labels $t = (+1, +1, -1)$. Compute `L_crit`, the perceptron criterion $\sum_i \max(0, -t_i z_i)$, as a 0-d tensor or a float, and `n_mistakes`, the number of examples with $t_i z_i \le 0$, as an `int`.

# %% [markdown] tags=["hint"]
# The scores are $(2, -2, -1)$, so the margins $t_i z_i$ are $(2, -2, 1)$. Only the second example is wrong, and it contributes $2$.

# %%
t3 = torch.tensor([1.0, 1.0, -1.0])
# BEGIN SOLUTION
margins3 = t3 * (P3 @ w3 + b3)
L_crit = torch.clamp(-margins3, min=0).sum()
n_mistakes = int((margins3 <= 0).sum())
# END SOLUTION
print("perceptron criterion:", float(L_crit), "  mistakes:", n_mistakes)

# %% tags=["check"]
check_close(float(L_crit), 2.0, "L_crit")
check(n_mistakes == 1, f"n_mistakes = {n_mistakes}, expected 1")
passed("Checkpoint 7.1")

# %% [markdown]
# ## 8. Limitations: overlap and non-uniqueness
#
# So far the classes could be separated by a line, and the perceptron found one. Two questions remain: what if no line separates them, and is the line it finds the only one, or the best one?
#
# ### When the classes overlap
#
# We keep the same class centers but widen the clouds from standard deviation $0.4$ to $0.9$, so they overlap and no line can separate them.

# %%
X_ov, y_ov = make_blobs(n=50, centers=((1.0, 1.0), (3.0, 3.0)), std=0.9, seed=0)
perm_ov = torch.randperm(len(y_ov), generator=torch.Generator().manual_seed(0))
X_ov, y_ov = X_ov[perm_ov], y_ov[perm_ov]
t_ov = (2 * y_ov - 1).float()

EPOCHS_OV = 60
w_ov, b_ov, m_ov, hist_ov = perceptron_train(X_ov, t_ov, EPOCHS_OV)
print(f"ran {len(m_ov)} epochs (the limit); mistakes in the last 10 epochs: {m_ov[-10:]}")
errors_ov = [int((predict(X_ov, w_, b_) != t_ov).sum()) for w_, b_ in hist_ov]
print("training errors of the end-of-epoch weights, last 10 epochs:", errors_ov[-10:])
print(f"fewest errors at the end of any epoch: {min(errors_ov)} (epoch {errors_ov.index(min(errors_ov)) + 1})")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
ax1.plot(range(1, len(m_ov) + 1), m_ov, ".-", color=COLORS[0], label="updates during the epoch")
ax1.plot(range(1, len(errors_ov) + 1), errors_ov, ".-", color=COLORS[1], label="errors of end-of-epoch weights")
ax1.set_ylim(bottom=0)
ax1.set_xlabel("epoch")
ax1.set_ylabel("number of examples (of 100)")
ax1.set_title("Overlapping classes: mistakes never reach zero")
ax1.legend()

plot_dataset(X_ov, y_ov, ax=ax2, alpha=0.6)
xlim = (X_ov[:, 0].min().item() - 0.3, X_ov[:, 0].max().item() + 0.3)
for k, e in enumerate(range(len(hist_ov) - 6, len(hist_ov))):
    w_, b_ = hist_ov[e]
    draw_line(ax2, w_, b_, xlim, color=plt.cm.viridis(k / 5), lw=1.5, label=f"after epoch {e + 1}: {errors_ov[e]} errors")
ax2.set_xlim(*xlim)
ax2.set_ylim(X_ov[:, 1].min().item() - 0.3, X_ov[:, 1].max().item() + 0.3)
ax2.set_title("Boundary after each of the last 6 epochs")
ax2.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.tight_layout()
plt.show()

# %% [markdown]
# Left: the number of updates per epoch hovers around a constant and never reaches zero, so the stopping rule never fires; the algorithm only ends because we capped the epochs. The error count of the end-of-epoch weights jumps around too. Right: from one epoch to the next the boundary keeps jumping, because some points are always on the wrong side and each one yanks the line toward itself. Where it is when we stop is an accident of the epoch cap.
#
# An easy patch is to remember the best weights seen so far (the "pocket" algorithm, Gallant 1990). It is a patch, not a cure: the perceptron rule has no notion of "the line that makes fewest mistakes" and never settles near it. Notebook 09 gives a loss that *does* have a sensible minimum on overlapping data.

# %% [markdown]
# ### Checkpoint 8.1
#
# From `hist_ov` (a list of `(w, b)` pairs, one per epoch), find the pair with the fewest training errors on `X_ov`, `t_ov`. Store its epoch index (0-based, the first one if there are ties) in `best_epoch` and its training accuracy in `best_acc`, a float. Compare `best_acc` with the accuracy of the final weights.

# %% [markdown] tags=["hint"]
# A list comprehension `[accuracy(X_ov, t_ov, w_, b_) for w_, b_ in hist_ov]` gives one accuracy per epoch; `max` and `.index` find the best.

# %%
# BEGIN SOLUTION
accs_ov = [accuracy(X_ov, t_ov, w_, b_) for w_, b_ in hist_ov]
best_acc = max(accs_ov)
best_epoch = accs_ov.index(best_acc)
# END SOLUTION
print(f"best end-of-epoch accuracy {best_acc:.2f} (epoch index {best_epoch}); final accuracy {accuracy(X_ov, t_ov, w_ov, b_ov):.2f}")

# %% tags=["check"]
_errs = [int((predict(X_ov, w_, b_) != t_ov).sum()) for w_, b_ in hist_ov]
check(best_epoch == _errs.index(min(_errs)), f"best_epoch = {best_epoch}, expected {_errs.index(min(_errs))}")
check_close(best_acc, 1 - min(_errs) / len(t_ov), "best_acc")
passed("Checkpoint 8.1")

# %% [markdown]
# ### The answer depends on the order
#
# Back to the separable blobs. Many lines separate them, and the perceptron stops at the *first* one it reaches. Which one that is depends on the sequence of updates, hence on the order in which examples are visited. Below, three runs differ only in `order_seed`.

# %%
fig, ax = plt.subplots(figsize=(6, 6))
plot_dataset(X, y, ax=ax, alpha=0.6)
print(f"{'order':>12s}  {'epochs':>6s}  {'updates':>7s}  {'(w1, w2, b) / |w|':>28s}  {'smallest distance to line':>26s}")
for k, seed in enumerate([None, 1, 2]):
    w_o, b_o, m_o, _ = perceptron_train(X, t, MAX_EPOCHS, order_seed=seed)
    nrm = torch.linalg.vector_norm(w_o)
    gap = (t * scores(X, w_o, b_o) / nrm).min().item()      # distance of the closest point
    label = "stored" if seed is None else f"seed {seed}"
    print(f"{label:>12s}  {len(m_o):6d}  {sum(m_o):7d}  {str([round(v, 3) for v in (torch.cat([w_o, b_o.reshape(1)]) / nrm).tolist()]):>28s}  {gap:26.3f}")
    draw_line(ax, w_o, b_o, (-0.5, 4.5), color=COLORS[k + 2], lw=2, label=f"order: {label}")
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 4.5)
ax.set_title("Three visiting orders, three separating lines")
ax.legend(fontsize=8, loc="upper left")
plt.show()

# %% [markdown]
# All three runs end with zero mistakes, yet they draw different lines (the table normalizes by $\|\mathbf{w}\|$, since only the direction and position matter). Some pass close to a training point — "smallest distance to line" is the gap between the boundary and the nearest example — which leaves little room for error on new data from the same clouds. The perceptron only promises *a* separating line, not the one with the widest gap. The widest-gap line, the **maximum-margin** separator, is what support vector machines find; the margin also controls how fast the perceptron converges (Exercise 4 and notebook 08).

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1 (modify): visiting order and initial weights
#
# The cell trains on the separable blobs with several settings, each a dictionary of keyword arguments for `perceptron_train`, and draws the final boundaries. The list starts with three visiting orders from a zero start, then three non-zero starts: $\mathbf{w} = (-3, 3)$, $b = 0$ (a line along the diagonal $x_2 = x_1$, which does not separate the clouds), the same with $\eta = 10$, and the hand-made rule $(1, 1, -4)$, which is already perfect.
#
# 1. Before running, predict for each setting whether it will converge, and whether it will need more or fewer updates than the zero start.
# 2. Run it, and compare the printed epochs and updates and the drawn lines with your predictions.
# 3. Modify `settings`: add at least two runs of your own, such as more `order_seed` values, or a start that is the learned classifier turned upside down, `w_init=-w_perc, b_init=-b_perc.item()`.
#
# In the markdown cell below, describe the effect of the order and of the initial weights on the final boundary and on the number of epochs and updates, and explain the case where nothing happens at all.

# %% [markdown] tags=["hint"]
# From a non-zero start, Section 5's argument about $\eta$ no longer applies: $\eta$ sets how big an update is compared with the initial weights. And recall when the algorithm stops: what happens in epoch 1 if no example is a mistake?

# %%
settings = [
    dict(order_seed=None),
    dict(order_seed=1),
    dict(order_seed=2),
    dict(w_init=torch.tensor([-3.0, 3.0]), b_init=0.0),
    dict(w_init=torch.tensor([-3.0, 3.0]), b_init=0.0, eta=10.0),
    dict(w_init=torch.tensor([1.0, 1.0]), b_init=-4.0),
]

fig, ax = plt.subplots(figsize=(9.5, 6))
plot_dataset(X, y, ax=ax, alpha=0.5)
for k, s in enumerate(settings):
    w_s1, b_s1, m_s1, _ = perceptron_train(X, t, MAX_EPOCHS, **s)
    label = ", ".join(f"{key}={val.tolist() if torch.is_tensor(val) else val}" for key, val in s.items())
    print(f"{label:55s} epochs {len(m_s1):3d}  updates {sum(m_s1):4d}  accuracy {accuracy(X, t, w_s1, b_s1):.2f}")
    draw_line(ax, w_s1, b_s1, (-0.5, 4.5), color=COLORS[k % len(COLORS)], lw=1.5, label=label)
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 4.5)
ax.set_title("Final boundaries for different settings")
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.show()

# %% [markdown]
# *Your answer:*

# %% [markdown] tags=["solution"]
# **Order.** All three orders separate the data with zero final mistakes, but they stop at different lines after different numbers of epochs and updates (see the printed table). The perceptron accepts the first separating line it reaches, and the order decides which one.
#
# **A bad start.** Starting from $\mathbf{w} = (-3, 3)$, $b = 0$ — a boundary along the diagonal $x_2 = x_1$, which does not separate the clouds at all — the perceptron still converges, but it needed 61 updates over 11 epochs, against 45 over 7 from zero. Each update adds a vector of length $\|\tilde{\mathbf{x}}_i\|$, between about 1.1 and 5 for these data, so the initial weights (length about 4.2) must first be "outvoted" by several updates. With $\eta = 10$ the same bad start was overcome in 13 updates and 2 epochs: each update is now ten times larger compared with the initial weights, which matter correspondingly less. (Dividing all weights by $\eta$ shows the general picture: rate $\eta$ from $\mathbf{w}_0$ is the same classifier as rate 1 from $\mathbf{w}_0/\eta$.) The final lines differ from run to run. The upside-down start `w_init=-w_perc, b_init=-b_perc.item()` gets every training point wrong at first; it still converges, in our run after 80 updates over 7 epochs, because the updates must first cancel the large wrong-way weights.
#
# **A perfect start.** Starting from the hand-made rule $(1, 1, -4)$, every example is already correct, so epoch 1 has zero mistakes, no update ever happens, and the algorithm stops after one epoch with the weights unchanged. The perceptron never improves a solution that is already correct, even one passing close to some points: it only reacts to mistakes.

# %% [markdown]
# ### Exercise 2 (implement): a perceptron in augmented form
#
# Write `train_perceptron(X, t, epochs)` for inputs `X` of shape `(N, D)` and labels `t` in $\{-1, +1\}$ of shape `(N,)`. Use the augmented form of Section 6 internally: one parameter vector of length $D + 1$, starting at zero, updated with $\eta = 1$ one example at a time in the stored order. Stop after the first epoch with no mistakes or after `epochs` epochs. Return `(w, b, mistakes_per_epoch)`: `w` of shape `(D,)`, `b` a 0-d tensor, and a list of ints.
#
# You may use vectorized code for anything that is not an update (for example building the augmented matrix), but the updates must happen per example. Do not call `perceptron_train`; the check compares your result against it.

# %% [markdown] tags=["hint"]
# Build `Xa = torch.cat([X, torch.ones(n, 1, dtype=X.dtype)], dim=1)`, start from `theta = torch.zeros(d + 1, dtype=X.dtype)`, and loop: for each epoch, for each `i`, test `t[i] * (Xa[i] @ theta) <= 0`. At the end, `theta[:d]` is `w` and `theta[d]` is `b`.

# %%
def train_perceptron(X, t, epochs):
    # BEGIN SOLUTION
    n, d = X.shape
    Xa = torch.cat([X, torch.ones(n, 1, dtype=X.dtype)], dim=1)   # (N, D + 1)
    theta = torch.zeros(d + 1, dtype=X.dtype)                      # (D + 1,)
    mistakes_per_epoch = []
    for _ in range(epochs):
        m = 0
        for i in range(n):
            if t[i] * (Xa[i] @ theta) <= 0:
                theta = theta + t[i] * Xa[i]
                m += 1
        mistakes_per_epoch.append(m)
        if m == 0:
            break
    return theta[:d], theta[d], mistakes_per_epoch
    # END SOLUTION

# %% tags=["check"]
_Xs, _ys = make_blobs(n=40, centers=((-1.0, 2.0), (2.0, 0.0)), std=0.5, seed=3)
_p = torch.randperm(len(_ys), generator=torch.Generator().manual_seed(3))
_Xs, _ts = _Xs[_p], (2 * _ys[_p] - 1).float()
w_i, b_i, m_i = train_perceptron(_Xs, _ts, 100)
check_shape(w_i, (2,), "w")
check_shape(b_i, (), "b")
check(isinstance(m_i, list) and len(m_i) >= 1, "mistakes_per_epoch should be a non-empty list")
check(m_i[-1] == 0, f"the last epoch should have zero mistakes on separable data, got {m_i[-1]}")
check(bool((_ts * (_Xs @ w_i + b_i) > 0).all()), "the returned (w, b) must classify every training point correctly")
_w_ref, _b_ref, _m_ref, _ = perceptron_train(_Xs, _ts, 100)
check(list(m_i) == _m_ref, f"mistakes per epoch {list(m_i)} differ from the reference {_m_ref}")
check_close(w_i, _w_ref, "w", atol=1e-5)
check_close(b_i, _b_ref, "b", atol=1e-5)
_w4, _b4, _m4 = train_perceptron(_Xs, _ts, 1)
check(len(_m4) == 1, "with epochs=1 the function must run exactly one epoch")
passed("Exercise 2")

# %% [markdown]
# ### Exercise 3 (debug): a perceptron that ignores half the data
#
# `perceptron_from_01_labels` is meant to train a perceptron directly from class labels `y` in $\{0, 1\}$, starting from zero with $\eta = 1$. The buggy version in the comments runs without error, but it never learns to separate the classes. It contains **two** bugs. Find both and write the corrected loop body.

# %% [markdown] tags=["hint"]
# Trace the very first example by hand, starting from $\mathbf{w} = \mathbf{0}$, $b = 0$: what is the score, and does the condition fire? Then ask what an update does when the label is $0$.

# %%
def perceptron_from_01_labels(X, y, epochs=100):
    n, d = X.shape
    w = torch.zeros(d)
    b = torch.tensor(0.0)
    for _ in range(epochs):
        mistakes = 0
        for i in range(n):
            # Buggy version:
            #     z = X[i] @ w + b
            #     if y[i] * z < 0:
            #         w = w + y[i] * X[i]
            #         b = b + y[i]
            #         mistakes += 1
            # BEGIN SOLUTION
            t_i = 2.0 * y[i] - 1.0                 # labels must be ±1 for the rule
            z = X[i] @ w + b
            if t_i * z <= 0:                       # ties (z = 0) are mistakes
                w = w + t_i * X[i]
                b = b + t_i
                mistakes += 1
            # END SOLUTION
        if mistakes == 0:
            break
    return w, b

# %% tags=["check"]
w_dbg, b_dbg = perceptron_from_01_labels(X, y)
acc_dbg = accuracy(X, t, w_dbg, b_dbg)
check(acc_dbg == 1.0, f"training accuracy is {acc_dbg:.2f}; a working perceptron separates these blobs perfectly")
check_close(w_dbg, w_perc, "w", atol=1e-5)
check_close(b_dbg, b_perc, "b")
passed("Exercise 3")

# %% [markdown]
# To see what each bug does on its own, here is the buggy loop with switches for the two fixes:

# %%
def perceptron_buggy(X, y, epochs=100, use_pm1=False, ties_are_mistakes=False):
    n, d = X.shape
    w, b, updates = torch.zeros(d), torch.tensor(0.0), 0
    for _ in range(epochs):
        mistakes = 0
        for i in range(n):
            lab = 2.0 * y[i] - 1.0 if use_pm1 else y[i].float()
            z = X[i] @ w + b
            if (lab * z <= 0) if ties_are_mistakes else (lab * z < 0):
                w, b = w + lab * X[i], b + lab
                mistakes += 1
                updates += (lab != 0).item()          # count updates that change something
        if mistakes == 0:
            break
    return w, b, updates


for pm1, ties, name in [(False, False, "both bugs"), (True, False, "only the tie bug"), (False, True, "only the 0/1 bug"), (True, True, "fixed")]:
    w_, b_, upd = perceptron_buggy(X, y, use_pm1=pm1, ties_are_mistakes=ties)
    print(f"{name:18s}: effective updates {upd:4d}   w = {[round(v, 2) for v in w_.tolist()]}, b = {b_.item():6.1f}   "
          f"accuracy {accuracy(X, t, w_, b_):.2f}   predicted class 1 for {int((predict(X, w_, b_) == 1).sum())} of {N}")

# %% [markdown] tags=["solution"]
# **Bug 1 (strict inequality).** The mistake test must be $t z \le 0$, not $< 0$. From the zero start every score is exactly $0$, so $t z = 0$ for every example and `< 0` never fires: no update ever happens, the first epoch "has no mistakes", and the loop stops with $\mathbf{w} = \mathbf{0}$, $b = 0$, which predicts $-1$ for everything (50% accuracy). The table shows this both for the original code and when only this bug remains.
#
# **Bug 2 (0/1 labels).** The rule needs $t \in \{-1, +1\}$. With $y \in \{0, 1\}$, a class-0 example gives $y z = 0$ whatever $z$ is, so under `< 0` it can never be detected as wrong; and under `<= 0` it *is* flagged on every visit, but the update adds $0\cdot\mathbf{x}$ and $0$ — negative examples never push the boundary. Only class-1 examples move it, and always toward predicting class 1. In the run with only this bug, the first class-1 example (a tie at $z = 0$) causes one real update, $\mathbf{w} = \mathbf{x}_i$, $b = 1$; all our data lie in the positive quadrant, so after that every score is positive, all 100 points are predicted class 1 and no class-1 example is ever wrong again. Every class-0 example still counts as a "mistake" on every visit, so no epoch is mistake-free and the loop runs all 100 epochs without changing anything: 50% accuracy again.
#
# **Fix:** convert $t = 2y - 1$ and test $t z \le 0$. Then the function reproduces `w_perc`, `b_perc` exactly. Lesson: label conventions are part of an algorithm's definition. The same mismatch reappears with loss functions in notebooks 09–10, which expect labels in $\{0, 1\}$ or class indices.

# %% [markdown]
# ### Exercise 4 (investigate): margin and the speed of learning
#
# How long does the perceptron take to converge? The scaffold makes datasets with a controlled **gap** between the classes: it draws two blobs centred at $(1, 2)$ and $(3, 2)$, then deletes every class-0 point with $x_1 > 2 - \text{gap}/2$ and every class-1 point with $x_1 < 2 + \text{gap}/2$. The vertical line $x_1 = 2$ then separates the classes with every point at distance at least $\gamma = \text{gap}/2$ from it — a **margin** of at least $\gamma$. For several gaps and 20 random datasets each, it counts the total number of updates until the perceptron converges (from zero, $\eta = 1$).
#
# Run it, then answer in the markdown cell:
#
# 1. How does the number of updates depend on the margin $\gamma$? Use both the mean and the median.
# 2. Why might the mean and the median disagree for small margins?
# 3. Explain in words why a bigger margin should mean fewer updates. (Think about how far the boundary can move without misclassifying any point.)
#
# Optionally, move the blobs away from the origin (change `OFFSET`) and see what happens to the number of updates at a fixed margin.

# %% [markdown] tags=["hint"]
# Each update changes $\mathbf{w}$ by a vector of bounded length, $\|\tilde{\mathbf{x}}\|$. With a large margin, a whole range of directions separates the data, and the perceptron only needs to land somewhere in that range; with a tiny margin, the range of acceptable boundaries is a thin sliver. Notebook 08 turns this into a proof.

# %%
def gapped_blobs(gap, seed, offset):
    """Blobs around (offset - 1, offset) and (offset + 1, offset) with every point at least gap/2 from x1 = offset."""
    Xg, yg = make_blobs(n=100, centers=((offset - 1.0, offset), (offset + 1.0, offset)), std=0.7, seed=seed)
    keep = ((yg == 0) & (Xg[:, 0] < offset - gap / 2)) | ((yg == 1) & (Xg[:, 0] > offset + gap / 2))
    Xg, yg = Xg[keep], yg[keep]
    p = torch.randperm(len(yg), generator=torch.Generator().manual_seed(seed))
    return Xg[p], (2 * yg[p] - 1).float()


OFFSET = 2.0
gaps = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
SEEDS = range(20)
results = {}
for gap in gaps:
    totals = []
    for seed in SEEDS:
        Xg, tg = gapped_blobs(gap, seed, OFFSET)
        _, _, m_g, _ = perceptron_train(Xg, tg, epochs=1000)
        check(m_g[-1] == 0, f"did not converge for gap {gap}, seed {seed}")
        totals.append(sum(m_g))
    results[gap] = torch.tensor(totals, dtype=torch.float32)
    print(f"gap {gap:4.2f} (margin ≥ {gap / 2:5.3f}): mean updates {results[gap].mean().item():6.1f}   "
          f"median {results[gap].median().item():6.1f}   max {int(results[gap].max()):4d}")

margins = [g / 2 for g in gaps]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.loglog(margins, [results[g].mean().item() for g in gaps], "o-", label="mean over 20 datasets")
ax.loglog(margins, [results[g].median().item() for g in gaps], "s--", label="median")
for g in gaps:
    ax.scatter([g / 2] * len(results[g]), results[g], s=6, color="gray", alpha=0.5)
ax.set_xlabel("margin γ = gap / 2 (log scale)")
ax.set_ylabel("updates until convergence (log scale)")
ax.set_title("Perceptron: bigger margin, fewer updates")
ax.legend()
plt.show()

# %% [markdown]
# *Your conclusion:*

# %% [markdown] tags=["solution"]
# **The trend.** The mean number of updates falls steadily as the margin grows: about 306 at $\gamma = 0.01$, 103 at $\gamma = 0.1$, 33 at $\gamma = 0.4$, and 11 at $\gamma = 0.8$. The median tells a slightly different story: it stays near 100 for all $\gamma \le 0.1$ and only then falls, to 10 at $\gamma = 0.8$. For margins from 0.1 upward, each doubling of the margin cuts the mean by roughly 40–70%. Below 0.1 the typical dataset hardly gets harder, because most points lie far from the boundary however small the gap is; what changes is the worst case.
#
# **Mean vs median.** For a given margin the counts vary a lot between datasets (the gray dots), and the distribution is skewed: at the two smallest margins one dataset needed 2507 updates while the median was about 100. The mean is pulled up by such rare hard cases, the median is not, so the two curves separate exactly where the spread is largest — at small margins, where a few points sit very close to the boundary and the perceptron can need many corrections to squeeze the line between them.
#
# **Why the margin matters.** If every point is at distance at least $\gamma$ from some separating line, then every line close enough to it also separates the data: there is a whole *cone* of acceptable weight directions, and its width grows with $\gamma$. Each update moves $\mathbf{w}$ by a bounded amount ($\|\tilde{\mathbf{x}}_i\|$), and (notebook 08 shows) always gains a definite amount of alignment with the good direction. A wide cone is easy to land in; a thin sliver needs many corrections.
#
# **Offset.** The data's distance from the origin also matters: with `OFFSET = 0` far fewer updates are needed at the same margin (in our runs, a mean of about 9 instead of 306 at $\gamma = 0.01$, and about 5 instead of 103 at $\gamma = 0.1$). The bias changes by only $\pm 1$ per update, while the changes to $\mathbf{w}$ are large when the inputs are far from the origin, so a boundary far from the origin takes many updates to shift into place.
#
# **The formal result** (stated here without proof; notebook 08 proves it): if some unit-length augmented weight vector $\mathbf{u}$ gives every example $t_i\,\mathbf{u}\cdot\tilde{\mathbf{x}}_i \ge \tilde\gamma > 0$, the perceptron makes at most $(R/\tilde\gamma)^2$ updates, whatever the order, where $R = \max_i\|\tilde{\mathbf{x}}_i\|$. Both quantities are measured on the augmented inputs, which is why the offset matters: for the line $x_1 = 2$ the augmented margin is $\tilde\gamma = \gamma/\sqrt{1 + 2^2}$, and $R$ grows with the offset too. For our datasets this bound is above 14 000 at $\gamma = 0.01$ and about 200 at $\gamma = 0.8$, comfortably above every count we observed.

# %% [markdown]
# ## Summary and what's next
#
# - **Classification** maps inputs to classes. A perceptron scores an input with $z = \mathbf{w}\cdot\mathbf{x} + b$ and predicts $+1$ if $z > 0$, else $-1$. We use labels $t = 2y - 1 \in \{-1, +1\}$, so "correct" means $t z > 0$.
# - **Geometry:** the decision boundary $\mathbf{w}\cdot\mathbf{x} + b = 0$ is a line (a hyperplane in $D$ dimensions); $\mathbf{w}$ is perpendicular to it and points to the positive side; $b$ shifts it; $z/\|\mathbf{w}\|$ is the signed distance. Scaling $(\mathbf{w}, b)$ by $c > 0$ changes nothing.
# - **Learning rule:** on a mistake ($t z \le 0$), $\mathbf{w} \leftarrow \mathbf{w} + \eta t\mathbf{x}$, $b \leftarrow b + \eta t$. Each update raises that example's $t z$ by $\eta(\|\mathbf{x}\|^2 + 1)$. From a zero start, $\eta$ only rescales the weights. The learned $\mathbf{w}$ is a combination of the misclassified training inputs.
# - **Training** visits examples in epochs and stops after an epoch without mistakes. On our separable blobs it converged in 7 epochs. Folding $b$ into $\mathbf{w}$ with a constant input 1 gives the same algorithm with one update line.
# - **As a loss:** the perceptron is stochastic gradient descent on the perceptron criterion $\sum_i \max(0, -t_i z_i)$, with ties treated as mistakes. Unlike MSE, it ignores correctly classified points. The step function has zero derivative almost everywhere, so a network of step units cannot be trained by backpropagation.
# - **Limits:** on overlapping classes the mistakes never stop and the boundary keeps jumping. On separable data the line found depends on the visiting order and need not leave a wide gap; a larger margin means fewer updates.
#
# Still unanswered: **when is convergence guaranteed, and how fast?** And **which problems can a single perceptron not solve at all**, however long it trains? Notebook 08 proves the perceptron convergence theorem — the $(R/\gamma)^2$ bound — and shows that something as simple as XOR is beyond any single line. Notebook 09 then replaces the hard step with a smooth, probabilistic output, logistic regression, which gives sensible answers on overlapping data and opens the door to gradient-based training of networks.

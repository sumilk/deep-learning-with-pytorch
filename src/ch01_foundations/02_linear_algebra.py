# %% [markdown]
# # 02 — Linear algebra through pictures
#
# | | |
# | --- | --- |
# | **Prerequisites** | Notebooks 00–01 (tensors, shapes, broadcasting) |
# | **You will learn** | Vectors as arrows and as lists of numbers; norms, dot products, angles, and projections; matrix–vector products two ways; matrices as transformations of the plane; determinants; composition as matrix multiplication; affine maps and what `nn.Linear` computes; why stacking affine layers without a nonlinearity gains nothing; batched products, `torch.bmm`, and `torch.einsum` |
# | **Estimated time** | About 2.5 hours |
# | **Compute** | CPU only. Runs in a few seconds. |

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

import torch.nn as nn
import torch.nn.functional as F

from dlcourse.plotting import COLORS

# %% [markdown]
# ## Motivating problem
#
# A layer of a neural network does something very simple: it **multiplies its input by a matrix and adds a vector**. Two questions follow, and this notebook answers both:
#
# 1. What does "multiply by a matrix, add a vector" do to data, *geometrically*? If the data are points in the plane, where do they go?
# 2. If one layer is that simple, can we get something more powerful by stacking many of them? (Spoiler: not without one more ingredient.)
#
# Everything in this notebook runs on the CPU with small tensors, so you can check every number by hand.

# %% [markdown]
# ## 1. Vectors: arrows and lists of numbers
#
# A **vector** is an ordered list of numbers, such as $\mathbf{a} = (3, 1)$. We write vectors in bold lowercase. A vector with $D$ entries lives in $\mathbb{R}^D$ ("D-dimensional real space"), and $a_i$ denotes its $i$-th entry.
#
# In two dimensions a vector has a picture: an **arrow** from the origin to the point $(a_1, a_2)$. Two operations matter most:
#
# - **Addition**, entry by entry: $\mathbf{a} + \mathbf{b} = (a_1 + b_1,\; a_2 + b_2)$. Picture: put the tail of $\mathbf{b}$ at the tip of $\mathbf{a}$.
# - **Scaling** by a number $c$ (a *scalar*): $c\,\mathbf{a} = (c\,a_1,\; c\,a_2)$. Picture: stretch the arrow by $|c|$, and flip it if $c < 0$.
#
# In PyTorch a vector is a 1-D tensor, and both operations are the elementwise arithmetic from notebook 01.

# %%
a = torch.tensor([3.0, 1.0])  # (2,)
b = torch.tensor([1.0, 2.0])  # (2,)

print("a + b    =", a + b)
print("2 * a    =", 2 * a)
print("-0.5 * b =", -0.5 * b)

# %%
def arrow(ax, start, vec, color, label=None, **kwargs):
    """Draw `vec` as an arrow whose tail is at `start` (both 2-D tensors)."""
    ax.quiver(start[0].item(), start[1].item(), vec[0].item(), vec[1].item(),
              angles="xy", scale_units="xy", scale=1, color=color, label=label, **kwargs)


origin = torch.zeros(2)
fig, ax = plt.subplots(figsize=(5, 4.5))
arrow(ax, origin, a, COLORS[0], "a = (3, 1)")
arrow(ax, origin, b, COLORS[1], "b = (1, 2)")
arrow(ax, a, b, COLORS[1], alpha=0.35)       # b, moved to start at the tip of a
arrow(ax, origin, a + b, COLORS[2], "a + b = (4, 3)")
arrow(ax, origin, -0.5 * b, COLORS[3], "−0.5 b = (−0.5, −1)")
ax.set_xlim(-1.5, 4.5)
ax.set_ylim(-1.5, 3.5)
ax.set_aspect("equal")
ax.set_xlabel("first entry $x_1$")
ax.set_ylabel("second entry $x_2$")
ax.set_title("Vector addition and scaling")
ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# Notice that $\mathbf{a} + \mathbf{b}$ ends exactly where the faded copy of $\mathbf{b}$ (placed at the tip of $\mathbf{a}$) ends: "tip to tail". Scaling by $-0.5$ halves $\mathbf{b}$ and reverses its direction.
#
# Where the picture stops being accurate: in deep learning, vectors usually have hundreds or thousands of entries. We cannot draw them, but every algebraic rule here holds unchanged in any dimension. That is why we learn the rules in 2-D, where we can see them.

# %% [markdown]
# ### Length: the Euclidean norm
#
# The **length** (Euclidean norm) of $\mathbf{a} \in \mathbb{R}^D$ is
#
# $$\|\mathbf{a}\| = \sqrt{a_1^2 + a_2^2 + \dots + a_D^2}.$$
#
# In 2-D this is Pythagoras' theorem: the arrow is the hypotenuse of a right triangle with sides $|a_1|$ and $|a_2|$. For $\mathbf{a} = (3, 1)$: $\|\mathbf{a}\| = \sqrt{9 + 1} = \sqrt{10} \approx 3.1623$.
#
# PyTorch computes it with `torch.linalg.vector_norm`. Dividing a vector by its norm gives a **unit vector** (length 1) pointing the same way.

# %%
print("by the formula:     ", (a ** 2).sum().sqrt().item())
print("vector_norm:        ", torch.linalg.vector_norm(a).item())
a_unit = a / torch.linalg.vector_norm(a)
print("unit vector:        ", a_unit, " length", torch.linalg.vector_norm(a_unit).item())

# %% [markdown]
# With a data matrix $X \in \mathbb{R}^{N \times D}$ — one example per **row**, as everywhere in this course — `dim=1` gives the length of every row at once:

# %%
X = torch.tensor([[3.0, 4.0],
                  [1.0, 0.0],
                  [-2.0, 2.0]])                     # (N=3, D=2)
row_norms = torch.linalg.vector_norm(X, dim=1)      # (3,)
print(row_norms)

# %% [markdown]
# The rows $(3, 4)$, $(1, 0)$, and $(-2, 2)$ have lengths $5$, $1$, and $\sqrt{8} \approx 2.83$.

# %% [markdown]
# ## 2. The dot product
#
# The **dot product** of two vectors of the same length multiplies matching entries and adds the results:
#
# $$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{D} a_i b_i.$$
#
# It turns two vectors into one number. For $\mathbf{a} = (3, 1)$ and $\mathbf{b} = (1, 2)$: $\mathbf{a} \cdot \mathbf{b} = 3 \cdot 1 + 1 \cdot 2 = 5$. Note also that $\mathbf{a} \cdot \mathbf{a} = \|\mathbf{a}\|^2$.
#
# PyTorch offers several equivalent spellings:

# %%
print("(a * b).sum() =", (a * b).sum().item())
print("torch.dot     =", torch.dot(a, b).item())
print("a @ b         =", (a @ b).item())

# %% [markdown]
# ### The geometric meaning
#
# The dot product measures how much two vectors point the same way. Precisely, if $\theta$ is the angle between them,
#
# $$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\,\|\mathbf{b}\|\cos\theta.$$
#
# **Derivation.** The dot product distributes over addition (this follows directly from the definition), so
# $\|\mathbf{a} - \mathbf{b}\|^2 = (\mathbf{a} - \mathbf{b}) \cdot (\mathbf{a} - \mathbf{b}) = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2\,\mathbf{a} \cdot \mathbf{b}$.
# The triangle with sides $\mathbf{a}$, $\mathbf{b}$, and $\mathbf{a} - \mathbf{b}$ also satisfies the **law of cosines** from school geometry (stated here without proof): $\|\mathbf{a} - \mathbf{b}\|^2 = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2\|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$. Comparing the two expressions gives the formula.
#
# Let us verify it. In 2-D we can measure each arrow's direction independently with `atan2`, take the difference of the angles, and compare.

# %%
angle_a = torch.atan2(a[1], a[0])        # direction of a, in radians
angle_b = torch.atan2(b[1], b[0])
theta = angle_b - angle_a
print(f"direction of a: {math.degrees(angle_a):6.2f}°   direction of b: {math.degrees(angle_b):6.2f}°   θ = {math.degrees(theta):.2f}°")

lhs = torch.dot(a, b)
rhs = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) * torch.cos(theta)
print(f"a · b = {lhs.item():.6f}     |a| |b| cos θ = {rhs.item():.6f}")

# %% [markdown]
# Both sides equal 5: the two arrows are exactly 45° apart, and $\sqrt{10}\cdot\sqrt{5}\cdot\cos 45° = \sqrt{50}/\sqrt{2} = 5$.
#
# Dividing out the lengths gives the **cosine similarity**, a number in $[-1, 1]$ that ignores length and keeps only direction:
#
# $$\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}.$$
#
# It is widely used to compare embeddings (Chapter 9). PyTorch has `F.cosine_similarity`:

# %%
print("by the formula:      ", (torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))).item())
print("F.cosine_similarity: ", F.cosine_similarity(a, b, dim=0).item())

# %% [markdown]
# ### Sign and orthogonality
#
# The sign of the dot product tells you the rough direction: positive when the angle is under 90°, zero at exactly 90°, negative beyond. The plot sweeps a unit vector $\mathbf{u}(\varphi) = (\cos\varphi, \sin\varphi)$ around the circle and records $\mathbf{a} \cdot \mathbf{u}(\varphi)$.

# %%
phis = torch.linspace(0, 2 * math.pi, 361)                 # (361,)
U = torch.stack([torch.cos(phis), torch.sin(phis)], dim=1)  # (361, 2): one unit vector per row
dots = U @ a                                                # (361,): a · u for every row at once

fig, ax = plt.subplots()
ax.plot(torch.rad2deg(phis), dots, color=COLORS[0])
ax.axhline(0, color="black", lw=0.8)
ax.axvline(math.degrees(angle_a), color=COLORS[1], ls="--", label=f"direction of a ({math.degrees(angle_a):.1f}°)")
ax.set_xlabel("direction φ of the unit vector u (degrees)")
ax.set_ylabel("a · u(φ)")
ax.set_title("Dot product of a = (3, 1) with a rotating unit vector")
ax.legend()
plt.show()
print(f"max of a·u on the 1° grid = {dots.max().item():.4f}   |a| = {torch.linalg.vector_norm(a).item():.4f}")

# %% [markdown]
# The curve is $\|\mathbf{a}\|\cos(\varphi - 18.4°)$: largest (equal to $\|\mathbf{a}\|$; the 1° grid of angles lands just beside the exact direction, so the printed maximum is a hair smaller) when $\mathbf{u}$ points along $\mathbf{a}$, zero when $\mathbf{u}$ is perpendicular to it, and most negative when $\mathbf{u}$ points the opposite way.
#
# Two vectors with dot product zero are called **orthogonal** (perpendicular). For example, $(1, 2) \cdot (-2, 1) = -2 + 2 = 0$.

# %%
print(torch.dot(torch.tensor([1.0, 2.0]), torch.tensor([-2.0, 1.0])).item())

# %% [markdown]
# ### Checkpoint 2.1
#
# For $\mathbf{u} = (1, 2, 3)$ and $\mathbf{v} = (4, -5, 6)$, compute
#
# - `dot_uv`: the dot product, using only elementwise multiplication and `.sum()`;
# - `cos_uv`: their cosine similarity, using `dot_uv` and `torch.linalg.vector_norm` (not `F.cosine_similarity`).
#
# Work out `dot_uv` by hand first.

# %% [markdown] tags=["hint"]
# $1 \cdot 4 + 2 \cdot (-5) + 3 \cdot 6 = 12$. For the cosine, divide by the product of the two lengths, $\sqrt{14}$ and $\sqrt{77}$.

# %%
u = torch.tensor([1.0, 2.0, 3.0])
v = torch.tensor([4.0, -5.0, 6.0])
# BEGIN SOLUTION
dot_uv = (u * v).sum()
cos_uv = dot_uv / (torch.linalg.vector_norm(u) * torch.linalg.vector_norm(v))
# END SOLUTION

# %% tags=["check"]
check_close(dot_uv, 12.0, "dot_uv")
check_close(cos_uv, 12.0 / math.sqrt(14 * 77), "cos_uv")
passed("Checkpoint 2.1")

# %% [markdown]
# ## 3. Projections
#
# How much of $\mathbf{a}$ points along $\mathbf{b}$? The **projection** of $\mathbf{a}$ onto $\mathbf{b}$ is the multiple $c\,\mathbf{b}$ that is closest to $\mathbf{a}$. What remains, the **residual** $\mathbf{r} = \mathbf{a} - c\,\mathbf{b}$, should be perpendicular to $\mathbf{b}$.
#
# **Derivation.** The squared distance from $\mathbf{a}$ to $c\,\mathbf{b}$ is
#
# $$\|\mathbf{a} - c\,\mathbf{b}\|^2 = \|\mathbf{a}\|^2 - 2c\,(\mathbf{a} \cdot \mathbf{b}) + c^2\|\mathbf{b}\|^2,$$
#
# a parabola in $c$ that opens upward. Its minimum is at its vertex, $c = \dfrac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}}$. So
#
# $$\operatorname{proj}_{\mathbf{b}}(\mathbf{a}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}}\,\mathbf{b}.$$
#
# With this $c$, the residual is orthogonal to $\mathbf{b}$: $\mathbf{r} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} - c\,(\mathbf{b} \cdot \mathbf{b}) = 0$.
#
# For $\mathbf{a} = (3, 1)$, $\mathbf{b} = (1, 2)$: $c = 5 / 5 = 1$, so the projection is $(1, 2)$ and the residual is $(2, -1)$.

# %%
c = torch.dot(a, b) / torch.dot(b, b)
proj = c * b
resid = a - proj
print("c =", c.item(), "  projection =", proj, "  residual =", resid)
print("residual · b =", torch.dot(resid, b).item())

# The vertex of the parabola really is the minimum: scan c and find the smallest distance.
cs = torch.linspace(-1, 3, 401)
dist = torch.linalg.vector_norm(a - cs[:, None] * b, dim=1)   # (401, 2) -> (401,)
print("c with the smallest distance on a grid:", cs[dist.argmin()].item())

# %%
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.axline((0, 0), (b[0].item(), b[1].item()), color="gray", lw=0.8, ls=":", label="line through b")
arrow(ax, origin, a, COLORS[0], "a")
arrow(ax, origin, 0.5 * b, COLORS[1], "b (drawn at half length)")
arrow(ax, origin, proj, COLORS[2], "projection of a onto b", width=0.006)
arrow(ax, proj, resid, COLORS[3], "residual a − proj")
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.0)
ax.set_aspect("equal")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Projection of a = (3, 1) onto b = (1, 2)")
ax.legend(loc="upper right", fontsize=8)
plt.show()

# %% [markdown]
# The residual (pink) meets the line through $\mathbf{b}$ at a right angle: dropping a perpendicular is the shortest way from a point to a line. The projection's length is $\|\mathbf{a}\|\cos\theta = \sqrt{10}\cos 45° = \sqrt{5}$, which is indeed $\|(1, 2)\|$.
#
# Projections return in least squares (notebook 06), in attention (Chapter 8), and whenever we split a vector into "the part along a direction" and "the rest".

# %% [markdown]
# ### Checkpoint 3.1
#
# Write `project(a, b)` that returns the projection of `a` onto `b` for 1-D tensors of any (equal) length. Assume `b` is not the zero vector.

# %% [markdown] tags=["hint"]
# Translate $\dfrac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}}\,\mathbf{b}$ directly: two calls to `torch.dot`, a division, and a multiplication.

# %%
def project(a, b):
    # BEGIN SOLUTION
    return (torch.dot(a, b) / torch.dot(b, b)) * b
    # END SOLUTION

# %% tags=["check"]
p = project(torch.tensor([3.0, 1.0]), torch.tensor([1.0, 2.0]))
check_close(p, torch.tensor([1.0, 2.0]), "project((3, 1), (1, 2))")
gen = torch.Generator().manual_seed(1)
a5, b5 = torch.randn(5, generator=gen), torch.randn(5, generator=gen)
p5 = project(a5, b5)
check_shape(p5, (5,), "project(a5, b5)")
check(abs(torch.dot(a5 - p5, b5).item()) < 1e-5, "the residual a - project(a, b) must be orthogonal to b")
check_close(project(a5, 3 * b5), p5, "project(a, 3b)", atol=1e-5)  # only b's direction matters
check_close(project(b5, b5), b5, "project(b, b)")
passed("Checkpoint 3.1")

# %% [markdown]
# ## 4. Matrix–vector products, two ways
#
# A **matrix** $A \in \mathbb{R}^{m \times n}$ is a rectangular grid of numbers with $m$ rows and $n$ columns; $A_{ij}$ is the entry in row $i$, column $j$. Multiplying it by a vector $\mathbf{x} \in \mathbb{R}^n$ gives a vector $A\mathbf{x} \in \mathbb{R}^m$:
#
# $$(A\mathbf{x})_i = \sum_{j=1}^{n} A_{ij}\, x_j \qquad \text{shapes: } (m, n) \cdot (n,) \to (m,).$$
#
# The inner sizes must match ($n$); the result has the outer size ($m$). There are two ways to read this formula, and both are worth having.
#
# **Rows view.** Entry $i$ of the result is the dot product of row $i$ of $A$ with $\mathbf{x}$.
#
# **Columns view.** The result is a weighted sum of the columns $\mathbf{c}_1, \dots, \mathbf{c}_n$ of $A$, with weights from $\mathbf{x}$: $A\mathbf{x} = x_1 \mathbf{c}_1 + x_2 \mathbf{c}_2 + \dots + x_n \mathbf{c}_n$. (Read the formula one output entry at a time to see that it is the same sum.)
#
# A small example to check by hand:
#
# $$A = \begin{pmatrix} 1 & 2 & 0 \\ 0 & 1 & -1 \end{pmatrix}, \quad \mathbf{x} = \begin{pmatrix} 1 \\ 1 \\ 2 \end{pmatrix}.$$
#
# Rows: $1 + 2 + 0 = 3$ and $0 + 1 - 2 = -1$. Columns: $1 \cdot (1, 0) + 1 \cdot (2, 1) + 2 \cdot (0, -1) = (3, -1)$.

# %%
A = torch.tensor([[1.0, 2.0, 0.0],
                  [0.0, 1.0, -1.0]])   # (m=2, n=3)
x = torch.tensor([1.0, 1.0, 2.0])     # (3,)

rows_view = torch.stack([torch.dot(A[i], x) for i in range(A.shape[0])])   # one dot product per row
cols_view = sum(x[j] * A[:, j] for j in range(A.shape[1]))                 # weighted sum of columns
print("rows view:   ", rows_view)
print("columns view:", cols_view)
print("A @ x:       ", A @ x, " shape", tuple((A @ x).shape))

# %% [markdown]
# PyTorch treats a 1-D tensor on the right of `@` as a column vector, so `(2, 3) @ (3,)` gives shape `(2,)`. If the inner sizes disagree, the product is undefined and PyTorch raises an error:

# %%
try:
    A @ torch.ones(2)
except RuntimeError as error:
    print("RuntimeError:", error)

# %% [markdown]
# The columns view has a consequence we use constantly: **every** $A\mathbf{x}$ is some combination of $A$'s columns. So the outputs can only reach the directions those columns span. Section 5 makes this visible.

# %% [markdown]
# ## 5. Matrices as transformations of the plane
#
# A $2 \times 2$ matrix $M$ sends every point $\mathbf{x}$ of the plane to a new point $M\mathbf{x}$. To see what it does, we transform a whole grid of points and redraw it.
#
# By the columns view, $M\mathbf{x} = x_1\,\mathbf{c}_1 + x_2\,\mathbf{c}_2$. Two facts follow:
#
# - The **basis vectors** $\mathbf{e}_1 = (1, 0)$ and $\mathbf{e}_2 = (0, 1)$ land on the columns: $M\mathbf{e}_1 = \mathbf{c}_1$, $M\mathbf{e}_2 = \mathbf{c}_2$. Reading the columns tells you where the axes go.
# - $M$ is **linear**: $M(\alpha\mathbf{x} + \beta\mathbf{y}) = \alpha M\mathbf{x} + \beta M\mathbf{y}$ for any scalars $\alpha, \beta$ (expand both sides entrywise). Consequently straight lines stay straight, parallel lines stay parallel, evenly spaced lines stay evenly spaced, and the origin stays put.
#
# The helper below draws the transformed grid. Horizontal lines are blue, vertical lines orange, the unit square $[0,1]^2$ is shaded green, and the arrows show where $\mathbf{e}_1$ and $\mathbf{e}_2$ land.

# %%
def draw_grid(ax, f, title, lim=2.0, n_lines=9, basis=True, view=3.2):
    """Draw the image of a square grid under f, which maps (P, 2) points to (P, 2) points."""
    ts = torch.linspace(-lim, lim, 201)
    for level in torch.linspace(-lim, lim, n_lines):
        horizontal = torch.stack([ts, torch.full_like(ts, level.item())], dim=1)   # (201, 2)
        vertical = torch.stack([torch.full_like(ts, level.item()), ts], dim=1)     # (201, 2)
        h, v = f(horizontal), f(vertical)
        ax.plot(h[:, 0].numpy(), h[:, 1].numpy(), color=COLORS[0], lw=0.8, alpha=0.7)
        ax.plot(v[:, 0].numpy(), v[:, 1].numpy(), color=COLORS[1], lw=0.8, alpha=0.7)
    square = f(torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    ax.fill(square[:, 0].numpy(), square[:, 1].numpy(), color=COLORS[2], alpha=0.45)
    if basis:
        o, img = f(torch.zeros(1, 2))[0], f(torch.eye(2))   # image of the origin and of e1, e2 (rows)
        arrow(ax, o, img[0] - o, "black", width=0.012)
        arrow(ax, o, img[1] - o, COLORS[3], width=0.012)
        ax.annotate("e₁", img[0].numpy(), xytext=(4, -10), textcoords="offset points")
        ax.annotate("e₂", img[1].numpy(), xytext=(4, 4), textcoords="offset points", color=COLORS[3])
    ax.set_xlim(-view, view)
    ax.set_ylim(-view, view)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title, fontsize=9)


def linear_map(M):
    """Return f(P) = P @ M.T: applies M to every row (point) of P."""
    return lambda P: P @ M.T


transforms = {
    "identity": torch.eye(2),
    "rotation by 90°": torch.tensor([[0.0, -1.0], [1.0, 0.0]]),
    "scaling diag(1.5, 0.5)": torch.tensor([[1.5, 0.0], [0.0, 0.5]]),
    "shear": torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
    "reflection across x₁-axis": torch.tensor([[1.0, 0.0], [0.0, -1.0]]),
    "projection onto the line x₂ = x₁": torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
}

fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
for ax, (name, M) in zip(axes.flat, transforms.items()):
    draw_grid(ax, linear_map(M), f"{name}\nM = {M.tolist()}")
fig.suptitle("Six 2×2 matrices acting on a grid of points")
plt.tight_layout()
plt.show()

# %% [markdown]
# Notice, panel by panel:
#
# - **Rotation by 90°:** the columns $(0, 1)$ and $(-1, 0)$ are $\mathbf{e}_1$ and $\mathbf{e}_2$ turned a quarter turn counterclockwise. Lengths and angles are preserved.
# - **Scaling:** the $x_1$ direction stretches by 1.5, the $x_2$ direction shrinks by half; the grid stays axis-aligned.
# - **Shear:** $\mathbf{e}_1$ stays put while $\mathbf{e}_2$ tilts to $(1, 1)$; squares become parallelograms.
# - **Reflection:** the plane is flipped over, so $\mathbf{e}_2$ (pink) now points down. Going counterclockwise from $\mathbf{e}_1$ to $\mathbf{e}_2$ became clockwise.
# - **Projection:** $M = \mathbf{w}\mathbf{w}^\top$ with $\mathbf{w} = (1, 1)/\sqrt{2}$, so $M\mathbf{x} = (\mathbf{w} \cdot \mathbf{x})\,\mathbf{w}$ — exactly Section 3's projection. The whole plane collapses onto one line; the unit square becomes a line segment with zero area.
#
# In every panel, grid lines stay straight, parallel, and evenly spaced, as linearity promised.

# %% [markdown]
# Let us confirm numerically that the columns are the images of the basis vectors:

# %%
M = transforms["shear"]
e1, e2 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
print("M @ e1 =", M @ e1, "  first column  =", M[:, 0])
print("M @ e2 =", M @ e2, "  second column =", M[:, 1])

# %% [markdown]
# ### The determinant: how areas scale
#
# Every panel scaled the area of the unit square by some factor. For a $2 \times 2$ matrix that factor is the **determinant**,
#
# $$\det\begin{pmatrix} p & q \\ r & s \end{pmatrix} = ps - qr,$$
#
# with a sign: negative when the transformation flips orientation (as a reflection does). We state this without proof, and verify it by measuring the transformed unit square directly with the **shoelace formula** (also stated without proof): a polygon with corners $(x_k, y_k)$ listed counterclockwise has signed area $\tfrac{1}{2}\sum_k (x_k y_{k+1} - x_{k+1} y_k)$, wrapping around at the end. Listing the corners in clockwise order instead flips the sign.

# %%
def shoelace_area(corners):
    """Signed area of a polygon with corners given as rows of a (K, 2) tensor."""
    x, y = corners[:, 0], corners[:, 1]
    x_next, y_next = torch.roll(x, -1), torch.roll(y, -1)
    return 0.5 * (x * y_next - x_next * y).sum()


unit_square = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])   # counterclockwise
print(f"{'transformation':36s} {'area of image':>14s} {'det(M)':>8s} {'rank':>5s}")
for name, M in transforms.items():
    area = shoelace_area(unit_square @ M.T)
    print(f"{name:36s} {area.item():14.3f} {torch.linalg.det(M).item():8.3f} {torch.linalg.matrix_rank(M).item():5d}")

# %% [markdown]
# The measured signed area matches $\det M$ in every case: rotations and shears preserve area (1), the scaling multiplies it by $1.5 \times 0.5 = 0.75$, the reflection gives $-1$ (orientation flipped), and the projection gives 0.
#
# The **rank** column counts how many independent directions the output occupies: 2 for all but the projection, whose outputs fill only a line (rank 1).
#
# A zero determinant means information is destroyed: different inputs land on the same output, so no matrix can undo the transformation. For the projection, $(1, -1)$ and the origin both map to $(0, 0)$, and asking for an inverse fails:

# %%
P = transforms["projection onto the line x₂ = x₁"]
print("P @ (1, -1) =", P @ torch.tensor([1.0, -1.0]), "   P @ (0, 0) =", P @ torch.zeros(2))
try:
    torch.linalg.inv(P)
except RuntimeError as error:   # torch.linalg.LinAlgError is a kind of RuntimeError
    print(type(error).__name__ + ":", str(error).splitlines()[0])

# %% [markdown]
# ### Checkpoint 5.1
#
# Write `rotation(theta)` returning the $2 \times 2$ matrix (shape `(2, 2)`, `float32`) that rotates the plane counterclockwise by `theta` radians. It must map $\mathbf{e}_1$ to $(\cos\theta, \sin\theta)$.

# %% [markdown] tags=["hint"]
# The first column is where $\mathbf{e}_1$ lands: $(\cos\theta, \sin\theta)$. The second column is where $\mathbf{e}_2$ lands: $\mathbf{e}_2$ starts at 90°, so it ends at angle $\theta + 90°$, which is the point $(-\sin\theta, \cos\theta)$.

# %%
def rotation(theta):
    # BEGIN SOLUTION
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s],
                         [s, c]])
    # END SOLUTION

# %% tags=["check"]
R = rotation(0.3)
check_shape(R, (2, 2), "rotation(0.3)")
check_close(R @ torch.tensor([1.0, 0.0]), torch.tensor([math.cos(0.3), math.sin(0.3)]), "rotation(0.3) @ e1")
check_close(rotation(math.pi / 2) @ torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), "rotation(pi/2) @ e1")
check_close(rotation(math.pi / 2) @ torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0]), "rotation(pi/2) @ e2")
check_close(torch.linalg.det(R), 1.0, "det(rotation(0.3))")
check_close(R.T @ R, torch.eye(2), "R.T @ R (a rotation preserves lengths and angles)")
passed("Checkpoint 5.1")

# %% [markdown]
# ## 6. Composition is matrix multiplication
#
# Apply $B$, then apply $A$: $\mathbf{x} \mapsto A(B\mathbf{x})$. Is there a single matrix that does both at once? Writing it out entrywise,
#
# $$\big(A(B\mathbf{x})\big)_i = \sum_j A_{ij} \sum_k B_{jk} x_k = \sum_k \Big(\sum_j A_{ij} B_{jk}\Big) x_k.$$
#
# The bracket defines the **matrix product** $C = AB$, with $C_{ik} = \sum_j A_{ij}B_{jk}$. So $A(B\mathbf{x}) = (AB)\mathbf{x}$: **multiplying matrices composes transformations**, and in $AB$ the right-hand matrix $B$ acts *first*.
#
# Shapes: $(m, n) \cdot (n, p) \to (m, p)$. Again, inner sizes must match. Entry $C_{ik}$ is the dot product of row $i$ of $A$ with column $k$ of $B$.

# %%
gen = torch.Generator().manual_seed(0)
A = torch.randn(2, 3, generator=gen)   # (2, 3)
B = torch.randn(3, 4, generator=gen)   # (3, 4)
x = torch.randn(4, generator=gen)      # (4,)
print("shape of A @ B:", tuple((A @ B).shape))
print("A(Bx) =", A @ (B @ x))
print("(AB)x =", (A @ B) @ x)

# %% [markdown]
# ### Order matters
#
# Take the 90° rotation $R$ and the shear $S$ from Section 5. "Rotate, then shear" is $SR$; "shear, then rotate" is $RS$.

# %%
R90 = transforms["rotation by 90°"]
S = transforms["shear"]
print("S @ R (rotate, then shear):\n", S @ R90)
print("R @ S (shear, then rotate):\n", R90 @ S)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
draw_grid(axes[0], linear_map(S @ R90), "rotate 90°, then shear: S R")
draw_grid(axes[1], linear_map(R90 @ S), "shear, then rotate 90°: R S")
fig.suptitle("The same two transformations in opposite orders")
plt.tight_layout()
plt.show()

# %% [markdown]
# The two pictures differ. Under $SR$ the originally vertical lines (orange) end up horizontal and the horizontal ones (blue) end up diagonal; under $RS$ the blue lines end up vertical and the orange ones diagonal. In general $AB \neq BA$. Matrix multiplication is **not commutative**, so the order of layers in a network matters.
#
# It *is* **associative**: $(AB)C = A(BC)$, because both sides apply $C$, then $B$, then $A$. In floating point the two groupings round differently (notebook 00), so we check with a tolerance. We also check a fact stated without proof: $\det(AB) = \det A \cdot \det B$ — area scale factors multiply.

# %%
gen = torch.Generator().manual_seed(1)
A, B, C = torch.randn(2, 3, generator=gen), torch.randn(3, 4, generator=gen), torch.randn(4, 5, generator=gen)
print("max |(AB)C - A(BC)| =", ((A @ B) @ C - A @ (B @ C)).abs().max().item())

A2, B2 = torch.randn(2, 2, generator=gen), torch.randn(2, 2, generator=gen)
print(f"det(AB) = {torch.linalg.det(A2 @ B2).item():.6f}   det(A)·det(B) = {(torch.linalg.det(A2) * torch.linalg.det(B2)).item():.6f}")

# %% [markdown]
# ### Checkpoint 6.1
#
# Build `M_ck`, the single $2 \times 2$ matrix that **first** rotates by 90° counterclockwise and **then** doubles the $x_1$ coordinate (scaling by `diag(2, 1)`). Use matrix multiplication of the two matrices; then check by hand where $\mathbf{e}_1$ and $\mathbf{e}_2$ should land.

# %% [markdown] tags=["hint"]
# The transformation applied first goes on the **right**: $M = D_2\,R_{90}$.

# %%
R90 = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
D2 = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
# BEGIN SOLUTION
M_ck = D2 @ R90
# END SOLUTION

# %% tags=["check"]
check_shape(M_ck, (2, 2), "M_ck")
check_close(M_ck @ torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), "M_ck @ e1 (e1 rotates to e2; doubling x1 leaves it alone)")
check_close(M_ck @ torch.tensor([0.0, 1.0]), torch.tensor([-2.0, 0.0]), "M_ck @ e2 (e2 rotates to -e1, then x1 doubles)")
passed("Checkpoint 6.1")

# %% [markdown]
# ## 7. Affine transformations: what a layer computes
#
# A linear map always keeps the origin fixed. To shift things around we add a vector $\mathbf{b}$:
#
# $$\mathbf{y} = W\mathbf{x} + \mathbf{b}, \qquad W \in \mathbb{R}^{M \times D},\; \mathbf{x} \in \mathbb{R}^{D},\; \mathbf{b}, \mathbf{y} \in \mathbb{R}^{M}.$$
#
# This is an **affine** transformation: a linear map followed by a translation. Strictly speaking it is not linear when $\mathbf{b} \neq \mathbf{0}$, since $f(\mathbf{0}) = \mathbf{b}$ and $f(2\mathbf{x}) \neq 2f(\mathbf{x})$. But it still keeps straight lines straight and parallel lines parallel.

# %%
W_shear = torch.tensor([[1.0, 0.5], [0.0, 1.0]])   # (M=2, D=2)
b_shift = torch.tensor([1.0, 0.5])                 # (2,)


def affine(P):
    return P @ W_shear.T + b_shift   # P: (points, 2)


x0 = torch.tensor([[1.0, 1.0]])
print("f(0)  =", affine(torch.zeros(1, 2))[0], "  (not the origin)")
print("f(2x) =", affine(2 * x0)[0], "  2 f(x) =", 2 * affine(x0)[0])

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
draw_grid(axes[0], linear_map(W_shear), "linear part only: x ↦ W x")
draw_grid(axes[1], affine, "affine: x ↦ W x + b,  b = (1, 0.5)")
fig.suptitle("Adding b translates the whole transformed grid")
plt.tight_layout()
plt.show()

# %% [markdown]
# The right panel is the left panel slid by $\mathbf{b} = (1, 0.5)$: the shape of the grid is unchanged, but the origin (where the arrows start) has moved.

# %% [markdown]
# ### The batch form, and `nn.Linear`
#
# A network processes a whole batch at once. Stack $N$ inputs as the rows of $X \in \mathbb{R}^{N \times D}$. Row $i$ is $\mathbf{x}_i$ written as a row, and transposing $W\mathbf{x}_i$ gives the row $\mathbf{x}_i^\top W^\top$ (the transpose of a product reverses the order; check it entrywise). Stacking these rows:
#
# $$Y = X W^\top + \mathbf{b}, \qquad (N, D)\cdot(D, M) \to (N, M).$$
#
# Adding $\mathbf{b}$ of shape $(M,)$ to $(N, M)$ is **broadcasting** from notebook 01: $\mathbf{b}$ is added to every row.
#
# This is precisely what `nn.Linear(D, M)` computes. It stores `weight` with shape `(out_features, in_features)` $= (M, D)$ — the shape of $W$ — and `bias` with shape $(M,)$.

# %%
gen = torch.Generator().manual_seed(0)
N, D, M_out = 5, 3, 2
X = torch.randn(N, D, generator=gen)       # (N, D)
W = torch.randn(M_out, D, generator=gen)   # (M, D)
b_vec = torch.randn(M_out, generator=gen)  # (M,)

Y = X @ W.T + b_vec                        # (N, D) @ (D, M) + (M,) -> (N, M)
print("Y shape:", tuple(Y.shape))

layer = nn.Linear(D, M_out)
print("layer.weight shape:", tuple(layer.weight.shape), "  layer.bias shape:", tuple(layer.bias.shape))
with torch.no_grad():                      # copying parameters is not part of any computation to differentiate
    layer.weight.copy_(W)
    layer.bias.copy_(b_vec)
print("max |Y - layer(X)|       =", (Y - layer(X)).abs().max().item())
print("max |Y[2] - (W x_2 + b)| =", (Y[2] - (W @ X[2] + b_vec)).abs().max().item())

# %% [markdown]
# Our formula and `nn.Linear` agree, and each row of $Y$ is $W\mathbf{x}_i + \mathbf{b}$ for the corresponding input. (Ignore `torch.no_grad()` for now; notebook 05 explains it.)

# %% [markdown]
# ## 8. Why depth needs a nonlinearity
#
# Now the second motivating question. Stack two affine layers:
#
# $$\mathbf{y} = W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = \underbrace{(W_2 W_1)}_{W}\,\mathbf{x} + \underbrace{(W_2\mathbf{b}_1 + \mathbf{b}_2)}_{\mathbf{b}}.$$
#
# The two layers collapse into **one** affine layer with $W = W_2W_1$ and $\mathbf{b} = W_2\mathbf{b}_1 + \mathbf{b}_2$. By induction, any number of stacked affine layers is still a single affine layer. Depth buys nothing. Let us verify the formula on a batch, with a hidden layer of width 4:

# %%
gen = torch.Generator().manual_seed(2)
W1, b1 = torch.randn(4, 2, generator=gen), torch.randn(4, generator=gen)   # layer 1: 2 -> 4
W2, b2 = torch.randn(2, 4, generator=gen), torch.randn(2, generator=gen)   # layer 2: 4 -> 2
X = torch.randn(100, 2, generator=gen)                                     # (N=100, D=2)

two_layers = (X @ W1.T + b1) @ W2.T + b2                                   # (100, 2)
W_one, b_one = W2 @ W1, W2 @ b1 + b2                                       # (2, 2), (2,)
one_layer = X @ W_one.T + b_one
print("max difference:", (two_layers - one_layer).abs().max().item())

# %% [markdown]
# The difference is at the level of float32 rounding: the two-layer stack *is* a single affine map.
#
# The fix is to apply a **nonlinear** function between the layers. The simplest common choice is the **ReLU**, $\operatorname{relu}(z) = \max(z, 0)$, applied to every entry. Chapter 2 develops this properly; here is a preview of what it does to the plane.
#
# Random weights make a messy picture, so we pick a small network by hand, with a hidden layer of width 3:
#
# $$W_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix},\; \mathbf{b}_1 = \begin{pmatrix} 2 \\ 2 \\ 0 \end{pmatrix}, \qquad W_2 = \begin{pmatrix} 1 & 0 & 0.8 \\ 0 & 1 & -0.8 \end{pmatrix},\; \mathbf{b}_2 = \begin{pmatrix} -2 \\ -2 \end{pmatrix}.$$
#
# On the drawn region $[-2, 2]^2$, hidden units 1 and 2 ($x_1 + 2$ and $x_2 + 2$) are never negative, so the ReLU leaves them alone. Hidden unit 3 is $x_1 + x_2$: the ReLU zeroes it on one side of the diagonal $x_1 + x_2 = 0$ and passes it on the other.

# %%
W1p = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])   # (3, 2)
b1p = torch.tensor([2.0, 2.0, 0.0])                        # (3,)
W2p = torch.tensor([[1.0, 0.0, 0.8], [0.0, 1.0, -0.8]])    # (2, 3)
b2p = torch.tensor([-2.0, -2.0])                           # (2,)


def affine_affine(P):
    return (P @ W1p.T + b1p) @ W2p.T + b2p


def affine_relu_affine(P):
    return torch.relu(P @ W1p.T + b1p) @ W2p.T + b2p


fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
draw_grid(axes[0], lambda P: P, "input grid", basis=False, view=5.5)
draw_grid(axes[1], affine_affine, "affine → affine\n(= one affine map)", basis=False, view=5.5)
draw_grid(axes[2], affine_relu_affine, "affine → ReLU → affine", basis=False, view=5.5)
fig.suptitle("Two layers without and with a ReLU in between")
plt.tight_layout()
plt.show()

# %% [markdown]
# Without the ReLU (middle), the whole grid is sheared and shifted by one affine map: every line stays straight and evenly spaced. With the ReLU (right), the half below the diagonal $x_1 + x_2 = 0$ is left exactly where it was (there the network computes $\mathbf{x} + (2, 2) - (2, 2) = \mathbf{x}$), while the half above it is sheared. Every grid line that crosses the diagonal **bends** there.
#
# So the ReLU network is affine within each region where the hidden units' on/off pattern is fixed, but with a different affine map in each region. With many hidden units there are many such regions, which is what lets deep networks represent curved decision boundaries.
#
# A precise test: an affine map sends the midpoint of two points to the midpoint of their images, because $f\big(\tfrac{\mathbf{x}+\mathbf{x}'}{2}\big) = \tfrac{f(\mathbf{x}) + f(\mathbf{x}')}{2}$ follows from linearity. The ReLU network violates this.

# %%
gen = torch.Generator().manual_seed(3)
X_p = 4 * torch.rand(100, 2, generator=gen) - 2       # 100 points in [-2, 2]^2
X_q = 4 * torch.rand(100, 2, generator=gen) - 2
mid = (X_p + X_q) / 2
for name, f in [("affine → affine", affine_affine), ("affine → ReLU → affine", affine_relu_affine)]:
    gap = (f(mid) - (f(X_p) + f(X_q)) / 2).abs().max().item()
    print(f"{name:24s} max |f(midpoint) − midpoint of f| = {gap:.2e}")

# %% [markdown]
# ### Checkpoint 8.1
#
# Collapse the two layers below (2 → 3 → 4) into a single affine layer: compute `W_combined` (shape `(4, 2)`) and `b_combined` (shape `(4,)`) so that `X @ W_combined.T + b_combined` equals the two-layer output.

# %% [markdown] tags=["hint"]
# Use the derivation at the start of this section: $W = W_2 W_1$ and $\mathbf{b} = W_2\mathbf{b}_1 + \mathbf{b}_2$. Check the shapes: $(4, 3)\cdot(3, 2) \to (4, 2)$.

# %%
gen = torch.Generator().manual_seed(3)
V1, c1 = torch.randn(3, 2, generator=gen), torch.randn(3, generator=gen)   # layer 1: 2 -> 3
V2, c2 = torch.randn(4, 3, generator=gen), torch.randn(4, generator=gen)   # layer 2: 3 -> 4
# BEGIN SOLUTION
W_combined = V2 @ V1
b_combined = V2 @ c1 + c2
# END SOLUTION

# %% tags=["check"]
check_shape(W_combined, (4, 2), "W_combined")
check_shape(b_combined, (4,), "b_combined")
X_test = torch.randn(10, 2, generator=torch.Generator().manual_seed(4))
expected = (X_test @ V1.T + c1) @ V2.T + c2
check_close(X_test @ W_combined.T + b_combined, expected, "single-layer output", atol=1e-5)
passed("Checkpoint 8.1")

# %% [markdown]
# ## 9. Batched and general products
#
# Real models multiply stacks of matrices: one per example, per attention head, and so on. `torch.matmul` (the `@` operator) handles this with a few rules:
#
# - **1-D × 1-D:** dot product, result shape `()`.
# - **2-D × 1-D:** matrix–vector, `(m, n) @ (n,) → (m,)`; **1-D × 2-D:** `(n,) @ (n, p) → (p,)`.
# - **2-D × 2-D:** matrix product, `(m, n) @ (n, p) → (m, p)`.
# - **More dimensions:** the last two dimensions are multiplied as matrices; all **leading (batch) dimensions broadcast** using notebook 01's rules.
#
# `torch.bmm` is the strict version: both inputs must be 3-D with the same batch size, `(B, m, n)` and `(B, n, p)`, and nothing broadcasts.

# %%
gen = torch.Generator().manual_seed(0)
def r(*shape):
    return torch.randn(*shape, generator=gen)

cases = [
    ("(3,) @ (3,)", r(3), r(3)),
    ("(2, 3) @ (3,)", r(2, 3), r(3)),
    ("(3,) @ (3, 4)", r(3), r(3, 4)),
    ("(2, 3) @ (3, 4)", r(2, 3), r(3, 4)),
    ("(10, 2, 3) @ (3, 4)", r(10, 2, 3), r(3, 4)),
    ("(10, 2, 3) @ (10, 3, 4)", r(10, 2, 3), r(10, 3, 4)),
    ("(5, 1, 2, 3) @ (7, 3, 4)", r(5, 1, 2, 3), r(7, 3, 4)),
]
print(f"{'expression':28s} result shape")
for text, left, right in cases:
    print(f"{text:28s} {tuple((left @ right).shape)}")

# %% [markdown]
# Read the last two rows carefully. In `(10, 2, 3) @ (3, 4)`, the single `(3, 4)` matrix is broadcast across all 10 batch entries. In `(5, 1, 2, 3) @ (7, 3, 4)`, the batch dimensions `(5, 1)` and `(7,)` broadcast to `(5, 7)`, giving 35 independent `(2, 3) @ (3, 4)` products.
#
# Each batch entry really is an ordinary matrix product, and `bmm` agrees with `@` when both apply:

# %%
Ab, Bb = r(10, 2, 3), r(10, 3, 4)
print("max |(Ab @ Bb)[7] - Ab[7] @ Bb[7]| =", ((Ab @ Bb)[7] - Ab[7] @ Bb[7]).abs().max().item())
print("max |bmm - @|                     =", (torch.bmm(Ab, Bb) - Ab @ Bb).abs().max().item())
try:
    torch.bmm(Ab, r(3, 4))
except RuntimeError as error:
    print("bmm with a 2-D input -> RuntimeError:", str(error).splitlines()[0])

# %% [markdown]
# ### `torch.einsum`: products written as index formulas
#
# `torch.einsum` lets you write a product the way you would write its sum. Give each dimension a letter; letters that appear in the inputs but **not** in the output (after `->`) are summed over. For example `"ij,jk->ik"` reads $C_{ik} = \sum_j A_{ij}B_{jk}$: matrix multiplication. Some common patterns, each checked against the operation it replaces:

# %%
A, B = r(2, 3), r(3, 4)
p, q = r(3), r(3)
X, W = r(5, 3), r(2, 3)

examples = [
    ("ij,jk->ik", (A, B), A @ B, "matrix product"),
    ("i,i->", (p, q), p @ q, "dot product"),
    ("ij->ji", (A,), A.T, "transpose (nothing summed)"),
    ("bij,bjk->bik", (Ab, Bb), Ab @ Bb, "batched matrix product"),
    ("nd,md->nm", (X, W), X @ W.T, "X Wᵀ, as in nn.Linear"),
    ("nd,nd->n", (X, X), (X * X).sum(dim=1), "row-wise dot products"),
]
print(f"{'einsum string':16s} {'shapes in':24s} {'out':10s} matches  meaning")
for spec, operands, reference, meaning in examples:
    out = torch.einsum(spec, *operands)
    shapes = ", ".join(str(tuple(t.shape)) for t in operands)
    print(f"{spec:16s} {shapes:24s} {str(tuple(out.shape)):10s} {torch.allclose(out, reference, atol=1e-6)!s:8s} {meaning}")

# %% [markdown]
# `einsum` is often the clearest way to express an unusual contraction, and the index string doubles as documentation of the shapes. For standard products, `@` is just as fast and more familiar.
#
# A **shape-tracing** habit worth building: before running a product, write the shapes and cancel the summed dimension. `(N, D) @ (D, M)`: the `D`s meet and are summed, leaving `(N, M)`.

# %% [markdown]
# ### Checkpoint 9.1
#
# Predict the result shapes **without running anything**, as tuples in the dictionary `predicted`. Then run the check, which computes them.
#
# 1. `(4, 5) @ (5,)`
# 2. `(8, 4, 5) @ (5, 6)`
# 3. `(8, 1, 4, 5) @ (3, 5, 6)`
# 4. `torch.einsum("bnd,md->bnm", t1, t2)` with `t1` of shape `(8, 4, 5)` and `t2` of shape `(6, 5)`

# %% [markdown] tags=["hint"]
# Handle the last two dimensions as a matrix product, then broadcast the leading (batch) dimensions. In case 3 the batch shapes are `(8, 1)` and `(3,)`. In case 4, `d` is summed over and the output keeps `b`, `n`, `m` in that order.

# %%
# BEGIN SOLUTION
predicted = {1: (4,), 2: (8, 4, 6), 3: (8, 3, 4, 6), 4: (8, 4, 6)}
# END SOLUTION

# %% tags=["check"]
z = torch.zeros
actual = {
    1: z(4, 5) @ z(5),
    2: z(8, 4, 5) @ z(5, 6),
    3: z(8, 1, 4, 5) @ z(3, 5, 6),
    4: torch.einsum("bnd,md->bnm", z(8, 4, 5), z(6, 5)),
}
for key, result in actual.items():
    check(tuple(predicted[key]) == tuple(result.shape), f"case {key}: you predicted {tuple(predicted[key])}, actual {tuple(result.shape)}")
passed("Checkpoint 9.1")

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1 (modify): matrices that flip and collapse
#
# The cell below draws the grid under a matrix `M_ex` and prints its determinant. Run it as given. Then change `M_ex` to each of these and run again:
#
# 1. `[[1.0, 2.0], [0.5, 1.0]]`
# 2. `[[0.0, 1.0], [1.0, 0.0]]`
#
# For each, describe what happens to the grid, to the area of the green unit square, and to the orientation of the $\mathbf{e}_1$, $\mathbf{e}_2$ arrows, and connect it to the printed determinant.

# %%
M_ex = torch.tensor([[1.0, 0.5], [0.5, 1.0]])

fig, ax = plt.subplots(figsize=(4.5, 4.5))
draw_grid(ax, linear_map(M_ex), f"M = {M_ex.tolist()}")
plt.show()
print(f"det(M) = {torch.linalg.det(M_ex).item():.3f}   rank = {torch.linalg.matrix_rank(M_ex).item()}")

# %% [markdown]
# *Your answer:*

# %% [markdown] tags=["solution"]
# **As given**, $\det M = 1 - 0.25 = 0.75$. Since $M(1, 1) = (1.5, 1.5)$ and $M(1, -1) = (0.5, -0.5)$, the plane is stretched by 1.5 along the diagonal $x_2 = x_1$ and squeezed by 0.5 along the other diagonal; the square becomes a parallelogram of area $1.5 \times 0.5 = 0.75$, with orientation preserved.
#
# **`[[1, 2], [0.5, 1]]`:** $\det M = 1 \cdot 1 - 2 \cdot 0.5 = 0$ and the rank is 1. The columns $(1, 0.5)$ and $(2, 1)$ are parallel, so every output $x_1(1, 0.5) + x_2(2, 1) = (x_1 + 2x_2)(1, 0.5)$ lies on the single line through $(1, 0.5)$. The whole grid collapses onto that line, the unit square flattens into a segment of zero area, and both arrows point along the same line. The map cannot be undone: every point on a line $x_1 + 2x_2 = \text{const}$ lands on the same output.
#
# **`[[0, 1], [1, 0]]`:** $\det M = -1$. The map swaps the two coordinates, which is a reflection across the diagonal $x_2 = x_1$. The unit square maps onto itself (area 1), but $\mathbf{e}_1$ now points up and $\mathbf{e}_2$ points right: turning from $\mathbf{e}_1$ to $\mathbf{e}_2$ is now clockwise. The minus sign of the determinant records exactly this orientation flip.

# %% [markdown]
# ### Exercise 2 (implement): matrix multiplication with loops
#
# Write `matmul_loops(A, B)` for 2-D tensors `A` of shape `(m, n)` and `B` of shape `(n, p)`, using explicit Python loops and only scalar multiply-adds — no `@`, `torch.matmul`, `torch.dot`, `torch.einsum`, or `.sum()`. Return a tensor of shape `(m, p)`.
#
# If either input is not 2-D, or the inner sizes differ, raise a `ValueError` with a message that states both shapes.

# %% [markdown] tags=["hint"]
# Three nested loops: over output rows `i`, output columns `k`, and the shared index `j`. Accumulate `total += A[i, j].item() * B[j, k].item()` in a Python float, then store it in `C[i, k]`. Check the shapes before the loops, using `A.ndim` and `A.shape`.

# %%
def matmul_loops(A, B):
    # BEGIN SOLUTION
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError(f"cannot multiply shapes {tuple(A.shape)} and {tuple(B.shape)}")
    m, n = A.shape
    p = B.shape[1]
    C = torch.zeros(m, p, dtype=torch.result_type(A, B))
    for i in range(m):
        for k in range(p):
            total = 0.0
            for j in range(n):
                total += A[i, j].item() * B[j, k].item()
            C[i, k] = total
    return C
    # END SOLUTION

# %% tags=["check"]
gen = torch.Generator().manual_seed(5)
for m, n, p in [(2, 3, 4), (1, 5, 1), (4, 1, 3), (3, 3, 3)]:
    A_t, B_t = torch.randn(m, n, generator=gen), torch.randn(n, p, generator=gen)
    C_t = matmul_loops(A_t, B_t)
    check_shape(C_t, (m, p), f"matmul_loops on ({m}, {n}) @ ({n}, {p})")
    check_close(C_t, A_t @ B_t, f"matmul_loops on ({m}, {n}) @ ({n}, {p})", atol=1e-5)
check_close(matmul_loops(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0], [4.0]])), torch.tensor([[11.0]]), "[[1, 2]] @ [[3], [4]]")
for bad_A, bad_B in [(torch.ones(2, 3), torch.ones(2, 3)), (torch.ones(3), torch.ones(3, 2))]:
    try:
        matmul_loops(bad_A, bad_B)
    except ValueError:
        pass
    else:
        check(False, f"matmul_loops should raise ValueError for shapes {tuple(bad_A.shape)} and {tuple(bad_B.shape)}")
passed("Exercise 2")

# %% [markdown]
# Your loop version computes the same numbers as `@`, but far more slowly: each scalar operation goes through the Python interpreter. Chapter 5 measures how large that gap is and why it grows on a GPU.

# %% [markdown]
# ### Exercise 3 (debug): a layer that only works when it is square
#
# A colleague wrote a batch affine layer. They tested it with a square weight matrix, and it ran without error:

# %%
def buggy_affine(X, W, b):
    return X @ W + b


gen = torch.Generator().manual_seed(6)
X_sq, W_sq, b_sq = torch.randn(4, 3, generator=gen), torch.randn(3, 3, generator=gen), torch.randn(3, generator=gen)
print("output shape:", tuple(buggy_affine(X_sq, W_sq, b_sq).shape))
print("max |buggy - F.linear| =", (buggy_affine(X_sq, W_sq, b_sq) - F.linear(X_sq, W_sq, b_sq)).abs().max().item())

# %% [markdown]
# The shape looks right, but the values do not match `F.linear` (the function behind `nn.Linear`), and with a non-square `W` the layer crashes. Fix `affine_layer` below so that, for `X` of shape `(N, D)`, `W` of shape `(M, D)` (the `nn.Linear` convention), and `b` of shape `(M,)`, it returns $Y = XW^\top + \mathbf{b}$ of shape `(N, M)`.

# %% [markdown] tags=["hint"]
# Trace the shapes of `X @ W`: `(N, D) @ (M, D)`. The inner sizes are `D` and `M`, which only agree when the matrix is square — and even then the product uses the rows of `W` as columns. Section 7 has the correct formula.

# %%
def affine_layer(X, W, b):
    # Buggy version:
    #     return X @ W + b
    # BEGIN SOLUTION
    return X @ W.T + b
    # END SOLUTION

# %% tags=["check"]
gen = torch.Generator().manual_seed(7)
X_t, W_t, b_t = torch.randn(5, 3, generator=gen), torch.randn(2, 3, generator=gen), torch.randn(2, generator=gen)
out = affine_layer(X_t, W_t, b_t)
check_shape(out, (5, 2), "affine_layer output for X (5, 3), W (2, 3)")
check_close(out, F.linear(X_t, W_t, b_t), "affine_layer vs F.linear (non-square W)")
check_close(affine_layer(X_sq, W_sq, b_sq), F.linear(X_sq, W_sq, b_sq), "affine_layer vs F.linear (square W)")
passed("Exercise 3")

# %% [markdown]
# Lesson: a test that only uses square matrices cannot catch a transposition bug when the shapes happen to agree. When testing shape-handling code, make every dimension a different size.

# %% [markdown]
# ### Exercise 4 (investigate): can a wide linear "network" do more than a 2×2 matrix?
#
# Section 8 showed that two affine layers collapse into one. Perhaps a very **wide** hidden layer changes that? Consider $f(\mathbf{x}) = W_2(W_1\mathbf{x})$ with $W_1 \in \mathbb{R}^{k \times 2}$ (2 → $k$) and $W_2 \in \mathbb{R}^{2 \times k}$ ($k$ → 2), for hidden widths $k$ from 1 to 1000.
#
# The scaffold draws random $W_1, W_2$ for each $k$, then
#
# - measures the largest difference between $W_2(W_1\mathbf{x})$ and $M\mathbf{x}$ with the single $2 \times 2$ matrix $M = W_2 W_1$, over a grid of points;
# - reports the rank of $M$;
# - draws the transformed grid for three widths.
#
# Run it (try other seeds too), then answer in the markdown cell: does width ever make the composition more expressive than a single $2 \times 2$ matrix? Can it make it *less* expressive? Why?

# %% [markdown] tags=["hint"]
# Look at whether the grid lines stay straight and evenly spaced, and at the "max difference" column. Then think about the shape of $W_2W_1$, and about how many directions the hidden layer can pass through when $k = 1$.

# %%
gen = torch.Generator().manual_seed(6)
grid_pts = torch.stack(torch.meshgrid(torch.linspace(-2, 2, 21), torch.linspace(-2, 2, 21), indexing="ij"), dim=-1).reshape(-1, 2)  # (441, 2)
widths = [1, 2, 3, 10, 100, 1000]
layers = {}
print(f"{'k':>5s} {'max |W2(W1 x) − M x|':>22s} {'rank of M':>10s} {'det M':>7s} {'shape of M':>11s}")
for k in widths:
    W1_k = torch.randn(k, 2, generator=gen)
    W2_k = torch.randn(2, k, generator=gen) / math.sqrt(k)   # keeps outputs a similar size for every k
    layers[k] = (W1_k, W2_k)
    M_k = W2_k @ W1_k
    through_hidden = (grid_pts @ W1_k.T) @ W2_k.T            # (441, k) on the way, (441, 2) at the end
    diff = (through_hidden - grid_pts @ M_k.T).abs().max().item()
    print(f"{k:5d} {diff:22.2e} {torch.linalg.matrix_rank(M_k).item():10d} {torch.linalg.det(M_k).item():7.2f} {str(tuple(M_k.shape)):>11s}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, k in zip(axes, [1, 2, 1000]):
    W1_k, W2_k = layers[k]
    draw_grid(ax, lambda P, W1_k=W1_k, W2_k=W2_k: (P @ W1_k.T) @ W2_k.T, f"hidden width k = {k}", view=4)
fig.suptitle("x ↦ W₂(W₁x) through a hidden layer of width k")
plt.tight_layout()
plt.show()

# %% [markdown]
# *Your conclusion:*

# %% [markdown] tags=["solution"]
# Width never helps. For every $k$, the two-step computation agrees with the single $2 \times 2$ matrix $M = W_2W_1$ up to float32 rounding (differences of at most a few times $10^{-6}$, largest for the widest layers, whose longer sums accumulate more rounding), and in every panel the grid lines stay straight, parallel, and evenly spaced: the composition is one linear map of the plane, whatever happens in between. The $k = 1000$ panel is an ordinary parallelogram grid, no different in kind from the $k = 2$ panel. A 1000-dimensional hidden layer is just a more expensive way to write down four numbers.
#
# Width *can* hurt. With $k = 1$, everything passes through a single number $W_1\mathbf{x}$, so $M = W_2W_1$ has rank 1 its determinant is 0, and the whole plane collapses onto a line (left panel). In general $\operatorname{rank}(W_2W_1) \le \min(k, 2)$: a narrow layer is a **bottleneck** that can only destroy information, while a wide layer cannot create any beyond what a $2 \times 2$ matrix already can. (With random matrices and $k \ge 2$ the rank is 2 in practice; it is not guaranteed.)
#
# The conclusion holds for any input and output sizes: a composition of linear (or affine) maps is linear (or affine), regardless of depth or width. Only a nonlinearity between layers, as in Section 8, makes stacking worthwhile.

# %% [markdown]
# ## Summary and what's next
#
# - A vector is both a list of numbers and an arrow. Its length is the Euclidean norm; the dot product $\mathbf{a}\cdot\mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$ measures alignment, and zero means orthogonal.
# - The projection $\frac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\mathbf{b}$ is the closest point to $\mathbf{a}$ on the line through $\mathbf{b}$, and its residual is orthogonal to $\mathbf{b}$.
# - $A\mathbf{x}$ is a stack of row dot products, and equally a combination of $A$'s columns. A matrix's columns are where the basis vectors land.
# - A $2\times2$ matrix rotates, scales, shears, reflects, or collapses the plane; its determinant is the signed area scale factor, and determinant 0 means information is lost.
# - Matrix multiplication composes transformations, right-most first. It is associative but not commutative.
# - A layer computes the affine map $Y = XW^\top + \mathbf{b}$ — exactly `nn.Linear`. Stacked affine layers collapse into one, whatever their width; a nonlinearity such as ReLU between them bends space and breaks the collapse.
# - `@` broadcasts over batch dimensions, `torch.bmm` is the strict 3-D version, and `torch.einsum` writes any product as an index formula.
#
# **What remains unsolved.** We now know what a layer *does* for given $W$ and $\mathbf{b}$, but not how to *choose* them. Learning means adjusting the parameters to make a loss smaller, and for that we need to know how the loss changes when each parameter is nudged. That is the job of derivatives, gradients, and Jacobians — the subject of notebook 03, which ends with our first optimization algorithm, gradient descent.

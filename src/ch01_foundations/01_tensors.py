# %% [markdown]
# # 01 — Tensors from first principles
#
# | | |
# | --- | --- |
# | **Prerequisites** | Notebook 00 |
# | **You will learn** | What a tensor is; creating tensors; shapes and data types; indexing; elementwise operations and reductions; broadcasting; reshaping; how tensors are stored in memory (views, copies, strides); moving tensors between devices and NumPy |
# | **Estimated time** | 2.5–3 hours |
# | **Compute** | CPU. One optional GPU cell. |

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

# %% [markdown]
# ## Motivating problem
#
# A batch of 32 color images, each 64×64 pixels, is a block of 32 × 3 × 64 × 64 = 393,216 numbers. A neural network transforms such blocks over and over: multiplying, adding, summing, reshaping. Every one of these steps can go silently wrong if a single dimension is misread.
#
# This notebook builds the vocabulary and the habits for working with such blocks — **tensors** — correctly: always know the shape, know which dimension means what, and know whether an operation copies data or shares it.

# %% [markdown]
# ## 1. What is a tensor?
#
# A **tensor** is a rectangular array of numbers, all of the same type. Its **number of dimensions** (`ndim`) says how many indices you need to pick out one number:
#
# | ndim | Name | Example | Shape |
# | --- | --- | --- | --- |
# | 0 | scalar | a loss value | `()` |
# | 1 | vector | one example's features | `(D,)` |
# | 2 | matrix | a batch of feature vectors | `(N, D)` |
# | 3 | 3-D tensor | a batch of sequences | `(N, T, D)` |
# | 4 | 4-D tensor | a batch of color images | `(N, C, H, W)` |
#
# The **shape** lists the size of each dimension. In this course, the first dimension is almost always the **batch** dimension, written `N`: one entry per example.

# %%
scalar = torch.tensor(3.5)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])

for name, example in [("scalar", scalar), ("vector", vector), ("matrix", matrix)]:
    print(f"{name:6}: ndim={example.ndim}, shape={tuple(example.shape)}, numel={example.numel()}, dtype={example.dtype}")

# %% [markdown]
# `numel()` is the total number of elements: the product of the shape's sizes. A scalar has shape `()` and one element.

# %% [markdown]
# ### Creating tensors
#
# You rarely type numbers in by hand. The common constructors are:

# %%
gen = torch.Generator().manual_seed(0)

print(torch.zeros(2, 3))                       # all zeros
print(torch.ones(2, 3))                        # all ones
print(torch.full((2, 3), 7.0))                 # all the same value
print(torch.arange(0, 10, 2))                  # 0, 2, 4, 6, 8 (end excluded)
print(torch.linspace(0, 1, 5))                 # 5 evenly spaced points, both ends included
print(torch.rand(2, 3, generator=gen))         # uniform on [0, 1)
print(torch.randn(2, 3, generator=gen))        # standard normal: mean 0, std 1
print(torch.zeros_like(matrix))                # same shape and dtype as another tensor

# %% [markdown]
# ### Data types
#
# Every tensor has one **dtype**. The ones you will meet most:
#
# | dtype | Holds | Typical use |
# | --- | --- | --- |
# | `torch.float32` | 32-bit floating point (~7 significant digits) | Default for model parameters and data |
# | `torch.float64` | 64-bit floating point (~16 digits) | Precise numerical checks (CPU only on MPS machines) |
# | `torch.float16`, `torch.bfloat16` | 16-bit floating point | Faster GPU training (Chapter 5) |
# | `torch.int64` | Integers | Class labels, indices, counts |
# | `torch.bool` | `True`/`False` | Masks |
#
# PyTorch infers the dtype from what you write: Python floats become `float32`, Python ints become `int64`.

# %%
print(torch.tensor([1, 2]).dtype, torch.tensor([1.0, 2.0]).dtype, torch.tensor([True, False]).dtype)

ints = torch.tensor([1, 2, 3])
print("ints / 2  =", ints / 2, "← true division always gives floats")
print("ints // 2 =", ints // 2, "← floor division keeps integers")
print("ints + 0.5 =", ints + 0.5, "← mixing promotes to float")
print("as float64:", ints.to(torch.float64))

# %% [markdown]
# When an operation mixes dtypes, PyTorch **promotes** to a type that can represent both — integer plus float gives float. Promotion is convenient but can hide mistakes, such as accidentally computing in `float64` (slow on GPUs, unsupported on MPS) or comparing float labels with integer predictions.

# %% [markdown]
# ### Checkpoint 1.1
#
# Create `t`: a tensor of ones with shape `(2, 3, 4)` and dtype `float64`. Then set `n` to its number of elements, using a tensor method rather than doing the multiplication yourself.

# %% [markdown] tags=["hint"]
# Constructors accept `dtype=`. The element count is `t.numel()`.

# %%
# BEGIN SOLUTION
t = torch.ones(2, 3, 4, dtype=torch.float64)
n = t.numel()
# END SOLUTION

# %% tags=["check"]
check_shape(t, (2, 3, 4), "t")
check(t.dtype == torch.float64, f"t has dtype {t.dtype}, expected torch.float64")
check(n == 24, f"n = {n}, expected 2 × 3 × 4 = 24")
passed("Checkpoint 1.1")

# %% [markdown]
# ## 2. Indexing and slicing
#
# Indexing works like Python lists, with one index (or slice) per dimension, separated by commas. Indices start at 0, negative indices count from the end, and a slice `start:stop:step` excludes `stop`.

# %%
M = torch.arange(12).reshape(3, 4)  # reshape is explained in section 5
print(M)
print("M[1, 2]   =", M[1, 2])          # row 1, column 2 → a 0-d tensor
print("M[1]      =", M[1])             # row 1 → shape (4,)
print("M[:, 2]   =", M[:, 2])          # column 2 → shape (3,)
print("M[-1]     =", M[-1])            # last row
print("M[0:2, 1:3] =\n", M[0:2, 1:3])  # rows 0–1, columns 1–2 → shape (2, 2)
print("M[:, ::2] =\n", M[:, ::2])      # every other column

# %% [markdown]
# Notice how the shape changes: an **integer** index removes that dimension, while a **slice** keeps it (possibly with a smaller size). `M[1]` has shape `(4,)`, but `M[1:2]` has shape `(1, 4)`.

# %%
print(tuple(M[1].shape), tuple(M[1:2].shape))

# %% [markdown]
# Two more powerful forms:
#
# - **Boolean masks:** index with a `bool` tensor of the same shape to select the elements where it is `True`. The result is always 1-D.
# - **Integer arrays** ("fancy indexing"): index with a tensor of positions to gather those entries in that order.

# %%
mask = M > 5
print("mask =\n", mask)
print("M[mask] =", M[mask])

rows = torch.tensor([2, 0, 0])
print("M[rows] =\n", M[rows])  # rows 2, 0, 0 — order and repeats allowed

labels = torch.tensor([3, 0, 1])
print("one entry per row, M[torch.arange(3), labels] =", M[torch.arange(3), labels])

# %% [markdown]
# The last line picks entry `(0, 3)`, `(1, 0)`, and `(2, 1)` — one column per row. You will use exactly this pattern in Chapter 2 to pick each example's predicted probability for its correct class.
#
# Indexing can also assign:

# %%
A = torch.zeros(3, 4)
A[:, 0] = 1.0       # a whole column
A[A == 0] = -1.0    # every element where the mask is True
print(A)

# %% [markdown]
# ### Checkpoint 2.1
#
# Using `M` from above (`torch.arange(12).reshape(3, 4)`):
#
# 1. `last_col`: the last column, with shape `(3,)`.
# 2. `even_values`: all even elements of `M`, as a 1-D tensor, using a boolean mask.

# %% [markdown] tags=["hint"]
# Use `-1` for the last position. `M % 2 == 0` is a boolean mask that is `True` for even elements.

# %%
# BEGIN SOLUTION
last_col = M[:, -1]
even_values = M[M % 2 == 0]
# END SOLUTION

# %% tags=["check"]
check_shape(last_col, (3,), "last_col")
check(torch.equal(last_col, torch.tensor([3, 7, 11])), f"last_col = {last_col}, expected the last column [3, 7, 11]")
check(torch.equal(even_values, torch.tensor([0, 2, 4, 6, 8, 10])), f"even_values = {even_values}")
passed("Checkpoint 2.1")

# %% [markdown]
# ## 3. Elementwise operations and reductions
#
# Arithmetic operators and most math functions act **elementwise**: on matching positions of tensors with the same shape.

# %%
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([10.0, 20.0, 30.0])
print(x + y, x * y, x ** 2, torch.exp(x), torch.sqrt(y))

# %% [markdown]
# `*` is elementwise. Matrix multiplication is a different operation, written `@` — notebook 02 covers it.
#
# A **reduction** combines many elements into fewer: `sum`, `mean`, `max`, `min`, `argmax`, `std`. With no arguments it reduces everything to a scalar. With `dim=d`, it reduces **along** dimension `d` — that dimension disappears from the shape.

# %%
S = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])  # shape (2, 3)

print("S.sum()      =", S.sum())              # everything → shape ()
print("S.sum(dim=0) =", S.sum(dim=0))         # down the rows → shape (3,): one total per column
print("S.sum(dim=1) =", S.sum(dim=1))         # across columns → shape (2,): one total per row
print("S.sum(dim=1, keepdim=True) =\n", S.sum(dim=1, keepdim=True))  # shape (2, 1)

# %% [markdown]
# A reliable way to remember it: **the dimension you name is the one that disappears.** `S` has shape `(2, 3)`; summing over `dim=0` removes the 2 and leaves `(3,)`.
#
# `keepdim=True` keeps the reduced dimension with size 1. That matters in the next section, where it lets the result line up with the original tensor.
#
# `max` along a dimension returns both values and their positions; `argmax` returns only the positions:

# %%
scores = torch.tensor([[0.1, 0.7, 0.2],
                       [0.5, 0.3, 0.2]])
values, positions = scores.max(dim=1)
print("max values:", values, " positions:", positions, " argmax:", scores.argmax(dim=1))

# %% [markdown]
# ### Checkpoint 3.1
#
# `G` below holds the grades of 4 students (rows) on 3 exams (columns). Compute:
#
# 1. `student_means`: each student's mean grade, shape `(4, 1)`.
# 2. `best_exam`: for each student, the index of their highest-scoring exam, shape `(4,)`.

# %%
G = torch.tensor([[70.0, 85.0, 90.0],
                  [60.0, 95.0, 80.0],
                  [88.0, 72.0, 64.0],
                  [91.0, 91.0, 99.0]])

# %% [markdown] tags=["hint"]
# Each student is a row, so reduce *across the columns*: `dim=1`. Use `keepdim=True` to get shape `(4, 1)`.

# %%
# BEGIN SOLUTION
student_means = G.mean(dim=1, keepdim=True)
best_exam = G.argmax(dim=1)
# END SOLUTION

# %% tags=["check"]
check_shape(student_means, (4, 1), "student_means")
check_close(student_means, torch.tensor([[245 / 3], [235 / 3], [224 / 3], [281 / 3]]), "student_means")
check(torch.equal(best_exam, torch.tensor([2, 1, 0, 2])), f"best_exam = {best_exam}, expected [2, 1, 0, 2]")
passed("Checkpoint 3.1")

# %% [markdown]
# ## 4. Broadcasting
#
# What should `G - row_means` mean, when `G` has shape `(4, 3)` and `row_means = G.mean(dim=1, keepdim=True)` has shape `(4, 1)` — the result you computed in checkpoint 3.1? We want each student's mean subtracted from each of their grades. PyTorch does this by **broadcasting**: it virtually repeats the size-1 dimension to match, without copying data.
#
# The rules, applied to two shapes:
#
# 1. Line the shapes up **from the right**. If one has fewer dimensions, pad it on the left with 1s.
# 2. In each position, the sizes must be **equal**, or one of them must be **1**.
# 3. The result takes the larger size in each position; a size-1 dimension is repeated to match.
#
# | Shape A | Shape B | Aligned | Result |
# | --- | --- | --- | --- |
# | `(4, 3)` | `(4, 1)` | `(4, 3)` vs `(4, 1)` | `(4, 3)` |
# | `(4, 3)` | `(3,)` | `(4, 3)` vs `(1, 3)` | `(4, 3)` |
# | `(3, 1)` | `(1, 4)` | `(3, 1)` vs `(1, 4)` | `(3, 4)` |
# | `(4, 3)` | `(4,)` | `(4, 3)` vs `(1, 4)` | error: 3 ≠ 4 |

# %%
row_means = G.mean(dim=1, keepdim=True)  # (4, 1)
centered = G - row_means                 # (4, 3) - (4, 1) → (4, 3)
print(centered)
print("row means after centering (zero up to float32 rounding):", centered.mean(dim=1))

# %% [markdown]
# The second row of the table subtracts something per **column** instead — for example, each exam's mean:

# %%
exam_means = G.mean(dim=0)            # shape (3,)
print((G - exam_means).mean(dim=0))   # (4, 3) - (3,) → (4, 3); column means are now 0

# %% [markdown]
# The third row of the table builds a full grid from two vectors. This is an **outer** operation — every combination of an `i` and a `j`:

# %%
a = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)
b = torch.tensor([10.0, 20.0, 30.0, 40.0])  # (4,) → treated as (1, 4)
print(a * b)  # (3, 4): entry [i, j] = a[i] * b[j]

# %% [markdown]
# ### The most common silent broadcasting bug
#
# Broadcasting never complains when shapes are compatible — even when the result is not what you meant. Suppose a model outputs predictions of shape `(N, 1)` and the targets have shape `(N,)`:

# %%
predictions = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)
targets = torch.tensor([1.0, 2.0, 3.0])            # (3,)

wrong = (predictions - targets) ** 2               # (3, 1) vs (1, 3) → (3, 3)!
right = (predictions.squeeze(1) - targets) ** 2    # (3,) vs (3,) → (3,)
print("wrong shape:", tuple(wrong.shape), " mean:", wrong.mean().item())
print("right shape:", tuple(right.shape), " mean:", right.mean().item())

# %% [markdown]
# The predictions are perfect, so the mean squared error should be 0. The broadcast version compares every prediction with *every* target and reports 1.33 instead. No error is raised. This exact bug can make a model train to the wrong answer; notebook 06 asks you to find it in a training loop. The defense is to **check shapes**: print them, or assert them.

# %% [markdown]
# ### Checkpoint 4.1
#
# Standardize the columns of `G`: subtract each column's mean and divide by each column's standard deviation, using broadcasting (no loops). Store the result in `G_std`, shape `(4, 3)`. Use `G.std(dim=0)` for the standard deviation.

# %% [markdown] tags=["hint"]
# `G.mean(dim=0)` and `G.std(dim=0)` both have shape `(3,)`, which broadcasts against `(4, 3)` as `(1, 3)`.

# %%
# BEGIN SOLUTION
G_std = (G - G.mean(dim=0)) / G.std(dim=0)
# END SOLUTION

# %% tags=["check"]
check_shape(G_std, (4, 3), "G_std")
check_close(G_std.mean(dim=0), torch.zeros(3), "column means of G_std", atol=1e-5)
check_close(G_std.std(dim=0), torch.ones(3), "column standard deviations of G_std", atol=1e-5)
passed("Checkpoint 4.1")

# %% [markdown]
# ## 5. Changing shape
#
# Many operations rearrange the same elements into a different shape:
#
# | Operation | Effect |
# | --- | --- |
# | `reshape(shape)` / `view(shape)` | Same elements in the same order, new shape. One size may be `-1`, meaning "whatever makes the count work". |
# | `unsqueeze(d)` / `squeeze(d)` | Insert / remove a dimension of size 1 at position `d`. |
# | `transpose(d0, d1)`, `.T` | Swap two dimensions. |
# | `permute(order)` | Reorder all dimensions. |
# | `flatten(start_dim)` | Merge dimensions from `start_dim` onward into one. |
# | `torch.cat(list, dim)` | Join tensors along an **existing** dimension. |
# | `torch.stack(list, dim)` | Join tensors along a **new** dimension. |

# %%
v = torch.arange(6)
print(v.reshape(2, 3))
print(v.reshape(3, -1))           # -1 is inferred as 2
print(tuple(v.unsqueeze(0).shape), tuple(v.unsqueeze(1).shape))

images = torch.zeros(8, 32, 32, 3)          # (N, H, W, C), as many image libraries store them
print("permuted:", tuple(images.permute(0, 3, 1, 2).shape))  # (N, C, H, W), as PyTorch expects
print("flattened:", tuple(images.flatten(start_dim=1).shape))  # one long vector per image

p, q = torch.zeros(2, 3), torch.ones(2, 3)
print("cat dim 0:", tuple(torch.cat([p, q], dim=0).shape), " stack dim 0:", tuple(torch.stack([p, q], dim=0).shape))

# %% [markdown]
# `reshape` keeps elements in **row-major** order: the last index changes fastest. Reshaping `(2, 3)` to `(3, 2)` is not a transpose:

# %%
R = torch.arange(6).reshape(2, 3)
print("R =\n", R)
print("R.reshape(3, 2) =\n", R.reshape(3, 2))
print("R.T =\n", R.T)

# %% [markdown]
# Both results have shape `(3, 2)`, but different contents. Confusing the two is another silent bug: shapes match, numbers are scrambled.

# %% [markdown]
# ### Checkpoint 5.1
#
# `batch` holds 5 grayscale images of size 4×4, shape `(5, 4, 4)`. A convolutional layer (Chapter 6) expects shape `(N, C, H, W)` with `C = 1` channel. Create `batch_nchw` with shape `(5, 1, 4, 4)`. Then create `batch_flat` with one row of 16 pixels per image, shape `(5, 16)`.

# %%
batch = torch.rand(5, 4, 4, generator=torch.Generator().manual_seed(1))

# %% [markdown] tags=["hint"]
# `unsqueeze(1)` inserts a size-1 dimension at position 1. For flattening, `reshape(5, -1)` or `flatten(start_dim=1)` both work.

# %%
# BEGIN SOLUTION
batch_nchw = batch.unsqueeze(1)
batch_flat = batch.reshape(5, -1)
# END SOLUTION

# %% tags=["check"]
check_shape(batch_nchw, (5, 1, 4, 4), "batch_nchw")
check_shape(batch_flat, (5, 16), "batch_flat")
check(torch.equal(batch_nchw[:, 0], batch), "batch_nchw should contain the same images, with a channel dimension inserted at position 1")
check(torch.equal(batch_flat[2], batch[2].reshape(-1)), "each row of batch_flat should be one image's pixels in row-major order")
passed("Checkpoint 5.1")

# %% [markdown]
# ## 6. How tensors live in memory: storage, strides, and views
#
# A tensor's elements are stored in one flat, one-dimensional block of memory called its **storage**. The shape is only an interpretation of that block. Two numbers per dimension tell PyTorch how to find element `[i, j]`:
#
# - the **shape**, and
# - the **strides**: how many storage positions to skip to move one step along each dimension.
#
# Element `[i, j]` lives at storage position `offset + i * stride[0] + j * stride[1]`.

# %%
from dlcourse.plotting import COLORS


def draw_tensor(t, title):
    """Draw a 2-D tensor as a grid, and its storage as a row, labelled with storage positions."""
    rows, cols = t.shape
    storage = t.untyped_storage()
    n_storage = storage.nbytes() // t.element_size()
    fig, (ax_grid, ax_store) = plt.subplots(
        2, 1, figsize=(max(4, 0.6 * n_storage), 3.2), gridspec_kw={"height_ratios": [rows, 1]}
    )
    used = set()
    for i in range(rows):
        for j in range(cols):
            pos = t.storage_offset() + i * t.stride(0) + j * t.stride(1)
            used.add(pos)
            ax_grid.add_patch(plt.Rectangle((j, rows - 1 - i), 1, 1, fill=False))
            ax_grid.text(j + 0.5, rows - 0.5 - i, f"{t[i, j].item()}\n@{pos}", ha="center", va="center", fontsize=8)
    ax_grid.set_xlim(0, cols)
    ax_grid.set_ylim(0, rows)
    ax_grid.set_aspect("equal")
    ax_grid.axis("off")
    ax_grid.set_title(f"{title}: shape {tuple(t.shape)}, strides {t.stride()}")
    flat = torch.tensor([], dtype=t.dtype).set_(storage)
    for k in range(n_storage):
        shade = COLORS[5] if k in used else "white"
        ax_store.add_patch(plt.Rectangle((k, 0), 1, 1, facecolor=shade, edgecolor="black", alpha=0.6))
        ax_store.text(k + 0.5, 0.5, f"{flat[k].item()}", ha="center", va="center", fontsize=8)
        ax_store.text(k + 0.5, -0.3, f"{k}", ha="center", va="center", fontsize=7, color="gray")
    ax_store.set_xlim(0, n_storage)
    ax_store.set_ylim(-0.6, 1)
    ax_store.axis("off")
    ax_store.set_title("storage (position below each cell; shaded = used by this tensor)", fontsize=9)
    plt.tight_layout()
    plt.show()


T = torch.arange(0, 120, 10).reshape(3, 4)  # values 0, 10, ..., 110 so they differ from positions
draw_tensor(T, "T")

# %% [markdown]
# Each cell shows the value and the storage position it is read from (`@k`). With strides `(4, 1)`, moving down one row skips 4 positions and moving right one column skips 1. This tensor is **contiguous**: its rows are laid out one after another in storage, in row-major order.
#
# Now transpose it:

# %%
draw_tensor(T.T, "T.T")
print("Same storage?", T.T.untyped_storage().data_ptr() == T.untyped_storage().data_ptr())
print("T.T is contiguous?", T.T.is_contiguous())

# %% [markdown]
# Transposing copies nothing. It returns a **view**: a new shape and new strides — `(1, 4)` instead of `(4, 1)` — over the *same* storage. That makes it instant, even for huge tensors. The price: the elements are no longer in row-major order in memory, so the view is **not contiguous**.
#
# Slicing also returns views. Here, every other column of `T`:

# %%
draw_tensor(T[:, ::2], "T[:, ::2]")

# %% [markdown]
# The stride along columns is now 2: the view skips every other storage position.

# %% [markdown]
# ### Views share data
#
# Because a view shares storage, **writing through a view changes the original**:

# %%
base = torch.zeros(2, 3)
row = base[0]          # a view of the first row
row[1] = 5.0
print(base)

# %% [markdown]
# This is efficient and often what you want — and a classic source of bugs when it is not. Use `.clone()` to get an independent copy:

# %%
base = torch.zeros(2, 3)
row_copy = base[0].clone()
row_copy[1] = 5.0
print(base)  # unchanged

# %% [markdown]
# ### `view` versus `reshape`
#
# `view` *never* copies: it only works when the new shape can be expressed with strides over the existing storage. `reshape` returns a view when it can and silently copies when it cannot. Flattening a transposed tensor is the standard example:

# %%
try:
    T.T.view(12)
except RuntimeError as error:
    print("RuntimeError:", str(error).splitlines()[0][:110], "...")

flat = T.T.reshape(12)  # works, by copying
print(flat)
print("reshape copied?", flat.untyped_storage().data_ptr() != T.untyped_storage().data_ptr())
print("contiguous() then view:", T.T.contiguous().view(12))

# %% [markdown]
# Rules of thumb:
#
# - Operations that only reinterpret memory — slicing, `transpose`, `permute`, `view`, `unsqueeze`, `expand` — return views.
# - Arithmetic (`x + 1`, `x * y`, `torch.exp(x)`) always creates new tensors.
# - Operations whose names end in `_` (`add_`, `zero_`, `mul_`) and indexed assignment (`x[i] = …`) modify a tensor **in place**. In-place operations interact badly with automatic differentiation; notebook 05 shows why.
# - When in doubt, compare `untyped_storage().data_ptr()`, or `clone()` defensively.

# %% [markdown]
# Broadcasting, by the way, is implemented with exactly this machinery: `expand` creates a view whose stride is **0** along the repeated dimension, so every "copy" reads the same memory.

# %%
col = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)
wide = col.expand(3, 4)
print(wide)
print("strides:", wide.stride(), "— moving along a row stays at the same storage position")

# %% [markdown]
# ### Checkpoint 6.1
#
# Without running any code first, predict the strides of these tensors, then store your predictions as tuples:
#
# 1. `stride_a`: strides of `torch.zeros(2, 3, 4)`.
# 2. `stride_b`: strides of `torch.zeros(2, 3, 4).permute(2, 0, 1)`.

# %% [markdown] tags=["hint"]
# For a contiguous tensor, the stride of a dimension is the product of the sizes of all dimensions to its right. `permute(2, 0, 1)` produces a view whose dimension 0 is the old dimension 2, and so on — the strides move with their dimensions.

# %%
# BEGIN SOLUTION
stride_a = (12, 4, 1)
stride_b = (1, 12, 4)
# END SOLUTION

# %% tags=["check"]
check(stride_a == torch.zeros(2, 3, 4).stride(), f"stride_a = {stride_a} is not right. Moving one step along the last dimension skips 1 element; along the middle one, a whole row of 4.")
check(stride_b == torch.zeros(2, 3, 4).permute(2, 0, 1).stride(), f"stride_b = {stride_b} is not right. permute reorders the strides the same way it reorders the shape.")
passed("Checkpoint 6.1")

# %% [markdown]
# ## 7. Devices and NumPy
#
# As notebook 00 showed, every tensor lives on a device. `.to(device)` returns a tensor on that device — a **copy** when the device changes, so it does not share memory with the original. `.cpu()` is shorthand for `.to("cpu")`.

# %%
on_device = T.to(device)
back = on_device.cpu()
print(on_device.device, back.device, torch.equal(back, T))

# %% [markdown]
# PyTorch and NumPy convert in both directions. On the CPU, `.numpy()` and `torch.from_numpy` **share memory** — another kind of view:

# %%
import numpy as np

t_np = torch.zeros(3)
arr = t_np.numpy()
arr[0] = 42.0
print("tensor after editing the NumPy array:", t_np)

from_np = torch.from_numpy(np.array([1.0, 2.0]))
print("dtype from NumPy:", from_np.dtype, "← NumPy defaults to float64")

# %% [markdown]
# Two practical notes:
#
# - NumPy's default float type is `float64`, PyTorch's is `float32`. Convert with `.float()` when moving data from NumPy into a model.
# - `.item()` turns a one-element tensor into a Python number. Use it for printing and logging; on a GPU it forces the CPU to wait for the result (Chapter 5).

# %% [markdown]
# ## 8. [GPU] Moving data is a copy
#
# If a GPU is selected, this cell shows that a tensor moved to the GPU is independent of the original: changing one does not change the other. It skips itself on a CPU-only machine.

# %% tags=["gpu"]
if device.type == "cpu":
    print("No GPU selected; skipping.")
else:
    host = torch.zeros(3)
    gpu_copy = host.to(device)
    gpu_copy += 1
    print("on CPU:", host, " on", device.type, ":", gpu_copy.cpu())
    print("Same tensor object?", host is gpu_copy)
    print("Moving a tensor already on the device returns it unchanged:", gpu_copy.to(device) is gpu_copy)

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1 (modify): broadcasting errors
#
# The cell below subtracts a vector from a matrix. Change `v` so that its shape is `(4,)` instead of `(3,)`, run the cell, and read the error. Then change `v` to shape `(4, 1)`. In the markdown cell below, explain using the three broadcasting rules why the first change fails and the second works, and say what the second version computes.

# %%
X_demo = torch.arange(12.0).reshape(4, 3)
v = torch.tensor([1.0, 2.0, 3.0])
try:
    print((X_demo - v).shape)
except RuntimeError as error:
    print("RuntimeError:", error)

# %% [markdown]
# *Your answer:*

# %% [markdown] tags=["solution"]
# Shapes are aligned from the right. With `v` of shape `(4,)`, the last dimensions are 3 (matrix) and 4 (vector): unequal and neither is 1, so broadcasting fails. With `v` of shape `(4, 1)`, the comparison is `(4, 3)` against `(4, 1)`: the first sizes match and the second is 1, so `v` is repeated across the 3 columns. The result subtracts `v[i]` from every element of row `i` — a per-row offset instead of a per-column one.

# %% [markdown]
# ### Exercise 2 (implement): pairwise squared distances
#
# Many algorithms need the squared Euclidean distance between every pair of points from two sets. Write `pairwise_sq_distances(A, B)` where `A` has shape `(N, D)` and `B` has shape `(M, D)`. It returns `dist` of shape `(N, M)` with
#
# $$\text{dist}[i, j] = \sum_{k=1}^{D} (A[i, k] - B[j, k])^2.$$
#
# Use broadcasting; no Python loops.

# %% [markdown] tags=["hint"]
# Insert size-1 dimensions so that `A` becomes `(N, 1, D)` and `B` becomes `(1, M, D)`. Their difference broadcasts to `(N, M, D)`. Square it and sum over the last dimension.

# %%
def pairwise_sq_distances(A, B):
    # BEGIN SOLUTION
    diff = A.unsqueeze(1) - B.unsqueeze(0)  # (N, 1, D) - (1, M, D) → (N, M, D)
    return (diff ** 2).sum(dim=-1)          # (N, M)
    # END SOLUTION

# %% tags=["check"]
gen = torch.Generator().manual_seed(0)
A_test = torch.randn(5, 3, generator=gen, dtype=torch.float64)
B_test = torch.randn(7, 3, generator=gen, dtype=torch.float64)
dist = pairwise_sq_distances(A_test, B_test)
check_shape(dist, (5, 7), "pairwise_sq_distances(A, B)")
check_close(dist, torch.cdist(A_test, B_test) ** 2, "pairwise_sq_distances(A, B)")
check_close(pairwise_sq_distances(A_test, A_test).diagonal(), torch.zeros(5), "distance of each point to itself")
passed("Exercise 2")

# %% [markdown]
# ### Exercise 3 (debug): a function that changes its input
#
# `center_first_feature` is meant to return a copy of `data` in which the first column has mean zero, **leaving `data` unchanged**. It returns the right values, but it also corrupts the caller's tensor. Find and fix the bug.

# %% [markdown] tags=["hint"]
# Is `result` a new tensor or a view of `data`? What does `-=` do to the storage it points at?

# %%
def center_first_feature(data):
    # Buggy version:
    #     result = data
    #     result[:, 0] -= result[:, 0].mean()
    #     return result
    # BEGIN SOLUTION
    result = data.clone()
    result[:, 0] -= result[:, 0].mean()
    return result
    # END SOLUTION

# %% tags=["check"]
data = torch.tensor([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
original = data.clone()
out = center_first_feature(data)
check(torch.equal(data, original), "data was modified: the function must not change its input")
check_close(out[:, 0], torch.tensor([-2.0, 0.0, 2.0]), "first column of the result")
check_close(out[:, 1], original[:, 1], "second column of the result")
passed("Exercise 3")

# %% [markdown]
# ### Exercise 4 (investigate): why vectorize?
#
# Exercise 2 asked for broadcasting instead of loops. Is that only about elegance? The scaffold below times a pure Python double loop against a broadcast version for growing numbers of points.
#
# Run it, then in the markdown cell below answer: (1) roughly how many times faster is the broadcast version at the largest size? (2) How does each runtime grow when `N` doubles, and why? (3) Why is a Python loop over tensor elements so slow?

# %%
import time


def loop_sq_distances(A, B):
    out = torch.empty(A.shape[0], B.shape[0])
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            out[i, j] = ((A[i] - B[j]) ** 2).sum()
    return out


def broadcast_sq_distances(A, B):
    return ((A.unsqueeze(1) - B.unsqueeze(0)) ** 2).sum(dim=-1)


def best_time(fn, *args, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
    return min(times)


sizes = [10, 20, 40, 80, 160]
loop_times, broadcast_times = [], []
gen = torch.Generator().manual_seed(0)
for n in sizes:
    P = torch.randn(n, 8, generator=gen)
    check_close(loop_sq_distances(P, P), broadcast_sq_distances(P, P), "loop vs broadcast", atol=1e-4)
    loop_times.append(best_time(loop_sq_distances, P, P, repeats=1))
    broadcast_times.append(best_time(broadcast_sq_distances, P, P))

plt.loglog(sizes, loop_times, "o-", label="Python double loop")
plt.loglog(sizes, broadcast_times, "s-", label="broadcasting")
plt.xlabel("number of points N (distances computed: N²)")
plt.ylabel("time (seconds)")
plt.title("Pairwise distances: loop vs broadcasting (CPU)")
plt.legend()
plt.show()
for n, lt, bt in zip(sizes, loop_times, broadcast_times):
    print(f"N = {n:>4}: loop {lt * 1e3:8.2f} ms   broadcast {bt * 1e3:6.3f} ms   ratio {lt / bt:7.0f}×")

# %% [markdown]
# *Your answer:*

# %% [markdown] tags=["solution"]
# (1) At N = 160 the broadcast version is on the order of a thousand times faster (the exact ratio depends on your machine; the printed table shows it). (2) The loop's time roughly quadruples each time N doubles, because it does N² iterations of the same work. The broadcast version grows much more slowly at these sizes: its time is dominated by fixed overhead, and the arithmetic itself — about N² × 8 subtract-square-add steps — is tiny for optimized code. At much larger N it too grows like N². (3) Each loop iteration runs Python code and launches several separate tensor operations, each with microseconds of overhead for checking shapes, dtypes, and devices and allocating a result — far more than the few floating-point operations they perform. A single broadcast operation pays that overhead once and then runs a compiled loop over all elements. On a GPU the gap is larger still, because each tiny operation also has to be launched on the device.

# %% [markdown]
# ## Summary and what's next
#
# - A tensor is a rectangular array with a **shape** and a **dtype**, living on a **device**.
# - Integer indices remove a dimension; slices keep it. Boolean masks and integer arrays select arbitrary elements.
# - Reductions with `dim=d` remove dimension `d`, unless `keepdim=True`.
# - **Broadcasting** aligns shapes from the right and repeats size-1 dimensions. It never warns when the result has an unintended shape — check shapes.
# - A tensor is a view of flat **storage** through **strides**. Slicing, transposing, and `view` share memory; arithmetic creates new tensors; `clone()` makes an independent copy.
# - Vectorized tensor operations are orders of magnitude faster than Python loops.
#
# We can now store and manipulate data. But a neural network layer does something specific to its input: it multiplies by a matrix and adds a vector. Notebook 02 builds a geometric picture of what that does to data — and why stacking such layers alone is not enough.

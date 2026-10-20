# %% [markdown]
# # Chapter 1 problem set — Foundations
#
# | | |
# | --- | --- |
# | **Prerequisites** | Notebooks 00–06 |
# | **Covers** | Tensor shapes and views, a custom autograd function, debugging training loops, likelihoods as losses, a complete regression fit |
# | **Estimated time** | 4–5 hours |
# | **Compute** | CPU. Problem 6 is optional and needs a GPU. |
#
# These problems combine ideas from the whole chapter. Each has automated checks; problems 4 and 5 also ask for a short written analysis. Attempt each problem before opening its hint.

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

# %% [markdown]
# ## Problem 1: Predict the shape
#
# *Notebooks 01–02.* For each expression below, predict the shape of the result **without running it**, or write `"error"` if PyTorch would raise an error. Record your predictions in the dictionary `predicted`, keyed by the expression's number. Shapes are tuples, e.g. `(3, 4)`; a scalar has shape `()`.
#
# The tensors are:
#
# ```python
# a = torch.zeros(3, 4)
# b = torch.zeros(4)
# c = torch.zeros(3, 1)
# d = torch.zeros(2, 3, 4)
# e = torch.zeros(5, 1, 4)
# ```
#
# | # | Expression | # | Expression |
# | --- | --- | --- | --- |
# | 1 | `a + b` | 11 | `a.T.reshape(12)` |
# | 2 | `a + c` | 12 | `a.T.view(12)` |
# | 3 | `b + c` | 13 | `a @ b` |
# | 4 | `a + a.T` | 14 | `d @ a.T` |
# | 5 | `d + a` | 15 | `a[:, None, :] - a[None, :, :]` |
# | 6 | `d + e` | 16 | `d.sum(dim=(0, 2))` |
# | 7 | `a.sum(dim=0)` | 17 | `d.permute(2, 0, 1)` |
# | 8 | `a.sum(dim=1, keepdim=True)` | 18 | `torch.stack([a, a], dim=1)` |
# | 9 | `a[0]` | 19 | `torch.cat([a, c], dim=1)` |
# | 10 | `a[0:1]` | 20 | `d[..., 0].unsqueeze(-1).expand(2, 3, 5)` |

# %%
a = torch.zeros(3, 4)
b = torch.zeros(4)
c = torch.zeros(3, 1)
d = torch.zeros(2, 3, 4)
e = torch.zeros(5, 1, 4)

expressions = {
    1: lambda: a + b,
    2: lambda: a + c,
    3: lambda: b + c,
    4: lambda: a + a.T,
    5: lambda: d + a,
    6: lambda: d + e,
    7: lambda: a.sum(dim=0),
    8: lambda: a.sum(dim=1, keepdim=True),
    9: lambda: a[0],
    10: lambda: a[0:1],
    11: lambda: a.T.reshape(12),
    12: lambda: a.T.view(12),
    13: lambda: a @ b,
    14: lambda: d @ a.T,
    15: lambda: a[:, None, :] - a[None, :, :],
    16: lambda: d.sum(dim=(0, 2)),
    17: lambda: d.permute(2, 0, 1),
    18: lambda: torch.stack([a, a], dim=1),
    19: lambda: torch.cat([a, c], dim=1),
    20: lambda: d[..., 0].unsqueeze(-1).expand(2, 3, 5),
}

# %%
# BEGIN SOLUTION
predicted = {
    1: (3, 4),
    2: (3, 4),
    3: (3, 4),
    4: "error",
    5: (2, 3, 4),
    6: "error",
    7: (4,),
    8: (3, 1),
    9: (4,),
    10: (1, 4),
    11: (12,),
    12: "error",
    13: (3,),
    14: (2, 3, 3),
    15: (3, 3, 4),
    16: (3,),
    17: (4, 2, 3),
    18: (3, 2, 4),
    19: (3, 5),
    20: (2, 3, 5),
}
# END SOLUTION

# %% tags=["check"]
def actual_shape(fn):
    try:
        return tuple(fn().shape)
    except RuntimeError:
        return "error"


wrong = {k: (predicted.get(k), actual_shape(fn)) for k, fn in expressions.items() if predicted.get(k) != actual_shape(fn)}
for k, (guess, truth) in wrong.items():
    print(f"#{k}: you predicted {guess}, actual {truth}")
check(not wrong, f"{len(wrong)} of 20 predictions are wrong; see the list above and work out why for each")
passed("Problem 1")

# %% [markdown] tags=["hint"]
# Broadcasting aligns shapes from the **right** and needs each pair of sizes to be equal or 1. `view` fails on non-contiguous tensors, while `reshape` copies. For `@` with more than two dimensions, the last two dimensions are multiplied as matrices and the leading dimensions are broadcast.

# %% [markdown] tags=["solution"]
# The three errors: #4 compares `(3, 4)` with `(4, 3)` — neither 3 vs 4 nor 4 vs 3 is broadcastable. #6 aligns `(2, 3, 4)` with `(5, 1, 4)` and fails on 2 vs 5. #12 cannot `view` the non-contiguous transpose. In #14, `d @ a.T` multiplies each `(3, 4)` slice of `d` by the `(4, 3)` matrix, giving `(2, 3, 3)`. #15 is the pairwise-difference pattern from notebook 01's exercise 2.

# %% [markdown]
# ## Problem 2: A numerically stable log-sum-exp with a custom backward
#
# *Notebooks 00, 03, 05.* The **log-sum-exp** of a vector $\mathbf{x} \in \mathbb{R}^K$ is
#
# $$\operatorname{lse}(\mathbf{x}) = \ln \sum_{k=1}^{K} e^{x_k}.$$
#
# It is a smooth version of $\max_k x_k$, and it sits inside every multiclass classification loss (notebook 10). Computed naively in `float32`, it fails for large or very negative inputs:

# %%
x_big = torch.tensor([[1000.0, 1000.0], [-1000.0, -1000.0], [1.0, 2.0]])  # (3, 2): one vector per row
print("naive lse:", torch.log(torch.exp(x_big).sum(dim=1)))
print("exp(1000) in float32:", torch.exp(torch.tensor(1000.0)), "  exp(-1000):", torch.exp(torch.tensor(-1000.0)))

# %% [markdown]
# For the first row the true answer is $1000 + \ln 2$, but $e^{1000}$ overflows to `inf`. For the second row it is $-1000 + \ln 2$, but $e^{-1000}$ underflows to 0 and the log gives `-inf`.
#
# **The fix.** For any constant $m$, $\operatorname{lse}(\mathbf{x}) = m + \ln \sum_k e^{x_k - m}$. Choosing $m = \max_k x_k$ makes the largest exponent exactly $e^0 = 1$: nothing overflows, and the sum is at least 1, so its log is safe.
#
# **The gradient.** Differentiating, $\dfrac{\partial\, \operatorname{lse}}{\partial x_j} = \dfrac{e^{x_j}}{\sum_k e^{x_k}} = e^{x_j - \operatorname{lse}(\mathbf{x})}$. These numbers are positive and sum to 1 — you will meet them again as the **softmax** in notebook 10.
#
# **Your task:** implement `StableLogSumExp`, a `torch.autograd.Function` that takes `x` of shape `(N, K)` and returns shape `(N,)`, one log-sum-exp per row:
#
# 1. **forward:** use the max-shift identity, row by row (`x.max(dim=1, keepdim=True)`). Save what the backward pass needs.
# 2. **backward:** `grad_output` has shape `(N,)`. Return `grad_output` times the gradient above, with shape `(N, K)`. Compute $e^{x_j - \operatorname{lse}}$ directly, never $e^{x_j}$ on its own.
#
# First prove the max-shift identity to yourself: factor $e^{m}$ out of the sum.

# %% [markdown] tags=["hint"]
# Forward: `m = x.max(dim=1, keepdim=True).values` (shape `(N, 1)`), `out = m.squeeze(1) + torch.log(torch.exp(x - m).sum(dim=1))`, then `ctx.save_for_backward(x, out)`. Backward: `torch.exp(x - out.unsqueeze(1)) * grad_output.unsqueeze(1)`. Every `unsqueeze` is there so `(N,)` broadcasts against `(N, K)` row by row (notebook 01).

# %%
class StableLogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # BEGIN SOLUTION
        m = x.max(dim=1, keepdim=True).values  # (N, 1)
        out = m.squeeze(1) + torch.log(torch.exp(x - m).sum(dim=1))  # (N,)
        ctx.save_for_backward(x, out)
        return out
        # END SOLUTION

    @staticmethod
    def backward(ctx, grad_output):
        # BEGIN SOLUTION
        x, out = ctx.saved_tensors
        return torch.exp(x - out.unsqueeze(1)) * grad_output.unsqueeze(1)  # (N, K)
        # END SOLUTION


stable_lse = StableLogSumExp.apply

# %% tags=["check"]
values = stable_lse(x_big)
check_shape(values, (3,), "stable_lse(x_big)")
check(bool(torch.isfinite(values).all()), f"stable_lse returned non-finite values: {values}")
check_close(values, torch.logsumexp(x_big, dim=1), "stable_lse(x_big)", rtol=1e-6)
x64 = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
check(torch.autograd.gradcheck(stable_lse, (x64,)), "gradcheck failed")
x32 = x_big.clone().requires_grad_()
weights = torch.tensor([1.0, 2.0, 3.0])
(stable_lse(x32) * weights).sum().backward()  # different grad_output per row
check(bool(torch.isfinite(x32.grad).all()), "gradient contains inf or nan for large inputs")
# Near 1000, float32 values are spaced about 6e-5 apart, so x - lse carries an error of that size.
check_close(x32.grad, torch.softmax(x_big, dim=1) * weights.unsqueeze(1), "gradient of stable_lse", atol=2e-4)
passed("Problem 2")

# %% [markdown]
# ## Problem 3: Two silent training bugs
#
# *Notebooks 05–06.* Both parts fit $y = 3x + 2$ by linear regression with gradient descent, and each contains one bug that raises no error. Find and fix each.

# %%
gen = torch.Generator().manual_seed(0)
x_train = torch.linspace(-1, 1, 50).unsqueeze(1)  # (50, 1)
y_train = 3 * x_train + 2 + 0.1 * torch.randn(50, 1, generator=gen)
design = torch.cat([x_train, torch.ones_like(x_train)], dim=1)  # (50, 2): columns x and 1
w_ls, b_ls = torch.linalg.lstsq(design, y_train).solution.squeeze(1).tolist()
print(f"least squares: w = {w_ls:.4f}, b = {b_ls:.4f}")

# %% [markdown]
# ### 3a. The model that never learns
#
# This version runs without errors, and its loss is computed correctly, yet the returned parameters are nowhere near the answer. It was "refactored" so the model is created by a helper — and something broke. Find the bug.

# %% [markdown] tags=["hint"]
# An optimizer holds references to the specific parameter tensors it was given when it was created. Print `model.weight` before and after training. Which tensors does `optimizer.step()` update?

# %%
def make_model():
    torch.manual_seed(0)
    return nn.Linear(1, 1)


def train_with_optimizer(steps=500, lr=0.1):
    # Buggy version:
    #     model = make_model()
    #     optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #     model = make_model()  # "reset to a known initialization"
    # BEGIN SOLUTION
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # END SOLUTION
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.mse_loss(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    return model.weight.item(), model.bias.item()

# %% [markdown]
# ### 3b. The penalty that does nothing
#
# To discourage large slopes, we add a **penalty** $\lambda w^2$ to the loss (Chapter 4 calls this L2 regularization):
#
# $$L(w, b) = \text{MSE}(w, b) + \lambda w^2, \qquad \lambda = 1.$$
#
# The function below reports the right loss *value*, but training with it gives exactly the same slope as training without the penalty. No error is raised. Find the bug and fix it, without changing the value the function returns.

# %%
lam = 1.0


def penalized_loss_buggy(w, b):
    mse = F.mse_loss(w * x_train + b, y_train)
    penalty = torch.tensor(lam * w.item() ** 2)
    return mse + penalty


w0 = torch.tensor(1.0, requires_grad=True)
b0 = torch.tensor(0.0, requires_grad=True)
penalized_loss_buggy(w0, b0).backward()
print("loss value:", penalized_loss_buggy(w0, b0).item(), "   gradient with respect to w:", w0.grad.item())

# %% [markdown] tags=["hint"]
# `.item()` returns a plain Python number, and `torch.tensor(...)` builds a brand-new leaf tensor from it. Is there any path in the computational graph from `penalty` back to `w`?

# %%
def penalized_loss(w, b):
    # BEGIN SOLUTION
    mse = F.mse_loss(w * x_train + b, y_train)
    penalty = lam * w**2
    return mse + penalty
    # END SOLUTION

# %% tags=["check"]
w0 = torch.tensor(1.0, requires_grad=True)
b0 = torch.tensor(0.0, requires_grad=True)
loss = penalized_loss(w0, b0)
loss.backward()
check_close(loss, penalized_loss_buggy(torch.tensor(1.0), torch.tensor(0.0)), "penalized_loss value")
residual = (1.0 * x_train + 0.0 - y_train).squeeze(1)
expected_w_grad = (2 * residual * x_train.squeeze(1)).mean() + 2 * lam * 1.0
check_close(w0.grad, expected_w_grad, "gradient with respect to w (MSE part + 2λw)", rtol=1e-5)
passed("Problem 3b")

# %% [markdown] tags=["solution"]
# `w.item()` leaves the graph: the penalty becomes a constant, so it adds to the loss value but contributes zero gradient. Any computation you want to train through must stay in tensor operations on `w` itself. A good habit when a term "has no effect": compute the gradient with and without it and compare.

# %% [markdown]
# ## Problem 4: Which likelihood? Means, medians, and outliers
#
# *Notebooks 04 and 06.* Suppose we measure one unknown quantity $\mu$ many times: $y_i = \mu + \varepsilon_i$. Different assumptions about the noise $\varepsilon$ give different losses:
#
# - **Gaussian** noise with scale 1: negative log-likelihood (NLL) $= \sum_i \tfrac12 (y_i - \mu)^2 + \text{const}$.
# - **Laplace** noise with scale 1, density $\tfrac12 e^{-|\varepsilon|}$: NLL $= \sum_i |y_i - \mu| + \text{const}$.
#
# ### 4a. Implement the two NLLs over a grid
#
# Write `gaussian_nll(mu_grid, y)` and `laplace_nll(mu_grid, y)`. `mu_grid` has shape `(G,)` — candidate values of $\mu$ — and `y` has shape `(N,)`. Each returns shape `(G,)`: the full NLL (including constants) of all the data, for each candidate. Use broadcasting, no loops.

# %% [markdown] tags=["hint"]
# `mu_grid.unsqueeze(1) - y.unsqueeze(0)` has shape `(G, N)`: every residual for every candidate. Sum over `dim=1`. Constants per data point: $\tfrac12 \ln(2\pi)$ for the Gaussian and $\ln 2$ for the Laplace.

# %%
def gaussian_nll(mu_grid, y):
    # BEGIN SOLUTION
    residuals = mu_grid.unsqueeze(1) - y.unsqueeze(0)  # (G, N)
    return (0.5 * residuals**2 + 0.5 * math.log(2 * math.pi)).sum(dim=1)
    # END SOLUTION


def laplace_nll(mu_grid, y):
    # BEGIN SOLUTION
    residuals = mu_grid.unsqueeze(1) - y.unsqueeze(0)  # (G, N)
    return (residuals.abs() + math.log(2)).sum(dim=1)
    # END SOLUTION

# %% tags=["check"]
y_small = torch.tensor([0.0, 1.0, 5.0], dtype=torch.float64)
grid_small = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
gauss_expected = -torch.distributions.Normal(grid_small.unsqueeze(1), 1.0).log_prob(y_small).sum(dim=1)
laplace_expected = -torch.distributions.Laplace(grid_small.unsqueeze(1), 1.0).log_prob(y_small).sum(dim=1)
check_shape(gaussian_nll(grid_small, y_small), (3,), "gaussian_nll(grid, y)")
check_close(gaussian_nll(grid_small, y_small), gauss_expected, "gaussian_nll")
check_close(laplace_nll(grid_small, y_small), laplace_expected, "laplace_nll")
passed("Problem 4a")

# %% [markdown]
# ### 4b. Which estimates do they produce?
#
# The cell below finds the grid minimizer of each NLL for one sample and compares it with the sample **mean** and the sample **median**. Run it. Then, in the markdown cell after it, explain why the Gaussian NLL is minimized at the mean and the Laplace NLL at a median.

# %% tags=["exercise"]
gen = torch.Generator().manual_seed(1)
sample = 3 + torch.randn(101, generator=gen, dtype=torch.float64)
mu_grid = torch.linspace(0, 6, 60_001, dtype=torch.float64)
print(f"argmin Gaussian NLL {mu_grid[gaussian_nll(mu_grid, sample).argmin()].item():.4f}   mean   {sample.mean().item():.4f}")
print(f"argmin Laplace NLL  {mu_grid[laplace_nll(mu_grid, sample).argmin()].item():.4f}   median {sample.median().item():.4f}")

# %% [markdown]
# *Your explanation:*

# %% [markdown] tags=["solution"]
# The Gaussian NLL is a smooth parabola in $\mu$; its derivative $\sum_i (\mu - y_i)$ is zero exactly when $\mu = \frac{1}{N}\sum_i y_i$, the mean. The Laplace NLL $\sum_i |y_i - \mu|$ is piecewise linear; its slope at any $\mu$ between data points is (number of points below $\mu$) − (number of points above $\mu$). The slope changes sign — so the function is minimized — where as many points lie below as above: a median. Both grid minima agree with these formulas to the grid spacing of 1e-4.

# %% [markdown]
# ### 4c. Investigate: contamination
#
# Now corrupt the measurements: replace a fraction $f$ of them with a gross outlier at $y = 50$ (say, a sensor glitch). The scaffold below computes both estimates for $f$ from 0 to 0.6 and plots them against the true value $\mu = 3$.
#
# Run it and write a conclusion: how does each estimate respond to outliers? At roughly what fraction does each estimate become useless, and why? Which loss would you choose for data with occasional glitches, and what would you give up?

# %% tags=["exercise"]
fractions = torch.linspace(0, 0.6, 25)
gauss_estimates, laplace_estimates = [], []
for f in fractions:
    corrupted = sample.clone()
    corrupted[: int(round(f.item() * len(sample)))] = 50.0
    gauss_estimates.append(mu_grid[gaussian_nll(mu_grid, corrupted).argmin()].item())
    laplace_estimates.append(mu_grid[laplace_nll(mu_grid, corrupted).argmin()].item())

plt.plot(fractions, gauss_estimates, "o-", label="Gaussian NLL (mean)")
plt.plot(fractions, laplace_estimates, "s-", label="Laplace NLL (median)")
plt.axhline(3, color="gray", linestyle="--", label="true μ = 3")
plt.xlabel("fraction of measurements replaced by the outlier 50")
plt.ylabel("estimate of μ (grid limited to [0, 6])")
plt.title("Effect of contamination on the two estimates")
plt.legend()
plt.show()

# %% [markdown]
# *Your conclusion:*

# %% [markdown] tags=["solution"]
# The Gaussian estimate (the mean) moves toward the outlier as soon as any contamination appears: each glitch shifts the mean by about 47/101 ≈ 0.47, so with a few percent of glitches the estimate is already far off — and it leaves the plotted grid entirely once the true mean exceeds 6. In the limit, one point placed far enough away can move the mean arbitrarily: its **breakdown point** is 0. The Laplace estimate (the median) barely moves until half of the data is corrupted, because it depends only on how many points lie on each side, not on how far away they are. Its breakdown point is 50%; beyond that the median itself becomes an outlier value. The same reasoning holds for gradients: the Gaussian NLL's gradient grows with the residual, the Laplace NLL's gradient is bounded by 1 per point. For data with occasional glitches, the Laplace (absolute-error) loss is far more robust. The price: with clean Gaussian data the median is a noisier estimate than the mean (its variance is about π/2 ≈ 1.57 times larger for large samples), and the absolute value is not differentiable at 0, which can make gradient-based optimization converge more slowly.

# %% [markdown]
# ## Problem 5: A complete fit — polynomial regression
#
# *Notebooks 01–06.* A linear model can fit curves if we give it nonlinear **features**. Here $y = \sin(2x) + \varepsilon$ on $x \in [-2, 2]$, and we model it as a cubic polynomial:
#
# $$\hat y = w_1 x + w_2 x^2 + w_3 x^3 + b.$$
#
# This is still *linear in the parameters*, so everything from notebook 06 applies.

# %%
gen = torch.Generator().manual_seed(2)
x_poly = torch.linspace(-2, 2, 200)
y_poly = torch.sin(2 * x_poly) + 0.1 * torch.randn(200, generator=gen)

# %% [markdown]
# ### 5a. Build and standardize the features
#
# Write `poly_features(x, degree)` returning a `(N, degree)` matrix whose column `k` (counting from 1) is $x^k$ — using broadcasting, not a loop. Then write `standardize(F)` returning `(F_std, mean, std)`, where each column of `F_std` has mean 0 and standard deviation 1, and `mean` and `std` have shape `(degree,)`.

# %% [markdown] tags=["hint"]
# `x.unsqueeze(1) ** torch.arange(1, degree + 1)` broadcasts `(N, 1)` against `(degree,)`. For `standardize`, reduce over `dim=0`.

# %%
def poly_features(x, degree):
    # BEGIN SOLUTION
    return x.unsqueeze(1) ** torch.arange(1, degree + 1, dtype=x.dtype)
    # END SOLUTION


def standardize(F):
    # BEGIN SOLUTION
    mean = F.mean(dim=0)
    std = F.std(dim=0)
    return (F - mean) / std, mean, std
    # END SOLUTION

# %% tags=["check"]
feats = poly_features(torch.tensor([1.0, 2.0, -1.0]), 3)
check_close(feats, torch.tensor([[1.0, 1.0, 1.0], [2.0, 4.0, 8.0], [-1.0, 1.0, -1.0]]), "poly_features([1, 2, -1], 3)")
F_std, mean, std = standardize(poly_features(x_poly, 3))
check_shape(F_std, (200, 3), "F_std")
check_shape(mean, (3,), "mean")
check_close(F_std.mean(dim=0), torch.zeros(3), "column means of F_std", atol=1e-5)
check_close(F_std.std(dim=0), torch.ones(3), "column stds of F_std", atol=1e-5)
passed("Problem 5a")

# %% [markdown]
# ### 5b. Train, and compare with least squares
#
# Fit `nn.Linear(3, 1)` to the standardized features with `torch.optim.SGD` and the MSE loss. Choose the learning rate and number of steps yourself. Store the trained model in `poly_model`. The check compares your training MSE with the exact least-squares optimum.

# %% [markdown] tags=["hint"]
# Targets must have shape `(N, 1)` to match the model's output — remember the broadcasting bug from notebook 01. Standardized features make a learning rate around 0.1–0.5 stable; a few thousand steps suffice.

# %%
# BEGIN SOLUTION
torch.manual_seed(0)
poly_model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(poly_model.parameters(), lr=0.3)
targets = y_poly.unsqueeze(1)
for _ in range(3000):
    optimizer.zero_grad()
    loss = F.mse_loss(poly_model(F_std), targets)
    loss.backward()
    optimizer.step()
# END SOLUTION

# %% tags=["check"]
design_poly = torch.cat([F_std, torch.ones(200, 1)], dim=1).double()
solution = torch.linalg.lstsq(design_poly, y_poly.double().unsqueeze(1)).solution
best_mse = ((design_poly @ solution).squeeze(1) - y_poly.double()).pow(2).mean().item()
with torch.no_grad():
    your_mse = F.mse_loss(poly_model(F_std).squeeze(1), y_poly).item()
print(f"your training MSE {your_mse:.6f}   least-squares optimum {best_mse:.6f}")
check(your_mse <= best_mse * 1.001 + 1e-6, "your model has not converged to the least-squares optimum; train longer or adjust the learning rate")
passed("Problem 5b")

# %% [markdown]
# ### 5c. Investigate: why standardize?
#
# The cell below computes the **condition number** of the matrix $\tfrac{2}{N}A^\top A$ — the Hessian of the MSE, where $A$ is the feature matrix with a column of ones (notebooks 03 and 06) — for raw and standardized cubic features. It does so for two input ranges: our $x \in [-2, 2]$, and $x \in [0, 4]$. It then runs plain gradient descent in each case, with the largest stable learning rate, and plots the loss relative to the optimum.
#
# Run it, then answer in the markdown cell below:
#
# 1. How much does standardization reduce the condition number on each interval, and why is the effect so different?
# 2. Explain the convergence curves using the condition numbers.
# 3. Even after standardization, the condition number on $[0, 4]$ stays large. What property of the features does standardization *not* fix?

# %% tags=["exercise"]
degree = 3
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (lo, hi) in zip(axes, [(-2.0, 2.0), (0.0, 4.0)]):
    x_range = torch.linspace(lo, hi, 200, dtype=torch.float64)
    y_range = torch.sin(2 * x_range) + 0.1 * torch.randn(200, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    raw = poly_features(x_range, degree)
    standardized, _, _ = standardize(raw)
    for name, feats in [("raw", raw), ("standardized", standardized)]:
        A = torch.cat([feats, torch.ones(len(feats), 1, dtype=torch.float64)], dim=1)
        hessian = 2 / len(A) * A.T @ A
        eigenvalues = torch.linalg.eigvalsh(hessian)
        condition = (eigenvalues.max() / eigenvalues.min()).item()
        lr = 1.9 / eigenvalues.max().item()  # just under the stability limit 2 / λ_max
        best = torch.linalg.lstsq(A, y_range.unsqueeze(1)).solution.squeeze(1)
        best_mse = (A @ best - y_range).pow(2).mean().item()
        theta = torch.zeros(A.shape[1], dtype=torch.float64)
        gaps = []
        for _ in range(3000):
            residual = A @ theta - y_range
            gaps.append(residual.pow(2).mean().item() - best_mse)
            theta -= lr * 2 / len(A) * A.T @ residual
        print(f"x in [{lo:g}, {hi:g}], {name:12}: condition number {condition:9.3g}, final gap {gaps[-1]:.2e}")
        ax.semilogy(gaps, label=f"{name} (κ ≈ {condition:.3g})")
    ax.set_title(f"x ∈ [{lo:g}, {hi:g}]")
    ax.set_xlabel("gradient-descent step")
    ax.legend()
axes[0].set_ylabel("training MSE − optimal MSE")
fig.suptitle(f"Degree-{degree} polynomial features: gradient descent, raw vs standardized")
plt.tight_layout()
plt.show()

# %% [markdown]
# *Your conclusion:*

# %% [markdown] tags=["solution"]
# 1. On $[-2, 2]$ standardization lowers the condition number only from about 55 to 23, because the raw features are already centered and of moderate scale ($|x^3| \le 8$). On $[0, 4]$ it drops from about 70,000 to about 2,500 — a factor of nearly 30 — because there the raw columns have very different scales ($x^3$ reaches 64) and means far from zero, and the column of ones is nearly aligned with them.
# 2. The largest stable learning rate is set by the largest curvature $\lambda_{\max}$, and the error along the flattest direction then shrinks by a factor of roughly $1 - 2/\kappa$ per step (notebook 03), so the number of steps needed grows roughly in proportion to $\kappa$. On $[-2, 2]$ both versions reach the optimum to rounding error within a few hundred steps. On $[0, 4]$, after 3,000 steps the raw version is still about 0.036 above the optimal MSE, the standardized version about 0.001 — better, but not converged.
# 3. Standardization fixes each column's **scale** and **center**, but not the **correlation** between columns. On $[0, 4]$, $x$, $x^2$, and $x^3$ all increase together (their correlations are about 0.92–0.99), so the loss surface is still a long, narrow valley. A simple fix is to standardize $x$ *before* taking powers: features $z, z^2, z^3$ with $z = (x - \bar x)/s_x$ give a condition number of about 23 on $[0, 4]$, the same as on the symmetric interval. Orthogonal polynomial bases remove the correlation entirely.

# %% [markdown]
# ## Problem 6 (optional, GPU): the custom function on a device
#
# A custom `autograd.Function` works on any device as long as its operations do. Run `StableLogSumExp` on the selected GPU, and check that the values and gradients match the CPU within `float32` tolerance. This cell skips itself on a CPU-only machine.

# %% tags=["gpu", "check"]
if device.type == "cpu":
    print("No GPU selected; skipping.")
else:
    gen = torch.Generator().manual_seed(5)
    x_cpu = (30 * torch.randn(256, 10, generator=gen)).requires_grad_()
    x_dev = x_cpu.detach().to(device).requires_grad_()
    stable_lse(x_cpu).sum().backward()
    stable_lse(x_dev).sum().backward()
    check_close(stable_lse(x_dev).cpu(), stable_lse(x_cpu), f"values on {device.type}", rtol=1e-5, atol=1e-5)
    check_close(x_dev.grad.cpu(), x_cpu.grad, f"gradients on {device.type}", rtol=1e-4, atol=1e-6)
    passed("Problem 6")

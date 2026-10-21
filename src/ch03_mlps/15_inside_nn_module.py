# %% [markdown]
# # 15 — Inside `nn.Module`
#
# | | |
# | --- | --- |
# | **Prerequisites** | Notebooks 00–14, especially autograd and `torch.no_grad` (05), `nn.Linear` and the training step (06), the MLP built from a list of raw tensors (13), and the layer-by-layer backward pass (14) |
# | **You will learn** | What `nn.Module` does for you and how: how assigning an attribute registers a parameter or a submodule, and why a plain Python list or tensor is invisible; containers (`nn.ModuleList`, `nn.ModuleDict`, `nn.Sequential`); buffers for non-trainable state; `state_dict`, saving, loading and the errors strict loading gives; `train()` and `eval()`; forward and backward hooks for recording activations and gradients; re-initializing, editing and freezing parameters; moving a whole model between dtypes and devices |
# | **Estimated time** | About 2.5 hours, plus 1–2 hours for the exercises |
# | **Compute** | CPU core well under a minute. A short optional GPU cell moves a model to the accelerator. |

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
import os
import tempfile
from collections import OrderedDict

import torch.nn.functional as F
from torch import nn

from dlcourse.data import make_spirals
from dlcourse.plotting import COLORS, plot_decision_boundary

# %% [markdown]
# ## The motivating problem
#
# In notebook 13 we built an MLP from raw tensors: we created each weight matrix and bias vector with `requires_grad=True`, kept them in a Python list, wrote the forward pass as a function, and updated every tensor in the list by hand. Here is a compact version of that model for the three-class spirals.

# %%
X, y = make_spirals(n=200, classes=3, seed=0)    # X: (600, 2) float32, y: (600,) int64
N, D = X.shape
K = 3

gen = torch.Generator().manual_seed(0)
W1 = (torch.randn(64, D, generator=gen) / math.sqrt(D)).requires_grad_()    # (n_1, n_0) = (64, 2)
b1 = torch.zeros(64, requires_grad=True)                                     # (64,)
W2 = (torch.randn(K, 64, generator=gen) / math.sqrt(64)).requires_grad_()   # (K, 64)
b2 = torch.zeros(K, requires_grad=True)                                      # (K,)
params = [W1, b1, W2, b2]


def raw_mlp(X):
    """X: (N, 2) -> logits Z: (N, K). A = H W^T + b, as in notebook 13."""
    H1 = torch.relu(X @ W1.T + b1)    # (N, 64)
    return H1 @ W2.T + b2              # (N, K)


print(f"initial loss {F.cross_entropy(raw_mlp(X), y).item():.3f} (ln 3 = {math.log(3):.3f})")
for step in range(500):
    loss = F.cross_entropy(raw_mlp(X), y)
    for p in params:
        p.grad = None
    loss.backward()
    with torch.no_grad():
        for p in params:
            p -= 0.5 * p.grad
print(f"raw-tensor MLP: loss {loss.item():.3f}, accuracy {(raw_mlp(X).argmax(1) == y).float().mean().item():.3f}")

# %% [markdown]
# It trains: the loss falls below its starting value of about $\ln 3$. (One hidden layer and 500 plain gradient steps fit the three spirals only partly; the deeper networks later in this notebook do much better.) But everything around the arithmetic is our job:
#
# 1. **Tracking parameters.** Every new layer means two more tensors to create *and* to remember to add to `params`. Forget one and it is silently never trained.
# 2. **Moving and converting.** To run on a GPU, or in `float64` for a gradient check, we must convert every tensor and rebind every name.
# 3. **Saving and loading.** To save the model we must decide on names for the tensors and keep the save and load code in sync with the architecture.
# 4. **Modes.** Some layers must behave differently during training and evaluation (dropout, notebook 23; batch normalization, notebook 24). Our function has no notion of a mode.
# 5. **Inspection.** To see what happens inside — activation statistics per layer, gradients per layer — we would have to edit the forward function itself.
#
# PyTorch's `nn.Module` solves all five, and it is not magic: it is ordinary Python, built on two dictionaries and one overridden method. This notebook opens it up, one mechanism per section, so that later, when a model trains badly, you can see inside it.

# %% [markdown]
# ## 1. Writing a module
#
# A module is a Python class that subclasses `nn.Module` and defines two methods:
#
# - `__init__`: first call `super().__init__()` (this creates the module's internal bookkeeping), then create the layers and parameters as attributes.
# - `forward(x)`: compute the output from the input, using those attributes.
#
# Here is our own version of `nn.Linear`, holding the parameters of one layer: $W$ of shape `(out, in)` and $\mathbf b$ of shape `(out,)`, computing $A = XW^\top + \mathbf b$.

# %%
class MyLinear(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        bound = 1 / math.sqrt(n_in)
        # nn.Parameter wraps a tensor and marks it as a trainable parameter of this module.
        self.weight = nn.Parameter(torch.empty(n_out, n_in).uniform_(-bound, bound))   # (out, in)
        self.bias = nn.Parameter(torch.empty(n_out).uniform_(-bound, bound))           # (out,)

    def forward(self, x):            # x: (N, in)
        return x @ self.weight.T + self.bias      # (N, out)


set_seed(0)
mine = MyLinear(2, 4)
print(mine)
print("parameters:", [(name, tuple(p.shape)) for name, p in mine.named_parameters()])
print("a Parameter is a Tensor:", isinstance(mine.weight, torch.Tensor), "| requires_grad:", mine.weight.requires_grad)

# %% [markdown]
# `nn.Parameter` is a subclass of `torch.Tensor` with `requires_grad=True` by default; its only extra meaning is "this belongs to the module as a trainable parameter". Our layer computes exactly what `nn.Linear` computes. To check, copy the same numbers into an `nn.Linear` (the in-place copy needs `torch.no_grad()`; Section 7 explains why) and compare outputs:

# %%
ref = nn.Linear(2, 4)
with torch.no_grad():
    ref.weight.copy_(mine.weight)
    ref.bias.copy_(mine.bias)
x5 = torch.randn(5, 2)                           # (5, 2)
print("same output as nn.Linear:", torch.allclose(mine(x5), ref(x5)))

# %% [markdown]
# **Why `super().__init__()` must come first.** Assigning a parameter or submodule needs the bookkeeping that `nn.Module.__init__` creates. Forget the call and the first such assignment fails:

# %%
class Forgetful(nn.Module):
    def __init__(self):
        # super().__init__() is missing
        self.layer = nn.Linear(2, 3)


try:
    Forgetful()
except AttributeError as error:
    print("AttributeError:", error)

# %% [markdown]
# ### Call the module, not `.forward`
#
# We wrote `mine(x5)`, not `mine.forward(x5)`. Calling a module runs `nn.Module.__call__`, which does more than call `forward`: it runs any **hooks** registered on the module before and after `forward` (Section 6 is about hooks). Calling `.forward` directly skips them. A one-line preview makes the difference visible: a forward hook that counts calls.

# %%
calls = []
handle = mine.register_forward_hook(lambda module, inputs, output: calls.append(tuple(output.shape)))
_ = mine(x5)            # goes through __call__: the hook runs
_ = mine.forward(x5)    # bypasses __call__: the hook does not run
print("hook saw", len(calls), "call(s):", calls)
handle.remove()

# %% [markdown]
# Two forward passes, one recorded. Any tool built on hooks — activation recorders below, many profilers and debugging tools — silently misses work done through `.forward`. Always call the module.

# %% [markdown]
# ### Checkpoint 1.1
#
# Write a module `Affine1D` that holds two **scalar** parameters, `scale` (initialized to 1) and `shift` (initialized to 0), and computes `scale * x + shift` elementwise for an input of any shape. Store an instance in `aff`.

# %% [markdown] tags=["hint"]
# `self.scale = nn.Parameter(torch.tensor(1.0))` creates a 0-d parameter. Do not forget `super().__init__()`.

# %%
# BEGIN SOLUTION
class Affine1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.scale * x + self.shift


aff = Affine1D()
# END SOLUTION

# %% tags=["check"]
_names = sorted(name for name, _ in aff.named_parameters())
check(_names == ["scale", "shift"], f"aff should have parameters ['scale', 'shift'], found {_names}")
check_shape(aff.scale, (), "scale")
_x = torch.randn(3, 4)
check_close(aff(_x), _x, "output at initialization (scale 1, shift 0)")
with torch.no_grad():
    aff.scale.fill_(2.0)
    aff.shift.fill_(-1.0)
check_close(aff(_x), 2 * _x - 1, "output with scale 2, shift -1")
passed("Checkpoint 1.1")

# %% [markdown]
# ## 2. Parameter registration
#
# How did `named_parameters()` know about `weight` and `bias`? We never put them in a list. The answer is that `nn.Module` overrides **attribute assignment**, `__setattr__`. When you write `self.name = value`, it checks the type of `value`:
#
# - an `nn.Parameter` → stored in the dictionary `self._parameters` under `name`;
# - an `nn.Module` → stored in the dictionary `self._modules` under `name`;
# - anything else (a float, a string, a plain tensor, a list) → stored as an ordinary Python attribute in the instance `__dict__`.
#
# Reading `self.name` back works for all three because `nn.Module.__getattr__` looks in `_parameters`, `_buffers` (Section 3) and `_modules` when the normal lookup fails. Let us look inside a small two-layer network.

# %%
class TwoLayer(nn.Module):
    def __init__(self, n_in=2, n_hidden=16, n_out=3):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.temperature = 1.0                      # a plain float: configuration, not a parameter

    def forward(self, x):                           # x: (N, n_in)
        h = torch.relu(self.fc1(x))                 # (N, n_hidden)
        return self.fc2(h) / self.temperature       # (N, n_out)


set_seed(0)
net = TwoLayer()
print("net._modules:   ", list(net._modules.keys()))
print("net._parameters:", list(net._parameters.keys()), "(its own, not its children's)")
print("net.fc1._parameters:", list(net.fc1._parameters.keys()))
print("'fc1' in the instance __dict__:", "fc1" in net.__dict__, "| 'temperature' in __dict__:", "temperature" in net.__dict__)

# %% [markdown]
# Each module stores only its **own** parameters and its **direct** children. Everything else is found by walking the tree:
#
# - `named_parameters()` recurses through `_modules` and yields every parameter with a dotted path, such as `fc1.weight`.
# - `children()` yields the direct submodules; `modules()` / `named_modules()` yield the module itself and every descendant, depth first.

# %%
for name, p in net.named_parameters():
    print(f"{name:12s} {str(tuple(p.shape)):10s} {p.numel():5d} values")
print("children:", [type(m).__name__ for m in net.children()])
print("modules: ", [(name or "<root>", type(m).__name__) for name, m in net.named_modules()])

# %% [markdown]
# **Counting parameters.** An `nn.Linear(n_in, n_out)` has $n_\text{out} n_\text{in}$ weights and $n_\text{out}$ biases, $n_\text{out}(n_\text{in} + 1)$ in total. For `TwoLayer`: $16 \cdot 3 + 3 \cdot 17 = 48 + 51 = 99$. The standard one-liner agrees:

# %%
n_params = sum(p.numel() for p in net.parameters())
print("parameters:", n_params, "| by formula:", 16 * (2 + 1) + 3 * (16 + 1))

# %% [markdown]
# `parameters()` is what we hand to an optimizer, so "registered" means "will be trained". It also means "will be moved by `.to()`" (Section 8) and "will be saved by `state_dict()`" (Section 4). One more detail: if the same parameter is reachable under two names (two attributes pointing at the same layer, as in weight tying), `parameters()` yields it **once**, so an optimizer does not update it twice.

# %%
tied = TwoLayer()
tied.fc1_again = tied.fc1                             # the same module, registered under a second name
print("named_parameters:", [name for name, _ in tied.named_parameters()])
print("count:", sum(p.numel() for p in tied.parameters()), "(still 99)")

# %% [markdown]
# ### What is *not* registered
#
# Because registration is decided by the type of the assigned value, two common patterns silently escape it.
#
# **A plain tensor**, even with `requires_grad=True`, is just an attribute. It gets gradients when you call `backward`, but no optimizer built from `parameters()` will ever update it, and `state_dict()` and `.to()` ignore it.

# %%
class PlainTensorScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 3)
        self.scale = torch.ones(3, requires_grad=True)       # a plain tensor: NOT registered

    def forward(self, x):
        return self.fc(x) * self.scale


m_plain = PlainTensorScale()
print("parameters:", [name for name, _ in m_plain.named_parameters()])
print("state_dict keys:", list(m_plain.state_dict().keys()))

# %% [markdown]
# `scale` is missing from both. The fix is `self.scale = nn.Parameter(torch.ones(3))`.
#
# **A plain Python list (or dict) of modules.** `__setattr__` sees a `list`, which is neither a `Parameter` nor a `Module`, so it stores the list as an ordinary attribute and never looks inside it. The layers in the list work in `forward` — the outputs are perfectly sensible — but the module does not know they exist.

# %%
class ListMLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = [nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])]   # plain list: NOT registered

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


list_mlp = ListMLP([2, 16, 16, 3])
print("output shape:", tuple(list_mlp(X).shape), "— forward works")
print("registered parameters:", sum(p.numel() for p in list_mlp.parameters()))
try:
    torch.optim.SGD(list_mlp.parameters(), lr=0.1)
except ValueError as error:
    print("ValueError:", error)

# %% [markdown]
# Here the failure is loud, because *no* parameter is registered. If the model also had one properly registered layer, the optimizer would be created without complaint and would train only that layer — a silent bug. The containers in `torch.nn` exist to fix exactly this. They are themselves modules, so assigning one registers it, and they register their contents as children:
#
# - `nn.ModuleList(list_of_modules)`: indexable and iterable like a list; children are named `"0"`, `"1"`, ….
# - `nn.ModuleDict(dict_of_modules)`: indexed by string keys, which become the child names.
# - `nn.Sequential(m1, m2, …)`: like `ModuleList`, plus a `forward` that applies the modules in order. Passing an `OrderedDict` gives the children meaningful names.
#
# (For raw parameters, `nn.ParameterList` and `nn.ParameterDict` play the same role.)

# %%
class ModuleListMLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


class HeadsNet(nn.Module):
    """A shared body with one output head per task, chosen by name."""
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(2, 8)
        self.heads = nn.ModuleDict({"color": nn.Linear(8, 3), "size": nn.Linear(8, 1)})

    def forward(self, x, task):
        return self.heads[task](torch.relu(self.body(x)))


seq = nn.Sequential(OrderedDict([("hidden", nn.Linear(2, 16)), ("act", nn.ReLU()), ("out", nn.Linear(16, 3))]))
for model in [ModuleListMLP([2, 16, 16, 3]), HeadsNet(), seq]:
    names = [name for name, _ in model.named_parameters()]
    print(f"{type(model).__name__:14s} {sum(p.numel() for p in model.parameters()):4d} parameters: {names}")

# %% [markdown]
# The dotted names show the path through the tree: `layers.1.weight` is the weight of child `"1"` of the `ModuleList` called `layers`. `ModuleListMLP([2, 16, 16, 3])` has $16 \cdot 3 + 16 \cdot 17 + 3 \cdot 17 = 371$ parameters, which a `ListMLP` with the same sizes also *uses* but does not *register*. (The chapter problem set asks you to find this bug in a model that trains badly without any error.)

# %% [markdown]
# ### Checkpoint 2.1
#
# Write `count_mlp_params(sizes)` that returns, **from the formula** (no modules), the number of parameters of an MLP whose `nn.Linear` layers map `sizes[0] → sizes[1] → … → sizes[-1]`. For example, `count_mlp_params([2, 16, 3])` is 99.

# %% [markdown] tags=["hint"]
# Each consecutive pair `(a, b)` contributes `b * (a + 1)`. Iterate over `zip(sizes[:-1], sizes[1:])`.

# %%
# BEGIN SOLUTION
def count_mlp_params(sizes):
    return sum(b * (a + 1) for a, b in zip(sizes[:-1], sizes[1:]))
# END SOLUTION

# %% tags=["check"]
check(count_mlp_params([2, 16, 3]) == 99, f"count_mlp_params([2, 16, 3]) = {count_mlp_params([2, 16, 3])}, expected 99")
for _sizes in [[2, 64, 64, 64, 3], [10, 1], [784, 256, 128, 10]]:
    _expected = sum(p.numel() for p in ModuleListMLP(_sizes).parameters())
    check(count_mlp_params(_sizes) == _expected, f"count_mlp_params({_sizes}) = {count_mlp_params(_sizes)}, expected {_expected}")
passed("Checkpoint 2.1")

# %% [markdown]
# ## 3. Buffers: state that is not trained
#
# Some modules need tensors that are part of the model's state but are **not** learned by gradient descent: a running average of the inputs, a fixed mask, a lookup table. They should move with the model to a device and be saved with it, like parameters, but the optimizer must not touch them and they need no gradients.
#
# `self.register_buffer(name, tensor)` stores such a tensor in a third dictionary, `self._buffers`. Buffers appear in `buffers()` / `named_buffers()` and in `state_dict()`, never in `parameters()`.
#
# As an example we build `RunningCenter`, a module that subtracts the mean of each input feature:
#
# - in **training mode** it subtracts the mean of the current batch, $\bar{\mathbf x}_t$ (shape `(D,)`), and updates a running average of those batch means,
#
# $$\mathbf r_t = (1 - m)\,\mathbf r_{t-1} + m\,\bar{\mathbf x}_t, \qquad \mathbf r_0 = \mathbf 0,$$
#
# with a **momentum** $m \in (0, 1]$;
# - in **evaluation mode** it subtracts the stored $\mathbf r$ and leaves it unchanged.
#
# This is exactly what batch normalization (notebook 24) does for the mean, using the same convention for $m$. The flag `self.training`, which tells the module which mode it is in, is the subject of Section 5.

# %%
class RunningCenter(nn.Module):
    """Subtract per-feature means. x: (N, D) -> (N, D)."""
    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.momentum = momentum                                       # plain float: configuration
        self.register_buffer("running_mean", torch.zeros(num_features))   # (D,) state, not trained
        self.register_buffer("num_batches_tracked", torch.tensor(0))      # () int64 counter

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)                                 # (D,)
            with torch.no_grad():                                      # updating state is not part of the graph
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
                self.num_batches_tracked += 1
            return x - batch_mean
        return x - self.running_mean


center = RunningCenter(2)
print("parameters:", list(center.named_parameters()))
print("buffers:   ", [(name, b.tolist()) for name, b in center.named_buffers()])
print("state_dict:", list(center.state_dict().keys()))
print("requires_grad of running_mean:", center.running_mean.requires_grad)

# %% [markdown]
# No parameters, two buffers, and both buffers in the `state_dict`. Now feed it a stream of mini-batches of 32 points drawn from the spirals shifted to have mean near $(2, -1)$. (Notebook 21 builds proper mini-batch pipelines; here we just draw random row indices.)

# %%
X_shift = X + torch.tensor([2.0, -1.0])                 # (600, 2)
true_mean = X_shift.mean(dim=0)                         # (2,)
set_seed(0)
center = RunningCenter(2, momentum=0.1)
history = []
for t in range(80):
    batch = X_shift[torch.randint(0, N, (32,))]         # (32, 2)
    center(batch)
    history.append(center.running_mean.clone())         # clone: the buffer itself keeps changing
history = torch.stack(history)                          # (80, 2)
print("data mean:            ", [round(v, 3) for v in true_mean.tolist()])
print("running mean after 80:", [round(v, 3) for v in center.running_mean.tolist()],
      "| batches tracked:", center.num_batches_tracked.item())

# %%
fig, ax = plt.subplots(figsize=(7, 4))
steps = torch.arange(1, 81)
for j, col in enumerate(COLORS[:2]):
    ax.plot(steps, history[:, j], color=col, lw=2, label=f"running mean, feature {j + 1}")
    ax.axhline(true_mean[j].item(), color=col, ls="--", lw=1, label=f"data mean, feature {j + 1}")
    ax.plot(steps, true_mean[j] * (1 - 0.9 ** steps), color="gray", ls=":", lw=1.5,
            label="$\\mu(1 - 0.9^t)$" if j == 0 else None)
ax.set_xlabel("number of training batches $t$")
ax.set_ylabel("running mean $r_t$")
ax.set_title("A buffer tracking the input mean (momentum 0.1)")
ax.legend(fontsize=8, ncol=2)
plt.show()

# %% [markdown]
# The running mean starts at 0 and approaches the data mean like $\mu(1 - 0.9^t)$ (dotted): unrolling the recursion with $\mathbf r_0 = 0$ gives $\mathbf r_t = m\sum_{s=1}^{t}(1-m)^{t-s}\bar{\mathbf x}_s$, and if every batch mean were exactly $\mu$, the geometric sum is $\mu(1 - (1-m)^t)$. The real curves wiggle around this because each batch of 32 has its own mean. After about 40 batches the start-up bias is gone and what remains is batch-to-batch noise.
#
# **Buffers get no gradients.** Because the buffer was created without `requires_grad` and only updated under `no_grad`, backpropagating through the module leaves it alone:

# %%
center.eval()                                           # use the stored mean (Section 5)
x_req = X_shift[:4].clone().requires_grad_()            # (4, 2)
center(x_req).sum().backward()
print("grad of the input:", x_req.grad[0].tolist(), "| grad of running_mean:", center.running_mean.grad)

# %% [markdown]
# **Persistent and non-persistent buffers.** `register_buffer(name, tensor, persistent=False)` creates a buffer that moves with the model but is **not** saved in the `state_dict`. Use it for values that can be recomputed from the configuration, such as a fixed mask, so checkpoints do not carry them around.

# %%
class MaskedLinear(nn.Module):
    """A linear layer whose weight is multiplied by a fixed 0/1 mask (out, in)."""
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.register_buffer("mask", torch.tril(torch.ones(n_out, n_in)), persistent=False)

    def forward(self, x):
        return x @ (self.fc.weight * self.mask).T + self.fc.bias


masked = MaskedLinear(3, 3)
print("buffers:   ", [name for name, _ in masked.named_buffers()])
print("state_dict:", list(masked.state_dict().keys()))

# %% [markdown]
# ### Checkpoint 3.1
#
# Suppose every batch has exactly the same mean $\mu$. With momentum $m = 0.1$ and $\mathbf r_0 = 0$, how many batches does it take before the running mean is within 1% of $\mu$, that is, $|r_t - \mu| < 0.01|\mu|$? Derive `k_needed` (the smallest such $t$) from the formula $r_t = \mu(1 - 0.9^t)$; the check simulates `RunningCenter` to confirm.

# %% [markdown] tags=["hint"]
# $|r_t - \mu| = 0.9^t |\mu|$, so you need $0.9^t < 0.01$, i.e. $t > \ln 0.01 / \ln 0.9$. Use `math.ceil`, and make sure the inequality is strict.

# %%
# BEGIN SOLUTION
k_needed = math.floor(math.log(0.01) / math.log(0.9)) + 1
# END SOLUTION
print("k_needed =", k_needed)

# %% tags=["check"]
_c = RunningCenter(1, momentum=0.1)
_batch = torch.full((8, 1), 5.0, dtype=torch.float64)
_c.double()
_close = []
for _t in range(1, 80):
    _c(_batch)
    _close.append(abs(_c.running_mean.item() - 5.0) < 0.05)
check(isinstance(k_needed, int), f"k_needed should be an int, got {type(k_needed).__name__}")
check(k_needed == _close.index(True) + 1, f"k_needed = {k_needed}, but the simulation first gets within 1% at t = {_close.index(True) + 1}")
passed("Checkpoint 3.1")

# %% [markdown]
# ## 4. `state_dict`: the model as a dictionary of tensors
#
# `model.state_dict()` returns an ordered dictionary from dotted names to tensors: every registered parameter and every persistent buffer, in registration order. It does **not** contain the code (the class), plain attributes like `temperature` or `momentum`, or non-persistent buffers. Saving a model means saving this dictionary; loading means building a model from the class and copying tensors into it by name.
#
# We use a model that combines everything so far: a `RunningCenter`, then an `nn.Sequential` hidden part, then an output layer.

# %%
class CenteredMLP(nn.Module):
    def __init__(self, n_in=2, n_hidden=64, n_out=3):
        super().__init__()
        self.center = RunningCenter(n_in)
        self.hidden = nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU())
        self.out = nn.Linear(n_hidden, n_out)

    def forward(self, x):                      # x: (N, n_in)
        return self.out(self.hidden(self.center(x)))   # (N, n_out) logits


set_seed(0)
model = CenteredMLP()
sd = model.state_dict()
for name, tensor in sd.items():
    print(f"{name:32s} {str(tuple(tensor.shape)):10s} {tensor.dtype}")

# %% [markdown]
# The names follow the module tree (`hidden.2.weight` is child `"2"` of `hidden`; the `ReLU`s at indices 1 and 3 have no parameters, so they do not appear). Two facts about these tensors matter in practice:
#
# - They are **detached** (`requires_grad=False`) but **share memory** with the live parameters. A `state_dict` is a view of the model, not a copy: if the model keeps training, the "saved" tensors in memory change too.
# - `torch.save` writes them to a file, so a file is a real snapshot.

# %%
print("shares memory with the parameter:", sd["out.bias"].data_ptr() == model.out.bias.data_ptr(),
      "| requires_grad:", sd["out.bias"].requires_grad)
before = sd["out.bias"][0].item()
with torch.no_grad():
    model.out.bias += 1.0                               # "keep training"
print(f"sd['out.bias'][0] before: {before:.4f}, after editing the model: {sd['out.bias'][0].item():.4f}")
with torch.no_grad():
    model.out.bias -= 1.0                               # undo

# %% [markdown]
# Now let us train the model briefly on the spirals (shifted, so the centering matters), save it to a temporary file, and load it into a **fresh** model created with a different seed.

# %%
def train(model, X, y, steps=2000, lr=0.3):
    """Full-batch gradient descent on cross-entropy. Returns the final loss."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
    return loss.item()


set_seed(0)
model = CenteredMLP()
print(f"final training loss: {train(model, X_shift, y):.4f}")
model.eval()

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "centered_mlp.pt")
    torch.save(model.state_dict(), path)
    n_floats = sum(t.numel() for t in model.state_dict().values())
    print(f"file size: {os.path.getsize(path)} bytes for {n_floats} numbers ({4 * n_floats} bytes as float32/int64 data)")
    loaded_sd = torch.load(path, weights_only=True)

set_seed(123)
fresh = CenteredMLP().eval()
print("fresh model, before loading: same logits?", torch.equal(fresh(X_shift), model(X_shift)))
result = fresh.load_state_dict(loaded_sd)
print("load_state_dict returned:", result)
print("fresh model, after loading:  same logits?", torch.equal(fresh(X_shift), model(X_shift)))

# %% [markdown]
# After loading, the outputs are **bit-for-bit identical** (`torch.equal`, not just close): the same numbers went through the same code. The file is 21,907 bytes: the 4,550 stored numbers take about 18,200 bytes at 4 bytes each (the one `int64` counter actually takes 8), and the rest is names and structure. The temporary directory, and the file in it, are deleted when the `with` block ends.
#
# `weights_only=True` tells `torch.load` to accept only tensors and plain containers. A `.pt` file is a Python pickle, and unpickling arbitrary data can execute arbitrary code; `weights_only=True` is the default since PyTorch 2.6, and you should keep it for files you did not create.
#
# A `state_dict` alone is enough to reuse a trained model. To **resume training** you also need the optimizer's state (momentum buffers, step counts) and more; notebook 27 builds full checkpoints.

# %% [markdown]
# ### Strict loading and its errors
#
# `load_state_dict` is **strict** by default: every key in the file must match a key in the model and vice versa, with identical shapes. The error messages are worth learning to read. First, a model whose hidden width differs:

# %%
try:
    CenteredMLP(n_hidden=32).load_state_dict(loaded_sd)
except RuntimeError as error:
    print(error)

# %% [markdown]
# Every mismatch is listed with both shapes: "from checkpoint" is the file, "current model" is the model you are loading into. Second, a model with the same layers under different names — for example, a refactor that renamed `out` to `classifier`:

# %%
class RenamedMLP(CenteredMLP):
    def __init__(self):
        super().__init__()
        self.classifier = self.out      # the same layer under a new name ...
        del self.out                    # ... and the old name removed

    def forward(self, x):
        return self.classifier(self.hidden(self.center(x)))


try:
    RenamedMLP().load_state_dict(loaded_sd)
except RuntimeError as error:
    print(error)

# %% [markdown]
# "Missing" keys are in the model but not in the file; "unexpected" keys are in the file but not in the model. Here the fix is to rename the keys in the dictionary, e.g. `{k.replace("out.", "classifier.", 1): v for k, v in sd.items()}`.
#
# `strict=False` turns the key errors into a report and loads whatever matches. This is useful when you *intend* a partial load (reusing a trained body with a new head), and dangerous otherwise: a model with every key mismatched loads "successfully" and stays randomly initialized. Always inspect the returned lists. (Shape mismatches remain errors even with `strict=False`.)

# %%
partial = RenamedMLP()
report = partial.load_state_dict(loaded_sd, strict=False)
print("missing:   ", report.missing_keys)
print("unexpected:", report.unexpected_keys)

# %% [markdown]
# ### Checkpoint 4.1
#
# Because a `state_dict` shares memory with the model, `best = model.state_dict()` does **not** remember the best weights seen during training. Make a real in-memory snapshot of `model` called `snapshot`: a dictionary with the same keys whose tensors are **independent copies**. The check then changes the model and restores it from your snapshot.

# %% [markdown] tags=["hint"]
# A dictionary comprehension over `model.state_dict().items()`, calling `.clone()` on each tensor.

# %%
# BEGIN SOLUTION
snapshot = {name: tensor.clone() for name, tensor in model.state_dict().items()}
# END SOLUTION

# %% tags=["check"]
check(set(snapshot) == set(model.state_dict()), "snapshot should have exactly the keys of model.state_dict()")
_backup = {_k: _v.clone() for _k, _v in model.state_dict().items()}
_logits = model.eval()(X_shift).detach()
with torch.no_grad():
    for _p in model.parameters():
        _p.mul_(0.5)
_changed = not torch.equal(model(X_shift), _logits)
model.load_state_dict(snapshot)
_restored = torch.equal(model(X_shift), _logits)
model.load_state_dict(_backup)                       # leave the model as it was, whatever happened
check(_changed, "the model should have changed")
check(_restored, "restoring from snapshot did not recover the original outputs: are the tensors copies?")
passed("Checkpoint 4.1")

# %% [markdown]
# ## 5. `train()` and `eval()`
#
# Every module has a boolean attribute `training`. `model.train()` sets it to `True` on the module **and all its descendants**; `model.eval()` is the same as `model.train(False)`. Modules created fresh start in training mode. Most layers (`nn.Linear`, `nn.ReLU`) ignore the flag; a few behave differently depending on it — our `RunningCenter`, dropout (notebook 23), and batch normalization (notebook 24).

# %%
def mode_report(model):
    return {name or "<root>": m.training for name, m in model.named_modules()}


model.train()
print("after train():", set(mode_report(model).values()))
model.eval()
print("after eval(): ", set(mode_report(model).values()))
model.hidden.train()                       # flags can be set on part of the tree
print("after model.hidden.train():", mode_report(model))
_ = model.eval()

# %% [markdown]
# Setting the mode on a subtree changes only that subtree, so a model can end up in a mixed state; calling `model.train()` or `model.eval()` on the root is the safe habit.
#
# **`eval()` does not turn off gradient tracking.** The flag only changes how mode-dependent layers compute. Turning off the graph is `torch.no_grad()` (notebook 05); for prediction you usually want both.

# %%
print("eval mode, output requires_grad:", model(X_shift[:2]).requires_grad)
with torch.no_grad():
    print("eval + no_grad, output requires_grad:", model(X_shift[:2]).requires_grad)

# %% [markdown]
# ### Why the mode matters: predicting one example at a time
#
# Our trained `CenteredMLP` contains a `RunningCenter`. In training mode it subtracts the **batch** mean. Predict a single example in training mode and the batch mean *is* that example, so the centered input is exactly $\mathbf 0$ — every example gets the same prediction. Worse, every call also moves the running mean.

# %%
pts = X_shift[[0, 250, 500]]                            # one point from each class, (3, 2)
model.train()
rm_before = model.center.running_mean.clone()
one_by_one_train = [model(p[None]).argmax(1).item() for p in pts]   # p[None]: a batch of one, (1, 2)
rm_moved = (model.center.running_mean - rm_before).abs().max().item()
model.eval()
one_by_one_eval = [model(p[None]).argmax(1).item() for p in pts]
print("true labels:                   ", y[[0, 250, 500]].tolist())
print("train mode, one at a time:     ", one_by_one_train, f"(running mean moved by up to {rm_moved:.3f})")
print("eval mode, one at a time:      ", one_by_one_eval)
with torch.no_grad():
    model.center.running_mean.copy_(rm_before)          # undo the drift (the counter we leave alone)

# %% [markdown]
# In training mode the three different points get the same predicted class, and the buffer drifted. In evaluation mode each point is centered with the stored mean and classified correctly. Forgetting `model.eval()` before evaluation is one of the most common bugs in practice; with BatchNorm (notebook 24) the effect is the same in kind.
#
# **Dropout as a black box.** `nn.Dropout(p)` is the most common mode-dependent layer. In training mode it sets each entry to 0 with probability $p$ and multiplies the survivors by $1/(1-p)$; in evaluation mode it returns its input unchanged. Notebook 23 explains why this helps; here we only watch the flag at work.

# %%
drop = nn.Dropout(p=0.5)
ones = torch.ones(12)
set_seed(0)
print("train mode:", drop(ones).tolist())
drop.eval()
print("eval mode: ", drop(ones).tolist())
drop.train()
avg = torch.stack([drop(ones) for _ in range(10_000)]).mean(dim=0)
print(f"train mode, average of 10,000 calls: min {avg.min().item():.3f}, max {avg.max().item():.3f}")

# %% [markdown]
# Each entry is 0 or 2 in training mode, and the scaling makes the **average** over many calls about 1 — the evaluation-mode output. (Exercise 4 asks whether this averaging argument still holds for a whole network.)

# %% [markdown]
# From now on we predict with a small helper that switches to evaluation mode, turns off gradient tracking, and then **restores** the mode the model was in (so a training loop that calls it for monitoring does not continue training in evaluation mode):

# %%
def predict(model, X):
    """Class labels (N,) int64, computed in eval mode without tracking gradients."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        labels = model(X).argmax(dim=1)
    model.train(was_training)
    return labels


model.train()
labels_full = predict(model, X_shift)
print("still in training mode afterwards:", model.training,
      "| accuracy:", round((labels_full == y).float().mean().item(), 3),
      "| one point alone gets its batch label:", torch.equal(predict(model, X_shift[:1]), labels_full[:1]))
_ = model.eval()

# %% [markdown]
# ### Checkpoint 5.1
#
# By hand, before running anything: a fresh `RunningCenter(1, momentum=0.5)` receives three calls, in this order:
#
# 1. in training mode, the batch $[2, 4]$ (shape `(2, 1)`);
# 2. in evaluation mode, the batch $[10]$;
# 3. in training mode, the batch $[1]$.
#
# Store the output of call 2 (a single number) in `out_call2`, the output of call 3 in `out_call3`, and the final `running_mean` in `r_final`. The check runs the calls.

# %% [markdown] tags=["hint"]
# Call 1: batch mean 3, so $r = 0.5 \cdot 0 + 0.5 \cdot 3$. Call 2 uses $r$ and does not change it. Call 3: a batch of one is centered with its own mean.

# %%
# BEGIN SOLUTION
out_call2 = 10 - 1.5          # eval: subtract r = 1.5, no update
out_call3 = 0.0               # train: a batch of one minus its own mean
r_final = 0.5 * 1.5 + 0.5 * 1.0
# END SOLUTION

# %% tags=["check"]
_rc = RunningCenter(1, momentum=0.5)
_rc.train()
_rc(torch.tensor([[2.0], [4.0]]))
_rc.eval()
_o2 = _rc(torch.tensor([[10.0]])).item()
_rc.train()
_o3 = _rc(torch.tensor([[1.0]])).item()
check_close(out_call2, _o2, "out_call2")
check_close(out_call3, _o3, "out_call3")
check_close(r_final, _rc.running_mean.item(), "r_final")
passed("Checkpoint 5.1")

# %% [markdown]
# ## 6. Hooks: looking inside without editing `forward`
#
# A **hook** is a function you attach to a module (or a tensor) that PyTorch calls at a fixed moment. The three you will use most:
#
# | Registration | Called | Signature |
# | --- | --- | --- |
# | `module.register_forward_hook(fn)` | after `module.forward` returns | `fn(module, inputs, output)`; `inputs` is a tuple |
# | `module.register_full_backward_hook(fn)` | when gradients with respect to the module's inputs have been computed | `fn(module, grad_input, grad_output)`; both tuples |
# | `tensor.register_hook(fn)` | when the gradient of `tensor` has been computed | `fn(grad)` |
#
# Each registration returns a **handle**; `handle.remove()` detaches the hook. A forward hook that returns a value **replaces** the module's output; one that returns `None` only observes. (There is also `register_forward_pre_hook`, called before `forward` with the inputs.)
#
# For a `nn.Linear` layer $\ell$ with output $A^{(\ell)}$, `grad_output[0]` is the upstream gradient $G = \partial L / \partial A^{(\ell)}$, shape `(N, n_ℓ)`, and `grad_input[0]` is $\partial L / \partial H^{(\ell-1)} = G\,W^{(\ell)}$ (notebook 14) — or `None` if the layer's input does not require gradients, as for the first layer, whose input is the data.
#
# The model for this section is a three-hidden-layer ReLU MLP, built as an `nn.Sequential` with named children so the hook records are readable.

# %%
def make_mlp(sizes=(2, 64, 64, 64, 3)):
    """ReLU MLP with children named fc1, act1, ..., out."""
    layers = OrderedDict()
    n_hidden = len(sizes) - 2
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]), start=1):
        if i <= n_hidden:
            layers[f"fc{i}"] = nn.Linear(a, b)
            layers[f"act{i}"] = nn.ReLU()
        else:
            layers["out"] = nn.Linear(a, b)
    return nn.Sequential(layers)


set_seed(0)
mlp = make_mlp()
print(mlp)

# %% [markdown]
# First, verify that a backward hook delivers the same gradient that autograd would compute directly. We capture the output $A^{(2)}$ of `fc2` with a forward hook (and ask autograd to keep its gradient with `retain_grad()`), capture `grad_output` with a backward hook, and compare. As a third route, a forward hook can attach a **tensor** hook to the output it sees.

# %%
captured = {}


def keep_output(module, inputs, output):
    output.retain_grad()                                            # keep .grad of this non-leaf tensor
    output.register_hook(lambda g: captured.__setitem__("tensor_hook", g))
    captured["A2"] = output


def keep_grad(module, grad_input, grad_output):
    captured["grad_output"] = grad_output[0]
    captured["grad_input"] = grad_input[0]


handles = [mlp.fc2.register_forward_hook(keep_output), mlp.fc2.register_full_backward_hook(keep_grad)]
loss = F.cross_entropy(mlp(X), y)
loss.backward()
for h in handles:
    h.remove()

G = captured["A2"].grad                                             # (N, 64): ∂L/∂A^(2)
print("shape of ∂L/∂A2:", tuple(G.shape))
print("backward hook grad_output == retain_grad:", torch.allclose(captured["grad_output"], G))
print("tensor hook gradient       == retain_grad:", torch.allclose(captured["tensor_hook"], G))
print("grad_input == G W2 (notebook 14):        ", torch.allclose(captured["grad_input"], G @ mlp.fc2.weight))
print("hooks left on fc2:", len(mlp.fc2._forward_hooks), "forward,", len(mlp.fc2._backward_hooks), "backward")

# %% [markdown]
# All three routes agree, and the backward formula from notebook 14 holds for `grad_input`. After `remove()`, no hooks remain.
#
# A hook's ability to **replace** the output is powerful (and occasionally a source of confusing bugs). For instance, a hook returning `output * 0` on `act3` erases the last hidden layer, so the logits become just the output bias:

# %%
h = mlp.act3.register_forward_hook(lambda module, inputs, output: output * 0)
with torch.no_grad():
    print("logits of 2 points with the hook:", mlp(X[:2]).tolist(), "\noutput bias:                     ", mlp.out.bias.tolist())
h.remove()

# %% [markdown]
# ### A reusable activation recorder
#
# Now the useful version. `ActivationRecorder` attaches a forward hook to every module of the chosen types and, on every forward call, stores **summary statistics** of the module's output (storing the full tensors from thousands of steps would waste memory). It also works as a context manager, so the hooks are removed even if the code inside raises an error.

# %%
class ActivationRecorder:
    """Record statistics of the outputs of selected modules on every forward pass.

    records[name] is a list with one dict {stat_name: float} per forward call.
    """
    def __init__(self, model, layer_types=(nn.ReLU,), stats=None):
        self.stats = stats or {
            "mean": lambda a: a.mean().item(),
            "frac_zero": lambda a: (a == 0).float().mean().item(),
        }
        self.records = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, layer_types):
                self.records[name] = []
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inputs, output):
            a = output.detach()
            self.records[name].append({k: f(a) for k, f in self.stats.items()})
        return hook

    def series(self, name, stat):
        """The recorded values of one statistic of one layer, as a tensor (calls,)."""
        return torch.tensor([r[stat] for r in self.records[name]])

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


# %% [markdown]
# For gradients we write a matching `GradientRecorder`. It records $\|\partial L/\partial A^{(\ell)}\|$, the norm of the upstream gradient arriving at the output of each `nn.Linear`. We could use full backward hooks, but on the first layer, whose input (the data) does not require gradients, PyTorch fires them with `grad_input = (None,)` and prints a warning. The tensor-hook route verified above avoids this: a forward hook attaches a tensor hook to each output it sees.

# %%
class GradientRecorder:
    """Record the norm of ∂L/∂output of selected modules on every backward pass."""
    def __init__(self, model, layer_types=(nn.Linear,)):
        self.records = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, layer_types):
                self.records[name] = []
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def forward_hook(module, inputs, output):
            if output.requires_grad:              # e.g. not under torch.no_grad()
                output.register_hook(lambda g: self.records[name].append(g.norm().item()))
        return forward_hook

    def series(self, name):
        return torch.tensor(self.records[name])

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


# %% [markdown]
# Train the MLP on the spirals for 2000 full-batch steps with both recorders attached, then plot what they saw. Only the training loop runs inside the `with` blocks, so the evaluation afterwards is not recorded.

# %%
set_seed(0)
mlp = make_mlp()
opt = torch.optim.SGD(mlp.parameters(), lr=0.3)
losses = []
with ActivationRecorder(mlp) as acts, GradientRecorder(mlp) as grads:
    for step in range(2000):
        opt.zero_grad()
        loss = F.cross_entropy(mlp(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
print("records per layer:", {name: len(r) for name, r in acts.records.items()}, {name: len(r) for name, r in grads.records.items()})
print("hooks left after the with block:", sum(len(m._forward_hooks) + len(m._backward_hooks) for m in mlp.modules()))
print(f"final loss {losses[-1]:.4f}, training accuracy {(predict(mlp, X) == y).float().mean().item():.3f}")

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
steps = torch.arange(len(losses))
ax = axes[0, 0]
ax.plot(steps, losses, color=COLORS[0])
ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel("cross-entropy loss")
ax.set_title("Training loss")

for k, name in enumerate(acts.records):
    axes[0, 1].plot(steps, acts.series(name, "frac_zero"), color=COLORS[k], label=f"{name} (after fc{k + 1})")
    axes[1, 0].plot(steps, acts.series(name, "mean"), color=COLORS[k], label=name)
axes[0, 1].set_xlabel("training step")
axes[0, 1].set_ylabel("fraction of activations exactly 0")
axes[0, 1].set_title("Fraction of zeros in each ReLU layer's output")
axes[0, 1].legend(fontsize=8)
axes[1, 0].set_xlabel("training step")
axes[1, 0].set_ylabel("mean activation")
axes[1, 0].set_title("Mean of each ReLU layer's output")
axes[1, 0].legend(fontsize=8)

for k, name in enumerate(grads.records):
    axes[1, 1].plot(steps, grads.series(name), color=COLORS[k], label=f"∂L/∂A at {name}")
axes[1, 1].set_yscale("log")
axes[1, 1].set_xlabel("training step")
axes[1, 1].set_ylabel("$\\|\\partial L / \\partial A^{(\\ell)}\\|$")
axes[1, 1].set_title("Upstream gradient norm at each linear layer's output")
axes[1, 1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# What the recorders show:
#
# - **Loss (top left).** It falls from about $\ln 3$ to below 0.1 (final 0.083, training accuracy 0.972), but not smoothly: from about step 400 on there are repeated **spikes**, some above the starting loss. With plain gradient descent at $\eta = 0.3$ this network is close to the edge of stability; learning rates are the subject of notebook 19.
# - **Fraction of zeros (top right).** At initialization about half of each layer's ReLU outputs are exactly 0 (0.55, 0.48 and 0.50), as expected when pre-activations are roughly symmetric around 0. During training the fraction rises to about 0.6 in every layer (0.62, 0.64, 0.59 at the end): units become more selective, each active for only part of the plane.
# - **Mean activation (bottom left).** The deeper layers start with smaller activations (about 0.18, 0.08, 0.04 for layers 1–3: the signal shrinks with depth under the default initialization, a theme of notebook 18) and grow during training until layer 3 has the largest.
# - **Gradient norms (bottom right).** At the start $\|\partial L/\partial A^{(\ell)}\|$ is smallest at `fc1` and largest at `out`: the gradient shrinks as it travels backward. Later the order reverses. Every spike in the loss shows up at the same step in **every** statistic of every layer, which tells us a spike is a whole-network event, not one misbehaving layer.
#
# None of this required touching the model's code: the same recorders can be attached to any `nn.Module`.

# %%
fig, ax = plt.subplots(figsize=(5.5, 5))
plot_decision_boundary(lambda G: predict(mlp, G), X, y, ax=ax, title="Spirals: the 3-hidden-layer MLP after 2000 steps")
plt.show()

# %% [markdown]
# The decision regions follow the arms. The recorders did not change the training at all: hooks that only observe leave the computation untouched. Notebooks 16 and 18 build similar recorders to study activation functions and initialization across depth.

# %% [markdown]
# ### Checkpoint 6.1
#
# Use forward hooks to collect the **output shape** of every `nn.Linear` in `mlp` during one forward pass on `X[:5]`. Store the shapes as tuples, in call order, in the list `linear_shapes`, and make sure **no hooks remain** on `mlp` afterwards.

# %% [markdown] tags=["hint"]
# Loop over `mlp.modules()`, and for each `nn.Linear` keep the handle of `m.register_forward_hook(lambda mod, inp, out: linear_shapes.append(tuple(out.shape)))`. After the forward pass, call `remove()` on every handle.

# %%
# BEGIN SOLUTION
linear_shapes = []
_handles = [m.register_forward_hook(lambda mod, inp, out: linear_shapes.append(tuple(out.shape)))
            for m in mlp.modules() if isinstance(m, nn.Linear)]
with torch.no_grad():
    mlp(X[:5])
for _h in _handles:
    _h.remove()
# END SOLUTION
print(linear_shapes)

# %% tags=["check"]
check(linear_shapes == [(5, 64), (5, 64), (5, 64), (5, 3)],
      f"linear_shapes = {linear_shapes}, expected [(5, 64), (5, 64), (5, 64), (5, 3)]")
_left = sum(len(m._forward_hooks) for m in mlp.modules())
check(_left == 0, f"{_left} forward hook(s) are still attached to mlp")
passed("Checkpoint 6.1")

# %% [markdown]
# ## 7. Initialization and editing parameters
#
# ### `apply`: visit every module
#
# `model.apply(fn)` calls `fn(m)` on every submodule and finally on the model itself, children before parents, and returns the model. It is the standard way to re-initialize all layers of one type.

# %%
visited = []
make_mlp((2, 4, 3)).apply(lambda m: visited.append(type(m).__name__))
print("apply visits:", visited)

# %% [markdown]
# The root `Sequential` comes last. A common use is to replace the default initialization. `nn.Linear(n_in, n_out)` initializes both weight and bias uniformly in $[-1/\sqrt{n_\text{in}},\, 1/\sqrt{n_\text{in}}]$ (stated from the PyTorch source; checked numerically below). Notebook 18 derives better choices; here we only practise the mechanics, initializing all weights from $\mathcal N(0, 0.01^2)$ and all biases to 0 with the functions in `nn.init`, which modify tensors in place without recording gradients.

# %%
def init_small(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.zeros_(m.bias)


set_seed(0)
fresh_mlp = make_mlp()
bounds_ok = all(m.weight.abs().max() <= 1 / math.sqrt(m.in_features) for m in fresh_mlp.modules() if isinstance(m, nn.Linear))
print("default init within ±1/sqrt(fan_in) for every layer:", bounds_ok)
with torch.no_grad():
    print(f"initial loss, default init: {F.cross_entropy(fresh_mlp(X), y).item():.4f}")
    fresh_mlp.apply(init_small)
    print(f"initial loss, init_small:   {F.cross_entropy(fresh_mlp(X), y).item():.4f}   (ln 3 = {math.log(3):.4f})")

# %% [markdown]
# With tiny weights the logits are all near 0, so the initial loss is almost exactly $\ln K$ — the sanity check from notebook 13. (Such small weights in a deep network make the signal shrink layer after layer; notebook 18 measures this.)
#
# ### Editing parameters in place
#
# A parameter is a leaf tensor with `requires_grad=True`, and autograd forbids in-place changes to such tensors while gradients are being tracked, because the change would not be part of any graph (notebook 05). Edits therefore go inside `torch.no_grad()`, exactly like the update step in our raw-tensor training loop:

# %%
layer = nn.Linear(2, 3)
try:
    layer.weight.mul_(2.0)
except RuntimeError as error:
    print("RuntimeError:", error)
with torch.no_grad():
    layer.weight.mul_(2.0)
    layer.bias.fill_(0.0)
print("after the no_grad edit, bias =", layer.bias.tolist(), "| still a Parameter:", isinstance(layer.bias, nn.Parameter))

# %% [markdown]
# Note the difference between **editing** and **replacing**: `layer.bias.fill_(0)` changes the registered tensor, so an optimizer already holding it sees the change. `layer.bias = nn.Parameter(torch.zeros(3))` registers a *new* tensor, and an optimizer created earlier would keep updating the old one.
#
# ### Freezing parameters
#
# `p.requires_grad_(False)` **freezes** a parameter: autograd computes no gradient for it, so its `.grad` stays `None`, and `torch.optim` optimizers skip parameters whose `.grad` is `None`. This is how a trained part of a model is kept fixed while another part is trained (fine-tuning, Chapter 9). Freeze the hidden layers of the trained `mlp`, re-initialize its output layer, and retrain only that layer:

# %%
for name, p in mlp.named_parameters():
    if not name.startswith("out."):
        p.requires_grad_(False)
n_trainable = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
print(f"trainable parameters: {n_trainable} of {sum(p.numel() for p in mlp.parameters())}")

frozen_before = {name: p.clone() for name, p in mlp.named_parameters() if not p.requires_grad}
set_seed(1)
mlp.out.reset_parameters()                                         # nn.Linear's default initialization again
print(f"accuracy with a re-initialized output layer: {(predict(mlp, X) == y).float().mean().item():.3f}")
opt = torch.optim.SGD(mlp.parameters(), lr=0.3)                    # given ALL parameters, on purpose
for _ in range(300):
    opt.zero_grad()
    F.cross_entropy(mlp(X), y).backward()
    opt.step()
print(f"accuracy after retraining only the output layer: {(predict(mlp, X) == y).float().mean().item():.3f}")
print("frozen parameters unchanged:", all(torch.equal(p, frozen_before[n]) for n, p in mlp.named_parameters() if not p.requires_grad))
print("their .grad:", {n: p.grad for n, p in mlp.named_parameters() if not p.requires_grad and n.startswith("fc1")})

# %% [markdown]
# The frozen layers did not move by a single bit even though the optimizer was given every parameter, and retraining only the 195 output-layer parameters on top of the frozen, already-trained features recovers the accuracy quickly. The cleaner habit is to give the optimizer only the trainable parameters, `[p for p in model.parameters() if p.requires_grad]`.

# %% [markdown]
# ### Checkpoint 7.1
#
# Write `freeze_except(model, prefix)` that freezes every parameter whose name does **not** start with `prefix` and makes every parameter whose name does start with `prefix` trainable. It should return the number of trainable values (elements, not tensors).

# %% [markdown] tags=["hint"]
# Loop over `model.named_parameters()` and call `p.requires_grad_(name.startswith(prefix))`. Then sum `p.numel()` over the trainable ones.

# %%
def freeze_except(model, prefix):
    # BEGIN SOLUTION
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(prefix))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # END SOLUTION

# %% tags=["check"]
_m = make_mlp()
check(freeze_except(_m, "fc3.") == 64 * 65, "freeze_except(make_mlp(), 'fc3.') should return 4160")
check(not _m.fc1.weight.requires_grad and _m.fc3.weight.requires_grad and _m.fc3.bias.requires_grad, "only fc3 should be trainable")
check(freeze_except(_m, "") == 8707, "prefix '' should make every parameter trainable again (8707 values)")
check(all(p.requires_grad for p in _m.parameters()), "every parameter should be trainable after freeze_except(model, '')")
passed("Checkpoint 7.1")

# %% [markdown]
# ## 8. Dtype and device: one call for the whole tree
#
# `model.to(dtype)` and `model.to(device)` convert or move every registered parameter **and buffer**, recursively. Unlike `tensor.to`, which returns a new tensor, `module.to` modifies the module in place and returns the same object (so `model = model.to(device)` and `model.to(device)` are equivalent). Converting the dtype only affects floating-point tensors: our integer counter stays `int64`. And a plain-tensor attribute, being invisible to the module, is not converted at all.

# %%
cm = CenteredMLP()
cm.plain = torch.zeros(2)                                    # not registered
returned = cm.to(torch.float64)
print("module.to returns the same object:", returned is cm)
print("out.weight:", cm.out.weight.dtype, "| center.running_mean:", cm.center.running_mean.dtype,
      "| num_batches_tracked:", cm.center.num_batches_tracked.dtype, "| plain attribute:", cm.plain.dtype)
print("float64 forward pass works:", cm(X_shift.double()).dtype)

# %% [markdown]
# Converting a whole model to `float64` is how you would run `torch.autograd.gradcheck` on it (notebooks 05 and 14). The device version works the same way. On a GPU machine, the cell below moves the trained `CenteredMLP` from Section 4, checks where its parameters and buffers live, and compares predictions. (Apple's MPS backend does not support `float64`; keep double-precision work on the CPU.)

# %% tags=["gpu"]
if device.type == "cpu":
    print("No GPU selected; skipping. (Set up Colab with a GPU runtime to try this.)")
else:
    model_cpu = CenteredMLP()
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()
    model_dev = CenteredMLP().eval()
    model_dev.load_state_dict(model.state_dict())
    model_dev.to(device)
    print("parameter devices:", {p.device.type for p in model_dev.parameters()},
          "| buffer devices:", {b.device.type for b in model_dev.buffers()})
    with torch.no_grad():
        logits_cpu = model_cpu(X_shift)
        logits_dev = model_dev(X_shift.to(device)).cpu()
    print(f"max |CPU logits − {device.type.upper()} logits| = {(logits_cpu - logits_dev).abs().max().item():.2e}")
    print("same predicted classes:", torch.equal(logits_cpu.argmax(1), logits_dev.argmax(1)))
    try:
        model_dev(X_shift)                                    # input still on the CPU
    except RuntimeError as error:
        print("RuntimeError:", str(error)[:150])

# %% [markdown]
# On an accelerator the logits agree with the CPU's to within floating-point rounding (different kernels add in different orders, notebook 00), and the predictions match. Feeding a CPU tensor to a model on another device is an error: input and model must share a device.

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1 (modify): record a different statistic
#
# The cell below trains the same 3-hidden-layer MLP as Section 6 with an `ActivationRecorder` whose `stats` dictionary decides what is recorded, and plots every recorded statistic for every layer. It currently records only the fraction of zeros.
#
# 1. Add two statistics: `"max"`, the largest activation in the batch, and `"rms"`, the root-mean-square activation $\sqrt{\operatorname{mean}(a^2)}$. Run the cell.
# 2. How do the maximum and RMS activations of the three layers change during training, and how do the layers compare? Is the largest activation a good summary of a typical activation?
# 3. Replace `layer_types=(nn.ReLU,)` with `layer_types=(nn.Linear,)`, so the statistics describe the pre-activations $A^{(\ell)}$ (including the logits). What does `"frac_zero"` show now, and why?
#
# Answer in the markdown cell below.

# %% [markdown] tags=["hint"]
# Each statistic is a function of the detached output tensor `a` returning a Python float: `lambda a: a.max().item()` and `lambda a: a.pow(2).mean().sqrt().item()`. For part 3: how likely is a pre-activation computed from floating-point inputs to be exactly 0?

# %%
stats_ex1 = {
    "frac_zero": lambda a: (a == 0).float().mean().item(),
}
layer_types_ex1 = (nn.ReLU,)

set_seed(0)
mlp_ex1 = make_mlp()
opt_ex1 = torch.optim.SGD(mlp_ex1.parameters(), lr=0.3)
with ActivationRecorder(mlp_ex1, layer_types=layer_types_ex1, stats=stats_ex1) as rec_ex1:
    for _ in range(2000):
        opt_ex1.zero_grad()
        F.cross_entropy(mlp_ex1(X), y).backward()
        opt_ex1.step()

fig, axes = plt.subplots(1, len(stats_ex1), figsize=(5.5 * len(stats_ex1), 4), squeeze=False)
for ax, stat in zip(axes[0], stats_ex1):
    for k, name in enumerate(rec_ex1.records):
        values = rec_ex1.series(name, stat)
        ax.plot(values, color=COLORS[k], label=name)
        print(f"{stat:9s} {name:5s}: step 0 {values[0].item():8.3f}   step 1999 {values[-1].item():8.3f}")
    ax.set_xlabel("training step")
    ax.set_ylabel(stat)
    ax.set_title(f"{stat} per layer during training")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# *Your answer:*

# %% [markdown] tags=["solution"]
# With `stats_ex1` extended by `"max": lambda a: a.max().item()` and `"rms": lambda a: a.pow(2).mean().sqrt().item()`:
#
# **Maximum and RMS of the ReLU outputs.** Both grow during training, and they grow most in the deeper layers. The RMS goes from 0.33, 0.14, 0.07 (layers 1–3) at step 0 to 0.34, 0.54, 0.67 at step 1999: layer 1 hardly changes, layer 3 grows almost tenfold. The maximum goes from 1.4, 0.6, 0.35 to 2.6, 4.8, 6.6, and it is very **spiky**: it jumps at the loss spikes (layer 3 reaches 12.8 at one point). So the maximum is about ten times the RMS and is dominated by a few extreme examples and units; it is a good alarm for blow-ups but a poor summary of a typical activation. The RMS (or the mean) describes the typical scale; the fraction of zeros describes how many units are active.
#
# **Pre-activations (`layer_types=(nn.Linear,)`).** `"frac_zero"` is now exactly 0 for every layer and every step: a pre-activation $A = HW^\top + \mathbf b$ computed in floating point from continuous inputs is essentially never exactly 0, while the ReLU maps every negative value to exactly 0. The zeros we counted before are produced by the ReLU, not by the linear layer. The `"max"` of `fc1`–`fc3` equals that of `act1`–`act3` (the ReLU leaves positive values unchanged), whereas the RMS is larger before the ReLU (e.g. 0.82 versus 0.34 for layer 1 at the end) because the negative half still counts. The logits (`out`) grow fastest of all, to an RMS of about 4.8 and a maximum above 18: the network becomes confident, which is what drives the cross-entropy down.

# %% [markdown]
# ### Exercise 2 (implement): a `Standardize` module
#
# Notebook 06 showed that standardizing features (subtracting the mean, dividing by the standard deviation) makes gradient descent much better conditioned. Package this as a module whose statistics are **buffers**, so they are saved and moved with the model, but never trained:
#
# - `Standardize(num_features, eps=1e-8)`: buffers `mean` (initialized to zeros) and `std` (initialized to ones), both of shape `(num_features,)`, and no parameters.
# - `fit(X)`: set `mean` and `std` to the per-feature mean and standard deviation of `X` (shape `(N, num_features)`), using `X.std(dim=0)` with its default correction. It must not build an autograd graph, must update the existing buffers (not create new attributes), and must return `self`.
# - `forward(x)`: return `(x - mean) / (std + eps)`.

# %% [markdown] tags=["hint"]
# In `__init__`: `self.register_buffer("mean", torch.zeros(num_features))`, likewise `std`, and store `self.eps`. In `fit`: inside `with torch.no_grad():`, `self.mean.copy_(X.mean(dim=0))` and `self.std.copy_(X.std(dim=0))`. Copying in place keeps the buffers registered (assigning `self.mean = ...` with a tensor would also work, because the name is already a buffer).

# %%
class Standardize(nn.Module):
    def __init__(self, num_features, eps=1e-8):
        super().__init__()
        # BEGIN SOLUTION
        self.eps = eps
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("std", torch.ones(num_features))
        # END SOLUTION

    def fit(self, X):
        # BEGIN SOLUTION
        with torch.no_grad():
            self.mean.copy_(X.mean(dim=0))
            self.std.copy_(X.std(dim=0))
        return self
        # END SOLUTION

    def forward(self, x):
        # BEGIN SOLUTION
        return (x - self.mean) / (self.std + self.eps)
        # END SOLUTION

# %% tags=["check"]
_s = Standardize(2)
check(len(list(_s.parameters())) == 0, "Standardize should have no parameters")
check(sorted(_s.state_dict().keys()) == ["mean", "std"], f"state_dict keys {sorted(_s.state_dict().keys())}, expected ['mean', 'std']")
check_close(_s(X_shift), X_shift, "output before fit (mean 0, std 1)")
_Xg = X_shift.clone().requires_grad_()
check(_s.fit(_Xg) is _s, "fit should return self")
check(not _s.mean.requires_grad and not _s.std.requires_grad, "fit should not build a graph: the buffers require grad")
check(sorted(n for n, _ in _s.named_buffers()) == ["mean", "std"], "mean and std should still be registered buffers after fit")
_Z = _s(X_shift)
check_close(_Z.mean(dim=0), torch.zeros(2), "mean of standardized data", atol=1e-5)
check_close(_Z.std(dim=0), torch.ones(2), "std of standardized data", atol=1e-5)
_s2 = Standardize(2)
_s2.load_state_dict(_s.state_dict())
check(torch.equal(_s2(X_shift), _Z), "a fresh Standardize loaded from the state_dict should give identical outputs")
_s2.to(torch.float64)
check(_s2.mean.dtype == torch.float64, "the buffers should follow .to(torch.float64)")
passed("Exercise 2")

# %% [markdown]
# ### Exercise 3 (debug): a layer that never learns
#
# `SpiralClassifier` should be a 2 → 64 → 64 → 3 ReLU network. The buggy version (in the comments) runs and trains without any error, yet calling it twice on the same input gives different outputs and its loss never moves away from about $\ln 3$. Find the bug and write the corrected class.

# %% [markdown] tags=["hint"]
# Count the parameters the buggy model registers and compare with `count_mlp_params([2, 64, 64, 3])`. Where is the output layer created, and how often?

# %%
# Buggy version:
#     class SpiralClassifier(nn.Module):
#         def __init__(self, n_hidden=64, n_classes=3):
#             super().__init__()
#             self.n_classes = n_classes
#             self.body = nn.Sequential(nn.Linear(2, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU())
#
#         def forward(self, x):
#             head = nn.Linear(self.body[2].out_features, self.n_classes)
#             return head(self.body(x))
# BEGIN SOLUTION
class SpiralClassifier(nn.Module):
    def __init__(self, n_hidden=64, n_classes=3):
        super().__init__()
        self.n_classes = n_classes
        self.body = nn.Sequential(nn.Linear(2, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU())
        self.head = nn.Linear(n_hidden, n_classes)        # created once, registered, trained

    def forward(self, x):
        return self.head(self.body(x))
# END SOLUTION

# %% tags=["check"]
set_seed(0)
_clf = SpiralClassifier()
_n = sum(p.numel() for p in _clf.parameters())
check(_n == 4547, f"the model registers {_n} parameters; a 2 → 64 → 64 → 3 MLP has 4547")
with torch.no_grad():
    check(torch.equal(_clf(X), _clf(X)), "two calls on the same input should give the same output")
_final = train(_clf, X, y, steps=2000, lr=0.3)
_acc = (predict(_clf, X) == y).float().mean().item()
check(_acc > 0.9, f"training accuracy {_acc:.3f} after 2000 steps; expected above 0.9")
passed("Exercise 3")

# %% [markdown]
# For comparison, here is the buggy version trained in exactly the same way:

# %%
class BuggySpiralClassifier(nn.Module):
    def __init__(self, n_hidden=64, n_classes=3):
        super().__init__()
        self.n_classes = n_classes
        self.body = nn.Sequential(nn.Linear(2, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU())

    def forward(self, x):
        head = nn.Linear(self.body[2].out_features, self.n_classes)
        return head(self.body(x))


set_seed(0)
buggy = BuggySpiralClassifier()
print("registered parameters:", sum(p.numel() for p in buggy.parameters()))
with torch.no_grad():
    print("two calls on the same input agree:", torch.equal(buggy(X[:5]), buggy(X[:5])))
with torch.no_grad():
    print(f"loss before training: {F.cross_entropy(buggy(X), y).item():.4f}")
print(f"loss after 2000 steps: {train(buggy, X, y, steps=2000, lr=0.3):.4f}")
print(f"training accuracy: {(predict(buggy, X) == y).float().mean().item():.3f}")

# %% [markdown] tags=["solution"]
# **The bug: the output layer is created inside `forward`.** Every call builds a brand-new `nn.Linear(64, 3)` with fresh random weights. It is a local variable, never assigned to `self`, so `__setattr__` never sees it: it is not registered, not in `parameters()` (4352 registered values instead of 4547 — exactly the missing $3 \cdot 65 = 195$), not trained by the optimizer, not saved by `state_dict()`, and not moved by `.to()`. Its weights are re-drawn on every call, so the same input gives different outputs, and the body is trained toward a different random head at every step. The result: the loss stays at about $\ln 3$ (1.0989 before, 1.1006 after 2000 steps) and the accuracy, 0.318, is chance level for three classes.
#
# **Fix:** create the layer once, in `__init__`, as an attribute (`self.head = nn.Linear(n_hidden, n_classes)`), and only *use* it in `forward`. The rule: `forward` computes; `__init__` creates everything that holds state. The corrected model registers 4547 parameters, is deterministic, and trains to above 90% accuracy.

# %% [markdown]
# ### Exercise 4 (investigate): is evaluation mode the average of training mode?
#
# Section 5 showed that for a single `nn.Dropout`, the average of many training-mode outputs approaches the evaluation-mode output. Does this still hold for a network? The scaffold builds two small networks with the same weights except for where dropout sits:
#
# - **A**: `Linear → ReLU → Linear → ReLU → Dropout → Linear` (dropout only directly before the last, linear layer);
# - **B**: `Linear → ReLU → Dropout → Linear → ReLU → Linear` (dropout earlier, followed by another nonlinearity).
#
# For 200 inputs it compares the evaluation-mode output with the average of $S$ training-mode outputs, for several $S$, and reports the largest absolute gap. Run it and answer in the markdown cell:
#
# 1. For each network, does the gap shrink toward 0 as $S$ grows? At what rate?
# 2. Explain the difference between A and B. (Hint: what is $E[W(\mathbf m \odot \mathbf h)/(1-p)]$ for a random 0/1 mask $\mathbf m$, and what is $E[\operatorname{relu}(\cdot)]$ of a random quantity?)
# 3. Optionally, change `p` and see how the gap for B depends on it. Or try **C**: `Linear → Dropout → ReLU → Linear → ReLU → Linear`, dropout *before* the first ReLU — and explain what you find.

# %% [markdown] tags=["hint"]
# Averaging $S$ independent random outputs has a Monte Carlo error that falls like $1/\sqrt S$ (notebook 04). Expectation passes through linear maps, $E[W\mathbf v] = W E[\mathbf v]$, but not through nonlinear ones: in general $E[\operatorname{relu}(v)] \neq \operatorname{relu}(E[v])$.

# %%
p = 0.5
set_seed(0)
lin1, lin2, lin3 = nn.Linear(2, 64), nn.Linear(64, 64), nn.Linear(64, 3)    # shared by both networks
net_A = nn.Sequential(lin1, nn.ReLU(), lin2, nn.ReLU(), nn.Dropout(p), lin3)
net_B = nn.Sequential(lin1, nn.ReLU(), nn.Dropout(p), lin2, nn.ReLU(), lin3)
X_ex4 = X[torch.randperm(N)[:200]]                               # (200, 2)

S_values = [10, 100, 1000, 10_000]
gaps = {"A": [], "B": []}
with torch.no_grad():
    for label, net in [("A", net_A), ("B", net_B)]:
        net.eval()
        out_eval = net(X_ex4)                                    # (200, 3)
        net.train()
        set_seed(1)
        total, done = torch.zeros_like(out_eval), 0
        for S in S_values:
            while done < S:
                total += net(X_ex4)
                done += 1
            gaps[label].append((total / S - out_eval).abs().max().item())
        print(f"net {label}: typical |eval output| {out_eval.abs().mean().item():.3f}; max gap for S = "
              + ", ".join(f"{S}: {g:.4f}" for S, g in zip(S_values, gaps[label])))

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for k, label in enumerate(gaps):
    ax.loglog(S_values, gaps[label], "o-", color=COLORS[k], label=f"net {label}")
ax.loglog(S_values, [gaps["A"][0] * math.sqrt(S_values[0] / S) for S in S_values], ":", color="gray", label="$\\propto 1/\\sqrt{S}$")
ax.set_xlabel("number of training-mode passes averaged, $S$")
ax.set_ylabel("max |average − eval output|")
ax.set_title(f"Training-mode average versus eval mode (p = {p})")
ax.legend()
plt.show()

# %% [markdown]
# *Your conclusion:*

# %% [markdown] tags=["solution"]
# **Network A: yes, at the Monte Carlo rate.** The largest gap falls from 0.083 ($S = 10$) to 0.029, 0.0097 and 0.0035 ($S = 10^4$): a factor of about $\sqrt{10} \approx 3.2$ per tenfold increase in $S$, the $1/\sqrt S$ line of the plot. The average of the training-mode outputs converges to the evaluation-mode output.
#
# **Network B: no.** Its gap goes 0.131, 0.093, 0.083, 0.082 — it levels off at about 0.08, almost as large as the typical output magnitude (0.094). More samples only make the average converge, accurately, to a **different** value.
#
# **Why.** Let $\mathbf h$ be the input to the dropout and $\mathbf m$ the random mask, with each $m_j = 1$ with probability $1 - p$. Dropout outputs $\mathbf m \odot \mathbf h/(1-p)$, whose expectation is $\mathbf h$, the eval-mode output. In A, everything after the dropout is the linear map $\mathbf v \mapsto W\mathbf v + \mathbf b$, and expectation passes through linear maps: $E[W(\mathbf m \odot \mathbf h)/(1-p) + \mathbf b] = W\mathbf h + \mathbf b$. So eval mode computes *exactly* the average of training mode. In B, the dropout is followed by `Linear → ReLU`, and $E[\operatorname{relu}(V)] \ne \operatorname{relu}(E[V])$ for a random $V$ (in fact $E[\operatorname{relu}(V)] \ge \operatorname{relu}(E[V])$ by Jensen's inequality, since relu is convex). Evaluation mode is then only an **approximation** to the average over masks — a cheap one, a single pass instead of thousands. Notebook 23 returns to this when it explains dropout.
#
# **Optional parts.** With smaller $p$ the plateau for B shrinks (in our runs about 0.032 for $p = 0.2$ and 0.017 for $p = 0.1$), because the masks perturb $\mathbf h$ less. Network **C** (dropout *before* the first ReLU) gives exactly the same numbers as B: for $m_j \in \{0, 1\}$ and $c = 1/(1-p) > 0$, $\operatorname{relu}(m_j c\, a_j) = m_j c\, \operatorname{relu}(a_j)$, so dropout and ReLU commute, and C computes the same function as B for every mask.

# %% [markdown]
# ## Summary and what's next
#
# - A module is a class with `__init__` (call `super().__init__()` first) and `forward`. Call the module, `model(x)`, so that `__call__` runs hooks around `forward`.
# - **Registration** happens in `__setattr__`: an `nn.Parameter` goes into `_parameters`, an `nn.Module` into `_modules`, anything else into the plain `__dict__`. `parameters()`, `named_parameters()`, `children()` and `modules()` walk the tree. A plain tensor or a Python list of layers is **not** registered: not trained, not saved, not moved. `nn.ModuleList`, `nn.ModuleDict` and `nn.Sequential` register their contents.
# - **Buffers** (`register_buffer`) hold non-trainable state such as running statistics: saved in the `state_dict` (unless `persistent=False`) and moved by `.to()`, but never returned by `parameters()` and given no gradients.
# - The **`state_dict`** maps dotted names to tensors — parameters and persistent buffers, not code or plain attributes. Its tensors share memory with the model (clone for a snapshot). `torch.save`/`torch.load(weights_only=True)` round-trip it exactly; strict loading reports missing keys, unexpected keys and shape mismatches by name.
# - **`train()`/`eval()`** set the `training` flag on the whole subtree. It changes how mode-dependent layers compute (our `RunningCenter`; dropout; BatchNorm) and does not disable gradients. Predicting in training mode can silently give wrong answers and corrupt buffers.
# - **Hooks** observe (or modify) outputs and gradients without touching `forward`: forward hooks see `(module, inputs, output)`, full backward hooks see `grad_input` and `grad_output = ∂L/∂output`, tensor hooks see a tensor's gradient. Keep the handles and remove them. With them we recorded per-layer activation and gradient statistics throughout training.
# - `apply` re-initializes every layer; in-place parameter edits go inside `torch.no_grad()`; `requires_grad_(False)` freezes parameters, which optimizers then leave unchanged; `module.to(dtype or device)` converts every parameter and buffer in place.
#
# We now have the tools to see inside a network. Notebook 16 uses them on the piece of the MLP we have taken for granted so far, the **activation function**: sigmoid, tanh, ReLU and its relatives, their derivatives, how they saturate, and how many ReLU units die during training — measured with hooks like the ones built here.

# Authoring guide

How course notebooks are written, built, and validated. Read [LESSON_PLAN.md](LESSON_PLAN.md) first for what each notebook covers.

## Workflow

1. Edit the source in `src/<chapter>/<notebook>.py`. Never edit the generated `.ipynb` files.
2. Build: `uv run python scripts/build_notebooks.py`. This writes the student notebook to `notebooks/` and the solution notebook to `solutions/`, executes each from a fresh kernel, and saves the outputs. Only notebooks whose cells changed are re-executed; `--force` re-executes everything.
3. Validate: `uv run python scripts/validate_notebooks.py <chapter>`. Both variants run from a fresh kernel; results go to `VALIDATION.md`.
4. Commit the source, both generated notebooks, and `VALIDATION.md` together.

Generated notebooks are committed **with outputs**, so figures are visible without running anything. In student notebooks, the outputs of `exercise` and `check` cells are cleared. Outputs reflect the machine that built them (the setup cell prints its device); rebuild with `DLCOURSE_DEVICE=cpu` to produce CPU-only outputs.

## Source format

Sources use the Jupytext "percent" format:

```python
# %% [markdown]
# ## A heading
# Markdown lines start with "# ". A line containing only "#" is a blank line.

# %%
x = torch.arange(6)  # a code cell
```

Cell headers can carry tags: `# %% [markdown] tags=["hint"]`.

| Markup | Effect |
| --- | --- |
| `# BEGIN SOLUTION` … `# END SOLUTION` in a code cell | Student notebook replaces the lines with `# YOUR CODE HERE` / `raise NotImplementedError`, at the same indentation. Cell is tagged `exercise`. |
| `tags=["check"]` on a code cell | Automated exercise check. May fail in the student notebook until the exercise is done. |
| `tags=["hint"]` on a markdown cell | Rendered as a collapsed hint. |
| `tags=["solution"]` on a markdown cell | Worked explanation; appears only in the solution notebook. |
| `tags=["gpu"]` on a code cell | GPU scale-up cell. Must skip cleanly on a machine without a GPU. |

## Notebook structure

Every teaching notebook follows this order:

1. **Title cell:** `# NN — Title`, then a short table with *Prerequisites*, *You will learn*, *Estimated time*, and *Compute* (CPU core; GPU optional).
2. **Setup cell:** the standard setup cell below, unchanged.
3. **Motivating problem:** a concrete question the notebook will answer.
4. **Numbered sections** (`## 1. …`), each: intuition → small numerical example → code → plot where useful → **inline checkpoint**.
5. **[GPU] section** where it adds something (optional; guarded).
6. **Exercises:** `## Exercises`, in the order *modify → implement → debug → investigate*.
7. **Summary and what's next:** what was learned, what remains unsolved, and how the next notebook addresses it.

Chapter problem sets (`chNN_problems.py`) use the same title and setup cells, then a sequence of larger problems.

### Standard setup cell

```python
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
```

Notebooks may add imports after this cell, but never change it.

## Exercises

- **Inline checkpoints** come after each section. Each is small (one to five lines), has a code cell containing `# BEGIN SOLUTION`/`# END SOLUTION`, and a following `check` cell that uses `check`, `check_close`, or `check_shape` from `dlcourse` and ends with `passed("…")`.
- **End-of-notebook exercises** progress through:
  - *Modify:* change a parameter or line of existing code and explain the effect.
  - *Implement:* write a function from scratch; checked against a reference.
  - *Debug:* fix deliberately broken code. The broken version is given in the cell; the fix sits inside solution markers. The check must fail on the broken code.
  - *Investigate:* an open question answered with an experiment and a short written conclusion in a markdown cell. Fade scaffolding across the chapter: at least one investigation requires learner-written sampling/training, aggregation, or plotting code. Put the worked implementation inside solution markers and the worked conclusion in a `solution` markdown cell.
- Each chapter problem set includes an independent mini-capstone: give the task, data contract, criteria, and CPU budget, but leave the experiment loop and plots to the learner. Require predictions before execution, justified variables and controls, multiple seeds where relevant, an independent reference, a failure case, and a report on evidence, variability, limitations, and a follow-up. Include a visible rubric; passing code checks does not grade the report. See `LESSON_PLAN.md` for chapter themes.
- Every coded task's check must call its implementation or consume its result. For debug tasks, demonstrate that the original buggy implementation fails the same check; when a debug task contains several bugs, the check must also fail with each bug on its own. An uncalled function body is not validated by executing its definition.
- Add deterministic edge cases to random tests where applicable: exact thresholds/ties, zero inputs, singleton/non-square shapes, and extreme logits. State boundary ownership and check it against the generator; avoid claiming exact classification from a random sample alone.
- State assumptions behind diagnostics and provide counterexamples or negative controls. Approximately uniform logits imply initial cross-entropy near ln K; random untrained logits need not. Avoid treating a heuristic as proof of a bug.
- Every exercise with code has a `hint` cell placed after the exercise's description and before its solution cell.
- **Teaching cells must never depend on exercise results.** If a later section needs a function from an exercise, define the reference version in a teaching cell instead. The validator enforces this by running the student notebook and allowing failures only in `exercise` and `check` cells.
- In problem sets, an experiment cell may deliberately use the learner's code. Tag it `exercise` explicitly (`# %% tags=["exercise"]`) so the validator treats it as part of the exercise.
- An exercise stub must fail with `NotImplementedError` (from the stub) or `NameError` (a name the stub did not define). Write checks so this holds: call the learner's function or read the learner's variable before asserting anything else.

## Writing style

- Plain language first, then the equation, then the code. Define every symbol.
- State tensor shapes explicitly in text and in code comments, e.g. `# (batch, features)`.
- Prefer small numerical examples a learner can verify by hand before larger ones.
- Say where an intuition or analogy stops being accurate.
- Keep markdown cells short; split long explanations across cells interleaved with code.

## Plots

- Use `matplotlib`, styled by `setup_style()`; colors come from `dlcourse.plotting.COLORS`.
- Label axes, include units or tensor dimensions, and title every plot with what it shows.
- Follow each plot with one or two sentences saying what to notice.
- Every plot must be generated by code in the notebook.

## Compute and data

- Use the shared generators in `dlcourse.data` (`make_blobs`, `make_xor`, `make_rings`, `make_spirals`, `make_checkerboard`) rather than writing new ones, so datasets recur across notebooks. Each takes a `seed`.
- Use `dlcourse.plotting.plot_dataset` and `plot_decision_boundary` for 2-D classification plots.

- The CPU core of each notebook should execute in under 3 minutes on a laptop.
- CPU core sections use generated or committed data only; no downloads.
- GPU cells select the device with `dlcourse.get_device()` and skip with a printed message when `device.type == "cpu"`. They must not fail on CPU-only machines.
- MPS is supported where PyTorch supports the operations; note known gaps where relevant.
- Seed with `set_seed` before any randomness whose result the text describes.

## Validation

`scripts/validate_notebooks.py` must pass for both variants of every notebook before committing. Record the run in `VALIDATION.md`. GPU sections are validated separately on a CUDA machine with `DLCOURSE_DEVICE=cuda`.

# Deep Learning with PyTorch

A gradual, visual, hands-on course: tensors and perceptrons → MLPs → GPU training → CNNs → recurrent sequence models → attention → Transformers.

## Start here

1. Read the [lesson plan](LESSON_PLAN.md) for the 65-notebook curriculum and teaching approach.
2. Set up the environment (below).
3. Open [notebooks/ch01_foundations/00_course_guide.ipynb](notebooks/ch01_foundations/00_course_guide.ipynb).

**Status:** Chapters 1–3 (notebooks 00–20 and their problem sets) are written and validated. Later chapters are in progress.

## Setup

The repository uses [uv](https://docs.astral.sh/uv/) to install pinned versions of Python, PyTorch, and every other package:

```bash
uv sync              # create .venv from uv.lock
uv run jupyter lab   # open the notebooks
```

Core labs run on any CPU. Optional **[GPU]** sections use CUDA (local, Colab, or cloud) or Apple Silicon (MPS) where supported. To use a GPU on Google Colab, select a GPU runtime and run:

```python
!git clone https://github.com/sumilk/deep-learning-with-pytorch.git
%cd deep-learning-with-pytorch/notebooks/ch01_foundations
```

Set `DLCOURSE_DEVICE=cpu`, `cuda`, or `mps` before starting Jupyter to override automatic device selection.

## Learning approach

- Introduce the required math alongside the code; assume basic Python.
- Build intuition using diagrams, plots, small numerical examples, and experiments.
- Implement important components from scratch, then compare with PyTorch.
- Practice throughout each chapter: inline checkpoint exercises, end-of-notebook exercises, and a problem set per chapter, with automated checks.
- Practice with implementation challenges, deliberate bugs, tricky datasets, and failure analysis.
- Keep core labs small enough for CPU execution using locally generated data; scale up on GPU (CUDA, with MPS where supported) in optional sections with real datasets and pretrained models.

## Organization

```text
LESSON_PLAN.md   Curriculum and notebook requirements
AUTHORING.md     How notebooks are written, built, and validated
VALIDATION.md    Latest validation results
notebooks/       Student notebooks (exercises blank), one folder per chapter
solutions/       The same notebooks with worked solutions
src/             Notebook sources; the .ipynb files are generated from these
dlcourse/        Small helper package used by every notebook (devices, seeds, checks, plots)
scripts/         build_notebooks.py and validate_notebooks.py
```

Work in `notebooks/`. Try each exercise before opening its hint, and each hint before looking in `solutions/`.

## Contributing

Edit sources in `src/`, never the generated notebooks. See [AUTHORING.md](AUTHORING.md):

```bash
uv run python scripts/build_notebooks.py
uv run python scripts/validate_notebooks.py
```

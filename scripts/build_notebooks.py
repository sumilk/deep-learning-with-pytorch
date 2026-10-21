"""Build student and solution notebooks from percent-format sources.

Sources live in ``src/<chapter>/<notebook>.py`` and use the Jupytext
"percent" format:

    # %% [markdown]
    # # A heading
    # Markdown lines are commented with "# ".

    # %%
    x = 1  # a code cell

Cell headers may carry tags, e.g. ``# %% [markdown] tags=["hint"]``.

Markup understood by this script:

- ``# BEGIN SOLUTION`` / ``# END SOLUTION`` inside a code cell: the lines
  between them are replaced in the student notebook by
  ``# YOUR CODE HERE`` + ``raise NotImplementedError``. The cell is tagged
  ``exercise``.
- Tag ``hint`` on a markdown cell: rendered as a collapsed hint block.
- Tag ``solution`` on a markdown cell: included only in the solution notebook.
- Tag ``check`` on a code cell: an automated exercise check. In the student
  notebook it is allowed to fail until the exercise is completed.

Outputs:

- ``notebooks/<chapter>/<notebook>.ipynb`` — student version.
- ``solutions/<chapter>/<notebook>.ipynb`` — solution version.

Each notebook is executed from a fresh kernel and saved **with its outputs**,
so figures are visible without running anything. In student notebooks the
outputs of ``exercise`` and ``check`` cells are cleared (they would only show
the unfinished-exercise errors). A notebook is re-executed only when its
cells change, or with ``--force``.

Usage:

    uv run python scripts/build_notebooks.py               # build and execute what changed
    uv run python scripts/build_notebooks.py --force       # re-execute everything
    uv run python scripts/build_notebooks.py --no-execute  # write notebooks without outputs
    uv run python scripts/build_notebooks.py --check       # fail if any notebook's cells are stale
    uv run python scripts/build_notebooks.py --match 01_tensors
"""

import argparse
import json
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
STUDENT_DIR = ROOT / "notebooks"
SOLUTION_DIR = ROOT / "solutions"

HEADER = re.compile(r"^# %%(?P<md> \[markdown\])?(?:\s+tags=(?P<tags>\[.*\]))?\s*$")
BEGIN = re.compile(r"^(?P<indent>\s*)# BEGIN SOLUTION\s*$")
END = re.compile(r"^\s*# END SOLUTION\s*$")


def parse_cells(text, path):
    cells = []
    current = None
    for lineno, line in enumerate(text.splitlines(), 1):
        match = HEADER.match(line)
        if match:
            tags = json.loads(match.group("tags")) if match.group("tags") else []
            current = {"type": "markdown" if match.group("md") else "code", "tags": tags, "lines": []}
            cells.append(current)
            continue
        if current is None:
            if line.strip():
                raise SystemExit(f"{path}:{lineno}: content before the first '# %%' cell header")
            continue
        current["lines"].append(line)
    for cell in cells:
        lines = cell["lines"]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if cell["type"] == "markdown":
            cell["lines"] = [uncomment(line, path) for line in lines]
    return cells


def uncomment(line, path):
    if line.startswith("# "):
        return line[2:]
    if line.strip() in ("", "#"):
        return ""
    raise SystemExit(f"{path}: markdown line must start with '# ': {line!r}")


def strip_solutions(lines, student):
    """Return (lines, had_solution)."""
    out = []
    had_solution = False
    in_solution = False
    for line in lines:
        begin = BEGIN.match(line)
        if begin:
            if in_solution:
                raise ValueError("nested BEGIN SOLUTION")
            in_solution = had_solution = True
            if student:
                indent = begin.group("indent")
                out.append(f"{indent}# YOUR CODE HERE")
                out.append(f"{indent}raise NotImplementedError")
            continue
        if END.match(line):
            if not in_solution:
                raise ValueError("END SOLUTION without BEGIN SOLUTION")
            in_solution = False
            continue
        if in_solution and student:
            continue
        out.append(line)
    if in_solution:
        raise ValueError("BEGIN SOLUTION without END SOLUTION")
    return out, had_solution


def render(cells, student, path):
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    for cell in cells:
        tags = list(cell["tags"])
        if cell["type"] == "markdown":
            if "solution" in tags and student:
                continue
            source = "\n".join(cell["lines"])
            if "hint" in tags:
                source = f"<details>\n<summary><b>💡 Hint</b></summary>\n\n{source}\n\n</details>"
            if "solution" in tags:
                source = f"> **Worked solution.**\n>\n" + "\n".join(f"> {l}" if l else ">" for l in cell["lines"])
            new = nbformat.v4.new_markdown_cell(source)
        else:
            try:
                lines, had_solution = strip_solutions(cell["lines"], student)
            except ValueError as error:
                raise SystemExit(f"{path}: {error} in cell starting {cell['lines'][:1]}")
            if had_solution and "exercise" not in tags:
                tags.append("exercise")
            new = nbformat.v4.new_code_cell("\n".join(lines))
        if tags:
            new.metadata["tags"] = tags
        nb.cells.append(new)
    # Deterministic cell ids keep rebuilds diff-free.
    for index, cell in enumerate(nb.cells):
        cell.id = f"cell-{index:03d}"
    return nb


def without_outputs(nb):
    """Serialized notebook with outputs removed: what --check and change detection compare."""
    nb = nbformat.from_dict(json.loads(nbformat.writes(nb)))
    nb.metadata.pop("language_info", None)
    nb.metadata["language_info"] = {"name": "python"}
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
    return nbformat.writes(nb)


ALLOWED_STUDENT_ERRORS = {"NotImplementedError", "NameError"}


def execute(nb, path, student):
    """Run nb in a fresh kernel from path's folder; clean up outputs for publishing."""
    client = NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=student,
                            resources={"metadata": {"path": str(path.parent)}})
    client.execute()
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        cell.metadata.pop("execution", None)
        tags = set(cell.metadata.get("tags", []))
        if student and tags & {"exercise", "check"}:
            cell.outputs = []
            cell.execution_count = None
            continue
        for output in cell.outputs:
            if output.get("output_type") == "error":
                raise SystemExit(f"{path}: cell {index} raised {output.get('ename')} in a teaching cell")
    return nb


def targets(source):
    relative = source.relative_to(SRC).with_suffix(".ipynb")
    return STUDENT_DIR / relative, SOLUTION_DIR / relative


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit non-zero if any built notebook's cells are out of date")
    parser.add_argument("--match", default="", help="only build sources whose path contains this text")
    parser.add_argument("--force", action="store_true", help="re-execute notebooks even if their cells are unchanged")
    parser.add_argument("--no-execute", action="store_true", help="write notebooks without executing them")
    args = parser.parse_args()

    sources = [s for s in sorted(SRC.glob("*/*.py")) if args.match in str(s.relative_to(SRC))]
    if not sources:
        raise SystemExit(f"no sources found under {SRC}")
    stale, written = [], 0
    for source in sources:
        cells = parse_cells(source.read_text(), source)
        student_path, solution_path = targets(source)
        for path, student in ((student_path, True), (solution_path, False)):
            nb = render(cells, student, source)
            existing = nbformat.read(path, as_version=4) if path.exists() else None
            unchanged = existing is not None and without_outputs(existing) == without_outputs(nb)
            has_outputs = existing is not None and any(c.get("outputs") for c in existing.cells if c.cell_type == "code")
            if args.check:
                if not unchanged:
                    stale.append(path)
                continue
            if unchanged and not args.force and (has_outputs or args.no_execute):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if not args.no_execute:
                print(f"executing {path.relative_to(ROOT)}", flush=True)
                execute(nb, path, student)
            path.write_text(nbformat.writes(nb) + "\n")
            written += 1
    if args.check:
        if stale:
            print("Out-of-date notebooks (run scripts/build_notebooks.py):")
            for path in stale:
                print(f"  {path.relative_to(ROOT)}")
            sys.exit(1)
        print(f"All {2 * len(sources)} notebooks are up to date.")
    else:
        print(f"Built {len(sources)} sources; {written} notebooks written.")


if __name__ == "__main__":
    main()

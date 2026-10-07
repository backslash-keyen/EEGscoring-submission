"""Stitch the 0*.py percent-format scripts into a3_walkthrough.ipynb and execute it so the outputs are saved in the file.
Usage (from anywhere): python task1/a3_walkthrough/make_notebook.py [--no-run]"""
import re, sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

here = Path(__file__).parent


def cells_from(script):
    """Split a '# %%' percent-format script into (kind, source) cells."""
    out = []
    text = script.read_text(encoding="utf-8")
    for m in re.finditer(r"^# %%( \[markdown\])?\n(.*?)(?=^# %%|\Z)", text, flags=re.M | re.S):
        is_md, body = bool(m.group(1)), m.group(2).strip("\n")
        if is_md:
            body = "\n".join(re.sub(r"^# ?", "", ln) for ln in body.splitlines())
        if body.strip():
            out.append(("markdown" if is_md else "code", body))
    return out


nb = nbformat.v4.new_notebook()
nb.cells.append(nbformat.v4.new_markdown_cell(
    "# A3 confound audit: walkthrough\nWhat else, besides motor imagery, separates left from right trials? Four steps: inputs and splits, "
    "classifier and chance threshold, every route, three surprises.\nThe first cell sets the working folder so `common.py` imports; "
    "the figures are also saved in `outputs/a3_walkthrough/`."))
nb.cells.append(nbformat.v4.new_code_cell("import os, sys\nsys.path.insert(0, os.getcwd())\n%matplotlib inline"))
for script in sorted(here.glob("0*.py")):
    nb.cells.append(nbformat.v4.new_markdown_cell(f"---\n## `{script.name}`"))
    for kind, src in cells_from(script):
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "markdown" else nbformat.v4.new_code_cell(src))
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
if "--no-run" not in sys.argv:
    NotebookClient(nb, timeout=900, resources={"metadata": {"path": str(here)}}).execute()
nbformat.write(nb, here / "a3_walkthrough.ipynb")
print("wrote a3_walkthrough.ipynb with", len(nb.cells), "cells", "(executed)" if "--no-run" not in sys.argv else "(not executed)")

"""Saved notebook outputs must be free of warnings and errors.

The notebooks in ``notebooks/`` are documentation. Their saved outputs are what
readers see, so a warning there shows a misuse of CMOR.
The notebooks are not executed here, only their saved outputs are checked.
After editing a notebook, rebuild its outputs with
``tools/rebuild_notebook_output.sh``.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
WARNING_RE = re.compile(r"\b\w*Warning\b:")


def _text(output: dict) -> str:
    text = output.get("text", "")
    if not text:
        text = output.get("data", {}).get("text/plain", "")
    return "".join(text) if isinstance(text, list) else str(text)


def _problems(notebook: Path) -> list[str]:
    problems = []
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    for index, cell in enumerate(cells):
        for output in cell.get("outputs", []):
            kind = output.get("output_type")
            if kind == "error":
                problems.append(f"cell {index}: error {output.get('ename')}")
            elif kind == "stream" and output.get("name") == "stderr":
                problems.append(f"cell {index}: stderr {_text(output)[:200]!r}")
            elif WARNING_RE.search(_text(output)):
                problems.append(f"cell {index}: warning {_text(output)[:200]!r}")
    return problems


def test_notebooks_are_present():
    assert NOTEBOOKS, f"no notebooks found in {NOTEBOOK_DIR}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_saved_outputs_have_no_warnings_or_errors(notebook):
    assert _problems(notebook) == []

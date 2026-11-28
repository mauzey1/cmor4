#!/usr/bin/env python3
"""Check that saved notebook outputs are free of warnings and errors.

The notebooks in ``notebooks/`` are documentation. Their saved outputs are what
readers see, so a warning there shows a misuse of CMOR. This script doesn't run
the notebooks; it reads their saved outputs and reports any cell that has an
error, stderr output, or warning text. After editing a notebook, rebuild its
outputs with ``tools/rebuild_notebook_output.sh``.

Usage:
    tools/check_notebook_outputs.py [NOTEBOOK.ipynb ...]

Without arguments, every notebook in ``notebooks/`` is checked. Exits with
status 1 if any problem is found.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
WARNING_RE = re.compile(r"\b\w*Warning\b:")


def _text(output: dict) -> str:
    text = output.get("text", "")
    if not text:
        text = output.get("data", {}).get("text/plain", "")
    return "".join(text) if isinstance(text, list) else str(text)


def problems(notebook: Path) -> list[str]:
    """Return one message for each output that has an error or a warning."""

    found = []
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    for index, cell in enumerate(cells):
        for output in cell.get("outputs", []):
            kind = output.get("output_type")
            if kind == "error":
                found.append(f"cell {index}: error {output.get('ename')}")
            elif kind == "stream" and output.get("name") == "stderr":
                found.append(f"cell {index}: stderr {_text(output)[:200]!r}")
            elif WARNING_RE.search(_text(output)):
                found.append(f"cell {index}: warning {_text(output)[:200]!r}")
    return found


def main(argv: list[str]) -> int:
    notebooks = [Path(arg) for arg in argv] or sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        print(f"error: no notebooks found in {NOTEBOOK_DIR}", file=sys.stderr)
        return 1

    failed = False
    for notebook in notebooks:
        found = problems(notebook)
        if found:
            failed = True
            print(f"FAIL {notebook}")
            for message in found:
                print(f"  {message}")
        else:
            print(f"ok   {notebook}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

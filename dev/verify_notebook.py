"""Execute the notebook in a throwaway copy and report any cell errors.

    python dev/verify_notebook.py

The committed notebook carries no outputs, which keeps the repository small and its diffs
readable. Verifying that it still runs therefore has to happen somewhere else, so this copies
it to a temporary path, executes that, and inspects the result. The committed file is never
touched.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "large_array_viz.ipynb"


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"{NOTEBOOK.name} is missing, run `pixi run notebook` first")
        return 1

    with tempfile.TemporaryDirectory() as work:
        copy = Path(work) / NOTEBOOK.name
        shutil.copy(NOTEBOOK, copy)
        print(f"executing a copy of {NOTEBOOK.name}")

        result = subprocess.run(
            [
                sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                "--inplace", "--ExecutePreprocessor.timeout=1800", str(copy),
            ],
            cwd=ROOT,  # relative data paths in the notebook resolve against the repo root
            capture_output=True,
            text=True,
        )
        if result.returncode:
            print(result.stderr[-4000:])
            return result.returncode

        notebook = json.loads(copy.read_text())

    errors = [
        (index, output["ename"], output["evalue"])
        for index, cell in enumerate(notebook["cells"])
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    print(f"{len(notebook['cells'])} cells executed, {len(errors)} errors")
    for index, name, value in errors:
        print(f"  cell {index}: {name}: {value[:160]}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

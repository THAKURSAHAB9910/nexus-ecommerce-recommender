"""
Executes cells in notebooks/nexus_walkthrough.ipynb and embeds execution outputs
so the notebook displays full rich outputs when viewed on GitHub.
"""

import sys
import io
import json
import contextlib
from pathlib import Path
import pandas as pd
import numpy as np

# Change directory to notebooks parent during execution so relative paths '../' work
notebook_dir = Path("notebooks").resolve()
root_dir = Path(".").resolve()

nb_path = notebook_dir / "nexus_walkthrough.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Shared execution environment
exec_globals = {
    "__file__": str(nb_path),
    "__name__": "__main__",
}

# Change working dir to notebooks dir so relative paths like '../data/' resolve
import os
os.chdir(str(notebook_dir))

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue

    code = "".join(cell["source"])
    print(f"Executing code cell {idx}...")

    stdout_buf = io.StringIO()
    outputs = []
    
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
            # Parse code to separate last expression if any
            lines = code.strip().split("\n")
            last_line = lines[-1].strip()
            # Only eval the last line if it is at top-level indentation (not inside for/if/def)
            if len(lines) > 0 and not lines[-1].startswith((" ", "\t")) and not last_line.startswith(("print", "import", "from", "with", "for", "if", "def", "class")) and "=" not in last_line:
                exec_code = "\n".join(lines[:-1])
                eval_expr = lines[-1]
                if exec_code.strip():
                    exec(exec_code, exec_globals)
                res = eval(eval_expr, exec_globals)
            else:
                exec(code, exec_globals)
                res = None

        captured = stdout_buf.getvalue()
        if captured:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": captured.splitlines(keepends=True)
            })

        if res is not None:
            if isinstance(res, pd.DataFrame):
                outputs.append({
                    "data": {
                        "text/html": [res.to_html(classes="dataframe", index=False)],
                        "text/plain": [res.to_string()]
                    },
                    "execution_count": cell.get("execution_count", 1),
                    "metadata": {},
                    "output_type": "execute_result"
                })
            else:
                outputs.append({
                    "data": {
                        "text/plain": [str(res)]
                    },
                    "execution_count": cell.get("execution_count", 1),
                    "metadata": {},
                    "output_type": "execute_result"
                })

    except Exception as e:
        print(f"Error in cell {idx}: {e}")
        outputs.append({
            "ename": type(e).__name__,
            "evalue": str(e),
            "output_type": "error",
            "traceback": [str(e)]
        })

    cell["outputs"] = outputs

os.chdir(str(root_dir))

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook outputs populated successfully!")

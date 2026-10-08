"""Builds the Google Colab notebook from `colab_runner.py`, so the notebook and the tested code can never differ.

    python -m nocodeml_engine.colab.notebook ../web/public/nocodeml-colab.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUNNER = Path(__file__).with_name("colab_runner.py")

INTRO = """# NoCodeML: train on Google Colab

Your data was prepared on the NoCodeML website. This notebook trains the models you chose and gives back their
**predictions**. NoCodeML then works out every score, chart and check itself.

1. **Run all** (menu *Runtime → Run all*). When asked, choose the `nocodeml_bundle_….zip` you downloaded from NoCodeML.
2. Wait for training to finish (the log shows each model).
3. Your browser downloads **`nocodeml_results.json`**. Upload that file back in NoCodeML (Training → *Train on Google Colab*).

Nothing here needs a login, and nothing is sent anywhere: the file stays in your browser until you upload it yourself."""

UPLOAD = '''# 1) Choose the bundle you downloaded from NoCodeML
import os, zipfile
try:
    from google.colab import files
    uploaded = files.upload()
    bundle_zip = next(iter(uploaded))
except ImportError:                      # running somewhere other than Colab
    bundle_zip = input("Path to the bundle .zip: ")
with zipfile.ZipFile(bundle_zip) as z:
    z.extractall("bundle")
print("Bundle ready:", sorted(os.listdir("bundle")))'''

RUN = '''# 3) Train every model and save the predictions
results = run("bundle")
path = save(results)
print("Saved", path, f"({os.path.getsize(path) / 1e6:.1f} MB)")
try:
    files.download(path)                 # the browser downloads it; upload it in NoCodeML
except NameError:
    print("Upload this file in NoCodeML.")'''


def _cell(kind: str, source: str, n: int) -> dict:
    cell = {"cell_type": kind, "id": f"nocodeml{n}", "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def build_notebook() -> dict:
    cells = [
        _cell("markdown", INTRO, 0),
        _cell("code", UPLOAD, 1),
        _cell("code", "# 2) The training code. Plain pandas and scikit-learn: read it, it is short.\n" + RUNNER.read_text().rstrip("\n"), 2),
        _cell("code", RUN, 3),
    ]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"colab": {"name": "NoCodeML training", "provenance": []},
                         "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


def render() -> str:
    return json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "nocodeml-colab.ipynb")
    out.write_text(render())
    print("wrote", out)

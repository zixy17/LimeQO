"""Export the plan-equivalence classes of a workload from its EXPLAIN-plan zip.

Hints that produce the same plan for a query form one class (the plan's
`hint_list`). Dataset needs only these classes for every strategy except
LimeQO+, so they are shipped in `dataset/{name}-hint-classes.npy` and the
zip is optional.

Usage (from the repository root, with dataset/{name}.zip present):
    python src/data/export_hint_classes.py ceb job stack dsb
"""
import json
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd

NUM_HINTS = 49


def export(name):
    filenames = pd.read_csv(f"dataset/{name}-matrix.csv", index_col="filename").index
    row_of = {fn: i for i, fn in enumerate(filenames)}
    # classes[i, j] is the smallest hint that gives the same plan as hint j for query i
    classes = np.full((len(filenames), NUM_HINTS), -1, dtype=np.int8)
    with ZipFile(f"dataset/{name}.zip", "r") as f:
        for fp in f.namelist():
            if not fp.endswith(".json") or "MACOSX" in fp:
                continue
            with f.open(fp) as pfd:
                plan = json.load(pfd)
            hints = sorted(plan["hint_list"])
            row = row_of[plan["filename"]]
            assert (classes[row, hints] == -1).all(), f"{name}: hint in two plans"
            classes[row, hints] = hints[0]
    assert (classes >= 0).all(), f"{name}: some (query, hint) pairs have no plan"
    np.save(f"dataset/{name}-hint-classes.npy", classes)
    plans = sum(len(np.unique(row)) for row in classes)
    print(f"{name}: {len(classes)} queries, {plans} plans")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        export(name)

"""Dataset must work without the EXPLAIN-plan zips (they are needed only by LimeQO+).

Run from the repository root: python -m pytest tests
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from data.dataset import Dataset  # noqa: E402
from strategies.limeqo_plus import LimeQOPlusStrategy  # noqa: E402

WORKLOADS = ["ceb", "job", "stack", "dsb"]


def link_dataset_without_zips(workdir):
    (workdir / "dataset").mkdir()
    for fn in os.listdir(os.path.join(REPO, "dataset")):
        if not fn.endswith(".zip"):
            os.symlink(os.path.join(REPO, "dataset", fn), workdir / "dataset" / fn)


@pytest.fixture
def no_zips(tmp_path, monkeypatch):
    """A working directory with the committed dataset files and no plan zips."""
    link_dataset_without_zips(tmp_path)
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("name", WORKLOADS)
def test_loads_without_plan_zip(no_zips, name):
    ds = Dataset(name)
    assert not ds.has_plans
    n, m = ds.matrix.shape
    for i in range(n):
        for j in range(m):
            same = ds.get_same_hints(i, j)
            assert j in same
            # hints with the same plan have the same measured latency
            assert np.all(ds.matrix[i, same] == ds.matrix[i, j])


def test_limeqo_plus_needs_plans(no_zips):
    ds = Dataset("job")
    with pytest.raises(FileNotFoundError, match="job.zip"):
        LimeQOPlusStrategy().run(ds, "unused.json")


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "dataset", "job.zip")),
                    reason="dataset/job.zip not downloaded")
def test_classes_match_plan_zip(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO)
    with_zip = Dataset("job")
    link_dataset_without_zips(tmp_path)
    monkeypatch.chdir(tmp_path)
    without_zip = Dataset("job")
    n, m = with_zip.matrix.shape
    for i in range(n):
        for j in range(m):
            assert with_zip.get_same_hints(i, j) == without_zip.get_same_hints(i, j)

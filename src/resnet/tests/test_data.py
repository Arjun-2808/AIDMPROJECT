import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aidm_project.data import cifar_split, make_step_imbalance


def test_cifar_split_is_stratified_and_disjoint():
    labels = np.repeat(np.arange(10), 600)
    train, val = cifar_split(labels, n_val_per_class=50)
    assert len(val) == 500 and len(train) == 5500
    assert np.bincount(labels[val]).tolist() == [50] * 10
    assert not set(train) & set(val)


def test_step_imbalance_keeps_two_percent_of_minority_classes():
    labels = np.repeat(np.arange(10), 500)
    kept = make_step_imbalance(np.arange(len(labels)), labels, [3, 4, 9], 0.02)
    counts = np.bincount(labels[kept], minlength=10)
    assert counts[[3, 4, 9]].tolist() == [10, 10, 10]
    assert counts[[0, 1, 2, 5, 6, 7, 8]].tolist() == [500] * 7

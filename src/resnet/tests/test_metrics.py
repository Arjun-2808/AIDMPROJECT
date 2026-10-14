import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aidm_project.metrics import classification_metrics


def test_binary_metrics():
    result = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0])
    assert result["accuracy"] == 0.75
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]

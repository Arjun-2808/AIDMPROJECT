from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(targets, predictions, average: str = "binary") -> dict[str, float | list[list[int]]]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, average=average, zero_division=0)),
        "recall": float(recall_score(targets, predictions, average=average, zero_division=0)),
        "f1": float(f1_score(targets, predictions, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
    }

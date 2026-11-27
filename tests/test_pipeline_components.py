"""Unit tests for evaluation metrics and inference predictor."""

import numpy as np

from src.training.evaluate_all import _eval_row
from src.training.train_classical_mlp import compute_metrics


def test_compute_metrics_binary_perfect() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.05, 0.95]])
    m = compute_metrics(y_true, y_pred, y_probs, is_binary=True)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["roc_auc"] == 1.0


def test_eval_row_multiclass_and_binary() -> None:
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 0])
    row = _eval_row("TestModel", "38-Class", y_true, y_pred)
    assert row["model"] == "TestModel"
    assert row["accuracy"] == 0.75
    assert np.isnan(row["roc_auc"])

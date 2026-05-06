"""
Unit tests for evaluation metrics (src/models/evaluation.py)
Run with: pytest tests/test_evaluation.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from models.evaluation import (
    precision_at_k,
    calculate_precision_at_k,
    precision_at_k_multi,
    top_k_accuracy,
    f1_macro,
    EvaluationMetrics,
)


# ── precision_at_k ────────────────────────────────────────────────────────────

def test_precision_at_k_perfect():
    """All top-K are in ground truth → 1.0."""
    pred = ["a", "b", "c"]
    gt = ["a", "b", "c", "d"]
    assert precision_at_k(pred, gt, k=3) == pytest.approx(1.0)


def test_precision_at_k_zero():
    """None of top-K in ground truth → 0.0."""
    pred = ["x", "y", "z"]
    gt = ["a", "b", "c"]
    assert precision_at_k(pred, gt, k=3) == pytest.approx(0.0)


def test_precision_at_k_partial():
    """a and c correct out of top-3 → 2/3."""
    pred = ["a", "b", "c", "d", "e"]
    gt = ["a", "c", "f", "g"]
    assert precision_at_k(pred, gt, k=3) == pytest.approx(2 / 3)


def test_precision_at_k_case_insensitive():
    """Matching must be case-insensitive."""
    pred = ["Software", "Cloud"]
    gt = ["software", "cloud"]
    assert precision_at_k(pred, gt, k=2) == pytest.approx(1.0)


def test_precision_at_k_k_zero():
    """k=0 must return 0.0 without error."""
    assert precision_at_k(["a"], ["a"], k=0) == pytest.approx(0.0)


def test_precision_at_k_k_larger_than_list():
    """k > len(extracted) should not crash; computes against top-k only."""
    pred = ["a", "b"]
    gt = ["a", "b", "c"]
    # k=5 → top_5 = ["a","b"], hits=2, result=2/5
    assert precision_at_k(pred, gt, k=5) == pytest.approx(2 / 5)


def test_calculate_precision_at_k_alias():
    """calculate_precision_at_k is an alias and must behave identically."""
    pred = ["a", "b", "c", "d", "e"]
    gt = ["a", "c", "f", "g"]
    assert calculate_precision_at_k(pred, gt, k=3) == pytest.approx(2 / 3)


# ── precision_at_k_multi ─────────────────────────────────────────────────────

def test_precision_at_k_multi_returns_dict():
    pred = ["a", "b", "c", "d", "e"]
    gt = ["a", "c"]
    result = precision_at_k_multi(pred, gt, k_values=[1, 3, 5])
    assert set(result.keys()) == {1, 3, 5}
    assert result[1] == pytest.approx(1.0)   # "a" correct at k=1
    assert result[3] == pytest.approx(2 / 3)  # a,c correct at k=3


# ── top_k_accuracy ────────────────────────────────────────────────────────────

def test_top1_accuracy_perfect():
    y_true = ["J", "G", "Q"]
    y_pred_top3 = [["J", "M", "K"], ["G", "J", "C"], ["Q", "P", "R"]]
    assert top_k_accuracy(y_true, y_pred_top3, k=1) == pytest.approx(1.0)


def test_top3_accuracy_partial():
    y_true = ["J", "G", "Q"]
    # J in top-3, G not in top-3, Q in top-3 → 2/3
    y_pred_top3 = [["M", "J", "K"], ["C", "P", "R"], ["P", "R", "Q"]]
    assert top_k_accuracy(y_true, y_pred_top3, k=3) == pytest.approx(2 / 3)


def test_top_k_accuracy_empty():
    assert top_k_accuracy([], [], k=1) == pytest.approx(0.0)


# ── f1_macro ──────────────────────────────────────────────────────────────────

def test_f1_macro_perfect():
    y_true = ["J", "G", "Q"]
    y_pred = ["J", "G", "Q"]
    assert f1_macro(y_true, y_pred) == pytest.approx(1.0)


def test_f1_macro_all_wrong():
    y_true = ["J", "J", "J"]
    y_pred = ["G", "G", "G"]
    # No true positives → F1 = 0
    assert f1_macro(y_true, y_pred) == pytest.approx(0.0)


# ── EvaluationMetrics ─────────────────────────────────────────────────────────

def test_evaluation_metrics_empty():
    ev = EvaluationMetrics(k_values=[5, 10])
    report = ev.compute()
    assert "error" in report


def test_evaluation_metrics_basic():
    ev = EvaluationMetrics(k_values=[3])
    ev.add(
        extracted=["a", "b", "c"],
        ground_truth=["a", "c"],
        true_sector="J",
        predicted_sector="J",
        predicted_top3=["J", "M", "K"],
    )
    report = ev.compute()
    assert report["n_documents"] == 1
    assert report["precision_at_3"] == pytest.approx(2 / 3)
    assert report["top1_accuracy"] == pytest.approx(1.0)
    assert report["top3_accuracy"] == pytest.approx(1.0)


def test_evaluation_metrics_multiple_docs():
    ev = EvaluationMetrics(k_values=[2])
    ev.add(["a", "b"], ["a", "b"], "J", "J", ["J"])
    ev.add(["x", "y"], ["a", "b"], "G", "G", ["G"])
    report = ev.compute()
    assert report["n_documents"] == 2
    assert report["precision_at_2"] == pytest.approx(0.5)  # 1.0 + 0.0 / 2


def test_evaluation_metrics_no_sector_labels():
    """Adding docs without sector labels must not crash."""
    ev = EvaluationMetrics(k_values=[5])
    ev.add(["a", "b", "c", "d", "e"], ["a", "c", "e"])
    report = ev.compute()
    assert "precision_at_5" in report
    assert "top1_accuracy" not in report
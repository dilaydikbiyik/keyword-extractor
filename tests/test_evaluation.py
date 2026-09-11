"""Unit tests for the evaluation metrics in experiments/metrics.py."""

import pytest

from experiments.metrics import (
    cohen_kappa,
    f1_macro,
    precision_at_k,
    top_k_accuracy,
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


# ── precision_at_k_multi ─────────────────────────────────────────────────────


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


def test_cohen_kappa_is_none_when_undefined():
    """One label category across both raters leaves kappa undefined, not zero."""
    assert cohen_kappa(["J", "J"], ["J", "J"]) is None

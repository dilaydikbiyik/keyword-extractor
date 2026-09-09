"""Evaluation metrics for the comparison tables.

Wraps the metrics already in ``src/models/evaluation.py`` and adds the two
things a reviewer asks for that the shipped pipeline does not need:
bootstrap confidence intervals and a paired significance test against the
baseline a result claims to beat.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.evaluation import (  # noqa: E402  (path set above)
    cohen_kappa,
    f1_macro,
    precision_at_k,
    top_k_accuracy,
)

from experiments.config import N_BOOTSTRAP, SEED  # noqa: E402


def bootstrap_ci(
    correct: Sequence[bool],
    n_resamples: int = N_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = SEED,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for a proportion.

    With 30 evaluation documents the interval is wide enough that reporting
    the point estimate alone would overstate what the sample can support.
    """
    arr = np.asarray(correct, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[draws].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Dict:
    """Exact (binomial) McNemar test between two systems on the same items.

    Returns the discordant counts and a two-sided p-value.  Exact rather than
    chi-square because the discordant pairs are few at this sample size.
    """
    from scipy.stats import binomtest

    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    n01 = int(np.sum(a & ~b))  # a right, b wrong
    n10 = int(np.sum(~a & b))  # a wrong, b right
    n = n01 + n10
    p = 1.0 if n == 0 else float(binomtest(n01, n, 0.5).pvalue)
    return {"a_only_correct": n01, "b_only_correct": n10, "discordant": n, "p_value": p}


def evaluate_sector_predictions(
    y_true: Sequence[str],
    y_pred_ranked: Sequence[Sequence[str]],
) -> Dict:
    """Top-1/Top-3 accuracy, F1-macro, Kappa, plus a CI on Top-1."""
    y_true = list(y_true)
    y_pred_ranked = [list(p) for p in y_pred_ranked]
    y_pred_top1 = [p[0] if p else "" for p in y_pred_ranked]
    correct_top1 = [t == p for t, p in zip(y_true, y_pred_top1)]

    lo, hi = bootstrap_ci(correct_top1)
    return {
        "n": len(y_true),
        "top1_accuracy": top_k_accuracy(y_true, y_pred_ranked, k=1),
        "top1_ci95": [lo, hi],
        "top3_accuracy": top_k_accuracy(y_true, y_pred_ranked, k=3),
        "f1_macro": f1_macro(y_true, y_pred_top1),
        "cohen_kappa": cohen_kappa(y_true, y_pred_top1),
        "correct_top1": correct_top1,
    }


def evaluate_keywords(
    extracted: Sequence[Sequence[str]],
    ground_truth: Sequence[Sequence[str]],
    k_values: Sequence[int] = (3, 5, 10),
) -> Dict:
    """Mean Precision@K over documents that have keyword annotations."""
    pairs = [(e, g) for e, g in zip(extracted, ground_truth) if g]
    if not pairs:
        return {f"precision_at_{k}": None for k in k_values}
    out: Dict[str, float] = {"n_with_keyword_labels": len(pairs)}
    for k in k_values:
        out[f"precision_at_{k}"] = float(
            np.mean([precision_at_k(list(e), list(g), k) for e, g in pairs])
        )
    return out


def confusion_pairs(y_true: Sequence[str], y_pred: Sequence[str]) -> List[Tuple[str, str, int]]:
    """Most frequent (true, predicted) confusions, errors only."""
    from collections import Counter

    counts = Counter(
        (t, p) for t, p in zip(y_true, y_pred) if t != p
    )
    return [(t, p, c) for (t, p), c in counts.most_common()]

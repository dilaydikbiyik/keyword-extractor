"""
Evaluation Metrics for Keyword Extraction Pipeline

Implements:
  - Precision@K (exact match)
  - Semantic Match Score (embedding-based, partial credit)
  - Sector Classification Accuracy — Top-1, Top-3, F1-Macro, Cohen's Kappa
  - Batch evaluation helpers
"""

from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Keyword evaluation
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(
    extracted: List[str],
    ground_truth: List[str],
    k: int = 10,
) -> float:
    """
    Calculate Precision@K: fraction of top-K extracted keywords that appear
    in the ground-truth set.

    Args:
        extracted: Ranked list of extracted keywords (best first)
        ground_truth: Reference set of correct keywords
        k: Cut-off rank

    Returns:
        Precision score in [0, 1]
    """
    if k <= 0:
        return 0.0
    top_k = extracted[:k]
    gt_set = set(kw.lower().strip() for kw in ground_truth)
    hits = sum(1 for kw in top_k if kw.lower().strip() in gt_set)
    return hits / k


def calculate_precision_at_k(
    predicted: List[str],
    ground_truth: List[str],
    k: int = 10,
) -> float:
    """Alias for precision_at_k (matches import in test files)."""
    return precision_at_k(predicted, ground_truth, k)


def precision_at_k_multi(
    extracted: List[str],
    ground_truth: List[str],
    k_values: List[int] = (5, 10, 20),
) -> Dict[int, float]:
    """
    Calculate Precision@K for multiple K values at once.

    Args:
        extracted: Ranked list of extracted keywords
        ground_truth: Reference keyword set
        k_values: List of K cut-offs to evaluate

    Returns:
        Dict mapping each K to its precision score
    """
    return {k: precision_at_k(extracted, ground_truth, k) for k in k_values}


def semantic_match_score(
    extracted_keywords: List[str],
    gold_standard_keywords: List[str],
    model,
) -> float:
    """
    Embedding-based evaluation: gives partial credit for semantically close
    but not exactly matching keywords.

    For each extracted keyword, finds its highest cosine similarity to any
    gold-standard keyword.  Returns the mean of those max similarities.

    Args:
        extracted_keywords: List of extracted keyword strings
        gold_standard_keywords: List of reference keyword strings
        model: Any object with an ``encode(texts) -> np.ndarray`` method
               (e.g. a SentenceTransformer instance or EmbeddingService)

    Returns:
        Mean max cosine similarity in [0, 1]
    """
    if not extracted_keywords or not gold_standard_keywords:
        return 0.0

    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

    # Encode both sets
    extracted_embs = _encode(model, extracted_keywords)   # (E, D)
    gold_embs = _encode(model, gold_standard_keywords)    # (G, D)

    # Cosine similarity matrix  (E, G)
    sim_matrix = _cos_sim(extracted_embs, gold_embs)

    # For each extracted keyword, max similarity over all gold keywords
    max_sims = sim_matrix.max(axis=1)
    return float(max_sims.mean())


def _encode(model, texts: List[str]) -> np.ndarray:
    """Helper: encode texts using either encode() or embed_texts()."""
    if hasattr(model, "encode"):
        result = model.encode(texts, convert_to_numpy=True)
    elif hasattr(model, "embed_texts"):
        result = np.vstack(model.embed_texts(texts))
    else:
        raise TypeError(f"model must have encode() or embed_texts(), got {type(model)}")
    return np.atleast_2d(result)


# ─────────────────────────────────────────────────────────────────────────────
# Sector classification evaluation
# ─────────────────────────────────────────────────────────────────────────────

def top_k_accuracy(
    y_true: List[str],
    y_pred_top_k: List[List[str]],
    k: int = 1,
) -> float:
    """
    Calculate Top-K accuracy for sector classification.

    Args:
        y_true: Ground-truth sector codes, one per document
        y_pred_top_k: Ranked predictions per document (best first)
        k: Consider a prediction correct if the true label is in top-K

    Returns:
        Accuracy in [0, 1]
    """
    if not y_true:
        return 0.0
    hits = sum(
        1 for true, preds in zip(y_true, y_pred_top_k)
        if true in preds[:k]
    )
    return hits / len(y_true)


def f1_macro(
    y_true: List[str],
    y_pred: List[str],
) -> float:
    """
    Macro-averaged F1 score for sector classification.

    Args:
        y_true: Ground-truth sector codes
        y_pred: Predicted sector codes (top-1)

    Returns:
        F1-macro score in [0, 1]
    """
    from sklearn.metrics import f1_score  # type: ignore
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def cohen_kappa(
    y_true: List[str],
    y_pred: List[str],
) -> float:
    """
    Cohen's Kappa for inter-annotator or classifier agreement.

    Args:
        y_true: Ground-truth labels
        y_pred: Predicted labels

    Returns:
        Kappa score in [-1, 1]
    """
    from sklearn.metrics import cohen_kappa_score  # type: ignore
    return float(cohen_kappa_score(y_true, y_pred))


# ─────────────────────────────────────────────────────────────────────────────
# Batch / convenience helpers
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationMetrics:
    """
    Convenience wrapper that accumulates results across many documents and
    computes aggregate metrics in one call.

    Usage::

        ev = EvaluationMetrics(k_values=[5, 10])
        ev.add(extracted=["bulut", "yazılım"], ground_truth=["yazılım", "api"],
               true_sector="J", predicted_sector="J", predicted_top3=["J","M","K"])
        report = ev.compute()
    """

    def __init__(self, k_values: List[int] = (5, 10, 20)):
        self.k_values = list(k_values)
        self._precision_sums: Dict[int, float] = defaultdict(float)
        self._count = 0
        self._y_true: List[str] = []
        self._y_pred: List[str] = []
        self._y_pred_top3: List[List[str]] = []

    def add(
        self,
        extracted: List[str],
        ground_truth: List[str],
        true_sector: Optional[str] = None,
        predicted_sector: Optional[str] = None,
        predicted_top3: Optional[List[str]] = None,
    ):
        """
        Add one document's results.

        Args:
            extracted: Ranked extracted keywords
            ground_truth: Reference keywords
            true_sector: Correct sector code (optional)
            predicted_sector: Top-1 predicted sector (optional)
            predicted_top3: Top-3 predicted sectors (optional)
        """
        for k in self.k_values:
            self._precision_sums[k] += precision_at_k(extracted, ground_truth, k)

        if true_sector is not None and predicted_sector is not None:
            self._y_true.append(true_sector)
            self._y_pred.append(predicted_sector)
            self._y_pred_top3.append(predicted_top3 or [predicted_sector])

        self._count += 1

    def compute(self) -> Dict:
        """
        Compute all aggregate metrics.

        Returns:
            Dict with Precision@K values, Top-1/3 accuracy, F1-macro, Kappa
        """
        if self._count == 0:
            return {"error": "No documents added"}

        report: Dict = {
            "n_documents": self._count,
        }

        # Precision@K averages
        for k in self.k_values:
            report[f"precision_at_{k}"] = self._precision_sums[k] / self._count

        # Classification metrics (only if labels were provided)
        if self._y_true:
            report["top1_accuracy"] = top_k_accuracy(
                self._y_true, self._y_pred_top3, k=1
            )
            report["top3_accuracy"] = top_k_accuracy(
                self._y_true, self._y_pred_top3, k=3
            )
            try:
                report["f1_macro"] = f1_macro(self._y_true, self._y_pred)
            except Exception:
                report["f1_macro"] = None
            try:
                report["cohen_kappa"] = cohen_kappa(self._y_true, self._y_pred)
            except Exception:
                report["cohen_kappa"] = None

        return report
"""Loaders for the evaluation set, the raw corpus and the taxonomy."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from experiments.config import CORPUS_CSV, LABELS_JSON, SECTORS_JSON


@dataclass
class LabeledSample:
    """One annotated Handelsregister record."""

    id: int
    legal_name: str
    purpose: str
    true_sector: str
    keywords_ground_truth: List[str] = field(default_factory=list)
    annotation_method: str = "unknown"


def load_labeled_samples() -> List[LabeledSample]:
    """Load the human-labelled evaluation set."""
    with open(LABELS_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)
    return [
        LabeledSample(
            id=item["id"],
            legal_name=item.get("legal_name", ""),
            purpose=item["purpose"],
            true_sector=item["true_sector"],
            keywords_ground_truth=item.get("keywords_ground_truth", []),
            annotation_method=item.get("annotation_method", "unknown"),
        )
        for item in payload["samples"]
    ]


def load_labels_metadata() -> Dict:
    with open(LABELS_JSON, encoding="utf-8") as fh:
        return json.load(fh)["metadata"]


def load_corpus(limit: int | None = None) -> List[str]:
    """Load the unlabelled business-purpose texts.

    The corpus is what the unsupervised baselines are allowed to fit on
    (TF-IDF statistics); no label ever touches it.
    """
    df = pd.read_csv(CORPUS_CSV)
    texts = [t for t in df["purpose"].fillna("").tolist() if t.strip()]
    return texts[:limit] if limit else texts


def load_corpus_frame() -> pd.DataFrame:
    return pd.read_csv(CORPUS_CSV)


def load_taxonomy() -> Dict[str, dict]:
    with open(SECTORS_JSON, encoding="utf-8") as fh:
        return json.load(fh)["sectors"]


def sector_names() -> Dict[str, str]:
    return {code: info.get("name", code) for code, info in load_taxonomy().items()}

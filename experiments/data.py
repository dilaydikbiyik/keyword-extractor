"""Loaders for the evaluation set, the raw corpus and the taxonomy."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    # "corpus_sample" = drawn from the raw corpus; "authored" = written or
    # edited by hand rather than sampled. The distinction decides whether the
    # measured accuracy transfers to the corpus at all.
    provenance: str = "authored"


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
            provenance=item.get("provenance", "authored"),
        )
        for item in payload["samples"]
    ]


def load_split() -> Dict[str, List[int]]:
    """Development / held-out test document ids, if a split has been made."""
    path = LABELS_JSON.parent / "split.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"dev": payload["dev"], "test": payload["test"]}


def load_labeled_samples_split(half: Optional[str] = None) -> List[LabeledSample]:
    """Load the evaluation set, or one half of it.

    ``half`` is "dev", "test", or None for everything.
    """
    samples = load_labeled_samples()
    if half is None:
        return samples
    split = load_split()
    if not split:
        raise FileNotFoundError("No split yet — run `python -m experiments.split`.")
    wanted = set(split[half])
    return [s for s in samples if s.id in wanted]


def load_labels_metadata() -> Dict:
    with open(LABELS_JSON, encoding="utf-8") as fh:
        return json.load(fh)["metadata"]


def corpus_available() -> bool:
    """Whether the raw corpus is present in this checkout."""
    return CORPUS_CSV.exists()


def load_corpus(limit: int | None = None) -> List[str]:
    """Load the unlabelled business-purpose texts.

    The corpus is what the unsupervised baselines are allowed to fit on
    (TF-IDF statistics); no label ever touches it.  Returns an empty list when
    the corpus is not in this checkout — the baselines then fall back to the
    committed corpus statistics.
    """
    if not CORPUS_CSV.exists():
        return []
    df = pd.read_csv(CORPUS_CSV)
    texts = [t for t in df["purpose"].fillna("").tolist() if t.strip()]
    return texts[:limit] if limit else texts


def load_corpus_frame() -> pd.DataFrame:
    return pd.read_csv(CORPUS_CSV)


def load_taxonomy(path=None) -> Dict[str, dict]:
    with open(path or SECTORS_JSON, encoding="utf-8") as fh:
        return json.load(fh)["sectors"]


def sector_names() -> Dict[str, str]:
    return {code: info.get("name", code) for code, info in load_taxonomy().items()}

"""Error taxonomy for misclassified documents.

The manual pass is the part reviewers actually read, so the automatic flags
here deliberately stop at what a rule can decide (length, decision margin,
boilerplate shape).  Everything judgemental is left to a ``manual_category``
column, filled in against the codebook below.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

# The codebook the manual pass annotates against.  Keep these stable once
# annotation starts — changing a definition mid-pass invalidates the counts.
CODEBOOK: Dict[str, str] = {
    "ambiguous_sector_definition": (
        "The NACE section boundary itself is unclear for this business; two "
        "sections are defensible."
    ),
    "multi_sector_company": (
        "The company genuinely operates in several sections; the gold label "
        "picks one."
    ),
    "taxonomy_granularity": (
        "The correct activity sits in a NACE division whose parent section is "
        "counterintuitive (e.g. repair, holdings)."
    ),
    "short_text": "Too little text to determine the sector.",
    "boilerplate_only": (
        "The purpose field is legal boilerplate (holding, asset management, "
        "participation) with no activity signal."
    ),
    "seed_gap": "The correct sector's seed list lacks the vocabulary in this text.",
    "seed_leakage": (
        "A wrong sector's seed list contains a term that dominates this text."
    ),
    "annotation_error": "The gold label is wrong; the prediction is defensible.",
}

_BOILERPLATE = re.compile(
    r"\b(beteiligung|holding|verwaltung eigenen verm|erwerb und verwaltung|"
    r"halten und verwalten|geschäftsführung|komplementär)",
    re.IGNORECASE,
)


def auto_flags(
    text: str,
    scores: Sequence[float],
    true_sector: str,
    top3: Sequence[str],
    short_chars: int = 50,
    margin_threshold: float = 0.02,
    band: float = 0.03,
) -> List[str]:
    """Rule-decidable properties of one error."""
    flags: List[str] = []
    stripped = text.strip()
    if len(stripped) < short_chars:
        flags.append("short_text")
    if len(scores) >= 2 and (scores[0] - scores[1]) < margin_threshold:
        flags.append("low_margin")
    if len(scores) >= 3 and (scores[0] - scores[2]) < band:
        flags.append("crowded_top3")
    if _BOILERPLATE.search(stripped):
        flags.append("boilerplate_shape")
    if true_sector in list(top3)[:3]:
        flags.append("recoverable_in_top3")
    else:
        flags.append("outside_top3")
    return flags


def build_error_records(predictions: Sequence[Dict], samples_by_id: Dict[int, object]) -> List[Dict]:
    """One record per misclassified document, ready for manual coding."""
    records = []
    for pred in predictions:
        if pred["true"] == pred["predicted"]:
            continue
        sample = samples_by_id[pred["id"]]
        text = sample.purpose
        scores = pred.get("scores") or []
        records.append(
            {
                "id": pred["id"],
                "legal_name": sample.legal_name,
                "purpose": text,
                "text_length": len(text.strip()),
                "true_sector": pred["true"],
                "predicted_sector": pred["predicted"],
                "top3": pred["top3"],
                "score_top1": scores[0] if scores else None,
                "margin_top1_top2": (scores[0] - scores[1]) if len(scores) >= 2 else None,
                "keywords": pred.get("keywords", []),
                "auto_flags": auto_flags(text, scores, pred["true"], pred["top3"]),
                "manual_category": "",  # filled in by hand, values from CODEBOOK
                "manual_note": "",
            }
        )
    return records

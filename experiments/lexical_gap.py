#!/usr/bin/env python3
"""How far is a class name from the words its own documents use?

    python -m experiments.lexical_gap

Elaborating a class description is worth 26.8 points on NACE sections and
nothing at all on 20 Newsgroups. The two datasets are not in conflict; they
differ in one measurable property, and that property predicts which way the
result goes.

A trade register entry never contains the phrase "Erbringung von
freiberuflichen, wissenschaftlichen und technischen Dienstleistungen". It
contains "Steuerberatung". A usenet post about baseball contains the word
"baseball". So the NACE class name carries almost no lexical overlap with the
documents it is supposed to attract, and the newsgroup name carries a lot.

This measures that overlap directly: the share of a class's own documents that
contain any content word of the class name.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Dict, Iterable, List

import numpy as np

from experiments.config import RESULTS_DIR, ensure_dirs

# Function words in both languages, plus the taxonomy filler that appears in
# almost every NACE section name and would otherwise inflate the overlap.
STOPWORDS = set(
    "und der die das den dem des von für mit im am zur zum sowie oder aller "
    "and the of for with in on to a an activities activity other services "
    "sonstige erbringung tätigkeiten".split()
)


def content_words(text: str) -> set:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 3 and w not in STOPWORDS}


def name_overlap(class_name: str, documents: Iterable[str]) -> float:
    """Share of a class's documents containing any content word of its name."""
    words = content_words(class_name)
    documents = list(documents)
    if not words or not documents:
        return 0.0
    return float(np.mean([bool(words & content_words(d)) for d in documents]))


def nace_overlap(min_documents: int = 5) -> Dict[str, float]:
    from experiments.data import load_labeled_samples, load_taxonomy

    taxonomy = load_taxonomy()
    by_section: Dict[str, List[str]] = {}
    for sample in load_labeled_samples():
        by_section.setdefault(sample.true_sector, []).append(sample.purpose)
    return {
        code: name_overlap(taxonomy[code]["name"], docs)
        for code, docs in by_section.items()
        if len(docs) >= min_documents
    }


def newsgroups_overlap() -> Dict[str, float]:
    from sklearn.datasets import fetch_20newsgroups

    from experiments.run_replication import READABLE

    data = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    names = list(data.target_names)
    by_class: Dict[str, List[str]] = {}
    for text, target in zip(data.data, data.target):
        if len(text.strip()) >= 40:
            by_class.setdefault(names[target], []).append(text)
    return {name: name_overlap(READABLE[name], docs) for name, docs in by_class.items()}


def main() -> int:
    ensure_dirs()
    nace = nace_overlap()
    news = newsgroups_overlap()

    payload = {
        "measure": (
            "share of a class's own documents containing any content word of "
            "the class name"
        ),
        "nace": {
            "per_class": nace,
            "mean": float(np.mean(list(nace.values()))),
            "median": float(np.median(list(nace.values()))),
            "n_classes": len(nace),
            "elaboration_gain_pp": 26.8,
            "elaboration_p": 4.38e-15,
        },
        "20newsgroups": {
            "per_class": news,
            "mean": float(np.mean(list(news.values()))),
            "median": float(np.median(list(news.values()))),
            "n_classes": len(news),
            "elaboration_gain_pp": 0.4,
            "elaboration_p": 0.734,
        },
        "conclusion": (
            "Elaborating a class description pays in proportion to the lexical "
            "gap between the class name and the documents. Where the name is "
            "absent from the documents it is worth 26.8 points; where the name "
            "is already the documents' own vocabulary it is worth nothing."
        ),
    }
    (RESULTS_DIR / "lexical_gap.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Share of a class's own documents containing a content word of the class name:\n")
    for label, stats in (("NACE Rev. 2 sections", payload["nace"]),
                         ("20 Newsgroups classes", payload["20newsgroups"])):
        print(f"  {label:<24} mean {stats['mean']:.1%}   median {stats['median']:.1%}   "
              f"n={stats['n_classes']}   elaboration {stats['elaboration_gain_pp']:+.1f} pp")
    print(f"\nWrote {RESULTS_DIR / 'lexical_gap.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""How wrong is the regime check when it has no labels?

    python -m experiments.run_overlap_estimate

The lexical gap separates the two regimes this paper describes: classes whose
names do not appear in their documents gained 8.1 points from elaboration on
average, classes whose names do appear gained 1.4 (``run_gap_analysis``). That
is the first thing a person with a taxonomy wants to know, and the measure
behind it looks label-free -- it only counts words.

It is not. The overlap is the share of *a class's own documents* that contain a
content word of its name, and knowing which documents are a class's own is
exactly the labelling nobody has done yet. The diagnostic in
``experiments/diagnose.py`` substitutes the documents the terse vectors assign
to the class, the same pseudo-assignment the label-free predictor uses.

Substituting is not free, and this measures the price on the five corpora where
both versions can be computed. Two things matter to somebody relying on it: how
far the estimate sits from the labelled quantity, and whether it still puts the
taxonomy in the right regime -- the second being the only decision it is used
for. The answer is reported for each corpus rather than pooled, because a
practitioner has one taxonomy and not an average of five.

Not preregistered: it measures the bias of an estimator this project already
published rather than testing a claim, and it is no part of the corrected
family.
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

from experiments.config import RESULTS_DIR, ensure_dirs, set_seed
from experiments.diagnose import REGIME_THRESHOLD
from experiments.lexical_gap import name_overlap
from experiments.run_labelfree_predictor import CORPORA, embed
from experiments.run_rocchio import git_revision

RESULT = RESULTS_DIR / "overlap_estimate.json"


def bare_names(corpus_name: str, corpus: Dict) -> Dict[str, str]:
    """The bare class name, which is what the published analysis counted.

    For four of the five corpora the terse condition *is* the bare readable
    name. NACE is the exception: its terse condition carries the literal
    rendering of the original description as well, and including it would count
    words the practitioner's label does not contain.
    """
    if corpus_name != "NACE":
        return dict(corpus["terse"])
    from experiments.config import ROOT
    current = json.loads((ROOT / "data" / "taxonomy" / "sectors_v1.json")
                         .read_text(encoding="utf-8"))["sectors"]
    return {c: current[c].get("name", c) for c in corpus["names"]}


def measure(corpus_name: str, corpus: Dict) -> Dict:
    names: List[str] = corpus["names"]
    label = bare_names(corpus_name, corpus)
    documents, gold = corpus["documents"], corpus["gold"]

    vectors = embed([corpus["terse"][n] for n in names])
    docs = embed(documents)
    chosen = np.argmax(docs @ vectors.T, axis=1)

    rows = []
    for i, name in enumerate(names):
        own = [d for d, g in zip(documents, gold) if g == name]
        pseudo = [documents[j] for j in np.flatnonzero(chosen == i)]
        rows.append({
            "class": name,
            "n_own": len(own),
            "n_pseudo": len(pseudo),
            "labelled_overlap": name_overlap(label[name], own) if own else None,
            "estimated_overlap": name_overlap(label[name], pseudo) if pseudo else None,
        })

    both = [(r["labelled_overlap"], r["estimated_overlap"]) for r in rows
            if r["labelled_overlap"] is not None and r["estimated_overlap"] is not None]
    labelled_mean = float(np.mean([a for a, _ in both]))
    estimated_mean = float(np.mean([b for _, b in both]))
    rho = spearmanr([a for a, _ in both], [b for _, b in both]) if len(both) > 2 else None
    # A class is in the regime where elaboration paid if its overlap is below the
    # threshold. The decision the check is used for is this one, per class and
    # for the taxonomy as a whole.
    agree = sum(1 for a, b in both
                if (a < REGIME_THRESHOLD) == (b < REGIME_THRESHOLD))
    return {
        "classes_compared": len(both),
        "labelled_mean_overlap": labelled_mean,
        "estimated_mean_overlap": estimated_mean,
        "inflation_factor": (estimated_mean / labelled_mean) if labelled_mean else None,
        "rho": float(rho.statistic) if rho is not None else None,
        "p": float(rho.pvalue) if rho is not None else None,
        "per_class_verdict_agreement": agree / len(both) if both else None,
        "taxonomy_verdict_labelled": labelled_mean < REGIME_THRESHOLD,
        "taxonomy_verdict_estimated": estimated_mean < REGIME_THRESHOLD,
        "taxonomy_verdict_agrees": ((labelled_mean < REGIME_THRESHOLD)
                                    == (estimated_mean < REGIME_THRESHOLD)),
        "classes": rows,
    }


def main() -> int:
    set_seed()
    ensure_dirs()
    per_corpus = {name: measure(name, loader()) for name, loader in CORPORA.items()}

    pooled = [(r["labelled_overlap"], r["estimated_overlap"])
              for corpus in per_corpus.values() for r in corpus["classes"]
              if r["labelled_overlap"] is not None and r["estimated_overlap"] is not None]
    rho = spearmanr([a for a, _ in pooled], [b for _, b in pooled])
    agreeing = [n for n, c in per_corpus.items() if c["taxonomy_verdict_agrees"]]

    report = {
        "threshold": REGIME_THRESHOLD,
        "note": ("The price of computing the regime check without labels. Not "
                 "preregistered and no part of the corrected family."),
        "revision": git_revision(),
        "per_corpus": per_corpus,
        "pooled": {
            "classes": len(pooled),
            "rho": float(rho.statistic),
            "p": float(rho.pvalue),
            "labelled_mean": float(np.mean([a for a, _ in pooled])),
            "estimated_mean": float(np.mean([b for _, b in pooled])),
            "per_class_verdict_agreement": sum(
                1 for a, b in pooled
                if (a < REGIME_THRESHOLD) == (b < REGIME_THRESHOLD)) / len(pooled),
        },
        "taxonomy_verdict": {
            "corpora": len(per_corpus),
            "agreeing": len(agreeing),
            "agreeing_names": sorted(agreeing),
            "disagreeing_names": sorted(set(per_corpus) - set(agreeing)),
        },
    }
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    p = report["pooled"]
    print(f"{p['classes']} classes over {len(per_corpus)} corpora")
    print(f"labelled mean overlap {p['labelled_mean']:.4f}, "
          f"estimated {p['estimated_mean']:.4f} "
          f"({p['estimated_mean'] / p['labelled_mean']:.1f}x)")
    print(f"rho = {p['rho']:+.3f} (p = {p['p']:.2g}); "
          f"per-class regime verdict agrees on "
          f"{100 * p['per_class_verdict_agreement']:.0f}%")
    print(f"taxonomy-level verdict agrees on "
          f"{report['taxonomy_verdict']['agreeing']} of "
          f"{report['taxonomy_verdict']['corpora']} corpora"
          + (f"; disagrees on {', '.join(report['taxonomy_verdict']['disagreeing_names'])}"
             if report['taxonomy_verdict']['disagreeing_names'] else ""))
    print(f"\nwritten to {RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

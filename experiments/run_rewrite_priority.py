#!/usr/bin/env python3
"""If you can only rewrite some classes, which ones, and what does that buy?

    python -m experiments.run_rewrite_priority

The label-free estimate is reported in this paper as a correlation: it tracks
the labelled quantity at rho = +0.818 and predicts captured headroom at
+0.564. A correlation is not a decision. The question a practitioner actually
faces is narrower and harder: rewriting a taxonomy costs effort per class, so
if only a fraction of the classes can be rewritten, which fraction, and how
much of the available gain does that capture?

This answers it by simulation rather than by correlation. Class vectors are
built from the elaborated description for the top k of classes under some
ranking and from the terse one for the rest, and the whole classifier is run.
The quantity reported is the share of the full rewrite's gain that the partial
rewrite captures:

    captured(k) = (acc(top k rewritten) - acc(none)) / (acc(all) - acc(none))

Four rankings are compared, and the comparison is the point:

  * labelled change    -- the quantity needs labels; the ceiling this can reach
  * label-free change  -- the estimate, for somebody who has drafted rewrites
                          and must choose which to keep
  * label-free level   -- alignment under the terse descriptions alone, for
                          somebody who has written nothing yet. This is what
                          ``experiments/diagnose.py`` currently puts first, and
                          it has never been validated. If it does not beat
                          random here, the tool's advice is wrong and has to
                          change.
  * random             -- the floor, averaged over shuffles

Not preregistered: it re-analyses a factor already established rather than
testing a registered prediction, and it is no part of the corrected family.
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from experiments.config import RESULTS_DIR, SEED, ensure_dirs, set_seed
from experiments.run_labelfree_predictor import CORPORA, measure, pseudo_alignment
from experiments.run_rocchio import git_revision

RESULT = RESULTS_DIR / "rewrite_priority.json"
FRACTIONS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
SHUFFLES = 200
# Below this, the denominator is noise: a corpus that gains nothing from a full
# rewrite cannot say which classes to rewrite first, and dividing by its span
# produces shares in the hundreds of percent.
MIN_GAIN_PP = 2.0


def accuracy(vectors: np.ndarray, docs: np.ndarray, names: List[str],
             gold: List[str]) -> float:
    predicted = [names[i] for i in np.argmax(docs @ vectors.T, axis=1)]
    return float(np.mean([p == g for p, g in zip(predicted, gold)]))


def mixed(terse: np.ndarray, elaborated: np.ndarray, names: List[str],
          rewrite: List[str]) -> np.ndarray:
    """Elaborated vectors for the chosen classes, terse for every other."""
    out = terse.copy()
    index = {n: i for i, n in enumerate(names)}
    for name in rewrite:
        out[index[name]] = elaborated[index[name]]
    return out


def capture_curve(report: Dict, order: List[str]) -> List[Dict]:
    names, docs, gold = report["names"], report["docs"], report["gold"]
    terse, elaborated = report["terse_vectors"], report["elaborated_vectors"]
    floor = accuracy(terse, docs, names, gold)
    # Rewriting every class this ranking covers, which is what 100% means here.
    ceiling = accuracy(mixed(terse, elaborated, names, list(order)), docs, names, gold)
    span = ceiling - floor
    curve = []
    for fraction in FRACTIONS:
        k = max(1, round(fraction * len(order)))
        acc = accuracy(mixed(terse, elaborated, names, order[:k]), docs, names, gold)
        curve.append({
            "fraction_rewritten": fraction,
            "classes_rewritten": k,
            "top1_accuracy": acc,
            "captured_share": (acc - floor) / span if span else None,
        })
    return curve


def corpus_report(name: str, corpus: Dict) -> Dict:
    report = measure(corpus)
    names, docs = report["names"], report["docs"]
    classes = report["classes"]
    terse = report["terse_vectors"]
    assigned = [names[i] for i in np.argmax(docs @ terse.T, axis=1)]
    level = pseudo_alignment(terse, docs, assigned, names)

    rankings = {
        # needs labels: the ceiling a label-free ranking could reach
        "labelled change": sorted(classes, key=lambda c: -report["true_change"][c]),
        # needs drafts but no labels
        "label-free change": sorted(classes, key=lambda c: -report["estimated_change"][c]),
        # needs neither: what the released diagnostic ranks by today
        "label-free level": sorted([c for c in classes if c in level],
                                   key=lambda c: level[c]),
    }
    out = {k: capture_curve(report, order) for k, order in rankings.items()}

    rng = np.random.default_rng(SEED)
    shuffled = []
    for _ in range(SHUFFLES):
        order = list(classes)
        rng.shuffle(order)
        shuffled.append([row["captured_share"] for row in capture_curve(report, order)])
    mean = np.mean(shuffled, axis=0)
    spread = np.percentile(shuffled, [5, 95], axis=0)
    out["random"] = [
        {"fraction_rewritten": f, "captured_share": float(mean[i]),
         "captured_share_p5": float(spread[0][i]), "captured_share_p95": float(spread[1][i])}
        for i, f in enumerate(FRACTIONS)]

    floor = accuracy(terse, docs, names, report["gold"])
    ceiling = accuracy(report["elaborated_vectors"], docs, names, report["gold"])
    return {
        "classes": len(classes),
        "documents": len(docs),
        "top1_no_rewrite": floor,
        "top1_full_rewrite": ceiling,
        "full_gain_pp": 100 * (ceiling - floor),
        "curves": out,
    }


def main() -> int:
    set_seed()
    ensure_dirs()
    per_corpus = {name: corpus_report(name, load()) for name, load in CORPORA.items()}

    ranked = [n for n, c in per_corpus.items() if c["full_gain_pp"] >= MIN_GAIN_PP]
    # Pooled across the corpora that have a gain to distribute, weighted by how
    # much each has, since a corpus with nothing to gain should not dominate.
    pooled = {}
    for ranking in ("labelled change", "label-free change", "label-free level", "random"):
        rows = []
        for i, fraction in enumerate(FRACTIONS):
            num = den = 0.0
            for name in ranked:
                corpus = per_corpus[name]
                gain = corpus["full_gain_pp"]
                share = corpus["curves"][ranking][i]["captured_share"]
                if gain > 0 and share is not None:
                    num += gain * share
                    den += gain
            rows.append({"fraction_rewritten": fraction,
                         "captured_share": num / den if den else None})
        pooled[ranking] = rows

    report = {
        "note": ("Decision utility of the label-free estimate: what a partial "
                 "rewrite captures. Not preregistered and no part of the "
                 "corrected family."),
        "min_gain_pp": MIN_GAIN_PP,
        "corpora_ranked": ranked,
        "corpora_excluded": [n for n in per_corpus if n not in ranked],
        "fractions": list(FRACTIONS),
        "shuffles": SHUFFLES,
        "revision": git_revision(),
        "per_corpus": per_corpus,
        "pooled": pooled,
    }
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"ranked over {', '.join(ranked)}; "
          f"excluded for gaining under {MIN_GAIN_PP:.0f} points: "
          f"{', '.join(report['corpora_excluded']) or 'none'}\n")
    print(f"{'ranking':20} " + "  ".join(f"{int(100*f):>4}%" for f in FRACTIONS))
    for ranking, rows in pooled.items():
        cells = "  ".join(f"{100 * r['captured_share']:>4.0f}%" if r["captured_share"] is not None
                          else "   --" for r in rows)
        print(f"{ranking:20} {cells}")
    print(f"\nwritten to {RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

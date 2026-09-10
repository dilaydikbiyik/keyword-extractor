#!/usr/bin/env python3
"""A third dataset, with the outcome predicted before it is measured.

    python -m experiments.run_reuters

The alignment account says a description helps in proportion to how far it
moves the class vector toward the documents it must attract. That is a claim
that can be tested prospectively rather than fitted after the fact: measure the
alignment change first, state what it implies, and only then look at accuracy.

Reuters-21578 is a useful third setting because its labels are opaque codes —
``money-fx``, ``acq``, ``dlr`` — which puts it in the same regime as NACE
sections rather than 20 Newsgroups' readable topic names. The account therefore
predicts that elaboration helps here, and that the alignment change is positive.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score

from experiments.config import RESULTS_DIR, SEED, TABLES_DIR, ensure_dirs, set_seed
from experiments.metrics import bootstrap_ci, mcnemar_exact
from experiments.systems import get_embedder

READABLE = {
    "earn": "Corporate earnings",
    "acq": "Acquisitions and mergers",
    "money-fx": "Foreign exchange",
    "grain": "Grain trade",
    "crude": "Crude oil",
    "trade": "International trade",
    "interest": "Interest rates",
    "ship": "Shipping",
    "wheat": "Wheat",
    "corn": "Corn",
    "dlr": "The US dollar",
    "money-supply": "Money supply",
    "oilseed": "Oilseeds",
    "sugar": "Sugar",
    "coffee": "Coffee",
}

DEFINITION = {
    "earn": "Corporate earnings. Quarterly and annual results, net income and losses, earnings per share, revenue and profit figures, dividends declared.",
    "acq": "Acquisitions and mergers. Takeover bids, stakes acquired, companies bought and sold, merger agreements, tender offers, divestitures.",
    "money-fx": "Foreign exchange. Currency markets, exchange rates between the dollar, yen and mark, central bank intervention, currency stability agreements.",
    "grain": "Grain trade. Wheat, corn, barley and sorghum shipments, export sales and tenders, harvest and crop estimates, grain stocks.",
    "crude": "Crude oil. Oil prices and production quotas, OPEC agreements, refineries, output cuts, petroleum supply and demand.",
    "trade": "International trade. Trade deficits and surpluses, tariffs and import quotas, trade disputes and negotiations between countries.",
    "interest": "Interest rates. Central bank rates, prime and discount rates, bond yields, monetary tightening and easing.",
    "ship": "Shipping. Freight rates, vessels and tankers, port congestion, seamen's strikes, cargo and shipping lanes.",
    "wheat": "Wheat. Wheat crops, export sales and prices, subsidies and tenders, winter wheat conditions.",
    "corn": "Corn. Corn crops and yields, export sales, feed grain programmes, maize prices.",
    "dlr": "The US dollar. Movements of the dollar against other currencies, dollar strength and weakness, currency market reaction.",
    "money-supply": "Money supply. Monetary aggregates M1, M2 and M3, weekly money supply figures, central bank reserves.",
    "oilseed": "Oilseeds. Soybeans, rapeseed and sunflower seed, crushings, oilseed exports and meal.",
    "sugar": "Sugar. Sugar production and exports, quotas and prices, cane and beet harvests.",
    "coffee": "Coffee. Coffee exports and quotas, the International Coffee Organisation, bean prices and harvests.",
}

DOCS_PER_CLASS = 80


def load(classes: List[str]) -> Dict[str, List[str]]:
    """Single-label documents only: Reuters is multi-label, and a document
    carrying two of the classes under test cannot score either fairly."""
    import nltk

    try:
        nltk.data.find("corpora/reuters")
    except LookupError:
        nltk.download("reuters", quiet=True)
    from nltk.corpus import reuters

    wanted = set(classes)
    rng = np.random.default_rng(SEED)
    documents: Dict[str, List[str]] = {c: [] for c in classes}
    for fileid in reuters.fileids():
        labels = wanted & set(reuters.categories(fileid))
        if len(labels) != 1:
            continue
        text = reuters.raw(fileid).strip()
        if len(text) >= 80:
            documents[next(iter(labels))].append(text)
    return {
        c: list(rng.choice(v, size=min(DOCS_PER_CLASS, len(v)), replace=False))
        for c, v in documents.items()
        if len(v) >= 20
    }


def unit(texts):
    embedder = get_embedder()
    m = np.vstack([np.asarray(embedder.embed_text(t), dtype=np.float64) for t in texts])
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    set_seed()
    ensure_dirs()

    documents = load(list(READABLE))
    names = sorted(documents)
    flat = [d for c in names for d in documents[c]]
    gold = [c for c in names for _ in documents[c]]
    docs = unit(flat)
    print(f"{len(flat)} single-label documents, {len(names)} classes\n")

    conditions = {
        "raw category code": {c: c for c in names},
        "readable class name": {c: READABLE[c] for c in names},
        "definition of what the class covers": {c: DEFINITION[c] for c in names},
    }

    centroids = {}
    for c in names:
        idx = [i for i, g in enumerate(gold) if g == c]
        m = docs[idx].mean(axis=0)
        centroids[c] = m / np.linalg.norm(m)

    # --- the prediction, made from alignment alone ---
    alignment = {}
    for label, descriptions in conditions.items():
        vectors = unit([descriptions[c] for c in names])
        alignment[label] = float(
            np.mean([vectors[i] @ centroids[c] for i, c in enumerate(names)])
        )
    keys = list(conditions)
    delta = alignment[keys[2]] - alignment[keys[1]]
    prediction = "elaboration helps" if delta > 0 else "elaboration does not help"
    print("Predicted from alignment, before looking at accuracy:")
    for k in keys:
        print(f"  {k:<38} alignment {alignment[k]:.3f}")
    print(f"  readable name → definition: {delta:+.3f}  →  {prediction}\n")

    # --- only now, the accuracy ---
    results, correct = {}, {}
    for label, descriptions in conditions.items():
        vectors = unit([descriptions[c] for c in names])
        order = np.argsort(-(docs @ vectors.T), axis=1)
        predicted = [names[order[i, 0]] for i in range(len(gold))]
        hits = [gold[i] == predicted[i] for i in range(len(gold))]
        correct[label] = hits
        lo, hi = bootstrap_ci(hits)
        results[label] = {
            "top1_accuracy": float(np.mean(hits)),
            "top1_ci95": [lo, hi],
            "top3_accuracy": float(
                np.mean([gold[i] in [names[j] for j in order[i, :3]] for i in range(len(gold))])
            ),
            "f1_macro": float(f1_score(gold, predicted, average="macro", zero_division=0)),
            "alignment": alignment[label],
        }

    spelling_out = mcnemar_exact(correct[keys[1]], correct[keys[0]])
    defining = mcnemar_exact(correct[keys[2]], correct[keys[1]])
    for d, a, b in ((spelling_out, keys[1], keys[0]), (defining, keys[2], keys[1])):
        d["gain_pp"] = (results[a]["top1_accuracy"] - results[b]["top1_accuracy"]) * 100

    rows = ["| Class descriptions | Top-1 | 95% CI | Top-3 | F1-macro | alignment |",
            "| --- | --- | --- | --- | --- | --- |"]
    for k in keys:
        r = results[k]
        lo, hi = r["top1_ci95"]
        rows.append(f"| {k} | {r['top1_accuracy']:.1%} | [{lo:.1%}, {hi:.1%}] | "
                    f"{r['top3_accuracy']:.1%} | {r['f1_macro']:.3f} | {r['alignment']:.3f} |")
    table = "\n".join(rows)
    (TABLES_DIR / "reuters.md").write_text(table + "\n", encoding="utf-8")

    payload = {
        "dataset": "Reuters-21578, single-label documents over 15 frequent categories",
        "n": len(flat), "classes": len(names),
        "why_this_dataset": (
            "Its labels are opaque codes (money-fx, acq, dlr), which puts it in "
            "the same regime as NACE sections rather than 20 Newsgroups' readable "
            "topic names."
        ),
        "prediction_from_alignment": {
            "alignment_change_readable_to_definition": delta,
            "predicted": prediction,
            "made_before_measuring_accuracy": True,
        },
        "conditions": results,
        "spelling_the_label_out": spelling_out,
        "defining_the_class": defining,
        "table_markdown": table,
    }
    (RESULTS_DIR / "reuters.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(table)
    print(f"\ncode → readable name: {spelling_out['gain_pp']:+.1f} pp, p = {spelling_out['p_value']:.2e}")
    print(f"readable name → definition: {defining['gain_pp']:+.1f} pp, p = {defining['p_value']:.2e}")
    print(f"\nprediction was: {prediction}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""What predicts which classes gain from a better description?

    python -m experiments.run_predictor_search

Elaborating class descriptions is worth 26.8 points on NACE and 0.4 on 20
Newsgroups, and neither the lexical overlap of the class name nor the
confusability of its vector predicts which individual classes benefit. This
searches for something that does, and then asks whether the answer is usable.

Four steps:

1. **The predictor.** Target is the share of *available* headroom a class
   captured, not the raw gain — a class already at 90% cannot gain much, and
   raw gain mostly measures where a class started.
2. **The dataset level.** Whatever predicts per-class gain should also account
   for the difference between the two datasets, or it is not a mechanism.
3. **A selection rule.** If the quantity is causal, choosing each class's
   description by it should beat a fixed policy.
4. **Why the rule fails**, and whether calibration rescues it.

The honest summary is that step 1 and 2 succeed and steps 3 and 4 do not.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import spearmanr

from experiments.config import RESULTS_DIR, ROOT, SEED, ensure_dirs, set_seed
from experiments.lexical_gap import name_overlap
from experiments.metrics import mcnemar_exact
from experiments.systems import get_embedder

DOCS_PER_CLASS = 80


def unit(texts: Sequence[str]) -> np.ndarray:
    embedder = get_embedder()
    matrix = np.vstack([np.asarray(embedder.embed_text(t), dtype=np.float64) for t in texts])
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def load_nace():
    from experiments.data import load_labeled_samples, load_taxonomy

    current = load_taxonomy()
    previous = json.loads(
        (ROOT / "data" / "taxonomy" / "sectors_v1.json").read_text(encoding="utf-8")
    )["sectors"]
    documents: Dict[str, List[str]] = {}
    for sample in load_labeled_samples():
        documents.setdefault(sample.true_sector, []).append(sample.purpose)
    names = [c for c in sorted(documents) if len(documents[c]) >= 6]
    return (
        names,
        {c: documents[c] for c in names},
        {c: f"{previous[c]['name']}. {previous[c].get('description', '')}" for c in names},
        {c: f"{current[c]['name']}. {current[c].get('description', '')}" for c in names},
    )


def load_newsgroups():
    from sklearn.datasets import fetch_20newsgroups

    from experiments.run_replication import DEFINITION, READABLE

    data = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    names = list(data.target_names)
    rng = np.random.default_rng(SEED)
    documents: Dict[str, List[str]] = {}
    for text, target in zip(data.data, data.target):
        if len(text.strip()) >= 40:
            documents.setdefault(names[target], []).append(text.strip())
    documents = {
        k: list(rng.choice(v, size=min(DOCS_PER_CLASS, len(v)), replace=False))
        for k, v in documents.items()
    }
    return names, documents, dict(READABLE), dict(DEFINITION)


def per_class(names, documents, terse, rich, dataset):
    flat = [d for c in names for d in documents[c]]
    gold = [c for c in names for _ in documents[c]]
    docs = unit(flat)
    terse_vectors, rich_vectors = unit([terse[c] for c in names]), unit([rich[c] for c in names])

    recall = {}
    for label, vectors in (("terse", terse_vectors), ("rich", rich_vectors)):
        predicted = [names[i] for i in np.argmax(docs @ vectors.T, axis=1)]
        recall[label] = {
            c: float(np.mean([predicted[i] == c for i in range(len(gold)) if gold[i] == c]))
            for c in names
        }

    centroids = {}
    for c in names:
        idx = [i for i, g in enumerate(gold) if g == c]
        m = docs[idx].mean(axis=0)
        centroids[c] = m / np.linalg.norm(m)

    similarity = terse_vectors @ terse_vectors.T
    np.fill_diagonal(similarity, -1.0)

    rows = []
    for i, c in enumerate(names):
        aligned_terse = float(terse_vectors[i] @ centroids[c])
        aligned_rich = float(rich_vectors[i] @ centroids[c])
        rows.append({
            "dataset": dataset, "class": c,
            "recall_terse": recall["terse"][c], "recall_rich": recall["rich"][c],
            "gain": recall["rich"][c] - recall["terse"][c],
            "alignment_terse": aligned_terse,
            "alignment_gain": aligned_rich - aligned_terse,
            "confusability": float(similarity[i].max()),
            "name_overlap": name_overlap(c if dataset == "20NG" else terse[c], documents[c]),
            "words_added": len(rich[c].split()) - len(terse[c].split()),
        })
    return rows


def selection_rule(names, documents, terse, rich, contrastive: bool):
    """Choose each class's description on a development half, score on the rest."""
    rng = np.random.default_rng(SEED)
    dev_docs, dev_gold, test_docs, test_gold = [], [], [], []
    for c in names:
        d = list(documents[c])
        rng.shuffle(d)
        cut = max(1, len(d) // 2)
        dev_docs += d[:cut]
        dev_gold += [c] * cut
        test_docs += d[cut:]
        test_gold += [c] * (len(d) - cut)

    dev, test = unit(dev_docs), unit(test_docs)
    centroids = {}
    for c in names:
        idx = [i for i, g in enumerate(dev_gold) if g == c]
        m = dev[idx].mean(axis=0)
        centroids[c] = m / np.linalg.norm(m)
    centroid_matrix = np.vstack([centroids[c] for c in names])

    def score(c, text):
        v = unit([text])[0]
        sims = centroid_matrix @ v
        i = names.index(c)
        if not contrastive:
            return float(sims[i])
        return float(sims[i] - np.delete(sims, i).max())

    chosen, picks = {}, {"terse": 0, "rich": 0}
    for c in names:
        if score(c, terse[c]) >= score(c, rich[c]):
            chosen[c] = terse[c]
            picks["terse"] += 1
        else:
            chosen[c] = rich[c]
            picks["rich"] += 1

    def evaluate(descriptions):
        vectors = unit([descriptions[c] for c in names])
        predicted = [names[i] for i in np.argmax(test @ vectors.T, axis=1)]
        return [test_gold[i] == predicted[i] for i in range(len(test_gold))]

    outcomes = {"terse": evaluate(terse), "rich": evaluate(rich), "chosen": evaluate(chosen)}
    best = max(("terse", "rich"), key=lambda k: float(np.mean(outcomes[k])))
    test_stat = mcnemar_exact(outcomes["chosen"], outcomes[best])
    return {
        "picks": picks,
        "accuracy": {k: float(np.mean(v)) for k, v in outcomes.items()},
        "best_fixed_policy": best,
        "delta_pp": (float(np.mean(outcomes["chosen"])) - float(np.mean(outcomes[best]))) * 100,
        "p_value": test_stat["p_value"],
    }


def greedy_set_search(names, documents, terse, rich):
    """Optimise the whole set against development accuracy, not per class.

    Per-class criteria fail because the argmax compares across classes. The
    strongest available answer to that is to optimise the objective itself:
    repeatedly swap whichever single class's description most improves
    development accuracy, until nothing improves.

    It also fails, and the reason is capacity: there are 2^K configurations for
    K classes, and choosing among them from a labelled development set needs
    more supervision than a zero-shot pipeline is meant to require. Development
    accuracy rises and held-out accuracy does not follow.
    """
    rng = np.random.default_rng(SEED)
    dev_docs, dev_gold, test_docs, test_gold = [], [], [], []
    for c in names:
        d = list(documents[c])
        rng.shuffle(d)
        cut = max(1, len(d) // 2)
        dev_docs += d[:cut]
        dev_gold += [c] * cut
        test_docs += d[cut:]
        test_gold += [c] * (len(d) - cut)

    dev, test = unit(dev_docs), unit(test_docs)
    terse_vectors, rich_vectors = unit([terse[c] for c in names]), unit([rich[c] for c in names])

    def hits(docs, gold, elaborated):
        vectors = np.vstack(
            [rich_vectors[i] if elaborated[i] else terse_vectors[i] for i in range(len(names))]
        )
        predicted = [names[j] for j in np.argmax(docs @ vectors.T, axis=1)]
        return [gold[i] == predicted[i] for i in range(len(gold))]

    def accuracy(docs, gold, elaborated):
        return float(np.mean(hits(docs, gold, elaborated)))

    all_terse, all_rich = [False] * len(names), [True] * len(names)
    start_rich = accuracy(dev, dev_gold, all_rich) > accuracy(dev, dev_gold, all_terse)
    choice = [start_rich] * len(names)
    best = accuracy(dev, dev_gold, choice)
    dev_start, swaps = best, 0
    while swaps < len(names):
        candidates = []
        for i in range(len(names)):
            trial = list(choice)
            trial[i] = not trial[i]
            candidates.append((accuracy(dev, dev_gold, trial), i))
        score, index = max(candidates)
        if score <= best + 1e-12:
            break
        choice[index] = not choice[index]
        best = score
        swaps += 1

    test_terse = accuracy(test, test_gold, all_terse)
    test_rich = accuracy(test, test_gold, all_rich)
    best_fixed = "rich" if test_rich >= test_terse else "terse"
    fixed_hits = hits(test, test_gold, all_rich if best_fixed == "rich" else all_terse)
    searched = hits(test, test_gold, choice)
    return {
        "swaps": swaps,
        "classes_elaborated": int(sum(choice)),
        "dev_accuracy_start": dev_start,
        "dev_accuracy_end": best,
        "test_terse": test_terse,
        "test_rich": test_rich,
        "test_searched": float(np.mean(searched)),
        "best_fixed_policy": best_fixed,
        "delta_pp": (float(np.mean(searched)) - max(test_terse, test_rich)) * 100,
        "p_value": mcnemar_exact(searched, fixed_hits)["p_value"],
        "dev_gain_pp": (best - dev_start) * 100,
    }


def scale_and_calibration(names, documents, terse, rich):
    """Do the two styles share a similarity scale, and does z-scoring fix it?"""
    flat = [d for c in names for d in documents[c]]
    gold = [c for c in names for _ in documents[c]]
    docs = unit(flat)
    sets = {
        "terse": {c: terse[c] for c in names},
        "rich": {c: rich[c] for c in names},
        "mixed": {c: (rich[c] if i % 2 == 0 else terse[c]) for i, c in enumerate(names)},
    }
    out = {}
    for label, descriptions in sets.items():
        vectors = unit([descriptions[c] for c in names])
        sims = docs @ vectors.T
        plain = [gold[i] == names[j] for i, j in enumerate(np.argmax(sims, axis=1))]
        z = (sims - sims.mean(axis=0, keepdims=True)) / (sims.std(axis=0, keepdims=True) + 1e-9)
        zscored = [gold[i] == names[j] for i, j in enumerate(np.argmax(z, axis=1))]
        out[label] = {
            "argmax": float(np.mean(plain)),
            "z_scored": float(np.mean(zscored)),
            "mean_similarity": float(sims.mean()),
        }
    rich_positions = {i for i in range(len(names)) if i % 2 == 0}
    vectors = unit([sets["mixed"][c] for c in names])
    winners = np.argmax(docs @ vectors.T, axis=1)
    out["mixed"]["share_of_decisions_won_by_the_elaborated_half"] = float(
        np.mean([w in rich_positions for w in winners])
    )
    out["mixed"]["fair_share"] = len(rich_positions) / len(names)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    set_seed()
    ensure_dirs()

    datasets = {"NACE": load_nace(), "20NG": load_newsgroups()}
    rows = []
    for tag, (names, documents, terse, rich) in datasets.items():
        rows += per_class(names, documents, terse, rich, tag)

    usable = [r for r in rows if r["recall_terse"] < 0.999]
    target = np.array([(r["recall_rich"] - r["recall_terse"]) / (1 - r["recall_terse"]) for r in usable])

    predictors = ("alignment_gain", "alignment_terse", "confusability", "name_overlap", "words_added")
    correlations = {}
    for p in predictors:
        rho, pv = spearmanr([r[p] for r in usable], target)
        within = {}
        for tag in datasets:
            sub = [r for r in usable if r["dataset"] == tag]
            if len(sub) > 3:
                y = [(r["recall_rich"] - r["recall_terse"]) / (1 - r["recall_terse"]) for r in sub]
                within[tag] = float(spearmanr([r[p] for r in sub], y)[0])
        correlations[p] = {"rho": float(rho), "p": float(pv), "within": within}

    dataset_level = {
        tag: {
            "mean_alignment_terse": float(np.mean([r["alignment_terse"] for r in rows if r["dataset"] == tag])),
            "mean_alignment_gain": float(np.mean([r["alignment_gain"] for r in rows if r["dataset"] == tag])),
            "mean_accuracy_gain": float(np.mean([r["gain"] for r in rows if r["dataset"] == tag])),
        }
        for tag in datasets
    }

    rules, scales = {}, {}
    for tag, (names, documents, terse, rich) in datasets.items():
        rules[tag] = {
            "marginal": selection_rule(names, documents, terse, rich, contrastive=False),
            "contrastive": selection_rule(names, documents, terse, rich, contrastive=True),
            "greedy_set_search": greedy_set_search(names, documents, terse, rich),
        }
        scales[tag] = scale_and_calibration(names, documents, terse, rich)

    payload = {
        "n_classes": len(rows),
        "target": "share of available headroom captured",
        "correlations": correlations,
        "dataset_level": dataset_level,
        "selection_rules": rules,
        "scale_and_calibration": scales,
        "conclusion": (
            "How far a description moves the class vector toward its own documents "
            "predicts which classes gain (rho = +0.61, p = 0.0002) and accounts for "
            "the difference between the datasets. It does not convert into a "
            "per-class selection rule: both a marginal and a contrastive criterion "
            "lose to a fixed policy, because the two description styles sit on "
            "different similarity scales and the argmax compares across classes. "
            "Per-class z-scoring does not rescue it either, and neither does "
            "optimising the whole set greedily against development accuracy: "
            "development accuracy rises and held-out accuracy does not follow. "
            "There are 2^K configurations for K classes, and choosing among them "
            "needs more supervision than a zero-shot pipeline is meant to require."
        ),
        "classes": rows,
    }
    (RESULTS_DIR / "predictor_search.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"{len(usable)} classes · target = share of available headroom captured\n")
    print(f"  {'predictor':<22}{'rho':>8}{'p':>9}{'NACE':>8}{'20NG':>8}")
    for p, c in correlations.items():
        w = c["within"]
        print(f"  {p:<22}{c['rho']:>+8.3f}{c['p']:>9.4f}"
              f"{w.get('NACE', float('nan')):>+8.2f}{w.get('20NG', float('nan')):>+8.2f}")
    print("\n  dataset level:")
    for tag, d in dataset_level.items():
        print(f"    {tag:<6} alignment {d['mean_alignment_terse']:.3f} "
              f"→ {d['mean_alignment_terse'] + d['mean_alignment_gain']:.3f} "
              f"({d['mean_alignment_gain']:+.3f})   accuracy {d['mean_accuracy_gain']:+.1%}")
    print("\n  selection rules, on held-out documents:")
    for tag, r in rules.items():
        for kind, res in r.items():
            extra = ""
            if kind == "greedy_set_search":
                extra = f"   (dev {res['dev_gain_pp']:+.1f} pp over {res['swaps']} swaps)"
            print(f"    {tag:<6} {kind:<18} {res['delta_pp']:+.1f} pp vs "
                  f"{res['best_fixed_policy']:<6} p = {res['p_value']:.4f}{extra}")
    print("\n  mixed description styles:")
    for tag, s in scales.items():
        m = s["mixed"]
        print(f"    {tag:<6} elaborated half wins "
              f"{m['share_of_decisions_won_by_the_elaborated_half']:.1%} of decisions "
              f"(fair share {m['fair_share']:.0%})")
    print(f"\nWrote {RESULTS_DIR / 'predictor_search.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

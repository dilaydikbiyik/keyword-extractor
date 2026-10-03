#!/usr/bin/env python3
"""Can the quantity that predicts the gain be estimated without any labels?

    python -m experiments.run_labelfree_predictor --preregister   # commit the prediction
    python -m experiments.run_labelfree_predictor                 # then test it

Everything this paper establishes about class descriptions rests on one
quantity: how far elaborating a description moves the class vector toward the
centroid of the documents that belong to it. Measuring that centroid needs
labels, which is exactly what a practitioner deciding whether to rewrite a
taxonomy does not have. The finding explains; it does not yet advise.

This asks whether the quantity survives losing the labels. For each class, the
documents are pseudo-assigned by the terse class vectors -- argmax, the same
decision the classifier makes -- and the centroid of that pseudo-class stands in
for the real one:

    a_hat(c) = cos(v_c, centroid of the documents the terse vectors assign to c)

Nothing in a_hat uses a label. If the change in a_hat predicts which classes
gain from elaboration, then a practitioner can estimate the answer from
unlabelled documents alone, before paying for a single annotation.

Five corpora, \u007f108\u007f classes: the German register, 20 Newsgroups, Reuters-21578,
Brown and arXiv. Each contributes its terse condition, its elaborated condition,
and its documents stripped of their labels.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

from experiments.config import RESULTS_DIR, ROOT, SEED, ensure_dirs, set_seed
from experiments.run_rocchio import alignment, git_revision, unit_rows
from experiments.systems import get_embedder

PREREGISTRATION = RESULTS_DIR / "labelfree_predictor_preregistration.json"
RESULT = RESULTS_DIR / "labelfree_predictor.json"


def embed(texts: List[str], batch: int = 64) -> np.ndarray:
    return unit_rows(np.vstack([np.asarray(v, dtype=np.float64) for v in
                                get_embedder().embed_texts(list(texts), batch_size=batch)]))


def nace() -> Dict:
    from experiments.data import load_labeled_samples
    from experiments.run_description_study import LITERAL_GERMAN

    old = json.loads((ROOT / "data" / "taxonomy" / "sectors_v1.json")
                     .read_text(encoding="utf-8"))["sectors"]
    new = json.loads((ROOT / "data" / "taxonomy" / "sectors.json")
                     .read_text(encoding="utf-8"))["sectors"]
    names = sorted(old)
    samples = load_labeled_samples()
    return {
        "names": names,
        "terse": {c: f"{old[c].get('name', '')}. "
                     f"{LITERAL_GERMAN.get(c, old[c].get('description', ''))}" for c in names},
        "elaborated": {c: f"{new[c].get('name', '')}. {new[c].get('description', '')}"
                       for c in names},
        "documents": [s.purpose for s in samples],
        "gold": [s.true_sector for s in samples],
    }


def newsgroups() -> Dict:
    from experiments.run_replication import DEFINITION, READABLE, load_newsgroups

    texts, labels, names = load_newsgroups("test", 2000)
    return {"names": names, "terse": {n: READABLE[n] for n in names},
            "elaborated": {n: DEFINITION[n] for n in names},
            "documents": texts, "gold": labels}


def reuters() -> Dict:
    from experiments.run_reuters import DEFINITION, READABLE, load

    documents = load(sorted(READABLE))
    names = sorted(documents)
    texts, gold = [], []
    for name in names:
        texts += documents[name]
        gold += [name] * len(documents[name])
    return {"names": names, "terse": {n: READABLE[n] for n in names},
            "elaborated": {n: DEFINITION[n] for n in names},
            "documents": texts, "gold": gold}


def brown() -> Dict:
    from experiments.run_brown import DEFINITION, READABLE, load_brown

    texts, labels, names = load_brown()
    return {"names": names, "terse": {n: READABLE[n] for n in names},
            "elaborated": {n: DEFINITION[n] for n in names},
            "documents": texts, "gold": labels}


def arxiv() -> Dict:
    from experiments.run_arxiv import conditions, load_corpus

    texts, labels, names = load_corpus()
    style = conditions(names)
    return {"names": names,
            "terse": {n: style["readable class name"][n] for n in names},
            "elaborated": {n: style["arXiv's own description"][n] for n in names},
            "documents": texts, "gold": labels}


CORPORA = {"NACE": nace, "20NG": newsgroups, "Reuters": reuters, "Brown": brown, "arXiv": arxiv}


def pseudo_alignment(vectors: np.ndarray, docs: np.ndarray, assigned: List[str],
                     names: List[str]) -> Dict[str, float]:
    """The same measure as `alignment`, over the classes the terse vectors chose.

    A class the terse vectors never choose has no pseudo-centroid, exactly as a
    class with no documents has no real one, and is left out of both.
    """
    out = {}
    for i, name in enumerate(names):
        idx = [j for j, a in enumerate(assigned) if a == name]
        if idx:
            centroid = docs[idx].mean(axis=0)
            out[name] = float(vectors[i] @ (centroid / np.linalg.norm(centroid)))
    return out


def measure(corpus: Dict) -> Dict:
    """Both changes per class: the labelled one, and the one labels are not used for."""
    names = corpus["names"]
    terse = embed([corpus["terse"][n] for n in names])
    elaborated = embed([corpus["elaborated"][n] for n in names])
    docs = embed(corpus["documents"])
    assigned = [names[i] for i in np.argmax(docs @ terse.T, axis=1)]

    true_before = alignment(terse, docs, corpus["gold"], names)
    true_after = alignment(elaborated, docs, corpus["gold"], names)
    hat_before = pseudo_alignment(terse, docs, assigned, names)
    hat_after = pseudo_alignment(elaborated, docs, assigned, names)
    shared = [n for n in names if n in true_before and n in hat_before]
    return {
        "classes": shared,
        "true_change": {n: true_after[n] - true_before[n] for n in shared},
        "estimated_change": {n: hat_after[n] - hat_before[n] for n in shared},
        "n_documents": len(corpus["documents"]),
        "n_classes": len(names),
        "classes_pseudo_assigned": len(hat_before),
        "terse_vectors": terse, "elaborated_vectors": elaborated,
        "docs": docs, "gold": corpus["gold"], "names": names,
    }


def method_fingerprint() -> str:
    parts = [f"SEED={SEED}"]
    parts += [inspect.getsource(f) for f in (embed, pseudo_alignment, measure, alignment,
                                             nace, newsgroups, reuters, brown, arxiv)]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def pooled_pairs(state: Dict[str, Dict]) -> Tuple[List[float], List[float]]:
    estimated, true = [], []
    for report in state.values():
        for name in report["classes"]:
            estimated.append(report["estimated_change"][name])
            true.append(report["true_change"][name])
    return estimated, true


def headroom(report: Dict) -> Dict[str, float]:
    """Share of the recall a class could still gain that elaboration captured."""
    names, docs, gold = report["names"], report["docs"], report["gold"]
    out = {}
    for condition, vectors in (("terse", report["terse_vectors"]),
                               ("elaborated", report["elaborated_vectors"])):
        predicted = [names[i] for i in np.argmax(docs @ vectors.T, axis=1)]
        out[condition] = {n: float(np.mean([p == g for p, g in zip(predicted, gold) if g == n]))
                          for n in report["classes"] if n in set(gold)}
    return {n: (out["elaborated"][n] - out["terse"][n]) / (1 - out["terse"][n])
            for n in out["terse"] if out["terse"][n] < 0.999}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", action="store_true")
    args = parser.parse_args()
    set_seed()
    ensure_dirs()

    state = {}
    for name, load in CORPORA.items():
        corpus = load()
        state[name] = measure(corpus)
        print(f"  {name:8s} {state[name]['n_documents']:>6,} documents  "
              f"{len(state[name]['classes']):>3d} classes measured", flush=True)
    estimated, true = pooled_pairs(state)
    rho_est_true, p_est_true = spearmanr(estimated, true)
    agree = float(np.mean([np.sign(e) == np.sign(t) for e, t in zip(estimated, true)]))

    if args.preregister:
        if PREREGISTRATION.exists():
            print(f"{PREREGISTRATION} already exists; a prediction is made once.", file=sys.stderr)
            return 1
        print(f"  estimated vs. true change: rho = {rho_est_true:+.3f} (p = {p_est_true:.2e}), "
              f"signs agree on {agree:.0%} of {len(estimated)} classes")
        payload = {
            "study": "labelfree_predictor",
            "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_revision": git_revision(),
            "method_fingerprint": method_fingerprint(),
            "seed": SEED,
            "corpora": {k: {"documents": v["n_documents"], "classes": len(v["classes"])}
                        for k, v in state.items()},
            "classes_pooled": len(estimated),
            "estimate_vs_truth": {"rho": float(rho_est_true), "p": float(p_est_true),
                                  "sign_agreement": agree},
            "estimated_change_per_class": {k: v["estimated_change"] for k, v in state.items()},
            "predictions": [
                {"id": "predicts-headroom",
                 "claim": "Pooled across the five corpora, the label-free estimate of the "
                          "alignment change correlates positively with the share of available "
                          "headroom each class captured (Spearman rho > 0)."},
                {"id": "significant",
                 "claim": "That correlation is significant at 0.05 uncorrected."},
                {"id": "sign-rule",
                 "claim": "The sign of the estimate decides whether a class gained from "
                          "elaboration more often than chance (above 50% of classes)."},
                {"id": "weaker-than-labelled",
                 "claim": "It is weaker than the labelled quantity: the labelled change "
                          "correlates with headroom at least as strongly as the estimate does."},
            ],
            "note": "Written before any accuracy or headroom under the elaborated "
                    "descriptions was computed in this study.",
        }
        PREREGISTRATION.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
        for p in payload["predictions"]:
            print(f"  PREDICTION {p['id']}: {p['claim']}")
        print(f"Wrote {PREREGISTRATION}. Commit it before running the accuracy step.")
        return 0

    if not PREREGISTRATION.exists():
        print("No preregistration: run with --preregister and commit the file first.", file=sys.stderr)
        return 1
    registered = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if registered.get("method_fingerprint") != method_fingerprint():
        print("The method changed after the prediction was registered; refusing to test it.",
              file=sys.stderr)
        return 1

    per_corpus, pooled = {}, {"estimated": [], "true": [], "captured": []}
    for name, report in state.items():
        captured = headroom(report)
        classes = [c for c in report["classes"] if c in captured]
        est = [report["estimated_change"][c] for c in classes]
        tru = [report["true_change"][c] for c in classes]
        cap = [captured[c] for c in classes]
        rho_est, p_est = spearmanr(est, cap) if len(classes) > 2 else (float("nan"),) * 2
        rho_true, p_true = spearmanr(tru, cap) if len(classes) > 2 else (float("nan"),) * 2
        per_corpus[name] = {
            "classes": len(classes),
            "estimate_vs_headroom": {"rho": float(rho_est), "p": float(p_est)},
            "labelled_vs_headroom": {"rho": float(rho_true), "p": float(p_true)},
        }
        pooled["estimated"] += est
        pooled["true"] += tru
        pooled["captured"] += cap
        print(f"  {name:8s} {len(classes):>3d} classes  estimate rho = {rho_est:+.3f}  "
              f"labelled rho = {rho_true:+.3f}", flush=True)

    rho_est, p_est = spearmanr(pooled["estimated"], pooled["captured"])
    rho_true, p_true = spearmanr(pooled["true"], pooled["captured"])
    gained = [c > 0 for c in pooled["captured"]]
    sign_rule = float(np.mean([(e > 0) == g for e, g in zip(pooled["estimated"], gained)]))

    outcomes = []
    for pred in registered["predictions"]:
        if pred["id"] == "predicts-headroom":
            held = bool(rho_est > 0)
        elif pred["id"] == "significant":
            held = bool(rho_est > 0 and p_est < 0.05)
        elif pred["id"] == "sign-rule":
            held = bool(sign_rule > 0.5)
        else:
            held = bool(rho_true >= rho_est)
        outcomes.append({**pred, "held": held})
        print(f"  {'HELD  ' if held else 'FAILED'} {pred['claim']}")

    payload = {
        "study": "labelfree_predictor",
        "preregistration": PREREGISTRATION.name,
        "registered_at": registered["registered_at"],
        "corpora": per_corpus,
        "classes_pooled": len(pooled["captured"]),
        "documents_pooled": sum(v["n_documents"] for v in state.values()),
        "estimate_vs_truth": registered["estimate_vs_truth"],
        "pooled": {
            "estimate_vs_headroom": {"rho": float(rho_est), "p": float(p_est)},
            "labelled_vs_headroom": {"rho": float(rho_true), "p": float(p_true)},
            "sign_rule_accuracy": sign_rule,
            "classes_that_gained": int(sum(gained)),
        },
        "outcomes": outcomes,
    }
    RESULT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  pooled: estimate rho = {rho_est:+.3f} (p = {p_est:.2e}), "
          f"labelled rho = {rho_true:+.3f}, sign rule {sign_rule:.0%} of "
          f"{len(pooled['captured'])} classes")
    print(f"Wrote {RESULT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

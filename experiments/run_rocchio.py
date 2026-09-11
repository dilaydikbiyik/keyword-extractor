#!/usr/bin/env python3
"""Move a class vector toward its documents without labels, and see if it helps.

    python -m experiments.run_rocchio --preregister   # alignment only, then commit
    python -m experiments.run_rocchio                 # accuracy, against the file

The paper's account is that a description helps in proportion to how far it
moves the class vector toward the documents it must attract. That is a
correlation across hand-written descriptions. If it is the mechanism, moving the
vector there directly should help in the same proportion, with no description
and no label at all: a Rocchio update toward the nearest documents of an
unlabelled pool,

    v' = normalise(v + BETA * mean of the K pool documents nearest to v),

from the terse starting point of each corpus (the literal German control for
NACE, the readable class name for 20 Newsgroups and Reuters). K and BETA are
fixed here, not tuned. The evaluation documents are removed from every pool.

The protocol is two-step. ``--preregister`` measures only the alignment change
(with gold labels, as the paper measures alignment) and writes the predictions
it implies to ``results/rocchio_preregistration.json``; it refuses to overwrite
an existing file. That file is committed before the second step computes any
accuracy, so the repository's history shows which came first.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import spearmanr

from experiments.config import RESULTS_DIR, ROOT, SEED, ensure_dirs, set_seed
from experiments.data import load_labeled_samples
from experiments.metrics import mcnemar_exact
from experiments.run_description_study import LITERAL_GERMAN
from experiments.run_replication import READABLE as NEWS_READABLE, load_newsgroups
from experiments.run_reuters import READABLE as REUTERS_READABLE, load as load_reuters
from experiments.systems import get_embedder

K = 25
BETA = 1.0
PREREGISTRATION = RESULTS_DIR / "rocchio_preregistration.json"
RESULT = RESULTS_DIR / "rocchio.json"


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def rocchio(vectors: np.ndarray, pool: np.ndarray, k: int = K, beta: float = BETA) -> np.ndarray:
    """Each unit class vector moved toward the mean of its k nearest pool rows."""
    nearest = np.argsort(-(pool @ vectors.T), axis=0)[:k]
    return unit_rows(vectors + beta * pool[nearest].mean(axis=0))


def alignment(vectors: np.ndarray, docs: np.ndarray, gold: Sequence[str], names: Sequence[str]) -> np.ndarray:
    """cos(class vector, centroid of that class's own documents), per class."""
    out = []
    for i, name in enumerate(names):
        centroid = docs[[j for j, g in enumerate(gold) if g == name]].mean(axis=0)
        out.append(float(vectors[i] @ (centroid / np.linalg.norm(centroid))))
    return np.array(out)


def encode(texts: Sequence[str]) -> np.ndarray:
    """One text at a time, exactly as the published studies embedded them."""
    embedder = get_embedder()
    return unit_rows(np.vstack([np.asarray(embedder.embed_text(t), dtype=np.float64) for t in texts]))


def encode_pool(texts: Sequence[str]) -> np.ndarray:
    """In batches: the pools run to thousands of documents and are only searched."""
    return unit_rows(np.vstack([np.asarray(v, dtype=np.float64)
                                for v in get_embedder().embed_texts(list(texts), batch_size=64)]))


def corpora() -> Dict[str, Dict]:
    """Evaluation documents, terse class texts and an unlabelled pool per corpus."""
    old = json.loads((ROOT / "data" / "taxonomy" / "sectors_v1.json").read_text(encoding="utf-8"))["sectors"]
    codes = sorted(old)
    samples = load_labeled_samples()
    evaluated = {s.purpose.strip() for s in samples}
    with open(ROOT / "data" / "raw" / "handelsregister_sample_10k.csv", encoding="utf-8") as fh:
        nace_pool = [r["purpose"].strip() for r in csv.DictReader(fh)
                     if r.get("purpose") and r["purpose"].strip() not in evaluated]

    news_texts, news_labels, news_names = load_newsgroups("test", 2000)
    news_pool = [t for t in load_newsgroups("train")[0] if t not in set(news_texts)]

    reuters_docs = load_reuters(list(REUTERS_READABLE))
    reuters_names = sorted(reuters_docs)
    reuters_flat = [d for c in reuters_names for d in reuters_docs[c]]
    from nltk.corpus import reuters
    held = set(reuters_flat)
    reuters_pool = [t for t in (reuters.raw(f).strip() for f in reuters.fileids()) if len(t) >= 80 and t not in held]

    return {
        "NACE": {"names": codes, "docs": [s.purpose for s in samples], "gold": [s.true_sector for s in samples],
                 "terse": {c: f"{old[c].get('name', '')}. {LITERAL_GERMAN.get(c, old[c].get('description', ''))}"
                           for c in codes},
                 "pool": nace_pool},
        "20NG": {"names": news_names, "docs": news_texts, "gold": news_labels,
                 "terse": {n: NEWS_READABLE[n] for n in news_names}, "pool": news_pool},
        "Reuters": {"names": reuters_names, "docs": reuters_flat,
                    "gold": [c for c in reuters_names for _ in reuters_docs[c]],
                    "terse": {c: REUTERS_READABLE[c] for c in reuters_names}, "pool": reuters_pool},
    }


def vectors_for(corpus: Dict) -> Dict[str, np.ndarray]:
    names = corpus["names"]
    terse = encode([corpus["terse"][n] for n in names])
    return {"terse": terse, "rocchio": rocchio(terse, encode_pool(corpus["pool"]))}


def predictions_from(changes: Dict[str, float]) -> List[Dict]:
    """What the alignment account implies, derived mechanically from the changes."""
    ranked = sorted(changes, key=changes.get, reverse=True)
    out = [{"id": f"direction-{name}",
            "claim": f"On {name}, the Rocchio update {'raises' if d > 0 else 'lowers'} Top-1 accuracy.",
            "corpus": name, "expected_sign": 1 if d > 0 else -1}
           for name, d in changes.items()]
    out.append({"id": "ordering", "claim": "Accuracy gains follow the order of the alignment changes: "
                + " > ".join(ranked) + ".", "order": ranked})
    out.append({"id": "per-class", "claim": "Across all classes, the alignment change correlates positively "
                "with the share of available headroom captured (Spearman rho > 0)."})
    return out


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def preregister() -> int:
    if PREREGISTRATION.exists():
        print(f"{PREREGISTRATION} already exists; a prediction is made once.", file=sys.stderr)
        return 1
    changes, per_class = {}, {}
    for name, corpus in corpora().items():
        vecs = vectors_for(corpus)
        docs = encode(corpus["docs"])
        delta = alignment(vecs["rocchio"], docs, corpus["gold"], corpus["names"]) - \
            alignment(vecs["terse"], docs, corpus["gold"], corpus["names"])
        changes[name] = float(delta.mean())
        per_class[name] = dict(zip(corpus["names"], map(float, delta)))
        print(f"  {name:8s} pool {len(corpus['pool']):6d}  mean alignment change {changes[name]:+.4f}")
    payload = {
        "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_revision": git_revision(),
        "k": K, "beta": BETA, "seed": SEED,
        "mean_alignment_change": changes,
        "alignment_change_per_class": per_class,
        "predictions": predictions_from(changes),
        "note": "Written before any accuracy under the Rocchio vectors was computed.",
    }
    PREREGISTRATION.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for p in payload["predictions"]:
        print(f"  PREDICTION {p['id']}: {p['claim']}")
    print(f"Wrote {PREREGISTRATION}. Commit it before running the accuracy step.")
    return 0


def evaluate() -> int:
    if not PREREGISTRATION.exists():
        print("No preregistration: run with --preregister and commit the file first.", file=sys.stderr)
        return 1
    registered = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    report, pooled = {}, []
    for name, corpus in corpora().items():
        vecs = vectors_for(corpus)
        docs, gold, names = encode(corpus["docs"]), corpus["gold"], corpus["names"]
        hits, recall = {}, {}
        for variant, v in vecs.items():
            predicted = [names[i] for i in np.argmax(docs @ v.T, axis=1)]
            hits[variant] = [g == p for g, p in zip(gold, predicted)]
            recall[variant] = {n: float(np.mean([h for h, g in zip(hits[variant], gold) if g == n])) for n in names}
        test = mcnemar_exact(hits["terse"], hits["rocchio"])
        report[name] = {
            "top1_terse": float(np.mean(hits["terse"])),
            "top1_rocchio": float(np.mean(hits["rocchio"])),
            "gain_pp": 100 * float(np.mean(hits["rocchio"]) - np.mean(hits["terse"])),
            "mcnemar": test,
        }
        for n in names:
            if recall["terse"][n] < 0.999:
                share = (recall["rocchio"][n] - recall["terse"][n]) / (1 - recall["terse"][n])
                pooled.append((registered["alignment_change_per_class"][name][n], share))
        print(f"  {name:8s} Top-1 {report[name]['top1_terse']:.1%} -> {report[name]['top1_rocchio']:.1%}"
              f"  ({report[name]['gain_pp']:+.1f} pp, p = {test['p_value']:.4f})")

    rho, p = spearmanr([a for a, _ in pooled], [s for _, s in pooled])
    outcomes = []
    for pred in registered["predictions"]:
        if pred["id"].startswith("direction-"):
            gain = report[pred["corpus"]]["gain_pp"]
            held = np.sign(gain) == pred["expected_sign"]
        elif pred["id"] == "ordering":
            gains = [report[c]["gain_pp"] for c in pred["order"]]
            held = all(a > b for a, b in zip(gains, gains[1:]))
        else:
            held = bool(rho > 0)
        outcomes.append({**pred, "held": bool(held)})
        print(f"  {'HELD  ' if held else 'FAILED'} {pred['claim']}")
    payload = {"preregistration": PREREGISTRATION.name, "registered_at": registered["registered_at"],
               "k": K, "beta": BETA, "corpora": report,
               "per_class": {"n": len(pooled), "rho": float(rho), "p": float(p)},
               "outcomes": outcomes}
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  per-class rho = {rho:+.3f} (p = {p:.4f}, n = {len(pooled)})\nWrote {RESULT}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", action="store_true")
    args = parser.parse_args()
    set_seed()
    ensure_dirs()
    return preregister() if args.preregister else evaluate()


if __name__ == "__main__":
    sys.exit(main())

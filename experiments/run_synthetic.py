#!/usr/bin/env python3
"""Why alignment is the right quantity, and where the account must break.

    python -m experiments.run_synthetic

The paper's account is empirical: a class text helps in proportion to how far it
moves its class vector toward the centroid of the documents it must attract, and
that relation has now replicated in every corpus tested. It is not derived, and
a reader is entitled to ask why alignment rather than any other distance.

The argument is short. The classifier assigns a document $d$ to
$\\arg\\max_c \\cos(f(d), v_c)$, so what decides the document is the margin
between its own class and the best competitor,
$m(d) = \\cos(f(d), v_{c(d)}) - \\max_{c' \\ne c(d)} \\cos(f(d), v_{c'})$. Write
$f(d) = \\mu_{c(d)} + \\varepsilon$ for the class centroid $\\mu$ and a
document-specific residual. Then moving $v_c$ toward $\\mu_c$ raises the first
term for every document of $c$ by the same amount, up to the residual, which is
what alignment measures -- and lowers the second term for the documents of
whichever classes sit nearest to $c$. The account therefore has to hold per
class and to fail exactly where classes overlap, because there the second effect
is as large as the first.

This simulation checks both halves on data where the truth is known: Gaussian
class clouds, prototypes moved toward their own centroids by a controlled
fraction, with separation swept from well-separated to overlapping. Seeded, pure
numpy, no corpus and no encoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

from experiments.config import RESULTS_DIR, ROOT, SEED, ensure_dirs

RESULT = RESULTS_DIR / "synthetic.json"
CLASSES = 21
DIMENSION = 64
PER_CLASS = 200
STEPS = (0.0, 0.1, 0.2, 0.4, 0.7, 1.0)
SEPARATIONS = (3.0, 1.5, 0.75, 0.4)


def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)


def world(separation: float, rng) -> Dict:
    """Class centroids, their documents, and prototypes that start off-target."""
    centroids = unit(rng.normal(size=(CLASSES, DIMENSION))) * separation
    documents = np.vstack([centroids[k] + rng.normal(size=(PER_CLASS, DIMENSION))
                           for k in range(CLASSES)])
    gold = np.repeat(np.arange(CLASSES), PER_CLASS)
    # A terse class text: pointing roughly at the class, but not at its documents.
    start = centroids + 2.0 * separation * unit(rng.normal(size=(CLASSES, DIMENSION)))
    return {"centroids": centroids, "documents": unit(documents), "gold": gold, "start": start}


def recall(prototypes: np.ndarray, state: Dict) -> np.ndarray:
    predicted = np.argmax(state["documents"] @ unit(prototypes).T, axis=1)
    return np.array([float(np.mean(predicted[state["gold"] == k] == k)) for k in range(CLASSES)])


def alignment(prototypes: np.ndarray, state: Dict) -> np.ndarray:
    observed = unit(np.vstack([state["documents"][state["gold"] == k].mean(axis=0)
                               for k in range(CLASSES)]))
    return np.sum(unit(prototypes) * observed, axis=1)


def one_at_a_time(state: Dict) -> Dict:
    """Move a single prototype onto its own centroid, and see who pays for it.

    This isolates the second term of the margin: the class whose vector moved
    gains, and the classes nearest to it lose, because their documents now sit
    closer to somebody else's prototype. If that is the mechanism, the loss
    should concentrate on the nearest neighbour rather than spread evenly.
    """
    base = recall(state["start"], state)
    similarity = unit(state["centroids"]) @ unit(state["centroids"]).T
    np.fill_diagonal(similarity, -2.0)
    own, neighbour, others, overall = [], [], [], []
    base_accuracy = float(np.mean(base))
    for k in range(CLASSES):
        moved = state["start"].copy()
        moved[k] = state["centroids"][k]
        after = recall(moved, state)
        nearest = int(np.argmax(similarity[k]))
        rest = [j for j in range(CLASSES) if j not in (k, nearest)]
        own.append(after[k] - base[k])
        neighbour.append(after[nearest] - base[nearest])
        others.append(float(np.mean([after[j] - base[j] for j in rest])))
        overall.append(float(np.mean(after)) - base_accuracy)
    return {"own_recall_change": float(np.mean(own)),
            "nearest_neighbour_change": float(np.mean(neighbour)),
            "other_classes_change": float(np.mean(others)),
            "overall_accuracy_change": float(np.mean(overall)),
            "loss_concentrated_on_neighbour": bool(np.mean(neighbour) < np.mean(others))}


def main() -> int:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    ensure_dirs()
    rng = np.random.default_rng(SEED)

    report: List[Dict] = []
    for separation in SEPARATIONS:
        state = world(separation, rng)
        base_recall = recall(state["start"], state)
        base_alignment = alignment(state["start"], state)
        rows = []
        for step in STEPS:
            moved = state["start"] + step * (state["centroids"] - state["start"])
            r, a = recall(moved, state), alignment(moved, state)
            headroom = [(r[k] - base_recall[k]) / (1 - base_recall[k])
                        for k in range(CLASSES) if base_recall[k] < 0.999]
            change = [a[k] - base_alignment[k]
                      for k in range(CLASSES) if base_recall[k] < 0.999]
            rho = float(spearmanr(change, headroom).statistic) if step > 0 and len(change) > 2 \
                else float("nan")
            rows.append({"step": step,
                         "mean_alignment_change": float(np.mean(a - base_alignment)),
                         "accuracy": float(np.mean(r)),
                         "per_class_rho": rho})
            print(f"  separation {separation:4.2f}  step {step:3.1f}  "
                  f"dalign {rows[-1]['mean_alignment_change']:+.3f}  "
                  f"accuracy {rows[-1]['accuracy']:.1%}  rho {rho:+.3f}", flush=True)
        gains = [r["accuracy"] for r in rows]
        single = one_at_a_time(state)
        print(f"  separation {separation:4.2f}  one prototype moved: own "
              f"{single['own_recall_change']:+.3f}, nearest neighbour "
              f"{single['nearest_neighbour_change']:+.3f}, others "
              f"{single['other_classes_change']:+.3f}", flush=True)
        report.append({
            "separation": separation,
            "one_prototype_at_a_time": single,
            "monotone_in_step": all(b >= a - 1e-9 for a, b in zip(gains, gains[1:])),
            "accuracy_gain_pp": 100 * (gains[-1] - gains[0]),
            "per_class_rho_at_full_step": rows[-1]["per_class_rho"],
            "curve": rows,
        })

    payload = {
        "classes": CLASSES, "dimension": DIMENSION, "documents_per_class": PER_CLASS,
        "seed": SEED,
        "argument": "the decision is an argmax over cosines, so what moves a document is the "
                    "margin between its own class vector and the nearest competitor; moving a "
                    "class vector toward its own centroid raises the first term for all of its "
                    "documents and lowers the second for its neighbours' -- hence a per-class "
                    "relation that must weaken as classes overlap",
        "separations": report,
        "note": "A simulation, not a measurement of any corpus; it states what the account "
                "predicts and where it predicts its own failure.",
    }
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("  " + " | ".join(f"sep {r['separation']}: {'monotone' if r['monotone_in_step'] else 'NOT monotone'}"
                            f", rho {r['per_class_rho_at_full_step']:+.2f}" for r in report))
    print(f"Wrote {RESULT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

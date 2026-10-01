#!/usr/bin/env python3
"""Does the description effect survive a change of author?

    python -m experiments.run_description_source --preregister   # commit the prediction
    python -m experiments.run_description_source                 # then test it

The descriptions that carry this paper's main effect were drafted by the same
assistant that produced the silver labels, and that assistant had seen the
corpora. A bias it holds toward one reading of a section would reach both the
labels and the descriptions, and would flatter the descriptions. The experiment
that separates them is to have a different writer write the definitions and
grade it against the same labels.

Three class texts, one encoder, the same documents:

  1. the terse German control: the section's short name, hand-translated,
     which is condition 2 of the description study;
  2. the assistant's definitions, the paper's own;
  3. definitions written by an open instruction-tuned model from a different
     family than the labeller, given the control text and nothing else
     (`experiments/write_descriptions.py`).

The prediction is registered before any accuracy is computed, and it is not a
guess: the alignment account says what a class text buys is alignment with the
documents it has to attract, so the measured alignment changes fix the expected
ordering of the gains. If the independent definitions raise alignment and do not
raise accuracy, the account is wrong here, and that is recorded as a failure.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import spearmanr

from experiments.config import RESULTS_DIR, ROOT, SEED, ensure_dirs, set_seed
from experiments.data import load_labeled_samples
from experiments.metrics import bootstrap_ci, mcnemar_exact
from experiments.run_description_study import LITERAL_GERMAN
from experiments.run_rocchio import alignment, encode, git_revision, unit_rows

PREREGISTRATION = RESULTS_DIR / "description_source_preregistration.json"
RESULT = RESULTS_DIR / "description_source.json"
INDEPENDENT = ROOT / "data" / "taxonomy" / "sectors_qwen.json"
CONTROL = "terse German name"
ASSISTANT = "definitions by the assistant"
WRITER = "definitions by an independent model"


def class_texts() -> Dict[str, Dict[str, str]]:
    """The three conditions, each a class text per section."""
    old = json.loads((ROOT / "data" / "taxonomy" / "sectors_v1.json")
                     .read_text(encoding="utf-8"))["sectors"]
    new = json.loads((ROOT / "data" / "taxonomy" / "sectors.json")
                     .read_text(encoding="utf-8"))["sectors"]
    independent = json.loads(INDEPENDENT.read_text(encoding="utf-8"))["sectors"]
    codes = sorted(old)
    return {
        CONTROL: {c: f"{old[c].get('name', '')}. "
                     f"{LITERAL_GERMAN.get(c, old[c].get('description', ''))}" for c in codes},
        ASSISTANT: {c: f"{new[c].get('name', '')}. {new[c].get('description', '')}" for c in codes},
        WRITER: {c: f"{old[c].get('name', '')}. {independent[c]['description']}" for c in codes},
    }


def vectors(texts: Dict[str, str], codes: Sequence[str]) -> np.ndarray:
    return unit_rows(np.vstack([encode([texts[c]]) for c in codes]))


def predictions_from(changes: Dict[str, float]) -> List[Dict]:
    """What the alignment account implies, derived mechanically from the changes."""
    ranked = sorted(changes, key=changes.get, reverse=True)
    out = [{"id": f"direction-{name}",
            "claim": f"Against the control, {name} {'raises' if d > 0 else 'lowers'} Top-1 accuracy.",
            "condition": name, "expected_sign": 1 if d > 0 else -1}
           for name, d in changes.items()]
    out.append({"id": "ordering",
                "claim": "The gains follow the order of the alignment changes: " + " > ".join(ranked) + ".",
                "order": ranked})
    out.append({"id": "per-class",
                "claim": "For the independent definitions, the per-class alignment change correlates "
                         "positively with the share of available headroom captured (Spearman rho > 0)."})
    return out


def method_fingerprint() -> str:
    """A hash of the class texts and of everything that turns them into a number."""
    parts = [f"SEED={SEED}"]
    parts += [inspect.getsource(f) for f in (class_texts, vectors, alignment, encode, unit_rows)]
    for path in (ROOT / "data" / "taxonomy" / "sectors_v1.json",
                 ROOT / "data" / "taxonomy" / "sectors.json", INDEPENDENT):
        parts.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def measured(samples) -> Dict:
    """Alignment per class for each condition, which needs no accuracy."""
    codes = sorted(json.loads((ROOT / "data" / "taxonomy" / "sectors_v1.json")
                              .read_text(encoding="utf-8"))["sectors"])
    docs = unit_rows(np.vstack([encode([s.purpose]) for s in samples]))
    gold = [s.true_sector for s in samples]
    texts = class_texts()
    per_class = {name: alignment(vectors(t, codes), docs, gold, codes) for name, t in texts.items()}
    return {"codes": codes, "docs": docs, "gold": gold, "texts": texts, "alignment": per_class}


def preregister() -> int:
    if PREREGISTRATION.exists():
        print(f"{PREREGISTRATION} already exists; a prediction is made once.", file=sys.stderr)
        return 1
    state = measured(load_labeled_samples())
    base = state["alignment"][CONTROL]
    per_class = {name: {c: state["alignment"][name][c] - base[c] for c in base}
                 for name in (ASSISTANT, WRITER)}
    changes = {name: float(np.mean(list(v.values()))) for name, v in per_class.items()}
    for name, change in changes.items():
        print(f"  {name:38s} mean alignment change {change:+.4f}")
    payload = {
        "study": "description_source",
        "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_revision": git_revision(),
        "method_fingerprint": method_fingerprint(),
        "seed": SEED,
        "writer": json.loads(INDEPENDENT.read_text(encoding="utf-8"))["written_by"],
        "classes_measured": len(base),
        "mean_alignment_change": changes,
        "alignment_change_per_class": per_class,
        "predictions": predictions_from(changes),
        "note": "Written before any accuracy under the independent definitions was computed.",
    }
    PREREGISTRATION.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    for p in payload["predictions"]:
        print(f"  PREDICTION {p['id']}: {p['claim']}")
    print(f"Wrote {PREREGISTRATION}. Commit it before running the accuracy step.")
    return 0


def evaluate() -> int:
    if not PREREGISTRATION.exists():
        print("No preregistration: run with --preregister and commit the file first.", file=sys.stderr)
        return 1
    registered = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if registered.get("method_fingerprint") != method_fingerprint():
        print("The class texts or the method changed after the prediction was registered; "
              "refusing to test it.", file=sys.stderr)
        return 1

    samples = load_labeled_samples()
    state = measured(samples)
    codes, docs, gold = state["codes"], state["docs"], state["gold"]
    present = [c for c in codes if c in set(gold)]

    hits, report = {}, {}
    for name, texts in state["texts"].items():
        order = np.argsort(-(docs @ vectors(texts, codes).T), axis=1)
        predicted = [codes[order[i, 0]] for i in range(len(gold))]
        hits[name] = [g == p for g, p in zip(gold, predicted)]
        lo, hi = bootstrap_ci(hits[name])
        report[name] = {
            "top1_accuracy": float(np.mean(hits[name])),
            "top1_ci95": [lo, hi],
            "top3_accuracy": float(np.mean([gold[i] in [codes[j] for j in order[i, :3]]
                                            for i in range(len(gold))])),
            "mean_words_per_class": float(np.mean([len(t.split()) for t in texts.values()])),
        }

    recall = {name: {c: float(np.mean([h for h, g in zip(hits[name], gold) if g == c]))
                     for c in present} for name in hits}
    for name in (ASSISTANT, WRITER):
        report[name]["gain_pp"] = 100 * float(np.mean(hits[name]) - np.mean(hits[CONTROL]))
        report[name]["mcnemar_vs_control"] = mcnemar_exact(hits[CONTROL], hits[name])
    report[WRITER]["mcnemar_vs_assistant"] = mcnemar_exact(hits[ASSISTANT], hits[WRITER])

    pooled = [(registered["alignment_change_per_class"][WRITER][c],
               (recall[WRITER][c] - recall[CONTROL][c]) / (1 - recall[CONTROL][c]))
              for c in present if recall[CONTROL][c] < 0.999]
    rho, p_rho = spearmanr([a for a, _ in pooled], [s for _, s in pooled])

    outcomes = []
    for pred in registered["predictions"]:
        if pred["id"].startswith("direction-"):
            held = np.sign(report[pred["condition"]]["gain_pp"]) == pred["expected_sign"]
        elif pred["id"] == "ordering":
            gains = [report[c]["gain_pp"] for c in pred["order"]]
            held = all(a > b for a, b in zip(gains, gains[1:]))
        else:
            held = bool(rho > 0)
        outcomes.append({**pred, "held": bool(held)})
        print(f"  {'HELD  ' if held else 'FAILED'} {pred['claim']}")

    payload = {
        "study": "description_source",
        "preregistration": PREREGISTRATION.name,
        "registered_at": registered["registered_at"],
        "writer": registered["writer"],
        "n": len(samples),
        "conditions": report,
        "mean_alignment_change": registered["mean_alignment_change"],
        "per_class": {"n": len(pooled), "rho": float(rho), "p": float(p_rho)},
        "outcomes": outcomes,
    }
    RESULT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name in state["texts"]:
        r = report[name]
        gain = f"{r['gain_pp']:+.1f} pp" if "gain_pp" in r else "control"
        print(f"  {name:38s} Top-1 {r['top1_accuracy']:.1%}  {gain}")
    print(f"  per-class rho = {rho:+.3f} (p = {p_rho:.4f}, n = {len(pooled)})")
    print(f"Wrote {RESULT}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", action="store_true")
    args = parser.parse_args()
    if not INDEPENDENT.exists():
        print("Write the independent definitions first: python -m experiments.write_descriptions",
              file=sys.stderr)
        return 1
    set_seed()
    ensure_dirs()
    return preregister() if args.preregister else evaluate()


if __name__ == "__main__":
    sys.exit(main())

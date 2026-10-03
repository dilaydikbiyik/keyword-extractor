#!/usr/bin/env python3
"""The paper's own findings, predicted in advance and tested on four times the documents.

    python -m experiments.run_extended_eval --preregister   # commit the prediction
    python -m experiments.run_extended_eval                 # then test it

The German evaluation set is \u007f299\u007f documents, which is what a hand-checked set
costs and what every interval in the paper inherits. It cannot be *replaced*:
every preregistered prediction was committed against exactly those documents.
It can be *extended*, and this tests the paper's two headline findings on
\u007f1200\u007f documents drawn afterwards from the same corpus
(``experiments/build_extended_set.py``), with none of the original documents
among them.

What is registered, before any accuracy on the new documents is computed, is
derived from the published numbers rather than guessed:

1. the content effect -- definitions against the terse German control -- is
   positive on the new documents;
2. it lands within ten points of the published \u007f+26.8\u007f, which is the paper's
   claim restated as a quantity rather than a direction;
3. the language effect stays null: the terse German control does not beat the
   terse original by more than two points;
4. per class, the alignment change still correlates with the share of headroom
   captured.

The new labels come from the same procedure as the old ones -- an assistant
applying ``docs/annotation_guidelines.md`` -- and a second labeller from another
model family labelled the same documents, so their agreement is reported with
every figure. It is lower than the human check on the first set, and the paper
says what that does and does not mean.
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
from sklearn.metrics import f1_score

from experiments.config import DATA_DIR, RESULTS_DIR, ROOT, SEED, ensure_dirs, set_seed
from experiments.metrics import bootstrap_ci, mcnemar_exact
from experiments.run_description_study import LITERAL_GERMAN
from experiments.run_rocchio import alignment, git_revision, unit_rows
from experiments.systems import get_embedder

PREREGISTRATION = RESULTS_DIR / "extended_eval_preregistration.json"
RESULT = RESULTS_DIR / "extended_eval.json"
EXTENDED = DATA_DIR / "evaluation" / "extended_labels.json"
PUBLISHED_CONTENT_EFFECT = 26.8
PUBLISHED_LANGUAGE_EFFECT = -2.3
TOLERANCE = 10.0


def load_extended() -> Tuple[List[str], List[str], Dict]:
    payload = json.loads(EXTENDED.read_text(encoding="utf-8"))
    samples = payload["samples"]
    return ([s["purpose"] for s in samples], [s["true_sector"] for s in samples],
            payload["metadata"])


def conditions(codes: List[str]) -> Dict[str, Dict[str, str]]:
    """The three class texts of the original study, unchanged."""
    old = json.loads((ROOT / "data" / "taxonomy" / "sectors_v1.json")
                     .read_text(encoding="utf-8"))["sectors"]
    new = json.loads((ROOT / "data" / "taxonomy" / "sectors.json")
                     .read_text(encoding="utf-8"))["sectors"]
    return {
        "original (terse, mostly Turkish)": {
            c: f"{old[c].get('name', '')}. {old[c].get('description', '')}" for c in codes},
        "same content, rendered in German": {
            c: f"{old[c].get('name', '')}. "
               f"{LITERAL_GERMAN.get(c, old[c].get('description', ''))}" for c in codes},
        "rewritten as NACE-style definitions": {
            c: f"{new[c].get('name', '')}. {new[c].get('description', '')}" for c in codes},
    }


def embed(texts: List[str], batch: int = 64) -> np.ndarray:
    return unit_rows(np.vstack([np.asarray(v, dtype=np.float64) for v in
                                get_embedder().embed_texts(list(texts), batch_size=batch)]))


def method_fingerprint() -> str:
    parts = [f"SEED={SEED}", f"TOLERANCE={TOLERANCE}",
             hashlib.sha256(EXTENDED.read_bytes()).hexdigest()]
    parts += [inspect.getsource(f) for f in (load_extended, conditions, embed, alignment)]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def measure() -> Dict:
    purposes, gold, metadata = load_extended()
    codes = sorted(json.loads((ROOT / "data" / "taxonomy" / "sectors_v1.json")
                              .read_text(encoding="utf-8"))["sectors"])
    docs = embed(purposes)
    texts = conditions(codes)
    vectors = {name: embed([t[c] for c in codes]) for name, t in texts.items()}
    names = list(texts)
    per_class = {name: alignment(v, docs, gold, codes) for name, v in vectors.items()}
    return {"codes": codes, "docs": docs, "gold": gold, "metadata": metadata,
            "names": names, "vectors": vectors, "per_class": per_class, "n": len(purposes)}


def predictions() -> List[Dict]:
    return [
        {"id": "content-positive",
         "claim": "On the extended set, rewriting the terse German control into definitions "
                  "raises Top-1 accuracy."},
        {"id": "content-magnitude",
         "claim": f"That gain is within {TOLERANCE:.0f} points of the published "
                  f"{PUBLISHED_CONTENT_EFFECT:+.1f}."},
        {"id": "language-null",
         "claim": "Rendering the same terse content in German does not raise Top-1 accuracy by "
                  "more than two points, as on the first set."},
        {"id": "per-class",
         "claim": "Across classes, the alignment change of the rewrite correlates positively "
                  "with the share of available headroom captured."},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preregister", action="store_true")
    args = parser.parse_args()
    if not EXTENDED.exists():
        print("Build the extended set first: python -m experiments.build_extended_set --merge",
              file=sys.stderr)
        return 1
    set_seed()
    ensure_dirs()

    if args.preregister:
        if PREREGISTRATION.exists():
            print(f"{PREREGISTRATION} already exists; a prediction is made once.", file=sys.stderr)
            return 1
        payload = json.loads(EXTENDED.read_text(encoding="utf-8"))["metadata"]
        registered = {
            "study": "extended_eval",
            "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_revision": git_revision(),
            "method_fingerprint": method_fingerprint(),
            "seed": SEED,
            "n": payload["total"],
            "labeller_agreement": payload["labeller_agreement"],
            "published": {"content_effect_pp": PUBLISHED_CONTENT_EFFECT,
                          "language_effect_pp": PUBLISHED_LANGUAGE_EFFECT,
                          "tolerance_pp": TOLERANCE},
            "predictions": predictions(),
            "note": "Written before any accuracy on the extended documents was computed. The "
                    "first evaluation set is untouched; this is a second set, not a replacement.",
        }
        PREREGISTRATION.write_text(json.dumps(registered, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
        for p in registered["predictions"]:
            print(f"  PREDICTION {p['id']}: {p['claim']}")
        print(f"Wrote {PREREGISTRATION}. Commit it before running the accuracy step.")
        return 0

    if not PREREGISTRATION.exists():
        print("No preregistration: run with --preregister and commit the file first.",
              file=sys.stderr)
        return 1
    registered = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if registered.get("method_fingerprint") != method_fingerprint():
        print("The set or the method changed after the prediction was registered; refusing "
              "to test it.", file=sys.stderr)
        return 1

    state = measure()
    codes, docs, gold, names = state["codes"], state["docs"], state["gold"], state["names"]
    print(f"{state['n']} documents, {len(set(gold))} sections present", flush=True)

    results, hits = {}, {}
    for name in names:
        order = np.argsort(-(docs @ state["vectors"][name].T), axis=1)
        predicted = [codes[order[i, 0]] for i in range(len(gold))]
        hits[name] = [g == p for g, p in zip(gold, predicted)]
        lo, hi = bootstrap_ci(hits[name])
        results[name] = {
            "top1_accuracy": float(np.mean(hits[name])),
            "top1_ci95": [lo, hi],
            "top3_accuracy": float(np.mean([gold[i] in [codes[j] for j in order[i, :3]]
                                            for i in range(len(gold))])),
            "f1_macro": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        }
        print(f"  {name:<40} {results[name]['top1_accuracy']:.1%}", flush=True)

    language = mcnemar_exact(hits[names[1]], hits[names[0]])
    language["gain_pp"] = 100 * (results[names[1]]["top1_accuracy"]
                                 - results[names[0]]["top1_accuracy"])
    content = mcnemar_exact(hits[names[2]], hits[names[1]])
    content["gain_pp"] = 100 * (results[names[2]]["top1_accuracy"]
                                - results[names[1]]["top1_accuracy"])

    present = [c for c in codes if c in set(gold)]
    recall = {name: {c: float(np.mean([h for h, g in zip(hits[name], gold) if g == c]))
                     for c in present} for name in names}
    change = {c: state["per_class"][names[2]][c] - state["per_class"][names[1]][c]
              for c in present if c in state["per_class"][names[1]]}
    pooled = [(change[c], (recall[names[2]][c] - recall[names[1]][c]) / (1 - recall[names[1]][c]))
              for c in change if recall[names[1]][c] < 0.999]
    rho, p_rho = spearmanr([a for a, _ in pooled], [s for _, s in pooled])

    outcomes = []
    for pred in registered["predictions"]:
        if pred["id"] == "content-positive":
            held = content["gain_pp"] > 0
        elif pred["id"] == "content-magnitude":
            held = abs(content["gain_pp"] - PUBLISHED_CONTENT_EFFECT) <= TOLERANCE
        elif pred["id"] == "language-null":
            held = language["gain_pp"] <= 2.0
        else:
            held = bool(rho > 0)
        outcomes.append({**pred, "held": bool(held)})
        print(f"  {'HELD  ' if held else 'FAILED'} {pred['claim']}")

    payload = {
        "study": "extended_eval",
        "preregistration": PREREGISTRATION.name,
        "registered_at": registered["registered_at"],
        "n": state["n"],
        "first_set_n": 299,
        "labeller_agreement": state["metadata"]["labeller_agreement"],
        "conditions": results,
        "language_effect": language,
        "content_effect": content,
        "published": registered["published"],
        "per_class": {"n": len(pooled), "rho": float(rho), "p": float(p_rho)},
        "outcomes": outcomes,
    }
    RESULT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  language {language['gain_pp']:+.1f} pp, content {content['gain_pp']:+.1f} pp, "
          f"per-class rho = {rho:+.3f} (p = {p_rho:.4f}, n = {len(pooled)})")
    print(f"Wrote {RESULT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Two robustness checks a reviewer asks for first.

    python -m experiments.robustness

1. **The labels.** The evaluation labels are model-assisted; only the 50
   documents marked ``human_verified`` carry an answer a person checked. Do
   the conclusions hold on those 50 alone? They were drawn half at random and
   half from the labeller's low-confidence cases, so their absolute accuracy
   is expected to be lower; what matters is whether the ordering and the
   paired comparisons survive.
2. **The mechanism.** The central correlation between alignment change and
   headroom captured is rho = +0.513 over 32 classes. How wide is its
   bootstrap interval, what does a permutation test say, and does any single
   class carry it?

Reads the committed predictions and per-class results; runs no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.config import RESULTS_DIR, ROOT, SEED
from experiments.metrics import bootstrap_ci, mcnemar_exact

LABELS = ROOT / "data" / "evaluation" / "human_labels.json"
N_PERMUTATIONS = 10_000


def predictions(results_dir: Path) -> dict:
    """Per-system predictions, baselines and ablations merged into one map."""
    merged = {}
    for name in ("baselines_predictions", "ablation_predictions"):
        merged.update(json.loads((results_dir / f"{name}.json").read_text(encoding="utf-8")))
    return merged


def label_check(preds: dict, samples: list) -> dict:
    """Per-system accuracy on the human-verified documents and on the rest."""
    labels = {s["id"]: s["true_sector"] for s in samples}
    verified = {s["id"] for s in samples if s.get("annotation_method") == "human_verified"}
    mismatched = sum(d["true"] != labels.get(d["id"]) for d in preds["full"])
    if mismatched:
        raise SystemExit(f"{mismatched} predictions disagree with the current labels; "
                         "re-run `make reproduce` before this check.")

    def correct(key, ids, k=1):
        by_id = {d["id"]: d for d in preds[key]}
        return [by_id[i]["true"] in (by_id[i]["top3"][:k] if k > 1 else [by_id[i]["predicted"]])
                for i in sorted(ids)]

    rest = set(labels) - verified
    full_verified = correct("full", verified)
    systems = {}
    for key in preds:
        on_verified = correct(key, verified)
        lo, hi = bootstrap_ci(on_verified)
        entry = {
            "top1_verified": float(np.mean(on_verified)),
            "top1_verified_ci95": [lo, hi],
            "top3_verified": float(np.mean(correct(key, verified, k=3))),
            "top1_rest": float(np.mean(correct(key, rest))),
        }
        if key != "full":
            entry["vs_full_on_verified"] = mcnemar_exact(full_verified, on_verified)
        systems[key] = entry
    return {"n_verified": len(verified), "n_rest": len(rest), "systems": systems}


def mechanism_check(search: dict, seed: int = SEED) -> dict:
    """Bootstrap interval, permutation test and leave-one-out range for rho."""
    rows = [r for r in search["classes"] if r["recall_terse"] < 0.999]
    x = np.array([r["alignment_gain"] for r in rows])
    y = np.array([(r["recall_rich"] - r["recall_terse"]) / (1 - r["recall_terse"]) for r in rows])
    rho = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)

    boot = []
    for _ in range(N_PERMUTATIONS):
        idx = rng.integers(0, len(x), len(x))
        if len(set(x[idx])) > 1 and len(set(y[idx])) > 1:
            boot.append(spearmanr(x[idx], y[idx])[0])
    perm = np.array([spearmanr(x, rng.permutation(y))[0] for _ in range(N_PERMUTATIONS)])
    loo = [float(spearmanr(np.delete(x, i), np.delete(y, i))[0]) for i in range(len(x))]
    return {
        "n_classes": len(rows),
        "rho": rho,
        "rho_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "permutation_p": float((np.sum(np.abs(perm) >= abs(rho)) + 1) / (len(perm) + 1)),
        "leave_one_out_min": min(loo),
        "leave_one_out_max": max(loo),
        "class_whose_removal_weakens_most": rows[int(np.argmin(loo))]["class"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR),
                        help="Where to read predictions and per-class results from.")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    samples = json.loads(LABELS.read_text(encoding="utf-8"))["samples"]
    search = json.loads((results_dir / "predictor_search.json").read_text(encoding="utf-8"))
    report = {
        "labels": label_check(predictions(results_dir), samples),
        "mechanism": mechanism_check(search),
        "note": "The verified documents are half random, half low-confidence cases, "
                "so their absolute accuracy is not an estimate for the whole set.",
    }
    (RESULTS_DIR / "robustness.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lab = report["labels"]
    print(f"Human-verified documents: {lab['n_verified']} (rest: {lab['n_rest']})")
    print(f"  {'system':24s}{'top1 verified':>15s}{'95% CI':>18s}{'top1 rest':>11s}{'p vs full':>11s}")
    for key, e in sorted(lab["systems"].items(), key=lambda kv: -kv[1]["top1_verified"]):
        p = e.get("vs_full_on_verified", {}).get("p_value")
        ci = e["top1_verified_ci95"]
        print(f"  {key:24s}{e['top1_verified']:>15.1%}   [{ci[0]:.1%}, {ci[1]:.1%}]"
              f"{e['top1_rest']:>11.1%}{'' if p is None else f'{p:>11.4f}'}")
    m = report["mechanism"]
    print(f"\nrho = {m['rho']:+.3f}  95% CI [{m['rho_ci95'][0]:+.3f}, {m['rho_ci95'][1]:+.3f}]"
          f"  permutation p = {m['permutation_p']:.4f}")
    print(f"leave-one-out range [{m['leave_one_out_min']:+.3f}, {m['leave_one_out_max']:+.3f}]"
          f"  (weakest without class {m['class_whose_removal_weakens_most']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

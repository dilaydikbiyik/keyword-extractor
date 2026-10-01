#!/usr/bin/env python3
"""Is the label-free update's behaviour a property of the method or of one setting?

    python -m experiments.rocchio_sensitivity

The paper reports the Rocchio update at one untuned setting, K = 25 neighbours
and beta = 1, chosen before any result and never changed: tuning it on the
evaluation labels would have made a label-free method depend on labels. That
leaves a fair question open -- whether what the paper reports is the method's
behaviour or an accident of two numbers.

This sweeps both over a grid on all three corpora and reports the gain in every
cell. It is a sensitivity analysis of an effect already reported, not a new
finding and not a search for a better setting: no cell is tested for
significance, nothing here enters the corrected family, and the headline setting
stays the registered one. What the grid can show is whether the sign of the gain
and the ordering across corpora hold away from that one point.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from experiments.config import CORPUS_CSV, RESULTS_DIR, ensure_dirs, set_seed
from experiments.run_rocchio import BETA, K, corpora, encode, encode_pool, rocchio, unit_rows

RESULT = RESULTS_DIR / "rocchio_sensitivity.json"
NEIGHBOURS = [5, 10, 25, 50, 100]
BETAS = [0.25, 0.5, 1.0, 2.0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="K and beta at three values each.")
    args = parser.parse_args()
    if not CORPUS_CSV.exists():
        print("The label-free studies need the raw trade register sample in data/raw/ for their "
              "unlabelled NACE pool, and it is not redistributed; see data/README.md.",
              file=sys.stderr)
        return 1
    set_seed()
    ensure_dirs()
    neighbours = NEIGHBOURS[1::2] if args.quick else NEIGHBOURS
    betas = BETAS[1::2] if args.quick else BETAS

    report = {}
    for name, corpus in corpora().items():
        names = corpus["names"]
        terse = encode([corpus["terse"][n] for n in names])
        docs = encode(corpus["docs"])
        pool = encode_pool(corpus["pool"])
        gold = corpus["gold"]
        baseline = float(np.mean([g == names[i] for g, i in
                                  zip(gold, np.argmax(docs @ terse.T, axis=1))]))
        grid = {}
        for k in neighbours:
            for beta in betas:
                moved = unit_rows(rocchio(terse, pool, k=k, beta=beta))
                accuracy = float(np.mean([g == names[i] for g, i in
                                          zip(gold, np.argmax(docs @ moved.T, axis=1))]))
                grid[f"k={k},beta={beta}"] = {"top1": accuracy,
                                              "gain_pp": 100 * (accuracy - baseline)}
        gains = [cell["gain_pp"] for cell in grid.values()]
        report[name] = {
            "top1_terse": baseline,
            "registered_cell": f"k={K},beta={BETA}",
            "gain_pp_registered": grid[f"k={K},beta={float(BETA)}"]["gain_pp"],
            "gain_pp_min": min(gains), "gain_pp_max": max(gains),
            "gain_pp_median": float(np.median(gains)),
            "cells_with_the_registered_sign": sum(
                1 for g in gains if np.sign(g) == np.sign(grid[f"k={K},beta={float(BETA)}"]["gain_pp"])),
            "cells": len(gains),
            "grid": grid,
        }
        r = report[name]
        print(f"  {name:8s} terse {baseline:.1%}  registered {r['gain_pp_registered']:+.1f} pp  "
              f"grid [{r['gain_pp_min']:+.1f}, {r['gain_pp_max']:+.1f}] pp  "
              f"same sign in {r['cells_with_the_registered_sign']}/{r['cells']} cells")

    payload = {
        "grid": {"k": neighbours, "beta": betas},
        "registered": {"k": K, "beta": BETA},
        "note": "A sensitivity analysis of the reported effect. No significance test is run on "
                "any cell and none of this enters the corrected family; the paper's setting "
                "remains the registered one.",
        "corpora": report,
    }
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {RESULT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

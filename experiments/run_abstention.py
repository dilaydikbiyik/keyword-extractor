#!/usr/bin/env python3
"""When should the classifier say "ask a human"?

    python -m experiments.run_abstention

Nearly half the coded errors are documents where NACE admits more than one
defensible answer, and the system is meant to propose candidates to a human
coder rather than to decide alone. A tool like that needs a rule for declining,
and the rule's cost has to be a number: how much of the work can it hand back,
and what does accuracy become on what it keeps?

The margin the decision already produces answers it. For a document, the margin
is the gap between the cosine of its chosen section and the next one; abstaining
below a threshold trades coverage for accuracy. Nothing is trained and no model
runs: this reads the saved per-document scores of the system the paper reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

import numpy as np

from experiments.config import RESULTS_DIR, ROOT, ensure_dirs

RESULT = RESULTS_DIR / "abstention.json"
PREDICTIONS = RESULTS_DIR / "baselines_predictions.json"
COVERAGES = (1.0, 0.9, 0.75, 0.5, 0.25)
TARGETS = (0.7, 0.8, 0.9)


def rows(system: str = "full") -> List[Dict]:
    payload = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    if system not in payload:
        raise SystemExit(f"{system} not in {PREDICTIONS.name}: {', '.join(payload)}")
    out = []
    for r in payload[system]:
        scores = r.get("scores") or []
        if len(scores) < 2:
            continue
        out.append({"id": r["id"], "correct": r["true"] == r["predicted"],
                    "in_top3": r["true"] in r.get("top3", []),
                    "margin": float(scores[0]) - float(scores[1])})
    return out


def curve(data: List[Dict]) -> List[Dict]:
    """Accuracy on what is kept, at each coverage, keeping the most confident."""
    ranked = sorted(data, key=lambda r: -r["margin"])
    out = []
    for coverage in COVERAGES:
        keep = ranked[:max(1, round(coverage * len(ranked)))]
        out.append({
            "coverage": len(keep) / len(ranked),
            "n_kept": len(keep),
            "margin_threshold": keep[-1]["margin"],
            "top1_on_kept": float(np.mean([r["correct"] for r in keep])),
            "top3_on_kept": float(np.mean([r["in_top3"] for r in keep])),
        })
    return out


def reach(data: List[Dict], target: float) -> Dict:
    """The largest coverage whose Top-1 accuracy still reaches a target."""
    ranked = sorted(data, key=lambda r: -r["margin"])
    best = None
    for n in range(1, len(ranked) + 1):
        accuracy = float(np.mean([r["correct"] for r in ranked[:n]]))
        if accuracy >= target:
            best = {"target": target, "coverage": n / len(ranked), "n_kept": n,
                    "margin_threshold": ranked[n - 1]["margin"], "top1_on_kept": accuracy}
    return best or {"target": target, "coverage": 0.0, "reachable": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", default="full")
    args = parser.parse_args()
    ensure_dirs()
    if not PREDICTIONS.exists():
        print("Run the baselines first: make reproduce", file=sys.stderr)
        return 1

    data = rows(args.system)
    points = curve(data)
    targets = [reach(data, t) for t in TARGETS]
    payload = {
        "system": args.system,
        "n": len(data),
        "rule": "abstain when the margin between the top two sections falls below a threshold; "
                "the margin is already computed by the decision, so nothing is trained",
        "coverage_curve": points,
        "coverage_at_target_accuracy": targets,
        "baseline_top1": points[0]["top1_on_kept"],
        "baseline_top3": points[0]["top3_on_kept"],
    }
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{len(data)} documents, system {args.system}")
    for p in points:
        print(f"  coverage {p['coverage']:5.0%}  margin >= {p['margin_threshold']:.3f}  "
              f"Top-1 {p['top1_on_kept']:.1%}  Top-3 {p['top3_on_kept']:.1%}")
    for t in targets:
        if t.get("reachable", True):
            print(f"  Top-1 {t['target']:.0%} reached at coverage {t['coverage']:.0%} "
                  f"({t['n_kept']} documents, margin >= {t['margin_threshold']:.3f})")
        else:
            print(f"  Top-1 {t['target']:.0%} is not reachable at any coverage")
    print(f"Wrote {RESULT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

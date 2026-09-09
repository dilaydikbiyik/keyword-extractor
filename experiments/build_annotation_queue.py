#!/usr/bin/env python3
"""Build a stratified annotation queue so the evaluation set can grow.

    python -m experiments.build_annotation_queue --target 300

Thirty labelled documents give a 95% CI roughly ±14 points wide, which is
wider than the gap between this pipeline and its baselines — so the current
evaluation set cannot support a comparative claim no matter how the numbers
fall.  This script picks which documents to label next.

Selection is stratified by *predicted* sector (never by a gold label, which
does not exist yet) and split within each sector between confident and
low-margin decisions, so the resulting set is neither a random dump nor a
pile of hard cases mined to look bad.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

from experiments.config import RESULTS_DIR, SEED, ensure_dirs, set_seed
from experiments.data import load_corpus_frame, load_labeled_samples, sector_names
from experiments.systems import EmbeddingRanker

QUEUE_CSV = RESULTS_DIR / "annotation_queue.csv"
QUEUE_META = RESULTS_DIR / "annotation_queue.json"


def ci_halfwidth(p: float, n: int) -> float:
    """Normal-approximation half-width of a 95% CI, for the sizing table."""
    return 1.96 * math.sqrt(p * (1 - p) / n)


def sizing_table(p: float = 0.8, sizes=(30, 100, 200, 300, 500)) -> List[Dict]:
    return [{"n": n, "ci_halfwidth_pp": round(ci_halfwidth(p, n) * 100, 1)} for n in sizes]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=300, help="Documents to queue.")
    parser.add_argument(
        "--pool", type=int, default=1500, help="Corpus documents to score for selection."
    )
    parser.add_argument(
        "--min-per-sector", type=int, default=5, help="Floor per predicted sector."
    )
    args = parser.parse_args()

    set_seed()
    ensure_dirs()

    df = load_corpus_frame()
    already = {s.purpose.strip() for s in load_labeled_samples()}

    rows = [
        {"row": int(i), "legal_name": str(r.get("legal_name", "")), "purpose": str(r["purpose"])}
        for i, r in df.iterrows()
        if isinstance(r.get("purpose"), str) and r["purpose"].strip()
        and r["purpose"].strip() not in already
    ]
    rng = np.random.default_rng(SEED)
    if len(rows) > args.pool:
        rows = [rows[i] for i in rng.choice(len(rows), size=args.pool, replace=False)]

    print(f"Scoring {len(rows)} candidate documents…", flush=True)
    ranker = EmbeddingRanker()
    for i, row in enumerate(rows):
        if i % 250 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
        ranked = ranker.rank(row["purpose"])
        row["predicted_sector"] = ranked[0][0]
        row["top3"] = [c for c, _ in ranked[:3]]
        row["margin"] = float(ranked[0][1] - ranked[1][1])

    by_sector: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_sector[row["predicted_sector"]].append(row)

    # Proportional quota with a floor, so rare sections are still represented.
    n_sectors = len(by_sector)
    floor = min(args.min_per_sector, args.target // max(n_sectors, 1))
    remaining = args.target - floor * n_sectors
    total = len(rows)
    quotas = {
        code: floor + int(round(remaining * len(items) / total))
        for code, items in by_sector.items()
    }

    selected: List[dict] = []
    for code, items in sorted(by_sector.items()):
        quota = min(quotas[code], len(items))
        if quota <= 0:
            continue
        ordered = sorted(items, key=lambda r: r["margin"])
        n_uncertain = quota // 2
        for row in ordered[:n_uncertain]:  # tightest decisions
            row["stratum"] = "low_margin"
            selected.append(row)
        rest = ordered[n_uncertain:]
        if rest and quota - n_uncertain > 0:
            idx = rng.choice(len(rest), size=min(quota - n_uncertain, len(rest)), replace=False)
            for i in idx:
                rest[i]["stratum"] = "representative"
                selected.append(rest[i])

    selected.sort(key=lambda r: (r["predicted_sector"], -r["margin"]))
    names = sector_names()

    with open(QUEUE_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "queue_id", "corpus_row", "legal_name", "purpose",
                "predicted_sector", "predicted_sector_name", "top3", "margin",
                "stratum", "true_sector", "keywords_ground_truth", "annotator_note",
            ]
        )
        for qid, row in enumerate(selected):
            writer.writerow(
                [
                    qid, row["row"], row["legal_name"], row["purpose"],
                    row["predicted_sector"], names.get(row["predicted_sector"], ""),
                    "|".join(row["top3"]), f"{row['margin']:.4f}", row["stratum"],
                    "", "", "",
                ]
            )

    meta = {
        "seed": SEED,
        "pool_size": len(rows),
        "target": args.target,
        "selected": len(selected),
        "per_sector": {c: sum(1 for r in selected if r["predicted_sector"] == c)
                       for c in sorted(by_sector)},
        "strata": {
            s: sum(1 for r in selected if r["stratum"] == s)
            for s in ("low_margin", "representative")
        },
        "ci_sizing_at_p_0.80": sizing_table(),
        "note": (
            "Stratified by PREDICTED sector; no gold label was used in selection. "
            "Fill in true_sector (a NACE section letter A–U) and "
            "keywords_ground_truth (pipe-separated) by hand, then merge with "
            "`python -m experiments.merge_annotations`."
        ),
    }
    QUEUE_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nQueued {len(selected)} documents → {QUEUE_CSV}")
    print("Per predicted sector:", meta["per_sector"])
    print("\n95% CI half-width at 80% accuracy:")
    for row in meta["ci_sizing_at_p_0.80"]:
        print(f"  n = {row['n']:>4}   ±{row['ci_halfwidth_pp']} pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())

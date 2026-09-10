#!/usr/bin/env python3
"""Produce the error analysis for the full system.

    python -m experiments.run_error_analysis

Reads ``results/baselines_predictions.json`` (so it never re-runs the model),
writes a machine-readable report, a confusion summary, and a CSV to annotate
by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from typing import Dict, List

from experiments.config import RESULTS_DIR, ROOT, TABLES_DIR, ensure_dirs, set_seed
from experiments.data import load_labeled_samples, sector_names
from experiments.error_analysis import (
    CODEBOOK,
    MANUAL_TARGET,
    build_error_records,
    carry_manual_coding,
    summarise_manual_coding,
)
from experiments.metrics import confusion_pairs
from experiments.report import markdown_table

PREDICTIONS_FILE = RESULTS_DIR / "baselines_predictions.json"
CSV_FILE = RESULTS_DIR / "error_analysis.csv"
SPLIT_FILE = ROOT / "data" / "evaluation" / "split.json"


def confusion_markdown(predictions: List[Dict]) -> str:
    names = sector_names()
    pairs = confusion_pairs(
        [p["true"] for p in predictions], [p["predicted"] for p in predictions]
    )
    rows = [
        {
            "true": f"{t} — {names.get(t, t)}",
            "pred": f"{p} — {names.get(p, p)}",
            "n": n,
        }
        for t, p, n in pairs
    ]
    return markdown_table(
        rows,
        [("Gold section", "true", str), ("Predicted", "pred", str), ("Errors", "n", str)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", default="full", help="System key to analyse.")
    args = parser.parse_args()

    set_seed()
    ensure_dirs()

    if not PREDICTIONS_FILE.exists():
        print(
            f"{PREDICTIONS_FILE} not found — run "
            "`python -m experiments.run_experiments` first.",
            file=sys.stderr,
        )
        return 1

    all_predictions = json.loads(PREDICTIONS_FILE.read_text(encoding="utf-8"))
    if args.system not in all_predictions:
        print(f"Unknown system '{args.system}'. Have: {list(all_predictions)}", file=sys.stderr)
        return 1
    predictions = all_predictions[args.system]

    samples_by_id = {s.id: s for s in load_labeled_samples()}
    records = build_error_records(predictions, samples_by_id)

    # Read the hand-coded categories before the CSV is rewritten: they are the
    # one part of this report that no code can regenerate.
    previous = []
    if CSV_FILE.exists():
        with open(CSV_FILE, newline="", encoding="utf-8") as fh:
            previous = list(csv.DictReader(fh))
    kept = carry_manual_coding(records, previous)
    dev_ids = set(json.loads(SPLIT_FILE.read_text(encoding="utf-8"))["dev"])
    coding = summarise_manual_coding(records, dev_ids)

    flag_counts = Counter(flag for r in records for flag in r["auto_flags"])
    n_total = len(predictions)

    report = {
        "system": args.system,
        "n_documents": n_total,
        "n_errors": len(records),
        "error_rate": len(records) / n_total if n_total else 0.0,
        "auto_flag_counts": dict(flag_counts.most_common()),
        "codebook": CODEBOOK,
        "manual_coding_complete": bool(coding) and coding["n_coded"] >= MANUAL_TARGET,
        "errors": records,
        "manual_coding": coding,
    }
    (RESULTS_DIR / "error_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "id", "legal_name", "purpose", "text_length", "true_sector",
                "predicted_sector", "top3", "score_top1", "margin_top1_top2",
                "auto_flags", "manual_category", "manual_note",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r["id"], r["legal_name"], r["purpose"], r["text_length"],
                    r["true_sector"], r["predicted_sector"], "|".join(r["top3"]),
                    "" if r["score_top1"] is None else f"{r['score_top1']:.4f}",
                    "" if r["margin_top1_top2"] is None else f"{r['margin_top1_top2']:.4f}",
                    "|".join(r["auto_flags"]), r["manual_category"], r["manual_note"],
                ]
            )

    table = confusion_markdown(predictions)
    (TABLES_DIR / "confusions.md").write_text(table + "\n", encoding="utf-8")

    print(f"System: {args.system}")
    print(f"Errors: {len(records)} / {n_total}")
    print("\nAutomatic flags:")
    for flag, count in flag_counts.most_common():
        print(f"  {flag:24s} {count}")
    print("\n" + table)
    print(f"\nManual coding: {kept['carried']} categories carried over, {kept['dropped']} dropped "
          "because the error changed or disappeared.")
    print(f"Annotate by hand: {CSV_FILE}")
    if len(records) < 50:
        print(
            f"\nNOTE: 50 hand-inspected errors is the target; this set yields "
            f"{len(records)}. Enlarge the labelled set: `make annotate`, label the "
            "queue, then `make merge ARGS=--replace`."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

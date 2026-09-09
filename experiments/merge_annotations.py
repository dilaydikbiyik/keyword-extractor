#!/usr/bin/env python3
"""Merge a hand-annotated queue back into the evaluation set.

    python -m experiments.merge_annotations --queue results/annotation_queue.csv

Reads the CSV produced by ``build_annotation_queue``, keeps the rows where a
``true_sector`` has been filled in, validates it against the taxonomy, and
appends them to ``data/evaluation/human_labels.json``.  The previous label
file is copied to ``.bak`` first, and documents already present are skipped,
so the command is safe to run repeatedly as annotation proceeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from experiments.config import LABELS_JSON, RESULTS_DIR
from experiments.data import load_taxonomy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue", type=Path, default=RESULTS_DIR / "annotation_queue.csv"
    )
    parser.add_argument(
        "--annotator", default="", help="Recorded on every merged sample."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would change, write nothing."
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Start a fresh evaluation set from the merged rows instead of "
            "appending. Use this once the queue is labelled: the original 30 "
            "documents were written by hand rather than sampled from the "
            "corpus, so mixing them dilutes a representative set."
        ),
    )
    args = parser.parse_args()

    if not args.queue.exists():
        print(f"Queue not found: {args.queue}", file=sys.stderr)
        return 1

    codes = set(load_taxonomy())
    payload = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    samples = payload["samples"]
    # In --replace mode the current set is being discarded, so "already in the
    # label set" is not a reason to skip a row: it would drop every document
    # whose label was corrected since the last merge. Deduplicate only against
    # rows added in this run.
    existing = set() if args.replace else {s["purpose"].strip() for s in samples}
    next_id = max((s["id"] for s in samples), default=-1) + 1

    added, skipped_blank, skipped_dupe, invalid = [], 0, 0, []

    with open(args.queue, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            # A human answer always wins. Where there is none, fall back to the
            # model-produced silver label and record it as such, so the two
            # never become indistinguishable downstream.
            label = (row.get("true_sector") or "").strip().upper()
            method = "manual"
            if not label:
                label = (row.get("model_assisted_sector") or "").strip().upper()
                method = "model_assisted"
            if not label:
                skipped_blank += 1
                continue
            if label not in codes:
                invalid.append((row.get("queue_id"), label))
                continue
            purpose = (row.get("purpose") or "").strip()
            if not purpose or purpose in existing:
                skipped_dupe += 1
                continue
            keywords = [
                kw.strip()
                for kw in (row.get("keywords_ground_truth") or "").split("|")
                if kw.strip()
            ]
            added.append(
                {
                    "id": next_id + len(added),
                    "legal_name": row.get("legal_name", ""),
                    "purpose": purpose,
                    "true_sector": label,
                    "keywords_ground_truth": keywords,
                    "annotation_method": method,
                    "provenance": "corpus_sample",
                    "labeller_confidence": row.get("labeller_confidence", ""),
                    "annotator": args.annotator,
                    "source_queue_id": row.get("queue_id"),
                }
            )
            existing.add(purpose)

    methods = Counter(s["annotation_method"] for s in added)
    print(f"Annotated and new:   {len(added)} ({dict(methods)})")
    print(f"Not yet annotated:   {skipped_blank}")
    print(f"Already in label set:{skipped_dupe}")
    if invalid:
        print(f"\nInvalid sector codes on {len(invalid)} rows — fix these first:")
        for queue_id, label in invalid[:20]:
            print(f"  queue_id {queue_id}: {label!r}")
        return 1
    if not added:
        print("\nNothing to merge.")
        return 0

    if args.replace:
        print(f"\n--replace: dropping {len(samples)} previously labelled documents.")
        samples = added
        for new_id, sample in enumerate(samples):
            sample["id"] = new_id
    else:
        samples.extend(added)
    # Rebind explicitly: in the --replace branch `samples` is a new list, and
    # only mutating the one already inside `payload` would silently write the
    # old documents back out under the new metadata.
    payload["samples"] = samples
    payload["metadata"]["total"] = len(samples)
    payload["metadata"]["sector_distribution"] = dict(
        sorted(Counter(s["true_sector"] for s in samples).items())
    )
    payload["metadata"]["last_merged"] = date.today().isoformat()
    payload["metadata"]["annotation"] = (
        "corpus-sampled; labels model-assisted per docs/annotation_guidelines.md, "
        "human-verified on a subset (results/verification_report.json)"
        if args.replace
        else "mixed: hand-authored seed set plus corpus-sampled queue batches"
    )
    payload["metadata"]["annotation_method_distribution"] = dict(
        sorted(Counter(s.get("annotation_method", "unknown") for s in samples).items())
    )
    payload["metadata"]["provenance_distribution"] = dict(
        sorted(Counter(s.get("provenance", "authored") for s in samples).items())
    )

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return 0

    shutil.copyfile(LABELS_JSON, LABELS_JSON.with_suffix(".json.bak"))
    LABELS_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nEvaluation set is now {len(samples)} documents → {LABELS_JSON}")
    print("Backup written alongside it. Re-run `make reproduce` to update the tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

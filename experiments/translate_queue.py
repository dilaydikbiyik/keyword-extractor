#!/usr/bin/env python3
"""Add an English translation column to the annotation queue.

    python -m experiments.translate_queue

Annotation should not require reading German. This translates each queued
business purpose with Helsinki-NLP/opus-mt-de-en and writes it back into the
CSV as `purpose_en`, which the annotation tool shows next to the original.

The translation is an aid for the annotator; the German text stays the source
of truth and is what the pipeline sees.
"""

from __future__ import annotations

import argparse
import csv
import sys

from experiments.config import RESULTS_DIR, set_seed
from experiments.systems import translate_de_en

QUEUE_CSV = RESULTS_DIR / "annotation_queue.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default=str(QUEUE_CSV))
    args = parser.parse_args()

    set_seed()
    path = args.queue
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = list(reader)

    if "purpose_en" not in header:
        header.insert(header.index("purpose") + 1, "purpose_en")

    todo = [r for r in rows if not (r.get("purpose_en") or "").strip()]
    print(f"{len(rows)} rows, {len(todo)} still to translate")

    for i, row in enumerate(todo, 1):
        if i % 25 == 0 or i == 1:
            print(f"  {i}/{len(todo)}", flush=True)
        row["purpose_en"] = translate_de_en(row["purpose"])

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})

    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

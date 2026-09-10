#!/usr/bin/env python3
"""Describe the retired hand-written evaluation set against the real corpus.

    python -m experiments.retired_set

The first evaluation set was 30 documents written by hand to exercise the
pipeline. It was replaced because it was measurably easier than the register
it stood in for, and the paper says by how much. This writes those figures to
``results/retired_eval_set.json`` so the sentence that quotes them is generated,
not typed.

The retired documents are kept in ``data/evaluation/retired_handwritten.json``;
on first run they are recovered from the commit that replaced them. Comparing
against the corpus needs ``data/raw/``, which is not redistributed, so the
output is committed like the other corpus-dependent results.
"""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys

from experiments.config import RESULTS_DIR, ROOT
from experiments.error_analysis import _BOILERPLATE

RETIRED = ROOT / "data" / "evaluation" / "retired_handwritten.json"
CORPUS = ROOT / "data" / "raw" / "handelsregister_sample_10k.csv"
# The last revision at which data/evaluation/human_labels.json held the 30
# hand-written documents; c571a1e replaced them with the corpus sample.
RETIRED_REVISION = "c571a1e^"
RETIRED_HEADLINE_TOP1 = 0.800


def retired_documents() -> list:
    if not RETIRED.exists():
        raw = subprocess.check_output(
            ["git", "show", f"{RETIRED_REVISION}:data/evaluation/human_labels.json"], cwd=ROOT
        )
        payload = json.loads(raw)
        payload["metadata"] = {
            **payload.get("metadata", {}),
            "status": "retired",
            "recovered_from": RETIRED_REVISION,
            "note": "Hand-written by the author to exercise the pipeline. Replaced by a "
                    "corpus sample because it was shorter and cleaner than real entries.",
        }
        RETIRED.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return json.loads(RETIRED.read_text(encoding="utf-8"))["samples"]


def describe(texts: list) -> dict:
    return {
        "n": len(texts),
        "median_chars": float(statistics.median(len(t) for t in texts)),
        "boilerplate_share": sum(bool(_BOILERPLATE.search(t.strip())) for t in texts) / len(texts),
    }


def main() -> int:
    retired = [d["purpose"] for d in retired_documents()]
    if not CORPUS.exists():
        raise SystemExit(f"{CORPUS} is missing; the corpus comparison needs the raw sample.")
    with CORPUS.open(encoding="utf-8") as handle:
        corpus = [r["purpose"] for r in csv.DictReader(handle) if r.get("purpose")]

    payload = {
        "retired": {**describe(retired), "reported_top1": RETIRED_HEADLINE_TOP1},
        "corpus": describe(corpus),
        "boilerplate_detector": "experiments.error_analysis._BOILERPLATE",
        "note": "The retired set's Top-1 is historical: it was measured before the set "
                "was replaced and is kept as the figure the paper retracts.",
    }
    (RESULTS_DIR / "retired_eval_set.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    for name, block in (("retired", payload["retired"]), ("corpus", payload["corpus"])):
        print(f"  {name:8s} n={block['n']:5d}  median {block['median_chars']:.0f} chars  "
              f"boilerplate {block['boilerplate_share']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Split the evaluation set into a development half and a held-out test half.

    python -m experiments.split

Any change motivated by looking at errors — seed lists, thresholds, rules — is
a form of fitting. Without a held-out half there is no way to tell a real
improvement from a tighter fit to the documents that were inspected, and
"we improved the taxonomy after error analysis" is exactly the claim a reviewer
will probe.

The split is stratified by gold section and seeded, so it is stable across runs
and reproducible from the repository alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

from experiments.config import LABELS_JSON, SEED, ensure_dirs
from experiments.data import load_labeled_samples

SPLIT_JSON = LABELS_JSON.parent / "split.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev-fraction", type=float, default=0.5,
        help="Share of documents in the development half (default: 0.5).",
    )
    args = parser.parse_args()

    ensure_dirs()
    samples = load_labeled_samples()
    by_section = defaultdict(list)
    for s in samples:
        by_section[s.true_sector].append(s.id)

    rng = np.random.default_rng(SEED)
    dev, test = [], []
    for section in sorted(by_section):
        ids = sorted(by_section[section])
        rng.shuffle(ids)
        # A section with a single document goes to dev: a held-out class the
        # test half has never seen would report a spurious zero.
        cut = max(1, round(len(ids) * args.dev_fraction)) if len(ids) > 1 else 1
        dev.extend(ids[:cut])
        test.extend(ids[cut:])

    payload = {
        "seed": SEED,
        "dev_fraction": args.dev_fraction,
        "stratified_by": "gold section",
        "note": (
            "Taxonomy edits, thresholds and any other change motivated by "
            "inspecting errors are developed on dev and reported on test. "
            "Test documents are not to be looked at."
        ),
        "dev": sorted(dev),
        "test": sorted(test),
    }
    SPLIT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"dev {len(dev)} · test {len(test)} → {SPLIT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

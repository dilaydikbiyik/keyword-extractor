#!/usr/bin/env python3
"""What a wrong class description costs.

    python -m experiments.run_description_noise

Every comparison in this paper puts a terse description against a good one, so
the reader learns the upside of writing and nothing about the risk. A
practitioner about to rewrite a taxonomy is exposed to both: descriptions that
are confident and wrong, or long and empty.

Four corruptions, each mechanical and seeded, applied to the paper's own
definitions:

* **permuted** -- a share of classes receive another class's definition, which
  is what a mislabelled or mis-pasted taxonomy looks like;
* **nearest-neighbour swapped** -- the swap happens between the classes whose
  names are already most similar, which is the worst case rather than the
  average one;
* **generic** -- every class gets the same style-matched text with no
  class-specific content, which separates "a definition helps" from "any longer
  text helps";
* **name only** -- the floor, for reference.

Not registered, and no part of the corrected family: this is a sensitivity
analysis of advice the paper gives.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

import numpy as np

from experiments.config import DATA_DIR, RESULTS_DIR, ROOT, SEED, ensure_dirs, set_seed
from experiments.run_dose_response import embed, sets

RESULT = RESULTS_DIR / "description_noise.json"
SHARES = (0.1, 0.25, 0.5, 1.0)
GENERIC = ("Unternehmen dieses Abschnitts erbringen Leistungen im Rahmen ihres "
           "Gesellschaftszwecks und betreiben alle damit zusammenhängenden Geschäfte, "
           "soweit dafür keine besondere Genehmigung erforderlich ist.")


def permute(codes: List[str], texts: Dict[str, str], share: float, rng) -> Dict[str, str]:
    """A share of classes receive another affected class's definition."""
    out = dict(texts)
    k = max(2, round(share * len(codes)))
    picked = list(rng.choice(codes, size=k, replace=False))
    shifted = picked[1:] + picked[:1]
    for target, source in zip(picked, shifted):
        out[target] = texts[source]
    return out


def swap_nearest(codes: List[str], texts: Dict[str, str], names: Dict[str, str],
                 share: float) -> Dict[str, str]:
    """Swap definitions between the most similar class names, worst case first."""
    vectors = embed([names[c] for c in codes])
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -2.0)
    pairs, used = [], set()
    order = np.dstack(np.unravel_index(np.argsort(-similarity, axis=None), similarity.shape))[0]
    for i, j in order:
        a, b = codes[int(i)], codes[int(j)]
        if a in used or b in used:
            continue
        pairs.append((a, b))
        used |= {a, b}
        if 2 * len(pairs) >= max(2, round(share * len(codes))):
            break
    out = dict(texts)
    for a, b in pairs:
        out[a], out[b] = texts[b], texts[a]
    return out, [list(p) for p in pairs]


def main() -> int:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    set_seed()
    ensure_dirs()
    taxonomy = json.loads((DATA_DIR / "taxonomy" / "sectors.json")
                          .read_text(encoding="utf-8"))["sectors"]
    codes = sorted(taxonomy)
    names = {c: taxonomy[c].get("name", "") for c in codes}
    clean = {c: f"{names[c]}. {taxonomy[c].get('description', '')}".strip() for c in codes}

    conditions: Dict[str, Dict[str, str]] = {"as written": clean}
    rng = np.random.default_rng(SEED)
    for share in SHARES:
        conditions[f"permuted, {share:.0%} of classes"] = permute(codes, clean, share, rng)
    swapped, pairs = swap_nearest(codes, clean, names, 0.25)
    conditions["nearest-neighbour swapped, 25%"] = swapped
    conditions["generic, every class the same"] = {c: f"{names[c]}. {GENERIC}" for c in codes}
    conditions["name only"] = dict(names)

    report = {}
    for label, data in sets().items():
        docs = embed(data["purposes"])
        gold = data["gold"]
        rows = {}
        for name, texts in conditions.items():
            vectors = embed([texts[c] for c in codes])
            predicted = [codes[i] for i in np.argmax(docs @ vectors.T, axis=1)]
            rows[name] = float(np.mean([g == p for g, p in zip(gold, predicted)]))
            print(f"  {label:26s} {name:34s} {rows[name]:.1%}", flush=True)
        clean_accuracy = rows["as written"]
        report[label] = {
            "n": len(gold),
            "top1_accuracy": rows,
            "cost_pp": {k: 100 * (v - clean_accuracy) for k, v in rows.items() if k != "as written"},
        }

    payload = {
        "seed": SEED,
        "swapped_pairs": pairs,
        "generic_text": GENERIC,
        "note": "A sensitivity analysis of the paper's advice. Not registered; no part of the "
                "corrected family.",
        "sets": report,
    }
    RESULT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {RESULT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

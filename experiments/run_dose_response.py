#!/usr/bin/env python3
"""How much description writing pays, and where it stops paying.

    python -m experiments.run_dose_response

"Write definitions instead of labels" is advice without a budget. The class
texts in this paper run from one word to fifty, and the gains do not scale with
length -- words added correlates negatively with the per-class gain. So the
practical question is not whether to write but how much: truncating each
definition to its first k words and measuring the curve answers it.

Both German sets are evaluated, the hand-checked \u007f299\u007f and the second set of
\u007f1200\u007f. Nothing is registered here: it is a dose-response curve over a factor
already shown to matter, and it enters no corrected family.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

import numpy as np

from experiments.config import DATA_DIR, RESULTS_DIR, ROOT, ensure_dirs, set_seed
from experiments.run_rocchio import unit_rows
from experiments.systems import get_embedder

RESULT = RESULTS_DIR / "dose_response.json"
BUDGETS = (0, 3, 5, 10, 20, 40, None)


def embed(texts: List[str], batch: int = 64) -> np.ndarray:
    return unit_rows(np.vstack([np.asarray(v, dtype=np.float64) for v in
                                get_embedder().embed_texts(list(texts), batch_size=batch)]))


def class_text(name: str, description: str, budget) -> str:
    """The class name, plus the first `budget` words of its definition."""
    if budget == 0:
        return name
    words = description.split()
    return f"{name}. {' '.join(words if budget is None else words[:budget])}".strip()


def sets() -> Dict[str, Dict]:
    """The two German evaluation sets, each as documents and gold sections."""
    from experiments.data import load_labeled_samples

    first = load_labeled_samples()
    out = {"first set (hand-checked)": {"purposes": [s.purpose for s in first],
                                        "gold": [s.true_sector for s in first]}}
    second = DATA_DIR / "evaluation" / "extended_labels.json"
    if second.exists():
        samples = json.loads(second.read_text(encoding="utf-8"))["samples"]
        out["second set (silver)"] = {"purposes": [s["purpose"] for s in samples],
                                      "gold": [s["true_sector"] for s in samples]}
    return out


def main() -> int:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    set_seed()
    ensure_dirs()
    taxonomy = json.loads((DATA_DIR / "taxonomy" / "sectors.json")
                          .read_text(encoding="utf-8"))["sectors"]
    codes = sorted(taxonomy)
    lengths = [len(taxonomy[c].get("description", "").split()) for c in codes]

    report = {}
    for label, data in sets().items():
        docs = embed(data["purposes"])
        gold = data["gold"]
        rows = []
        for budget in BUDGETS:
            texts = [class_text(taxonomy[c].get("name", ""),
                                taxonomy[c].get("description", ""), budget) for c in codes]
            vectors = embed(texts)
            predicted = [codes[i] for i in np.argmax(docs @ vectors.T, axis=1)]
            accuracy = float(np.mean([g == p for g, p in zip(gold, predicted)]))
            rows.append({"budget_words": budget, "top1_accuracy": accuracy,
                         "mean_words_per_class": float(np.mean([len(t.split()) for t in texts]))})
            print(f"  {label:26s} first {('all' if budget is None else budget):>3} words  "
                  f"{accuracy:.1%}", flush=True)
        full = rows[-1]["top1_accuracy"]
        name_only = rows[0]["top1_accuracy"]
        gain = full - name_only
        saturation = next((r["budget_words"] for r in rows
                           if gain > 0 and r["top1_accuracy"] - name_only >= 0.9 * gain), None)
        report[label] = {"n": len(gold), "curve": rows,
                         "gain_full_over_name_only_pp": 100 * gain,
                         "words_for_90pc_of_gain": saturation}

    payload = {
        "budgets": [b if b is not None else "all" for b in BUDGETS],
        "class_text": "the class name, plus the first k words of its definition",
        "definition_length_words": {"median": float(np.median(lengths)),
                                    "min": int(min(lengths)), "max": int(max(lengths))},
        "note": "A dose-response curve over a factor already established; not registered and "
                "not part of the corrected family.",
        "sets": report,
    }
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for label, r in report.items():
        print(f"  {label}: {r['gain_full_over_name_only_pp']:+.1f} pp from the name alone to the "
              f"full definition; 90% of it by {r['words_for_90pc_of_gain']} words")
    print(f"Wrote {RESULT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

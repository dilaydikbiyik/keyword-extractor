#!/usr/bin/env python3
"""The author's blind second coding of the error sample.

    python -m experiments.code_errors --build   # write the blind coding sheet
    python -m experiments.code_errors           # score it once it is filled in

The fifty development-half errors in ``results/error_analysis.csv`` carry a
first coding against the codebook in ``experiments/error_analysis.py``. This
builds the same fifty for the author, blind: in a shuffled order, without the
first coding or the automatic flags, and with an English machine translation
beside the German, as in the label verification. Scoring reports the raw
agreement and Cohen's kappa between the two codings and where they differ; the
paper quotes the result as soon as the report exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from typing import Callable, Dict, List

import numpy as np

from experiments.config import RESULTS_DIR, SEED, ensure_dirs
from experiments.data import sector_names
from experiments.error_analysis import CODEBOOK
from experiments.metrics import cohen_kappa

ERRORS = RESULTS_DIR / "error_analysis.csv"
SHEET = RESULTS_DIR / "author_error_coding.csv"
GUIDE = RESULTS_DIR / "author_error_coding_codebook.md"
REPORT = RESULTS_DIR / "author_error_coding_report.json"
COLUMNS = ["id", "purpose_de", "purpose_en", "gold_section", "predicted_section", "top3_sections",
           "author_category", "author_note"]


def first_coding() -> List[Dict]:
    with open(ERRORS, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["manual_category"].strip()]


def section(code: str, names: Dict[str, str]) -> str:
    return f"{code} ({names.get(code, '?')})"


def guide_text() -> str:
    lines = ["# Coding the errors",
             "",
             "Fill in `author_category` for every row of `author_error_coding.csv` with exactly one",
             "of the category names below, and `author_note` if a row needs a word of explanation.",
             "Read the English translation, the gold section and the predicted section, and ask why",
             "the system chose the predicted one. Code each row on its own; do not look at",
             "`error_analysis.csv` until you are done, or the coding is no longer blind.",
             "",
             "| Category | Use it when |",
             "| --- | --- |"]
    lines += [f"| `{name}` | {text} |" for name, text in CODEBOOK.items()]
    lines += ["", "Then run `make author-coding-score`.", ""]
    return "\n".join(lines)


def build(translate: Callable[[str], str] = None) -> int:
    if SHEET.exists():
        print(f"{SHEET.name} already exists; it may hold your coding, so it is not overwritten.",
              file=sys.stderr)
        return 1
    if translate is None:
        from experiments.systems import translate_de_en as translate
    rows = first_coding()
    names = sector_names()
    order = np.random.default_rng(SEED).permutation(len(rows))
    with open(SHEET, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for i in order:
            r = rows[i]
            writer.writerow({
                "id": r["id"], "purpose_de": r["purpose"], "purpose_en": translate(r["purpose"]),
                "gold_section": section(r["true_sector"], names),
                "predicted_section": section(r["predicted_sector"], names),
                "top3_sections": " | ".join(section(c, names) for c in r["top3"].split("|") if c),
                "author_category": "", "author_note": "",
            })
    GUIDE.write_text(guide_text(), encoding="utf-8")
    print(f"Wrote {SHEET.name} ({len(rows)} rows, shuffled, first coding hidden) and {GUIDE.name}.")
    return 0


def score() -> int:
    if not SHEET.exists():
        print("No coding sheet: run with --build first.", file=sys.stderr)
        return 1
    with open(SHEET, newline="", encoding="utf-8-sig") as fh:
        sheet = list(csv.DictReader(fh))
    invalid = sorted({r["author_category"].strip() for r in sheet
                      if r["author_category"].strip() and r["author_category"].strip() not in CODEBOOK})
    if invalid:
        print(f"Not codebook categories: {invalid}. Use one of {list(CODEBOOK)}.", file=sys.stderr)
        return 1
    filled = [r for r in sheet if r["author_category"].strip()]
    if not filled:
        print(f"No row of {SHEET.name} is coded yet.", file=sys.stderr)
        return 1
    first = {r["id"]: r["manual_category"].strip() for r in first_coding()}
    theirs = [first[r["id"]] for r in filled]
    mine = [r["author_category"].strip() for r in filled]
    agree = [a == b for a, b in zip(theirs, mine)]
    report = {
        "n_coded": len(filled), "n_sheet": len(sheet),
        "agreement": float(np.mean(agree)), "cohen_kappa": cohen_kappa(theirs, mine),
        "author_distribution": dict(Counter(mine).most_common()),
        "first_distribution": dict(Counter(theirs).most_common()),
        "disagreements": [{"first": a, "author": b, "count": n}
                          for (a, b), n in Counter((a, b) for a, b in zip(theirs, mine) if a != b).most_common()],
        "note": "The author coded blind to the first coding, from English machine translations.",
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    kappa = report["cohen_kappa"]
    print(f"  {len(filled)} of {len(sheet)} coded; agreement {report['agreement']:.0%}, "
          f"kappa {'undefined' if kappa is None else f'{kappa:.3f}'}\nWrote {REPORT.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true", help="Write the blind coding sheet.")
    args = parser.parse_args()
    ensure_dirs()
    return build() if args.build else score()


if __name__ == "__main__":
    sys.exit(main())

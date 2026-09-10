#!/usr/bin/env python3
"""Build and score the human verification sample for the silver labels.

    python -m experiments.verify_labels --build     # draw the sample
    python -m experiments.verify_labels             # score it once filled in

The labels in the annotation queue were produced by a language model applying
``docs/annotation_guidelines.md``. They are silver, not gold. This draws a
sample for a human pass and reports the agreement between the two, which is
the figure a paper has to quote when it uses model-assisted annotation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np

from experiments.config import RESULTS_DIR, SEED, ensure_dirs

QUEUE_CSV = RESULTS_DIR / "annotation_queue.csv"
SAMPLE_CSV = RESULTS_DIR / "verification_sample.csv"
REPORT_JSON = RESULTS_DIR / "verification_report.json"
SECOND_CSV = RESULTS_DIR / "second_annotator_sample.csv"
SECOND_REPORT_JSON = RESULTS_DIR / "second_annotator_report.json"

SAMPLE_SIZE = 50


def previously_sampled() -> set:
    """Queue ids already used in an earlier verification pass."""
    used = set()
    for path in (SAMPLE_CSV, RESULTS_DIR / "verification_sample_pilot.csv"):
        if path.exists():
            with open(path, newline="", encoding="utf-8") as fh:
                used.update(r["queue_id"] for r in csv.DictReader(fh))
    return used


def build(size: int = SAMPLE_SIZE, fresh: bool = False) -> int:
    """Draw a verification sample: random draw plus every low-confidence case.

    Both halves matter. A purely random sample under-represents exactly the
    documents where the labeller was unsure, and a purely low-confidence sample
    would overstate the disagreement rate.
    """
    with open(QUEUE_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    labelled = [r for r in rows if r.get("model_assisted_sector")]
    if fresh:
        used = previously_sampled()
        if used:
            # Keep the pilot answers; the new pass is scored on its own file.
            if SAMPLE_CSV.exists():
                SAMPLE_CSV.replace(RESULTS_DIR / "verification_sample_pilot.csv")
            labelled = [r for r in labelled if r["queue_id"] not in used]
            print(f"Excluding {len(used)} documents used in the pilot pass.")
    if not labelled:
        print("The queue has no silver labels yet.", file=sys.stderr)
        return 1

    rng = np.random.default_rng(SEED)
    low = [r for r in labelled if r.get("labeller_confidence") == "low"]
    high = [r for r in labelled if r.get("labeller_confidence") != "low"]

    n_low = min(len(low), size // 2)
    n_high = min(len(high), size - n_low)
    picked = [low[i] for i in rng.choice(len(low), n_low, replace=False)]
    picked += [high[i] for i in rng.choice(len(high), n_high, replace=False)]
    picked.sort(key=lambda r: int(r["queue_id"]))

    header = [
        "queue_id", "legal_name", "purpose_en", "purpose",
        "silver_sector", "labeller_confidence", "top3",
        "true_sector", "annotator_note",
    ]
    with open(SAMPLE_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for r in picked:
            writer.writerow({
                "queue_id": r["queue_id"],
                "legal_name": r["legal_name"],
                "purpose_en": r.get("purpose_en", ""),
                "purpose": r["purpose"],
                "silver_sector": r["model_assisted_sector"],
                "labeller_confidence": r.get("labeller_confidence", ""),
                "top3": r.get("top3", ""),
                "true_sector": "",
                "annotator_note": "",
            })

    print(f"Wrote {SAMPLE_CSV}: {len(picked)} documents "
          f"({n_low} low-confidence, {n_high} random).")
    print("Open tools/annotate.html, load it, and fill in true_sector.")
    print("The silver label is deliberately not shown while you decide.")
    return 0


def score() -> int:
    """Report agreement between the human pass and the silver labels."""
    if not SAMPLE_CSV.exists():
        print(f"{SAMPLE_CSV} not found — run with --build first.", file=sys.stderr)
        return 1
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    done = [r for r in rows if (r.get("true_sector") or "").strip()]
    if not done:
        print(f"None of the {len(rows)} documents have been verified yet.")
        return 1

    human = [r["true_sector"].strip().upper() for r in done]
    silver = [r["silver_sector"].strip().upper() for r in done]
    agree = sum(1 for h, s in zip(human, silver) if h == s)

    from sklearn.metrics import cohen_kappa_score

    kappa = float(cohen_kappa_score(human, silver)) if len(set(human)) > 1 else None

    by_conf = {}
    for level in ("low", "high"):
        subset = [r for r in done if r.get("labeller_confidence") == level]
        if subset:
            hits = sum(1 for r in subset
                       if r["true_sector"].strip().upper() == r["silver_sector"].strip().upper())
            by_conf[level] = {"n": len(subset), "agreement": hits / len(subset)}

    disagreements = [
        {"queue_id": r["queue_id"], "human": r["true_sector"].strip().upper(),
         "silver": r["silver_sector"], "note": r.get("annotator_note", "")}
        for r in done
        if r["true_sector"].strip().upper() != r["silver_sector"].strip().upper()
    ]

    report = {
        "n_verified": len(done),
        "n_sample": len(rows),
        "raw_agreement": agree / len(done),
        "cohen_kappa": kappa,
        "by_labeller_confidence": by_conf,
        "disagreements": disagreements,
        "note": (
            "Agreement between a human pass and model-produced silver labels. "
            "Quote this alongside any result computed on those labels."
        ),
    }
    ensure_dirs()
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Verified {len(done)} of {len(rows)} documents")
    print(f"Raw agreement : {agree}/{len(done)} = {agree / len(done):.1%}")
    print(f"Cohen's kappa : {kappa:.3f}" if kappa is not None else "Cohen's kappa : n/a")
    for level, stats in by_conf.items():
        print(f"  {level:<5} confidence: {stats['agreement']:.1%} of {stats['n']}")
    if disagreements:
        print(f"\n{len(disagreements)} disagreements:")
        for d in disagreements[:15]:
            print(f"  queue {d['queue_id']:>3}: you said {d['human']}, silver said {d['silver']}")
    print(f"\nWrote {REPORT_JSON}")
    return 0


def apply_verified() -> int:
    """Promote the human answers over the silver labels for verified documents.

    Standard annotation practice: agreement is reported from the pass *before*
    adjudication, and the released labels are the adjudicated ones. Running
    this before scoring would make the agreement figure meaningless.
    """
    if not REPORT_JSON.exists():
        print("Score the sample first: make verify", file=sys.stderr)
        return 1
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as fh:
        verified = {
            r["queue_id"]: r["true_sector"].strip().upper()
            for r in csv.DictReader(fh)
            if (r.get("true_sector") or "").strip()
        }
    if not verified:
        print("No verified answers in the sample.", file=sys.stderr)
        return 1

    with open(QUEUE_CSV, newline="", encoding="utf-8") as fh:
        header = list(csv.DictReader(fh).fieldnames or [])
        fh.seek(0)
        rows = list(csv.DictReader(fh))
    for column in ("human_verified", "adjudication_note"):
        if column not in header:
            header.append(column)

    changed = 0
    for row in rows:
        answer = verified.get(row["queue_id"])
        if not answer:
            continue
        row["human_verified"] = "yes"
        if row.get("model_assisted_sector") != answer:
            row["adjudication_note"] = (
                f"{row.get('model_assisted_sector')}->{answer}: human verification"
            )
            row["model_assisted_sector"] = answer
            row["labeller_confidence"] = "high"
            changed += 1

    with open(QUEUE_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})

    print(f"{len(verified)} documents marked human-verified; {changed} labels changed.")
    print("Re-merge and re-measure: make merge ARGS=--replace && make reproduce")
    return 0


def build_second() -> int:
    """Hand the measurement documents to a second annotator who reads German.

    Agreement so far is human-versus-model, and the human worked from machine
    translations. A second annotator gets the German text only: no
    translation, no model suggestion, no earlier answer, and a shuffled order,
    so nothing in the file hints at what anyone else said.
    """
    if SECOND_CSV.exists():
        with open(SECOND_CSV, newline="", encoding="utf-8") as fh:
            if any((r.get("true_sector") or "").strip() for r in csv.DictReader(fh)):
                print(f"{SECOND_CSV} already holds answers; refusing to overwrite it.",
                      file=sys.stderr)
                return 1
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("true_sector") or "").strip()]
    if not rows:
        print("Score the measurement sample first: make verify", file=sys.stderr)
        return 1
    order = np.random.default_rng(SEED).permutation(len(rows))
    header = ["queue_id", "legal_name", "purpose", "true_sector", "annotator_note"]
    with open(SECOND_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for i in order:
            r = rows[i]
            writer.writerow({"queue_id": r["queue_id"], "legal_name": r.get("legal_name", ""),
                             "purpose": r["purpose"], "true_sector": "", "annotator_note": ""})
    print(f"Wrote {len(rows)} documents to {SECOND_CSV}.")
    print("Give it to a second annotator with docs/second_annotator.md, then run:")
    print("  make second-annotator-score")
    return 0


def pairwise(a: dict, b: dict) -> dict:
    """Raw agreement and Cohen's kappa over the documents both have answered."""
    from sklearn.metrics import cohen_kappa_score

    common = sorted(set(a) & set(b))
    x, y = [a[k] for k in common], [b[k] for k in common]
    agree = sum(p == q for p, q in zip(x, y))
    kappa = float(cohen_kappa_score(x, y)) if len(set(x) | set(y)) > 1 else None
    return {"n": len(common), "raw_agreement": agree / len(common) if common else None,
            "cohen_kappa": kappa}


def score_second() -> int:
    """Human-versus-human agreement, the figure reviewers ask for first."""
    if not SECOND_CSV.exists():
        print("Build the sample first: make second-annotator", file=sys.stderr)
        return 1
    with open(SECOND_CSV, newline="", encoding="utf-8") as fh:
        second = {r["queue_id"]: r["true_sector"].strip().upper()
                  for r in csv.DictReader(fh) if (r.get("true_sector") or "").strip()}
    if not second:
        print("None of the documents have a second answer yet.", file=sys.stderr)
        return 1
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    first = {r["queue_id"]: r["true_sector"].strip().upper()
             for r in rows if (r.get("true_sector") or "").strip()}
    silver = {r["queue_id"]: r["silver_sector"].strip().upper()
              for r in rows if (r.get("silver_sector") or "").strip()}
    report = {
        "second_vs_first_human": pairwise(second, first),
        "second_vs_silver": pairwise(second, silver),
        "note": "The first human worked from English machine translations; the "
                "second annotator read the German text. Quote both figures.",
    }
    ensure_dirs()
    SECOND_REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, block in (("second vs first human", report["second_vs_first_human"]),
                        ("second vs silver", report["second_vs_silver"])):
        kappa = "n/a" if block["cohen_kappa"] is None else f"{block['cohen_kappa']:.3f}"
        print(f"  {name:22s} n={block['n']:3d}  agreement {block['raw_agreement']:.1%}  kappa {kappa}")
    print(f"\nWrote {SECOND_REPORT_JSON}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Draw the sample.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Draw a sample that excludes documents already used in a pilot "
             "pass. Measuring again on documents whose labels were just "
             "adjudicated would score them on their own training data.",
    )
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--second-build", action="store_true",
                        help="Write the blind German-only sample for a second annotator.")
    parser.add_argument("--second-score", action="store_true",
                        help="Score the second annotator against the first and the silver labels.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the verified answers back into the queue as the final "
             "labels for those documents. Run after scoring: kappa is measured "
             "before adjudication, the released labels come after it.",
    )
    args = parser.parse_args()
    if args.build or args.fresh:
        return build(args.size, fresh=args.fresh)
    if args.second_build:
        return build_second()
    if args.second_score:
        return score_second()
    if args.apply:
        return apply_verified()
    return score()


if __name__ == "__main__":
    sys.exit(main())

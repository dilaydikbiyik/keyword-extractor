#!/usr/bin/env python3
"""Will rewriting your class descriptions pay, and which classes first?

    python -m experiments.diagnose --taxonomy classes.json --documents pool.txt
    python -m experiments.diagnose --taxonomy classes.json --documents pool.txt \
        --rewritten classes_v2.json

Everything this project measures concerns somebody else's taxonomy: a list of
classes, a description for each, and a pile of documents nobody has labelled.
The findings were published as findings, which left the person holding that pile
to read a paper and guess. This runs the measurements on their taxonomy instead.

Nothing here needs a label. Two modes:

*Before writing anything* -- the regime check. Elaborating class descriptions
was worth 26.8 points on one taxonomy and 0.4 on another, and the two are not
in conflict: they differ in how much lexical overlap the class names have with
their documents, and that property separates the regimes (``run_gap_analysis``).
This reports which regime a taxonomy is in, which classes sit furthest from the
documents they attract, and which ones a careless rewrite would damage most.

*After rewriting* -- the change check. The quantity that predicts the gain is
how far the rewrite moves each class vector toward the centroid of its
documents, and it survives losing the labels (``run_labelfree_predictor``). So
a rewrite can be checked before a single annotation is paid for: did the vectors
move toward the documents, or away?

Every calibration figure printed below is read out of ``results/`` at run time,
from the study that established it. None of them is typed into this file, so the
advice cannot drift from the evidence, and the report says plainly where each
number came from and how weak it is.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from experiments.config import EMBEDDING_MODEL, RESULTS_DIR, ensure_dirs, set_seed
from experiments.lexical_gap import content_words, name_overlap
from experiments.run_rocchio import unit_rows
from experiments.systems import get_embedder

# The overlap below which elaboration paid on the corpora studied here. Set by
# run_gap_analysis, which binned classes at this value; read, not invented.
REGIME_THRESHOLD = 0.05


# ── reading somebody else's files ────────────────────────────────────────────

def load_taxonomy(path: Path) -> Dict[str, Dict[str, str]]:
    """Each class to its bare name and the full text that gets embedded.

    Accepted: ``{"A": "text"}``, ``{"classes": {...}}``, ``{"sectors": {...}}``,
    and either of those with ``{"name": ..., "description": ...}`` objects
    instead of strings.

    The two are kept apart on purpose. What gets embedded is everything the
    class says. The regime check counts only the words of the *name*, because
    that is what the published measure counted; counting the description's
    words as well inflates the overlap and can flip the verdict -- which it did,
    on this project's own taxonomy, before this distinction was made.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in ("classes", "sectors", "taxonomy", "labels"):
        if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
            raw = raw[key]
            break
    if not isinstance(raw, dict) or not raw:
        raise SystemExit(f"{path}: expected an object mapping class names to descriptions")
    out = {}
    for key, value in raw.items():
        key = str(key)
        if isinstance(value, str):
            # No separate name field: the key is the label the practitioner has.
            out[key] = {"name": key, "text": value.strip() or key}
        elif isinstance(value, dict):
            label = str(value.get("name", "")).strip() or key
            description = str(value.get("description", "")).strip()
            out[key] = {"name": label,
                        "text": f"{label}. {description}" if description else label}
        else:
            raise SystemExit(f"{path}: class {key!r} is neither a string nor an object")
    if len(out) < 2:
        raise SystemExit(f"{path}: a taxonomy needs at least two classes")
    return out


def load_documents(path: Path, column: str | None) -> List[str]:
    """One document per line, or a JSON list, or one column of a CSV."""
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("documents", raw.get("texts", []))
        docs = [str(d).strip() if isinstance(d, str) else str(d.get(column or "text", "")).strip()
                for d in raw]
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if column is None:
                raise SystemExit("--column is required for a CSV pool")
            if reader.fieldnames and column not in reader.fieldnames:
                raise SystemExit(f"{path}: no column {column!r}; found "
                                 f"{', '.join(reader.fieldnames or [])}")
            docs = [str(row.get(column, "")).strip() for row in reader]
    else:
        docs = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    docs = [d for d in docs if d]
    if len(docs) < 2:
        raise SystemExit(f"{path}: found {len(docs)} documents; the pool needs at least two")
    return docs


# ── the measurements, none of which touch a label ────────────────────────────

def embed(texts: Sequence[str], model: str, batch: int = 64) -> np.ndarray:
    embedder = get_embedder(model)
    return unit_rows(np.vstack([np.asarray(v, dtype=np.float64)
                                for v in embedder.embed_texts(list(texts), batch_size=batch)]))


def assign(docs: np.ndarray, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The decision itself: argmax over cosines, and the margin behind it.

    The margin is the gap to the runner-up. It is the quantity the abstention
    study thresholds, and it costs nothing extra to compute.
    """
    scores = docs @ vectors.T
    order = np.argsort(-scores, axis=1)
    top = order[:, 0]
    margin = scores[np.arange(len(scores)), top] - scores[np.arange(len(scores)), order[:, 1]]
    return top, margin


def pseudo_alignment(vectors: np.ndarray, docs: np.ndarray,
                     chosen: np.ndarray, n_classes: int) -> Dict[int, float]:
    """cos(class vector, centroid of the documents it attracts).

    The labelled version of this quantity is what predicts the gain; this is the
    version that pseudo-assigns by argmax instead, and it is the one a person
    without labels can compute. A class the vectors never choose has no
    centroid and is reported separately rather than scored.
    """
    out = {}
    for i in range(n_classes):
        idx = np.flatnonzero(chosen == i)
        if idx.size:
            centroid = docs[idx].mean(axis=0)
            out[i] = float(vectors[i] @ (centroid / np.linalg.norm(centroid)))
    return out


def nearest_competitor(vectors: np.ndarray, names: Sequence[str]) -> List[Tuple[str, float]]:
    """For each class, the class whose vector sits closest to it.

    Confusability does not predict which classes gain -- that was measured and
    it does not (run_gap_analysis). It predicts something else: where a wrong
    description is expensive. Swapping descriptions between nearest neighbours
    cost several times what swapping them at random cost.
    """
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    out = []
    for i in range(len(names)):
        j = int(np.argmax(similarity[i]))
        out.append((names[j], float(similarity[i, j])))
    return out


# ── the calibration, read from the studies rather than typed here ────────────

def regime_bias(bias: Dict) -> Dict:
    """What the overlap estimate costs when it has no labels to group by.

    The decision this feeds is one comparison against a threshold, so the
    quantity that matters is not the estimate's error but how often the
    comparison still comes out right -- and, where it does not, in which
    direction. Measured in run_overlap_estimate over the five corpora here.

    The error is largest in the low-overlap regime, which is where the decision
    is actually taken: pseudo-assignment hands a class documents that are not
    its own, those documents raise the count of name words, and the estimate
    reads high. So an estimate just above the threshold is not evidence of the
    regime where elaboration does not pay. The band in which that happened is
    read off the corpora rather than guessed: the highest estimate produced by a
    taxonomy whose labelled overlap was in fact below the threshold.
    """
    low = [c for c in bias["per_corpus"].values()
           if c["labelled_mean_overlap"] < bias["threshold"]]
    return {
        "rho": bias["pooled"]["rho"],
        "classes": bias["pooled"]["classes"],
        "per_class_agreement": bias["pooled"]["per_class_verdict_agreement"],
        "corpora": bias["taxonomy_verdict"]["corpora"],
        "corpora_agreeing": bias["taxonomy_verdict"]["agreeing"],
        "corpora_disagreeing": bias["taxonomy_verdict"]["disagreeing_names"],
        "worst_inflation": max((c["inflation_factor"] for c in low
                                if c["inflation_factor"]), default=None),
        "undecided_upper": max((c["estimated_mean_overlap"] for c in low), default=None),
        "source": "results/overlap_estimate.json",
    }


def verdict(overlap: float, bias: Dict) -> str:
    """Three answers, not two, because the measured error permits only three."""
    if overlap < REGIME_THRESHOLD:
        return ("below the threshold: the regime where elaboration paid")
    if bias["undecided_upper"] and overlap <= bias["undecided_upper"]:
        return ("above the threshold but inside the band where this estimate has "
                "been wrong before: undecided, and worth trying on a few classes")
    return "above the threshold: the regime where elaboration mostly did not pay"


def calibration() -> Dict:
    """What the project's own results say about how far to trust each number."""

    def read(name: str) -> Dict:
        path = RESULTS_DIR / name
        if not path.exists():
            raise SystemExit(f"missing {path}; run `make study` first, or `make reproduce`")
        return json.loads(path.read_text(encoding="utf-8"))

    gap, labelfree = read("gap_analysis.json"), read("labelfree_predictor.json")
    dose, noise = read("dose_response.json"), read("description_noise.json")
    bias = read("overlap_estimate.json")
    binned = gap["binned_by_overlap"]
    pooled = labelfree["pooled"]
    # The corpus where the estimate did worst, found rather than remembered: the
    # weakest case is the one a stranger's taxonomy might resemble.
    weakest = min(labelfree["corpora"].items(),
                  key=lambda kv: kv[1]["estimate_vs_headroom"]["rho"])
    first = dose["sets"]["first set (hand-checked)"]
    costs = noise["sets"]["first set (hand-checked)"]["cost_pp"]
    return {
        "regime": {
            "threshold": REGIME_THRESHOLD,
            "mean_gain_below_pp": 100 * binned["below_5_percent"]["mean_gain"],
            "mean_gain_at_or_above_pp": 100 * binned["at_or_above_5_percent"]["mean_gain"],
            "n_below": binned["below_5_percent"]["n"],
            "n_at_or_above": binned["at_or_above_5_percent"]["n"],
            "within_regime_rho": gap["correlations"]["name_overlap"]["rho"],
            "within_regime_p": gap["correlations"]["name_overlap"]["p"],
            "source": "results/gap_analysis.json",
        },
        "regime_without_labels": regime_bias(bias),
        "estimate": {
            "vs_labelled_rho": labelfree["estimate_vs_truth"]["rho"],
            "headroom_rho": pooled["estimate_vs_headroom"]["rho"],
            "headroom_p": pooled["estimate_vs_headroom"]["p"],
            "labelled_headroom_rho": pooled["labelled_vs_headroom"]["rho"],
            "classes": labelfree["classes_pooled"],
            "documents": labelfree["documents_pooled"],
            "corpora": len(labelfree["corpora"]),
            "sign_rule_accuracy": pooled["sign_rule_accuracy"],
            "weakest_corpus": weakest[0],
            "weakest_rho": weakest[1]["estimate_vs_headroom"]["rho"],
            "source": "results/labelfree_predictor.json",
        },
        "budget": {
            "words_for_90pc": first["words_for_90pc_of_gain"],
            "full_gain_pp": first["gain_full_over_name_only_pp"],
            "source": "results/dose_response.json",
        },
        "risk": {
            "neighbour_quarter_pp": costs["nearest-neighbour swapped, 25%"],
            "random_quarter_pp": costs["permuted, 25% of classes"],
            "generic_pp": costs["generic, every class the same"],
            "source": "results/description_noise.json",
        },
        "abstention": {"source": "results/abstention.json",
                       "curve": read("abstention.json")["coverage_curve"]},
    }


# ── the report ───────────────────────────────────────────────────────────────

def diagnose(taxonomy: Dict[str, Dict[str, str]], documents: List[str], model: str,
             rewritten: Dict[str, Dict[str, str]] | None = None) -> Dict:
    names = sorted(taxonomy)
    vectors = embed([taxonomy[n]["text"] for n in names], model)
    docs = embed(documents, model)
    chosen, margin = assign(docs, vectors)
    hat = pseudo_alignment(vectors, docs, chosen, len(names))
    competitor = nearest_competitor(vectors, names)

    rows = []
    for i, name in enumerate(names):
        mine = np.flatnonzero(chosen == i)
        overlap = (name_overlap(taxonomy[name]["name"], [documents[j] for j in mine])
                   if mine.size else None)
        rows.append({
            "class": name,
            "share_of_pool_attracted": float(mine.size) / len(documents),
            "documents_attracted": int(mine.size),
            "alignment_hat": hat.get(i),
            "name_overlap_in_attracted": overlap,
            "nearest_other_class": competitor[i][0],
            "similarity_to_nearest": competitor[i][1],
            "median_margin": float(np.median(margin[mine])) if mine.size else None,
        })

    # A class whose name carries no content word scores zero overlap whatever the
    # documents say, and zero reads as "the regime where rewriting pays". That is
    # the wrong answer arrived at silently, and it is what a taxonomy of opaque
    # codes produces, so those classes are excluded and counted rather than
    # averaged in.
    unusable = [n for n in names if not content_words(taxonomy[n]["name"])]
    scored = [r for r in rows if r["name_overlap_in_attracted"] is not None
              and r["class"] not in unusable]
    enough = len(scored) >= 2 and len(scored) >= len(names) / 2
    pool_overlap = float(np.mean([r["name_overlap_in_attracted"] for r in scored])) \
        if scored else None
    cal = calibration()

    report = {
        "taxonomy_classes": len(names),
        "pool_documents": len(documents),
        "encoder": model,
        "regime": {
            "mean_name_overlap": pool_overlap,
            "classes_scored": len(scored),
            "classes_without_usable_name": unusable,
            "verdict": (verdict(pool_overlap, cal["regime_without_labels"]) if enough else
                        "not computable: too few class names carry a content word. "
                        "The regime check counts the words of the name, so a taxonomy "
                        "of opaque codes has to supply them -- give each class as "
                        '{"name": ..., "description": ...} rather than as one string.'),
            "undecided": bool(enough and REGIME_THRESHOLD <= pool_overlap
                              <= (cal["regime_without_labels"]["undecided_upper"] or 0)),
            "classes_below_threshold": [r["class"] for r in scored
                                        if r["name_overlap_in_attracted"] < REGIME_THRESHOLD],
        },
        "classes_attracting_nothing": [r["class"] for r in rows if not r["documents_attracted"]],
        "rewrite_first": [r["class"] for r in sorted(
            (r for r in scored if r["alignment_hat"] is not None),
            key=lambda r: r["alignment_hat"])[:10]],
        "write_carefully": [r["class"] for r in sorted(
            rows, key=lambda r: -r["similarity_to_nearest"])[:5]],
        "coverage_at_margin": [
            {"coverage": q,
             "margin_threshold": float(np.quantile(margin, 1 - q)) if q < 1 else float(margin.min())}
            for q in (1.0, 0.9, 0.75, 0.5, 0.25)],
        "classes": rows,
        "calibration": cal,
        "what_this_cannot_tell_you": [
            "the accuracy you will get: that needs labels, and this reads none",
            "whether a particular class will gain, as opposed to which gain more: "
            "the sign of the estimate called the direction barely above chance",
            "anything about a taxonomy whose classes the encoder cannot separate "
            "at all, where the pseudo-assignment this rests on is itself noise",
        ],
    }

    if rewritten is not None:
        missing = sorted(set(names) - set(rewritten))
        if missing:
            raise SystemExit("the rewritten taxonomy is missing: " + ", ".join(missing))
        after = embed([rewritten[n]["text"] for n in names], model)
        hat_after = pseudo_alignment(after, docs, chosen, len(names))
        moved = []
        for i, name in enumerate(names):
            if i in hat and i in hat_after:
                moved.append({
                    "class": name,
                    "alignment_hat_before": hat[i],
                    "alignment_hat_after": hat_after[i],
                    "change": hat_after[i] - hat[i],
                    "words_before": len(taxonomy[name]["text"].split()),
                    "words_after": len(rewritten[name]["text"].split()),
                })
        moved.sort(key=lambda r: -r["change"])
        report["rewrite"] = {
            "classes_measured": len(moved),
            "moved_toward_documents": sum(1 for r in moved if r["change"] > 0),
            "moved_away": sum(1 for r in moved if r["change"] < 0),
            "mean_change": float(np.mean([r["change"] for r in moved])) if moved else 0.0,
            "largest_gains": moved[:5],
            "largest_losses": moved[-5:][::-1],
            "under_budget": [r["class"] for r in moved
                             if r["words_after"] < cal["budget"]["words_for_90pc"]],
            "per_class": moved,
        }
    return report


def render(report: Dict) -> str:
    cal, out = report["calibration"], []
    w = out.append
    w(f"{report['taxonomy_classes']} classes, {report['pool_documents']} unlabelled "
      f"documents, encoder {report['encoder']}.")
    w("")
    w("REGIME — does rewriting pay for this taxonomy at all?")
    reg, mine = cal["regime"], report["regime"]
    bias = cal["regime_without_labels"]
    if mine["mean_name_overlap"] is None:
        w(f"  {mine['verdict']}")
        if mine["classes_without_usable_name"]:
            w("  no content word in the name of: "
              + ", ".join(mine["classes_without_usable_name"][:12]))
    else:
        w(f"  mean overlap between class names and the documents they attract: "
          f"{mine['mean_name_overlap']:.3f}, over {mine['classes_scored']} classes")
        if mine["classes_without_usable_name"]:
            w("  excluded, because their names carry no content word to count: "
              + ", ".join(mine["classes_without_usable_name"][:12]))
        w(f"  {mine['verdict']}")
        w(f"  measured: classes below {reg['threshold']:.2f} overlap gained "
          f"{reg['mean_gain_below_pp']:+.1f} points on average (n={reg['n_below']}), "
          f"those at or above it {reg['mean_gain_at_or_above_pp']:+.1f} "
          f"(n={reg['n_at_or_above']})")
        w(f"  it separates the regimes and does not rank inside one "
          f"(rho = {reg['within_regime_rho']:+.2f}, p = {reg['within_regime_p']:.2f}) — "
          f"{reg['source']}")
        w(f"  and this reading has no labels to group documents by, which was measured "
          f"too: it tracks the labelled overlap at rho = {bias['rho']:+.3f} over "
          f"{bias['classes']} classes and gives the same per-class verdict "
          f"{100 * bias['per_class_agreement']:.0f}% of the time, agreeing at the "
          f"taxonomy level on {bias['corpora_agreeing']} of {bias['corpora']} corpora"
          + (f" and failing on {', '.join(bias['corpora_disagreeing'])}"
             if bias["corpora_disagreeing"] else "")
          + f" — {bias['source']}")
        w(f"  every failure ran in one direction: without labels the overlap reads up "
          f"to {bias['worst_inflation']:.1f}x high in the regime that matters, so an "
          f"estimate between {reg['threshold']:.2f} and {bias['undecided_upper']:.2f} "
          f"is not evidence against rewriting.")
    if report["classes_attracting_nothing"]:
        w("")
        w("  these classes attract no document in the pool, which is the worst case "
          "and the clearest thing to fix first:")
        w("    " + ", ".join(report["classes_attracting_nothing"]))
    w("")
    w("WHICH CLASSES FIRST — lowest alignment with what they already attract")
    w("  " + ", ".join(report["rewrite_first"]) or "  (none scored)")
    est = cal["estimate"]
    if est.get("headroom_rho") is not None:
        w(f"  how far to trust this: the label-free estimate tracked the labelled "
          f"quantity at rho = {est['vs_labelled_rho']:+.3f} and predicted the share of "
          f"headroom captured at rho = {est['headroom_rho']:+.3f} over {est['classes']} "
          f"classes in {est['corpora']} corpora, against "
          f"{est['labelled_headroom_rho']:+.3f} with labels")
        w(f"  it was weakest where the pseudo-assignment is least accurate: "
          f"rho = {est['weakest_rho']:+.3f} on {est['weakest_corpus']} — so read this as a "
          f"ranking, not a forecast ({est['source']})")
        w(f"  and the sign alone called the direction for only "
          f"{100 * est['sign_rule_accuracy']:.0f}% of classes: it says which classes gain "
          f"more, not whether a given one gains at all")
    w("")
    w("WRITE THESE CAREFULLY — closest to another class, so a wrong description costs most")
    w("  " + ", ".join(report["write_carefully"]))
    risk = cal["risk"]
    w(f"  measured: descriptions swapped between nearest neighbours cost "
      f"{risk['neighbour_quarter_pp']:+.1f} points against {risk['random_quarter_pp']:+.1f} "
      f"for the same number swapped at random; one generic description for every class "
      f"cost {risk['generic_pp']:+.1f} — {risk['source']}")
    w("  (closeness predicts this cost. It does not predict which classes gain: "
      "that was measured too, and it does not.)")
    w("")
    bud = cal["budget"]
    w(f"HOW MUCH TO WRITE — {bud['words_for_90pc']} words per class carried ninety per cent "
      f"of a {bud['full_gain_pp']:+.1f} point gain; past that the curve is flat "
      f"({bud['source']})")
    w("")
    w("IF IT IS STILL WRONG — abstain below the margin the decision already computes")
    for row in report["coverage_at_margin"]:
        w(f"  keep {row['coverage']:.0%} of this pool: margin >= {row['margin_threshold']:.4f}")
    w("  the coverage above is yours; the accuracy it buys is not knowable without labels. "
      f"On the set in {cal['abstention']['source']}, half coverage raised Top-1 from "
      f"{100 * cal['abstention']['curve'][0]['top1_on_kept']:.1f}% to "
      f"{100 * cal['abstention']['curve'][3]['top1_on_kept']:.1f}%.")

    if "rewrite" in report:
        rw = report["rewrite"]
        w("")
        w("YOUR REWRITE — did the class vectors move toward the documents?")
        w(f"  {rw['moved_toward_documents']} of {rw['classes_measured']} classes moved toward, "
          f"{rw['moved_away']} away; mean change {rw['mean_change']:+.4f}")
        for row in rw["largest_gains"][:3]:
            w(f"    {row['class']}: {row['change']:+.4f} "
              f"({row['words_before']} -> {row['words_after']} words)")
        if rw["moved_away"]:
            w("  moved away from their documents, which is the case to look at:")
            for row in rw["largest_losses"][:3]:
                if row["change"] < 0:
                    w(f"    {row['class']}: {row['change']:+.4f}")
        if rw["under_budget"]:
            w(f"  under {cal['budget']['words_for_90pc']} words, where the curve has not "
              f"flattened yet: " + ", ".join(rw["under_budget"][:10]))
    w("")
    w("WHAT THIS CANNOT TELL YOU")
    for line in report["what_this_cannot_tell_you"]:
        w(f"  - {line}")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--taxonomy", type=Path, required=True,
                        help="JSON: class name -> description")
    parser.add_argument("--documents", type=Path, required=True,
                        help="unlabelled pool: .txt one per line, .json list, or .csv")
    parser.add_argument("--column", help="which column of a CSV pool holds the text")
    parser.add_argument("--rewritten", type=Path,
                        help="a second taxonomy file, to check a rewrite before paying "
                             "for labels")
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--out", type=Path, help="write the full report as JSON here")
    args = parser.parse_args(argv)

    set_seed()
    ensure_dirs()
    report = diagnose(
        load_taxonomy(args.taxonomy),
        load_documents(args.documents, args.column),
        args.model,
        load_taxonomy(args.rewritten) if args.rewritten else None,
    )
    print(render(report))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"\nfull report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

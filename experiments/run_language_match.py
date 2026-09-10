#!/usr/bin/env python3
"""Is it the language, or is it the match? A 2x2 factorial.

    python -m experiments.run_language_match

Rewriting the taxonomy's class descriptions from Turkish into German gained
17.7 points. Two explanations survive that observation:

  (a) the descriptions must be in the *same* language as the documents, or
  (b) German descriptions are simply better than Turkish ones, because the
      encoder happens to represent German more sharply.

They are told apart by crossing document language with description language.
If (a) holds, the diagonal wins: German descriptions on German documents and
English descriptions on English documents both beat the off-diagonal cells. If
(b) holds, one description language wins in both rows.

Documents are translated with the same Marian system used elsewhere in this
repository, and the class descriptions with it too, so the translation quality
is held constant across the cells rather than being a free variable.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score

from experiments.config import RESULTS_DIR, SECTORS_JSON, TABLES_DIR, ensure_dirs, set_seed
from experiments.data import load_labeled_samples_split
from experiments.metrics import bootstrap_ci, mcnemar_exact
from experiments.systems import get_embedder, translate_de_en


def sector_vectors(descriptions: Dict[str, str], seeds: Dict[str, List[str]], use_seeds: bool):
    """Build one vector per section from a given set of descriptions."""
    embedder = get_embedder()
    codes = sorted(descriptions)
    vectors = []
    for code in codes:
        parts = [embedder.embed_text(descriptions[code])]
        if use_seeds and seeds.get(code):
            parts.append(embedder.embed_text(", ".join(seeds[code])))
        vectors.append(np.asarray(np.mean(np.vstack(parts), axis=0), dtype=np.float64))
    return codes, np.vstack(vectors)


def evaluate(codes, vectors, documents: List[str], gold: List[str]) -> Dict:
    embedder = get_embedder()
    docs = np.vstack(
        [np.asarray(embedder.embed_text(d), dtype=np.float64) for d in documents]
    )
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    vecs = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    order = np.argsort(-(docs @ vecs.T), axis=1)
    predicted = [codes[order[i, 0]] for i in range(len(gold))]
    correct = [gold[i] == predicted[i] for i in range(len(gold))]
    lo, hi = bootstrap_ci(correct)
    return {
        "top1_accuracy": float(np.mean(correct)),
        "top1_ci95": [lo, hi],
        "top3_accuracy": float(
            np.mean([gold[i] in [codes[j] for j in order[i, :3]] for i in range(len(gold))])
        ),
        "f1_macro": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        "correct": correct,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half", choices=["dev", "test", "all"], default="all")
    parser.add_argument(
        "--no-seeds",
        action="store_true",
        help="Build sector vectors from the description alone, isolating the "
             "factor under test.",
    )
    args = parser.parse_args()

    set_seed()
    ensure_dirs()

    taxonomy = json.loads(SECTORS_JSON.read_text(encoding="utf-8"))["sectors"]
    codes = sorted(taxonomy)
    de_desc = {c: f"{taxonomy[c].get('name', '')}. {taxonomy[c].get('description', '')}" for c in codes}
    seeds = {c: taxonomy[c].get("seed_keywords", []) for c in codes}

    print("Translating the class descriptions…", flush=True)
    en_desc = {c: translate_de_en(de_desc[c]) for c in codes}

    samples = load_labeled_samples_split(None if args.half == "all" else args.half)
    gold = [s.true_sector for s in samples]
    de_docs = [s.purpose for s in samples]
    print(f"Translating {len(samples)} documents…", flush=True)
    en_docs = [translate_de_en(d) for d in de_docs]

    use_seeds = not args.no_seeds
    cells = {}
    for desc_lang, descriptions in (("German", de_desc), ("English", en_desc)):
        c, V = sector_vectors(descriptions, seeds, use_seeds)
        for doc_lang, documents in (("German", de_docs), ("English", en_docs)):
            key = f"{doc_lang} documents / {desc_lang} descriptions"
            cells[key] = evaluate(c, V, documents, gold)
            cells[key]["matched"] = doc_lang == desc_lang
            print(f"  {key:<44} {cells[key]['top1_accuracy']:.1%}", flush=True)

    matched = [k for k, v in cells.items() if v["matched"]]
    crossed = [k for k, v in cells.items() if not v["matched"]]
    mean_matched = float(np.mean([cells[k]["top1_accuracy"] for k in matched]))
    mean_crossed = float(np.mean([cells[k]["top1_accuracy"] for k in crossed]))

    # Each matched cell against the crossed cell that shares its documents.
    tests = {}
    for doc_lang in ("German", "English"):
        same = f"{doc_lang} documents / {doc_lang} descriptions"
        other = "English" if doc_lang == "German" else "German"
        cross = f"{doc_lang} documents / {other} descriptions"
        tests[doc_lang] = mcnemar_exact(cells[same]["correct"], cells[cross]["correct"])
        tests[doc_lang]["gain_pp"] = (
            cells[same]["top1_accuracy"] - cells[cross]["top1_accuracy"]
        ) * 100

    rows = ["| Documents | Descriptions | Top-1 | 95% CI | Top-3 | F1-macro |",
            "| --- | --- | --- | --- | --- | --- |"]
    for key, cell in cells.items():
        doc_lang, desc_lang = key.split(" documents / ")
        desc_lang = desc_lang.replace(" descriptions", "")
        mark = "**" if cell["matched"] else ""
        lo, hi = cell["top1_ci95"]
        rows.append(
            f"| {mark}{doc_lang}{mark} | {mark}{desc_lang}{mark} | "
            f"{mark}{cell['top1_accuracy']:.1%}{mark} | [{lo:.1%}, {hi:.1%}] | "
            f"{cell['top3_accuracy']:.1%} | {cell['f1_macro']:.3f} |"
        )
    table = "\n".join(rows)
    (TABLES_DIR / "language_match.md").write_text(table + "\n", encoding="utf-8")

    payload = {
        "half": args.half,
        "n": len(samples),
        "sector_vector": "description only" if args.no_seeds else "description + seeds",
        "cells": {k: {kk: vv for kk, vv in v.items() if kk != "correct"} for k, v in cells.items()},
        "mean_matched": mean_matched,
        "mean_crossed": mean_crossed,
        "matched_advantage_pp": (mean_matched - mean_crossed) * 100,
        "within_row_tests": tests,
        "table_markdown": table,
    }
    (RESULTS_DIR / "language_match.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + table)
    print(f"\nmatched mean {mean_matched:.1%} · crossed mean {mean_crossed:.1%} "
          f"· advantage {payload['matched_advantage_pp']:+.1f} pp")
    for doc_lang, t in tests.items():
        print(f"  {doc_lang} documents: matched − crossed = {t['gain_pp']:+.1f} pp, p = {t['p_value']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

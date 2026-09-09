#!/usr/bin/env python3
"""Run the baseline comparison and the ablation study.

    python -m experiments.run_experiments --suite all

Writes machine-readable results to ``results/*.json`` and the paper tables to
``results/tables/*.md``.  Nothing here reads a number that is hard-coded in the
README; the README is regenerated from these files.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List

from experiments import report
from experiments.config import EMBEDDING_MODEL, RESULTS_DIR, SEED, TABLES_DIR, ensure_dirs, set_seed
from experiments.data import load_corpus, load_labeled_samples, load_labels_metadata
from experiments.metrics import (
    evaluate_keywords,
    evaluate_sector_predictions,
    mcnemar_exact,
)
from experiments.systems import (
    LLMRanker,
    NoKeywords,
    System,
    build_ablations,
    build_baselines,
)

TOP_N_KEYWORDS = 10


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def run_system(system: System, samples, corpus) -> Dict:
    """Score one system over the labelled evaluation set."""
    print(f"  → {system.label}", flush=True)
    system.fit(corpus)

    outputs = [system.run(s.purpose, top_n_keywords=TOP_N_KEYWORDS) for s in samples]
    y_true = [s.true_sector for s in samples]
    ranked = [o["ranked_sectors"] for o in outputs]

    sector = evaluate_sector_predictions(y_true, ranked)
    keywords = evaluate_keywords(
        [o["keywords"] for o in outputs], [s.keywords_ground_truth for s in samples]
    )
    return {
        "key": system.key,
        "label": system.label,
        "description": system.description,
        "is_oracle": system.is_oracle,
        "sector": sector,
        "keywords": keywords,
        "predictions": [
            {
                "id": s.id,
                "true": s.true_sector,
                "predicted": o["top_sector"],
                "top3": o["ranked_sectors"][:3],
                "scores": o["scores"],
                "keywords": o["keywords"],
            }
            for s, o in zip(samples, outputs)
        ],
    }


def add_significance(results: List[Dict], reference_key: str = "full") -> None:
    """Paired exact McNemar of every system against the reference system."""
    ref = next((r for r in results if r["key"] == reference_key), None)
    if ref is None:
        return
    for r in results:
        if r["key"] == reference_key:
            r["p_value_vs_full"] = None
            continue
        test = mcnemar_exact(ref["sector"]["correct_top1"], r["sector"]["correct_top1"])
        r["p_value_vs_full"] = test["p_value"]
        r["mcnemar_vs_full"] = test


def strip_per_item(results: List[Dict]) -> List[Dict]:
    """Drop the per-document boolean vector from the saved summary."""
    slim = []
    for r in results:
        copy = {k: v for k, v in r.items() if k != "predictions"}
        copy["sector"] = {k: v for k, v in r["sector"].items() if k != "correct_top1"}
        slim.append(copy)
    return slim


def provenance(samples, extra: Dict | None = None) -> Dict:
    meta = load_labels_metadata()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "seed": SEED,
        "python": platform.python_version(),
        "embedding_model": EMBEDDING_MODEL,
        "eval_set_size": len(samples),
        "eval_set_source": meta.get("source"),
        "eval_set_annotation": meta.get("annotation"),
        **(extra or {}),
    }


def run_suite(name: str, systems: List[System], samples, corpus, table_fn) -> Dict:
    print(f"\n[{name}]")
    results = [run_system(s, samples, corpus) for s in systems]
    add_significance(results)

    table = table_fn(results)
    (TABLES_DIR / f"{name}.md").write_text(table + "\n", encoding="utf-8")

    payload = {
        "provenance": provenance(samples),
        "systems": strip_per_item(results),
        "table_markdown": table,
    }
    (RESULTS_DIR / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Per-document predictions live in their own file; the error analysis reads them.
    (RESULTS_DIR / f"{name}_predictions.json").write_text(
        json.dumps(
            {r["key"]: r["predictions"] for r in results}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print("\n" + table)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=["baselines", "ablation", "all"], default="all"
    )
    parser.add_argument(
        "--corpus-limit",
        type=int,
        default=None,
        help="Fit TF-IDF on the first N corpus documents (default: all).",
    )
    parser.add_argument(
        "--extra-ablations",
        action="store_true",
        help="Add the mpnet and translation ablations (downloads two models).",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Add the paid LLM zero-shot baseline (needs OPENAI_API_KEY).",
    )
    args = parser.parse_args()

    set_seed()
    ensure_dirs()

    samples = load_labeled_samples()
    corpus = load_corpus(limit=args.corpus_limit)
    label_counts = Counter(s.true_sector for s in samples)
    print(f"Evaluation set: {len(samples)} documents, {len(label_counts)} sectors")
    print(f"Corpus for TF-IDF statistics: {len(corpus)} documents")

    metrics: Dict[str, Dict] = {}

    if args.suite in ("baselines", "all"):
        systems = build_baselines(label_counts)
        if args.with_llm:
            systems.insert(
                -1,
                System(
                    key="llm-zeroshot",
                    label="LLM zero-shot (gpt-4o-mini)",
                    description="Sector letters asked directly of a small LLM.",
                    ranker=LLMRanker(),
                    keywords=NoKeywords(),
                ),
            )
        metrics["baselines"] = run_suite(
            "baselines", systems, samples, corpus, report.comparison_table
        )

    if args.suite in ("ablation", "all"):
        metrics["ablation"] = run_suite(
            "ablation",
            build_ablations(extra=args.extra_ablations),
            samples,
            corpus,
            report.ablation_table,
        )

    # results/metrics.json is the single file the README and the paper cite.
    headline = {}
    if "baselines" in metrics:
        full = next(
            (s for s in metrics["baselines"]["systems"] if s["key"] == "full"), None
        )
        if full:
            headline = {
                "top1_accuracy": full["sector"]["top1_accuracy"],
                "top1_ci95": full["sector"]["top1_ci95"],
                "top3_accuracy": full["sector"]["top3_accuracy"],
                "f1_macro": full["sector"]["f1_macro"],
                "cohen_kappa": full["sector"]["cohen_kappa"],
                "precision_at_5": full["keywords"].get("precision_at_5"),
            }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "provenance": provenance(samples),
                "headline": headline,
                "suites": {k: v["systems"] for k, v in metrics.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {RESULTS_DIR / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

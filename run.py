#!/usr/bin/env python3
"""Single entry point that reproduces every number in the README.

    python run.py --config config/config.yaml

Runs the baseline comparison, the ablation study and the error analysis in
order, writing everything under ``results/``.  ``make reproduce`` calls this.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import utils.quiet  # noqa: F401,E402  (quiet environment warnings before heavy imports)

from experiments import run_error_analysis, run_experiments  # noqa: E402
from experiments.config import RESULTS_DIR, ensure_dirs, set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Pipeline configuration file (recorded in the results provenance).",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Include the paid LLM zero-shot baseline (needs OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--half",
        choices=["dev", "test", "all"],
        default="all",
        help="Evaluate on the development half, the held-out test half, or all.",
    )
    parser.add_argument(
        "--extra-ablations",
        action="store_true",
        help="Add the mpnet and translation ablations (downloads two models).",
    )
    parser.add_argument(
        "--corpus-limit",
        type=int,
        default=None,
        help="Fit TF-IDF on the first N corpus documents (default: all 9,993).",
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    set_seed()
    ensure_dirs()

    experiment_args = ["--suite", "all"]
    if args.with_llm:
        experiment_args.append("--with-llm")
    if args.extra_ablations:
        experiment_args.append("--extra-ablations")
    if args.half != "all":
        experiment_args += ["--half", args.half]
    if args.corpus_limit:
        experiment_args += ["--corpus-limit", str(args.corpus_limit)]

    sys.argv = ["run_experiments"] + experiment_args
    code = run_experiments.main()
    if code != 0:
        return code

    print("\n" + "=" * 72)
    sys.argv = ["run_error_analysis"]
    code = run_error_analysis.main()
    if code != 0:
        return code

    print(f"\nAll results written to {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

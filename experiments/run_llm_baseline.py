#!/usr/bin/env python3
"""The LLM baseline, run locally on an open instruction-tuned model.

    python -m experiments.run_llm_baseline            # all documents
    python -m experiments.run_llm_baseline --limit 5  # smoke test, separate file

A 2026 reviewer expects to see how a general-purpose language model does when
simply asked for the section. This asks Qwen2.5-7B-Instruct, with the same
prompt as the paid API baseline, and compares it with the full system on the
same documents, on all of them and on the human-verified ones alone. The model
must not come from the family that produced the labels, or it would be graded
on its own answers.

Kept out of ``make reproduce``: it downloads 15.2 GB of weights on first use,
and reproducing the paper should never depend on that.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np

from experiments.config import RESULTS_DIR, ensure_dirs, set_seed
from experiments.data import load_labeled_samples
from experiments.metrics import bootstrap_ci, evaluate_sector_predictions, mcnemar_exact
from experiments.systems import LOCAL_LLM, LocalLLMRanker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=LOCAL_LLM)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N documents; writes llm_baseline_smoke.json.")
    args = parser.parse_args()

    set_seed()
    ensure_dirs()
    samples = load_labeled_samples()
    if args.limit:
        samples = samples[: args.limit]

    import torch
    import transformers

    ranker = LocalLLMRanker(args.model)
    print(f"{args.model} on {ranker.device}: {len(samples)} documents")
    started = time.perf_counter()
    ranked = []
    for i, sample in enumerate(samples, 1):
        ranked.append([code for code, _ in ranker.rank(sample.purpose)])
        if i % 25 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)}  ({time.perf_counter() - started:.0f} s)", flush=True)
    seconds = time.perf_counter() - started

    full = {d["id"]: d for d in json.loads(
        (RESULTS_DIR / "baselines_predictions.json").read_text(encoding="utf-8"))["full"]}
    llm_ok = [r[0] == s.true_sector for s, r in zip(samples, ranked)]
    full_ok = [full[s.id]["predicted"] == s.true_sector for s in samples]
    verified = [i for i, s in enumerate(samples) if s.annotation_method == "human_verified"]
    llm_verified = [llm_ok[i] for i in verified]
    full_verified = [full_ok[i] for i in verified]

    payload = {
        "model": args.model,
        "device": ranker.device,
        "decoding": "greedy, at most 16 new tokens",
        "prompt": "experiments.systems.llm_prompt, the same as the API baseline",
        "complete": args.limit is None,
        "wall_clock_seconds": round(seconds, 1),
        "off_format_replies": ranker.off_format,
        "sector": evaluate_sector_predictions([s.true_sector for s in samples], ranked),
        "mcnemar_vs_full": mcnemar_exact(full_ok, llm_ok),
        "verified": {
            "n": len(verified),
            "top1_llm": float(np.mean(llm_verified)) if verified else None,
            "top1_llm_ci95": list(bootstrap_ci(llm_verified)) if verified else None,
            "top1_full": float(np.mean(full_verified)) if verified else None,
            "mcnemar_vs_full": mcnemar_exact(full_verified, llm_verified) if verified else None,
        },
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "python": platform.python_version()},
        "predictions": [{"id": s.id, "true": s.true_sector, "predicted": r[0], "top3": r[:3]}
                        for s, r in zip(samples, ranked)],
    }
    out = RESULTS_DIR / ("llm_baseline.json" if payload["complete"] else "llm_baseline_smoke.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    sector = payload["sector"]
    print(f"\nTop-1 {sector['top1_accuracy']:.1%}  Top-3 {sector['top3_accuracy']:.1%}  "
          f"F1 {sector['f1_macro']:.3f}  p vs full {payload['mcnemar_vs_full']['p_value']:.4f}")
    if verified:
        v = payload["verified"]
        print(f"Human-verified ({v['n']}): LLM {v['top1_llm']:.1%} vs full {v['top1_full']:.1%}  "
              f"p = {v['mcnemar_vs_full']['p_value']:.4f}")
    print(f"Replies not in the requested format: {ranker.off_format}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

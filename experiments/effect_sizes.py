#!/usr/bin/env python3
"""Every comparison the paper states as a finding, with an interval and a correction.

    python -m experiments.effect_sizes

The paper reports exact McNemar p-values one comparison at a time. A reviewer
asks two further questions: how large is each difference, with what
uncertainty, and how many of the "significant" results survive once all the
tests the paper runs are counted together?

Both can be answered from the saved results without running a model. For paired
correct/incorrect outcomes the difference in accuracy depends only on the two
discordant counts and the number of documents, so a bootstrap over documents
can be drawn from those counts exactly. The Holm correction then runs over the
whole family of headline tests, the correlation tests included.
"""

from __future__ import annotations

import json
import sys
from typing import Dict, List, Optional

import numpy as np

from experiments.config import N_BOOTSTRAP, RESULTS_DIR, SEED

ALPHA = 0.05


def diff_ci(a_only: int, b_only: int, n: int, sign: int = 1, seed: int = SEED) -> List[float]:
    """Percentile bootstrap interval for sign * (a_only - b_only) / n, in points."""
    if n == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, [a_only / n, b_only / n, 1 - (a_only + b_only) / n], size=N_BOOTSTRAP)
    diffs = sign * 100 * (draws[:, 0] - draws[:, 1]) / n
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm–Bonferroni adjusted p-values, monotone and capped at 1."""
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def load(name: str) -> Optional[Dict]:
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


ROCCHIO_STUDIES = (("rocchio", "terse"), ("rocchio_definitions", "definitions"),
                   ("rocchio_mpnet", "terse, mpnet"))


def family() -> List[Dict]:
    """The headline comparisons: (label, discordant counts, n, orientation)."""
    rows = []

    def add(group, label, counts, n, sign=1):
        rows.append({"group": group, "label": label, "n": n, "sign": sign,
                     "a_only": counts["a_only_correct"], "b_only": counts["b_only_correct"],
                     "p": counts["p_value"]})

    study = load("description_study")
    add("Descriptions", "NACE: definitions vs. terse German", study["content_effect"], study["n"])
    add("Descriptions", "NACE: terse German vs. terse original", study["language_effect"], study["n"])
    news = load("replication_20newsgroups")
    add("Descriptions", "20NG: definitions vs. readable names", news["defining_the_class"], news["n"])
    add("Descriptions", "20NG: readable names vs. identifiers", news["spelling_the_label_out"], news["n"])
    reuters = load("reuters")
    add("Descriptions", "Reuters: definitions vs. readable names", reuters["defining_the_class"], reuters["n"])
    add("Descriptions", "Reuters: readable names vs. codes", reuters["spelling_the_label_out"], reuters["n"])

    for s in load("baselines")["systems"]:
        if s["key"] in {"tfidf-nace", "embed-zeroshot"}:
            add("System", f"full system vs. {s['label']}", s["mcnemar_vs_full"], s["sector"]["n"])
    for s in load("ablation")["systems"]:
        if s["key"] != "full":
            add("Ablation", f"full system vs. {s['label']}", s["mcnemar_vs_full"], s["sector"]["n"])

    llm = load("llm_baseline")
    if llm:
        add("LLM", "full system vs. LLM, Top-1", llm["mcnemar_vs_full"], llm["sector"]["n"])
        add("LLM", "full system vs. LLM, Top-3", llm["mcnemar_top3_vs_full"], llm["sector"]["n"])
        add("LLM", "full system vs. LLM, verified labels", llm["verified"]["mcnemar_vs_full"], llm["verified"]["n"])
    robustness = load("robustness")
    add("Verified labels", "full system vs. previous taxonomy",
        robustness["labels"]["systems"]["taxonomy-v1"]["vs_full_on_verified"], robustness["labels"]["n_verified"])

    for name, label in ROCCHIO_STUDIES:
        result = load(name)
        if not result:
            continue
        sizes = {"NACE": study["n"], "20NG": news["n"], "Reuters": reuters["n"]}
        for corpus, r in result["corpora"].items():
            add("Label-free update", f"{corpus}: {label} + update vs. {label}", r["mcnemar"], sizes[corpus], sign=-1)
    return rows


def selection_rules() -> List[Dict]:
    """The three per-class selection criteria, each against the best fixed policy.

    The paper reports all six as a finding (none generalises), so all six belong
    in the family. Their result files keep the difference and the p-value but not
    the discordant counts, so they enter the correction without an interval.
    """
    names = {"marginal": "marginal alignment", "contrastive": "contrastive margin",
             "greedy_set_search": "greedy search on development accuracy"}
    return [{"group": "Selection rule", "label": f"{corpus}: {names[rule]} vs. best fixed policy",
             "gain_pp": r["delta_pp"], "p": r["p_value"]}
            for corpus, rules in load("predictor_search")["selection_rules"].items()
            for rule, r in rules.items()]


def correlations() -> List[Dict]:
    out = [{"group": "Correlation", "label": "alignment change vs. headroom captured",
            "p": load("predictor_search")["correlations"]["alignment_gain"]["p"]}]
    for name, label in ROCCHIO_STUDIES:
        result = load(name)
        if result:
            out.append({"group": "Correlation", "label": f"label-free update ({label}), per class",
                        "p": result["per_class"]["p"]})
    return out


def main() -> int:
    rows = family()
    for r in rows:
        r["gain_pp"] = r["sign"] * 100 * (r["a_only"] - r["b_only"]) / r["n"]
        r["ci95"] = diff_ci(r["a_only"], r["b_only"], r["n"], r["sign"])
    tests = rows + selection_rules() + correlations()
    adjusted = holm({str(i): t["p"] for i, t in enumerate(tests)})
    for i, t in enumerate(tests):
        t["p_holm"] = adjusted[str(i)]
    nominal = [t for t in tests if t["p"] < ALPHA]
    survive = [t for t in nominal if t["p_holm"] < ALPHA]
    payload = {
        "alpha": ALPHA, "n_tests": len(tests),
        "nominally_significant": len(nominal), "significant_after_holm": len(survive),
        "lost_to_correction": [t["label"] for t in nominal if t["p_holm"] >= ALPHA],
        "tests": tests,
    }
    (RESULTS_DIR / "effect_sizes.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{len(tests)} tests; {len(nominal)} significant at {ALPHA}, {len(survive)} after Holm.")
    for t in tests:
        ci = t.get("ci95")
        interval = f"[{ci[0]:+6.1f}, {ci[1]:+6.1f}]" if ci else " " * 16
        gain = f"{t['gain_pp']:+6.1f}" if "gain_pp" in t else "      "
        print(f"  {t['label'][:52]:52s} {gain} {interval}  p {t['p']:.4f}  Holm {t['p_holm']:.4f}")
    if payload["lost_to_correction"]:
        print("Lost to the correction:", "; ".join(payload["lost_to_correction"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

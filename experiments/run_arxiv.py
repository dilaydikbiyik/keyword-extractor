#!/usr/bin/env python3
"""The claim at scale, on labels we did not make and descriptions we did not write.

    python -m experiments.run_arxiv --preregister   # commit the prediction
    python -m experiments.run_arxiv                 # then test it

Three objections apply to every result in this paper: the evaluation set is
small, its labels are model-assisted, and the elaborated class descriptions were
written for the study by the assistant that also produced those labels. This
corpus answers all three at once.

  * the labels are the primary arXiv category each paper's own authors chose at
    submission, kept only where the paper has no other Computer Science
    category, so the label is unambiguous;
  * the elaborated class text is arXiv's published description of the category,
    quoted verbatim -- written by arXiv, for its own submitters, years before
    this study;
  * the sample runs to thousands of abstracts across \u007fARXIV_CLASSES\u007f categories.

The three conditions mirror 20 Newsgroups and Brown: the bare identifier
(``cs.CL``), the readable name (``Computation and Language``), and the official
description. arXiv's descriptions are uneven by nature -- some enumerate a field,
others say only which ACM subject class a category covers -- and that unevenness
is the point: it varies the one quantity the account says matters, without
anyone writing a word for the experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from experiments.config import RESULTS_DIR, SEED, TABLES_DIR, ensure_dirs, set_seed
from experiments.fetch_arxiv import CACHE, SAMPLE_IDS, TAXONOMY
from experiments.metrics import bootstrap_ci, mcnemar_exact
from experiments.run_rocchio import alignment, git_revision
from experiments.systems import get_embedder

PREREGISTRATION = RESULTS_DIR / "arxiv_preregistration.json"
RESULT = RESULTS_DIR / "arxiv.json"


def load_corpus() -> Tuple[List[str], List[str], List[str]]:
    """The committed identifier list, with its abstracts from the local cache."""
    if not SAMPLE_IDS.exists() or not CACHE.exists():
        raise SystemExit("Fetch the corpus first: python -m experiments.fetch_arxiv --taxonomy "
                         "then python -m experiments.fetch_arxiv")
    wanted = json.loads(SAMPLE_IDS.read_text(encoding="utf-8"))["documents"]
    cached = json.loads(CACHE.read_text(encoding="utf-8"))
    texts, labels = [], []
    for category in sorted(wanted):
        abstracts = {e["id"]: e["abstract"] for e in cached.get(category, [])}
        for identifier in wanted[category]:
            if identifier in abstracts:
                texts.append(abstracts[identifier])
                labels.append(category)
    names = sorted(wanted)
    missing = sum(len(v) for v in wanted.values()) - len(texts)
    if missing:
        print(f"  {missing} identifiers are not in the local cache; re-run the fetch to restore them",
              file=sys.stderr)
    return texts, labels, names


def conditions(names: List[str]) -> Dict[str, Dict[str, str]]:
    """Identifier, readable name, and arXiv's own description of the category."""
    official = json.loads(TAXONOMY.read_text(encoding="utf-8"))["categories"]
    return {
        "raw class identifier": {n: n for n in names},
        "readable class name": {n: official[n]["name"] for n in names},
        "arXiv's own description": {n: f"{official[n]['name']}. {official[n]['description']}"
                                    for n in names},
    }


def embed(texts: List[str], batch: int = 64) -> np.ndarray:
    matrix = np.vstack([np.asarray(v, dtype=np.float64)
                        for v in get_embedder().embed_texts(list(texts), batch_size=batch)])
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def method_fingerprint() -> str:
    """The corpus, the class texts and the measurement, as registered."""
    parts = [f"SEED={SEED}",
             hashlib.sha256(SAMPLE_IDS.read_bytes()).hexdigest(),
             hashlib.sha256(TAXONOMY.read_bytes()).hexdigest()]
    parts += [inspect.getsource(f) for f in (load_corpus, conditions, embed, alignment)]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def alignment_per_condition(docs: np.ndarray, labels: List[str], names: List[str]) -> Dict:
    return {name: alignment(embed([texts[n] for n in names]), docs, labels, names)
            for name, texts in conditions(names).items()}


def predictions_from(changes: Dict[str, float]) -> List[Dict]:
    out = [{"id": f"direction-{step}", "step": step, "expected_sign": 1 if d > 0 else -1,
            "claim": f"On arXiv, {step} {'raises' if d > 0 else 'does not raise'} Top-1 accuracy."}
           for step, d in changes.items()]
    out.append({"id": "per-class",
                "claim": "Across the categories, the alignment change from the official "
                         "description correlates positively with the share of available headroom "
                         "captured (Spearman rho > 0)."})
    out.append({"id": "per-class-significant",
                "claim": "With this many categories that correlation is significant at 0.05 "
                         "uncorrected, which the three-corpus version of the test could not show."})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", action="store_true")
    args = parser.parse_args()
    set_seed()
    ensure_dirs()

    texts, labels, names = load_corpus()
    print(f"{len(texts)} abstracts, {len(names)} categories", flush=True)
    docs = embed(texts)
    per_condition = alignment_per_condition(docs, labels, names)
    keys = list(conditions(names))
    steps = {
        "spelling the label out": {n: per_condition[keys[1]][n] - per_condition[keys[0]][n]
                                   for n in per_condition[keys[0]]},
        "the official description": {n: per_condition[keys[2]][n] - per_condition[keys[1]][n]
                                     for n in per_condition[keys[0]]},
    }
    changes = {step: float(np.mean(list(v.values()))) for step, v in steps.items()}

    if args.preregister:
        if PREREGISTRATION.exists():
            print(f"{PREREGISTRATION} already exists; a prediction is made once.", file=sys.stderr)
            return 1
        for step, change in changes.items():
            print(f"  {step:26s} mean alignment change {change:+.4f}")
        payload = {
            "study": "arxiv",
            "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_revision": git_revision(),
            "method_fingerprint": method_fingerprint(),
            "seed": SEED, "n": len(texts), "classes": len(names),
            "labels": "the primary arXiv category chosen by each paper's authors",
            "class_text": "arXiv's published category descriptions, quoted verbatim",
            "mean_alignment_change": changes,
            "alignment_change_per_class": steps,
            "predictions": predictions_from(changes),
            "note": "Written before any accuracy on this corpus was computed.",
        }
        PREREGISTRATION.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
        for p in payload["predictions"]:
            print(f"  PREDICTION {p['id']}: {p['claim']}")
        print(f"Wrote {PREREGISTRATION}. Commit it before running the accuracy step.")
        return 0

    if not PREREGISTRATION.exists():
        print("No preregistration: run with --preregister and commit the file first.", file=sys.stderr)
        return 1
    registered = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if registered.get("method_fingerprint") != method_fingerprint():
        print("The corpus or the class texts changed after the prediction was registered; "
              "refusing to test it.", file=sys.stderr)
        return 1

    results, correct = {}, {}
    for name, texts_by_class in conditions(names).items():
        vectors = embed([texts_by_class[n] for n in names])
        order = np.argsort(-(docs @ vectors.T), axis=1)
        predicted = [names[order[i, 0]] for i in range(len(labels))]
        hits = [labels[i] == predicted[i] for i in range(len(labels))]
        correct[name] = hits
        lo, hi = bootstrap_ci(hits)
        results[name] = {
            "top1_accuracy": float(np.mean(hits)),
            "top1_ci95": [lo, hi],
            "top3_accuracy": float(np.mean([labels[i] in [names[j] for j in order[i, :3]]
                                            for i in range(len(labels))])),
            "f1_macro": float(f1_score(labels, predicted, average="macro", zero_division=0)),
            "mean_words_per_class": float(np.mean([len(d.split())
                                                   for d in texts_by_class.values()])),
        }
        print(f"  {name:<26} {results[name]['top1_accuracy']:.1%}", flush=True)

    spelling = mcnemar_exact(correct[keys[1]], correct[keys[0]])
    spelling["gain_pp"] = 100 * (results[keys[1]]["top1_accuracy"]
                                 - results[keys[0]]["top1_accuracy"])
    describing = mcnemar_exact(correct[keys[2]], correct[keys[1]])
    describing["gain_pp"] = 100 * (results[keys[2]]["top1_accuracy"]
                                   - results[keys[1]]["top1_accuracy"])

    present = [n for n in names if n in set(labels)]
    recall = {name: {n: float(np.mean([h for h, g in zip(correct[name], labels) if g == n]))
                     for n in present} for name in correct}
    pooled = [(registered["alignment_change_per_class"]["the official description"][n],
               (recall[keys[2]][n] - recall[keys[1]][n]) / (1 - recall[keys[1]][n]))
              for n in present if recall[keys[1]][n] < 0.999]
    rho, p_rho = spearmanr([a for a, _ in pooled], [s for _, s in pooled])

    gains = {"spelling the label out": spelling["gain_pp"],
             "the official description": describing["gain_pp"]}
    outcomes = []
    for pred in registered["predictions"]:
        if pred["id"].startswith("direction-"):
            held = np.sign(gains[pred["step"]]) == pred["expected_sign"]
        elif pred["id"] == "per-class":
            held = bool(rho > 0)
        else:
            held = bool(rho > 0 and p_rho < 0.05)
        outcomes.append({**pred, "held": bool(held)})
        print(f"  {'HELD  ' if held else 'FAILED'} {pred['claim']}")

    rows = ["| Class descriptions | Top-1 | 95% CI | Top-3 | F1-macro | words/class |",
            "| --- | --- | --- | --- | --- | --- |"]
    for name in keys:
        r = results[name]
        lo, hi = r["top1_ci95"]
        rows.append(f"| {name} | {r['top1_accuracy']:.1%} | [{lo:.1%}, {hi:.1%}] | "
                    f"{r['top3_accuracy']:.1%} | {r['f1_macro']:.3f} | "
                    f"{r['mean_words_per_class']:.0f} |")
    (TABLES_DIR / "arxiv.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    payload = {
        "study": "arxiv",
        "preregistration": PREREGISTRATION.name,
        "registered_at": registered["registered_at"],
        "n": len(texts), "classes": len(names), "classes_with_documents": len(present),
        "labels": registered["labels"], "class_text": registered["class_text"],
        "conditions": results,
        "spelling_the_label_out": spelling,
        "defining_the_class": describing,
        "mean_alignment_change": changes,
        "per_class": {"n": len(pooled), "rho": float(rho), "p": float(p_rho)},
        "outcomes": outcomes,
    }
    RESULT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  per-class rho = {rho:+.3f} (p = {p_rho:.2e}, n = {len(pooled)})")
    print(f"Wrote {RESULT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

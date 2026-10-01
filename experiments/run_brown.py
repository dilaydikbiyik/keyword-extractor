#!/usr/bin/env python3
"""A fourth corpus, chosen for the case the account finds hardest.

    python -m experiments.run_brown --preregister   # commit the prediction
    python -m experiments.run_brown                 # then test it

The account says what a class text buys is alignment with the documents it has
to attract. Its three corpora have well-separated alignment changes, which is
what makes the ordering visible and equally what keeps the test away from the
hard case: a corpus where the change is small, and the account therefore has to
predict a small gain rather than a large one.

Brown is that case by construction. Its classes are genres -- news, editorial,
romance, learned -- not topics, so writing out what a genre covers names the
subjects its documents discuss without naming the property that makes a
document belong to it. The three conditions mirror 20 Newsgroups:

  1. the raw category identifier, as the corpus ships it ("belles_lettres");
  2. a readable name for the same category ("Belles lettres");
  3. a definition enumerating what the category covers.

The documents are long -- Brown samples run to some two thousand words -- and
the encoder truncates, so each document enters as its first \u007fTRUNCATE\u007f characters.
That is a property of the corpus, not a choice made after seeing a result: it is
fixed here, covered by the fingerprint, and the same for every condition.
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
from experiments.metrics import bootstrap_ci, mcnemar_exact
from experiments.run_rocchio import alignment, git_revision
from experiments.systems import get_embedder

PREREGISTRATION = RESULTS_DIR / "brown_preregistration.json"
RESULT = RESULTS_DIR / "brown.json"
TRUNCATE = 1200
DOCS_PER_CLASS = 40

READABLE = {
    "adventure": "Adventure fiction",
    "belles_lettres": "Belles lettres",
    "editorial": "Newspaper editorial",
    "fiction": "General fiction",
    "government": "Government documents",
    "hobbies": "Hobbies and pastimes",
    "humor": "Humour",
    "learned": "Academic and scientific writing",
    "lore": "Popular lore",
    "mystery": "Mystery and detective fiction",
    "news": "Newspaper reporting",
    "religion": "Religious writing",
    "reviews": "Arts reviews",
    "romance": "Romance fiction",
    "science_fiction": "Science fiction",
}

# Written in the same style as the other corpora's definitions: name the class,
# then enumerate what it covers, in the vocabulary its documents use.
DEFINITION = {
    "adventure": "Adventure fiction. Westerns and tales of pursuit, gunfights and horses, ranches and frontier towns, danger and escape, men acting under physical threat.",
    "belles_lettres": "Belles lettres. Essays, biography and memoir, literary and cultural criticism, reflections on art, history and society, written for a general educated reader.",
    "editorial": "Newspaper editorial. Opinion and comment on public affairs, argument for or against a policy, elections and taxation, criticism of officials, the newspaper's own position.",
    "fiction": "General fiction. Novels and short stories of ordinary life, families and marriages, towns and neighbours, work and money, remembered childhoods.",
    "government": "Government documents. Statutes and regulations, official reports and budgets, agencies and committees, programmes and appropriations, administrative procedure.",
    "hobbies": "Hobbies and pastimes. Practical instruction for amateurs: gardening, sailing, antiques, photography, model building, home repair, collecting, equipment and technique.",
    "humor": "Humour. Comic sketches and satire, jokes and wordplay, mock advice and parody, the writer's own absurd predicaments.",
    "learned": "Academic and scientific writing. Research reports and scholarly argument, data and measurement, theory and method, citations and footnotes, medicine, physics, law and the social sciences.",
    "lore": "Popular lore. Magazine articles explaining a subject to the general reader: travel and places, history retold, personalities, science popularised, customs and curiosities.",
    "mystery": "Mystery and detective fiction. Murders and investigations, detectives and suspects, clues and alibis, police and courtrooms, crime in the city.",
    "news": "Newspaper reporting. Factual accounts of events: city and state politics, courts and crime, business and labour, sports results, named officials and dates.",
    "religion": "Religious writing. God and faith, scripture and its interpretation, churches and congregations, prayer and worship, sin and salvation, theology and the Christian life.",
    "reviews": "Arts reviews. Criticism of performances and works: concerts and recordings, theatre and film, exhibitions and books, judgement of the artist's achievement.",
    "romance": "Romance fiction. Love and courtship, longing and jealousy, domestic scenes and intimate conversation, emotional crises between a man and a woman.",
    "science_fiction": "Science fiction. Space travel and other planets, machines and experiments, the future and alternative worlds, aliens and mutants, science pushed past its limits.",
}


def load_brown() -> Tuple[List[str], List[str], List[str]]:
    """Brown documents, truncated, balanced as far as the corpus allows."""
    import nltk

    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)
    from nltk.corpus import brown

    rng = np.random.default_rng(SEED)
    names = sorted(brown.categories())
    texts, labels = [], []
    for category in names:
        ids = sorted(brown.fileids(category))
        take = min(DOCS_PER_CLASS, len(ids))
        for fileid in rng.choice(ids, size=take, replace=False):
            texts.append(" ".join(brown.words(fileid))[:TRUNCATE])
            labels.append(category)
    return texts, labels, names


def conditions(names: List[str]) -> Dict[str, Dict[str, str]]:
    return {
        "raw class identifier": {n: n for n in names},
        "readable class name": {n: READABLE[n] for n in names},
        "definition of what the class covers": {n: DEFINITION[n] for n in names},
    }


def embed(texts: List[str]) -> np.ndarray:
    embedder = get_embedder()
    matrix = np.vstack([np.asarray(embedder.embed_text(t), dtype=np.float64) for t in texts])
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def method_fingerprint() -> str:
    """Everything that fixes the documents, the class texts and the measurement."""
    parts = [f"SEED={SEED}", f"TRUNCATE={TRUNCATE}", f"DOCS_PER_CLASS={DOCS_PER_CLASS}",
             json.dumps(READABLE, sort_keys=True), json.dumps(DEFINITION, sort_keys=True)]
    parts += [inspect.getsource(f) for f in (load_brown, conditions, embed, alignment)]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def alignment_changes(docs: np.ndarray, labels: List[str], names: List[str]) -> Dict:
    """Per-class alignment for each condition, which needs no accuracy."""
    out = {}
    for name, texts in conditions(names).items():
        vectors = embed([texts[n] for n in names])
        out[name] = alignment(vectors, docs, labels, names)
    return out


def predictions_from(changes: Dict[str, float], reference: Dict[str, float]) -> List[Dict]:
    """Mechanical consequences of the account for this corpus."""
    out = [{"id": f"direction-{step}",
            "claim": f"On Brown, {step} {'raises' if d > 0 else 'does not raise'} Top-1 accuracy.",
            "step": step, "expected_sign": 1 if d > 0 else -1}
           for step, d in changes.items()]
    smaller = [c for c, d in reference.items() if changes["definition"] < d]
    out.append({"id": "smaller-than", "corpora": smaller,
                "claim": "Brown's gain from a definition is smaller than the gain of every corpus "
                         "whose alignment change is larger: " + ", ".join(smaller) + "."})
    out.append({"id": "per-class",
                "claim": "Across classes, the alignment change from the definition correlates "
                         "positively with the share of available headroom captured (rho > 0)."})
    return out


def reference_changes() -> Dict[str, float]:
    """The alignment changes the three published corpora reported, for the ordering.

    Read from the same files the paper's own table reads, so the comparison is
    against the published numbers and not a fresh measurement of them.
    """
    def read(name):
        return json.loads((RESULTS_DIR / f"{name}.json").read_text(encoding="utf-8"))

    level = read("predictor_search")["dataset_level"]
    return {
        "NACE": float(level["NACE"]["mean_alignment_gain"]),
        "Reuters": float(read("reuters")["prediction_from_alignment"]
                         ["alignment_change_readable_to_definition"]),
        "20NG": float(level["20NG"]["mean_alignment_gain"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregister", action="store_true")
    args = parser.parse_args()
    set_seed()
    ensure_dirs()

    texts, labels, names = load_brown()
    print(f"{len(texts)} documents, {len(names)} classes", flush=True)
    docs = embed(texts)
    per_class = alignment_changes(docs, labels, names)
    keys = list(conditions(names))
    steps = {
        "spelling the label out": {n: per_class[keys[1]][n] - per_class[keys[0]][n]
                                   for n in per_class[keys[0]]},
        "definition": {n: per_class[keys[2]][n] - per_class[keys[1]][n]
                       for n in per_class[keys[0]]},
    }
    changes = {step: float(np.mean(list(v.values()))) for step, v in steps.items()}

    if args.preregister:
        if PREREGISTRATION.exists():
            print(f"{PREREGISTRATION} already exists; a prediction is made once.", file=sys.stderr)
            return 1
        reference = reference_changes()
        for step, change in changes.items():
            print(f"  {step:24s} mean alignment change {change:+.4f}")
        print(f"  published corpora: {', '.join(f'{c} {d:+.4f}' for c, d in reference.items())}")
        payload = {
            "study": "brown",
            "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_revision": git_revision(),
            "method_fingerprint": method_fingerprint(),
            "seed": SEED, "truncate": TRUNCATE, "docs_per_class": DOCS_PER_CLASS,
            "n": len(texts), "classes": len(names),
            "mean_alignment_change": changes,
            "alignment_change_per_class": steps,
            "published_corpora_alignment_change": reference,
            "predictions": predictions_from(changes, reference),
            "note": "Written before any accuracy on Brown was computed.",
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
        print(f"  {name:<38} {results[name]['top1_accuracy']:.1%}", flush=True)

    spelling_out = mcnemar_exact(correct[keys[1]], correct[keys[0]])
    spelling_out["gain_pp"] = 100 * (results[keys[1]]["top1_accuracy"]
                                     - results[keys[0]]["top1_accuracy"])
    defining = mcnemar_exact(correct[keys[2]], correct[keys[1]])
    defining["gain_pp"] = 100 * (results[keys[2]]["top1_accuracy"]
                                 - results[keys[1]]["top1_accuracy"])

    recall = {name: {n: float(np.mean([h for h, g in zip(correct[name], labels) if g == n]))
                     for n in names} for name in correct}
    pooled = [(registered["alignment_change_per_class"]["definition"][n],
               (recall[keys[2]][n] - recall[keys[1]][n]) / (1 - recall[keys[1]][n]))
              for n in names if recall[keys[1]][n] < 0.999]
    rho, p_rho = spearmanr([a for a, _ in pooled], [s for _, s in pooled])

    gains = {"spelling the label out": spelling_out["gain_pp"], "definition": defining["gain_pp"]}
    outcomes = []
    for pred in registered["predictions"]:
        if pred["id"].startswith("direction-"):
            held = np.sign(gains[pred["step"]]) == pred["expected_sign"]
        elif pred["id"] == "smaller-than":
            reference = json.loads((RESULTS_DIR / "effect_sizes.json").read_text(encoding="utf-8"))
            published = {t["label"]: t["gain_pp"] for t in reference["tests"] if "gain_pp" in t}
            larger = {"NACE": published.get("NACE: definitions vs. terse German"),
                      "20NG": published.get("20NG: definitions vs. readable names"),
                      "Reuters": published.get("Reuters: definitions vs. readable names")}
            held = all(gains["definition"] < larger[c] for c in pred["corpora"] if larger.get(c))
        else:
            held = bool(rho > 0)
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
    (TABLES_DIR / "brown.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    payload = {
        "study": "brown",
        "preregistration": PREREGISTRATION.name,
        "registered_at": registered["registered_at"],
        "n": len(texts), "classes": len(names), "truncate": TRUNCATE,
        "conditions": results,
        "spelling_the_label_out": spelling_out,
        "defining_the_class": defining,
        "mean_alignment_change": changes,
        "per_class": {"n": len(pooled), "rho": float(rho), "p": float(p_rho)},
        "outcomes": outcomes,
    }
    RESULT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  per-class rho = {rho:+.3f} (p = {p_rho:.4f}, n = {len(pooled)})")
    print(f"Wrote {RESULT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

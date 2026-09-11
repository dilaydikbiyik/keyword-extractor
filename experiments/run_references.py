#!/usr/bin/env python3
"""Two questions a reviewer asks of any zero-shot result, answered on the same documents.

    python -m experiments.run_references            # uses the committed statistics
    python -m experiments.run_references --refit    # re-derive them from data/raw/

1. **Was the lexical baseline given a fair chance?** The TF-IDF baseline matches
   whole words, and German builds words by compounding: *Bodenbelagsarbeiten*
   never matches *Boden*. Two stronger lexical spaces, over the same sector texts
   the baseline uses: German Snowball stems, and character 3-5-grams within word
   boundaries, which see inside compounds.
2. **What would labels buy?** Supervised classifiers trained on the evaluation
   documents themselves, scored out of fold under stratified 5-fold
   cross-validation: TF-IDF with multinomial Naive Bayes, and logistic
   regression on the same sentence embeddings the zero-shot system uses. They
   see four fifths of the labels; the zero-shot system sees none. They are a
   reference for what labels are worth here, not competitors.

Neither was preregistered; the paper reports both as exploratory and they enter
the Holm family with everything else.

Each lexical space is frozen as a vocabulary and rounded IDF weights under
``data/derived/``, like the baseline's, because the raw corpus is not
redistributed. Both the refit and the committed path rank through the same
frozen arithmetic, so a clone reproduces these numbers exactly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from experiments.config import RESULTS_DIR, ROOT, SEED, ensure_dirs, set_seed
from experiments.data import load_corpus, load_labeled_samples, load_taxonomy
from experiments.metrics import bootstrap_ci, cohen_kappa, f1_macro, mcnemar_exact

DERIVED = ROOT / "data" / "derived"
RESULT = RESULTS_DIR / "references.json"
FOLDS = 5
MIN_DF, MAX_DF = 2, 0.6
CHAR_MAX_FEATURES = 100_000
TOKEN = re.compile(r"(?u)\b\w\w+\b")


def snowball_analyzer() -> Callable[[str], List[str]]:
    """Lower-cased word tokens, German Snowball stems, then unigrams and bigrams."""
    from nltk.stem.snowball import SnowballStemmer

    stem = SnowballStemmer("german").stem

    def analyze(text: str) -> List[str]:
        stems = [stem(t) for t in TOKEN.findall(text.lower())]
        return stems + [f"{a} {b}" for a, b in zip(stems, stems[1:])]

    return analyze


def char_analyzer() -> Callable[[str], List[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5)).build_analyzer()


SPACES = {
    "tfidf-snowball": ("TF-IDF, German Snowball stems", snowball_analyzer, None),
    "tfidf-char": ("TF-IDF, character 3-5-grams", char_analyzer, CHAR_MAX_FEATURES),
}


@dataclass
class FrozenSpace:
    """A TF-IDF space fixed as vocabulary and IDF: sublinear tf, times IDF, L2-normalised."""

    vocabulary: Dict[str, int]
    idf: np.ndarray
    analyze: Callable[[str], List[str]]

    def transform(self, text: str) -> np.ndarray:
        counts = np.zeros(len(self.vocabulary))
        for term in self.analyze(text):
            index = self.vocabulary.get(term)
            if index is not None:
                counts[index] += 1.0
        present = counts > 0
        counts[present] = 1.0 + np.log(counts[present])
        weights = counts * self.idf
        norm = np.linalg.norm(weights)
        return weights / norm if norm > 0 else weights


def stats_path(key: str):
    return DERIVED / f"{key.replace('-', '_')}_corpus_stats.json"


def fit_space(key: str, corpus: Sequence[str], sector_docs: Sequence[str]) -> None:
    """Derive the vocabulary and IDF from the unlabelled corpus and write them."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    label, make_analyzer, max_features = SPACES[key]
    vectorizer = TfidfVectorizer(analyzer=make_analyzer(), min_df=MIN_DF, max_df=MAX_DF,
                                 sublinear_tf=True, max_features=max_features)
    vectorizer.fit(list(corpus) + list(sector_docs))
    payload = {
        "description": f"{label}: vocabulary and IDF weights derived from the Handelsregister "
                       "sample. Aggregate statistics only; the records are not redistributed.",
        "n_documents": len(corpus),
        "n_features": len(vectorizer.vocabulary_),
        "idf": [round(float(v), 6) for v in vectorizer.idf_],
        "vocabulary": {term: int(i) for term, i in vectorizer.vocabulary_.items()},
    }
    path = stats_path(key)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  {path.relative_to(ROOT)}: {payload['n_features']:,} features, "
          f"{path.stat().st_size / 1024 / 1024:.1f} MB")


def load_space(key: str) -> FrozenSpace:
    payload = json.loads(stats_path(key).read_text(encoding="utf-8"))
    return FrozenSpace(payload["vocabulary"], np.asarray(payload["idf"]), SPACES[key][1]())


def sector_documents(taxonomy: Dict[str, dict], codes: Sequence[str]) -> List[str]:
    """Name, description and seed list: the pseudo-document the TF-IDF baseline uses."""
    return [" ".join([taxonomy[c].get("name", ""), taxonomy[c].get("description", ""),
                      *taxonomy[c].get("seed_keywords", [])]) for c in codes]


def lexical_rankings(space: FrozenSpace, texts: Sequence[str], sector_docs: Sequence[str],
                     codes: Sequence[str]) -> List[List[str]]:
    sectors = np.vstack([space.transform(d) for d in sector_docs])
    docs = np.vstack([space.transform(t) for t in texts])
    return [[codes[i] for i in np.argsort(-row, kind="stable")] for row in docs @ sectors.T]


def out_of_fold(features: Callable[[List[int], List[int]], Tuple[np.ndarray, np.ndarray]],
                make_model: Callable, gold: Sequence[str]) -> List[List[str]]:
    """Ranked predictions for every document from a model that never saw its label."""
    from sklearn.model_selection import StratifiedKFold

    ranked: List[List[str]] = [[] for _ in gold]
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    with warnings.catch_warnings():
        # Sections with fewer documents than folds are expected; they are scored all the same.
        warnings.filterwarnings("ignore", message="The least populated class")
        for train, test in folds.split(np.zeros(len(gold)), gold):
            x_train, x_test = features(list(train), list(test))
            model = make_model().fit(x_train, [gold[i] for i in train])
            for i, probs in zip(test, model.predict_proba(x_test)):
                ranked[i] = [model.classes_[j] for j in np.argsort(-probs, kind="stable")]
    return ranked


def naive_bayes_features(texts: Sequence[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    def features(train: List[int], test: List[int]):
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
        x_train = vectorizer.fit_transform([texts[i] for i in train])
        return x_train, vectorizer.transform([texts[i] for i in test])

    return features


def embedding_features(texts: Sequence[str]):
    from experiments.systems import get_embedder

    embedder = get_embedder()
    matrix = np.vstack([np.asarray(embedder.embed_text(t), dtype=np.float64) for t in texts])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return lambda train, test: (matrix[train], matrix[test])


def scored(key: str, label: str, kind: str, ranked: List[List[str]], gold: Sequence[str],
           full_correct: Sequence[bool]) -> Dict:
    top1 = [r[0] if r else "" for r in ranked]
    hits = [g == p for g, p in zip(gold, top1)]
    lo, hi = bootstrap_ci(hits)
    return {
        "key": key, "label": label, "kind": kind, "n": len(gold),
        "top1_accuracy": float(np.mean(hits)), "top1_ci95": [lo, hi],
        "top3_accuracy": float(np.mean([g in r[:3] for g, r in zip(gold, ranked)])),
        "f1_macro": f1_macro(list(gold), top1), "cohen_kappa": cohen_kappa(list(gold), top1),
        "mcnemar_vs_full": mcnemar_exact(full_correct, hits),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refit", action="store_true",
                        help="Re-derive the lexical statistics from the raw corpus before scoring.")
    args = parser.parse_args()
    set_seed()
    ensure_dirs()

    taxonomy = load_taxonomy()
    codes = sorted(taxonomy)
    sector_docs = sector_documents(taxonomy, codes)
    missing = [k for k in SPACES if not stats_path(k).exists()]
    if args.refit or missing:
        corpus = load_corpus()
        if not corpus:
            print(f"No corpus in this checkout and no statistics for {missing}; see data/README.md.",
                  file=sys.stderr)
            return 1
        for key in SPACES:
            fit_space(key, corpus, sector_docs)

    samples = load_labeled_samples()
    texts, gold = [s.purpose for s in samples], [s.true_sector for s in samples]
    full = {d["id"]: d for d in json.loads((RESULTS_DIR / "baselines_predictions.json")
                                           .read_text(encoding="utf-8"))["full"]}
    if any(full[s.id]["true"] != s.true_sector for s in samples):
        print("The committed predictions disagree with the current labels; run `make reproduce` first.",
              file=sys.stderr)
        return 1
    full_correct = [full[s.id]["predicted"] == s.true_sector for s in samples]
    full_top3 = [s.true_sector in full[s.id]["top3"] for s in samples]

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB

    spaces = {key: load_space(key) for key in SPACES}
    ranked = {key: lexical_rankings(spaces[key], texts, sector_docs, codes) for key in SPACES}
    previous = load_taxonomy(ROOT / "data" / "taxonomy" / "sectors_v1.json")
    assert sorted(previous) == codes
    ranked["tfidf-char-v1"] = lexical_rankings(spaces["tfidf-char"], texts, sector_documents(previous, codes), codes)
    labels = {**{key: label for key, (label, _, _) in SPACES.items()},
              "tfidf-char-v1": "TF-IDF, character 3-5-grams, previous descriptions"}
    systems = [scored(key, labels[key], "zero-shot", ranked[key], gold, full_correct)
               for key in ("tfidf-snowball", "tfidf-char", "tfidf-char-v1")]
    current, before = ([g == r[0] for g, r in zip(gold, ranked[k])] for k in ("tfidf-char", "tfidf-char-v1"))
    description_effect = {"gain_pp": 100 * float(np.mean(current) - np.mean(before)),
                          "mcnemar": mcnemar_exact(current, before)}
    systems.append(scored("supervised-nb", "TF-IDF + Naive Bayes, 5-fold", "supervised",
                          out_of_fold(naive_bayes_features(texts), lambda: MultinomialNB(alpha=1.0), gold),
                          gold, full_correct))
    systems.append(scored("supervised-lr", "Embeddings + logistic regression, 5-fold", "supervised",
                          out_of_fold(embedding_features(texts),
                                      lambda: LogisticRegression(max_iter=5000, random_state=SEED), gold),
                          gold, full_correct))

    payload = {"n": len(samples), "folds": FOLDS, "seed": SEED,
               "full_top1_accuracy": float(np.mean(full_correct)), "full_top3_accuracy": float(np.mean(full_top3)),
               "systems": systems, "description_effect_char": description_effect,
               "note": "Exploratory, not preregistered. Supervised rows are trained out of fold on the "
                       "evaluation labels and are a reference, not a competing zero-shot method."}
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for s in systems:
        test = s["mcnemar_vs_full"]
        print(f"  {s['label']:45s} Top-1 {s['top1_accuracy']:.1%}  Top-3 {s['top3_accuracy']:.1%}  "
              f"vs full {payload['full_top1_accuracy']:.1%}  p = {test['p_value']:.4f}")
    print(f"  character TF-IDF, current vs. previous descriptions: {description_effect['gain_pp']:+.1f} points, "
          f"p = {description_effect['mcnemar']['p_value']:.4f}")
    print(f"Wrote {RESULT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

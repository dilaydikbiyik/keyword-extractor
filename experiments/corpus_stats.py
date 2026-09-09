"""Corpus-derived TF-IDF statistics, so a clone can reproduce without the corpus.

The 9,993 trade register records are not redistributed (see ``data/README.md``).
What the TF-IDF baseline actually needs from them is a vocabulary and a set of
IDF weights, which are aggregate statistics rather than the records themselves.
Those are committed, and this module applies them with the same arithmetic
scikit-learn would: sublinear term frequency, multiply by IDF, L2-normalise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from experiments.config import TFIDF_STATS


@dataclass
class CorpusStatistics:
    """A frozen TF-IDF space: vocabulary, IDF weights, and how to apply them."""

    vocabulary: Dict[str, int]
    idf: np.ndarray
    n_documents: int
    ngram_range: List[int]
    lowercase: bool = True
    sublinear_tf: bool = True

    @property
    def feature_names(self) -> List[str]:
        names = [""] * len(self.vocabulary)
        for term, index in self.vocabulary.items():
            names[index] = term
        return names

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_vectorizer(cls, vectorizer, n_documents: int) -> "CorpusStatistics":
        return cls(
            vocabulary={term: int(i) for term, i in vectorizer.vocabulary_.items()},
            idf=np.asarray(vectorizer.idf_, dtype=np.float64),
            n_documents=n_documents,
            ngram_range=list(vectorizer.ngram_range),
            lowercase=vectorizer.lowercase,
            sublinear_tf=vectorizer.sublinear_tf,
        )

    @classmethod
    def load(cls, path=None) -> "CorpusStatistics":
        path = path or TFIDF_STATS
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing and the raw corpus is not present either. "
                "Regenerate it with `python -m experiments.corpus_stats` on a "
                "checkout that has data/raw/, or see data/README.md."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            vocabulary=payload["vocabulary"],
            idf=np.asarray(payload["idf"], dtype=np.float64),
            n_documents=payload["n_documents"],
            ngram_range=payload["ngram_range"],
            lowercase=payload.get("lowercase", True),
            sublinear_tf=payload.get("sublinear_tf", True),
        )

    def save(self, path=None) -> None:
        path = path or TFIDF_STATS
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "description": (
                        "TF-IDF vocabulary and IDF weights derived from the "
                        "Handelsregister sample. Aggregate statistics only; the "
                        "records themselves are not redistributed."
                    ),
                    "n_documents": self.n_documents,
                    "n_features": len(self.vocabulary),
                    "ngram_range": self.ngram_range,
                    "lowercase": self.lowercase,
                    "sublinear_tf": self.sublinear_tf,
                    # Rounded: six decimals is far below the precision at which
                    # any ranking in this baseline changes, and it halves the file.
                    "idf": [round(float(v), 6) for v in self.idf],
                    "vocabulary": self.vocabulary,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

    # ── use ─────────────────────────────────────────────────────────────────

    def _analyzer(self):
        """The exact analyzer scikit-learn would build for these parameters.

        ``build_analyzer`` does not require a fitted vectorizer, so tokenisation
        stays identical to the fitted path without carrying a pickle around.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        return TfidfVectorizer(
            lowercase=self.lowercase, ngram_range=tuple(self.ngram_range)
        ).build_analyzer()

    def transform(self, text: str) -> np.ndarray:
        """One document to a dense L2-normalised TF-IDF vector."""
        counts = np.zeros(len(self.vocabulary), dtype=np.float64)
        for term in self._analyzer()(text):
            index = self.vocabulary.get(term)
            if index is not None:
                counts[index] += 1.0

        present = counts > 0
        if self.sublinear_tf:
            counts[present] = 1.0 + np.log(counts[present])
        weights = counts * self.idf
        norm = np.linalg.norm(weights)
        return weights / norm if norm > 0 else weights


def main(argv: Optional[List[str]] = None) -> int:
    """Regenerate the committed statistics from the raw corpus."""
    import argparse

    from experiments.data import load_corpus
    from experiments.systems import TfidfRanker

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--out", default=None, help="Override the output path.")
    args = parser.parse_args(argv)

    corpus = load_corpus()
    if not corpus:
        print("No corpus in this checkout — nothing to derive. See data/README.md.")
        return 1

    ranker = TfidfRanker()
    ranker.fit(corpus)
    path = TFIDF_STATS if args.out is None else __import__("pathlib").Path(args.out)
    ranker.stats.save(path)
    size_mb = path.stat().st_size / 1024 / 1024
    print(
        f"{path} — {len(ranker.stats.vocabulary):,} features from "
        f"{ranker.stats.n_documents:,} documents, {size_mb:.1f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Systems under comparison: baselines, the full pipeline, and its ablations.

Every system is a pair of components — a *sector classifier* and a *keyword
extractor* — so a baseline and an ablation are the same thing structurally:
a different composition.  That is what makes the ablation table honest; the
"full" row here is the same code path the shipped pipeline runs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from services.embedder import EmbeddingService  # noqa: E402
from services.extractor import KeywordExtractor  # noqa: E402
from services.filter import KeywordFilter  # noqa: E402
from utils.preprocessing import TextPreprocessor  # noqa: E402

from experiments.config import EMBEDDING_MODEL, SEED  # noqa: E402
from experiments.corpus_stats import CorpusStatistics  # noqa: E402
from experiments.data import load_taxonomy  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Shared, expensive singletons
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def get_embedder(model_name: str = EMBEDDING_MODEL) -> EmbeddingService:
    return EmbeddingService(model_name=model_name)


@lru_cache(maxsize=4)
def get_extractor(model_name: str = EMBEDDING_MODEL) -> KeywordExtractor:
    return KeywordExtractor(model_name=model_name)


@lru_cache(maxsize=1)
def get_filter() -> KeywordFilter:
    return KeywordFilter()


@lru_cache(maxsize=1)
def get_preprocessor() -> TextPreprocessor:
    return TextPreprocessor()


MPNET_MODEL = "paraphrase-multilingual-mpnet-base-v2"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-de-en"


@lru_cache(maxsize=1)
def _translation_pipeline(model_name: str = TRANSLATION_MODEL):
    from transformers import pipeline

    return pipeline("translation", model=model_name, device=-1)


@lru_cache(maxsize=8192)
def translate_de_en(text: str) -> str:
    """German → English, cached per text.

    Used only by the translation ablation: it asks whether the multilingual
    model is doing real cross-lingual work or would do just as well on an
    English pivot.
    """
    if not text.strip():
        return text
    # The marian model truncates hard; chunk on sentence boundaries so long
    # legalistic purposes are not silently cut.
    chunks, current = [], ""
    for sentence in re.split(r"(?<=[.;])\s+", text):
        if len(current) + len(sentence) > 400 and current:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    pipe = _translation_pipeline()
    return " ".join(
        pipe(chunk, max_length=512)[0]["translation_text"] for chunk in chunks
    )


class TranslatingRanker:
    """Wraps a ranker so it sees an English translation of the input."""

    needs_corpus = False

    def __init__(self, inner):
        self.inner = inner

    def fit(self, corpus: Sequence[str]) -> None:
        self.inner.fit([translate_de_en(t) for t in corpus] if self.inner.needs_corpus else corpus)

    def rank(self, text: str) -> List[Tuple[str, float]]:
        return self.inner.rank(translate_de_en(text))


class TranslatingKeywords:
    """Wraps a keyword strategy so it extracts from the translation."""

    def __init__(self, inner):
        self.inner = inner

    def keywords(self, text: str, sector: str, top_n: int) -> List[str]:
        return self.inner.keywords(translate_de_en(text), sector, top_n)


# ─────────────────────────────────────────────────────────────────────────────
# Sector classifiers
# ─────────────────────────────────────────────────────────────────────────────

class SectorRanker(Protocol):
    """Ranks all 21 NACE sectors for a text, best first."""

    def fit(self, corpus: Sequence[str]) -> None: ...

    def rank(self, text: str) -> List[Tuple[str, float]]: ...


class RandomRanker:
    """Uniform random ranking — the floor any real method must clear."""

    needs_corpus = False

    def __init__(self, seed: int = SEED):
        self.codes = sorted(load_taxonomy())
        self.rng = np.random.default_rng(seed)

    def fit(self, corpus: Sequence[str]) -> None:
        return None

    def rank(self, text: str) -> List[Tuple[str, float]]:
        order = self.rng.permutation(len(self.codes))
        return [(self.codes[i], float(len(order) - r)) for r, i in enumerate(order)]


class PriorRanker:
    """Always predicts the most frequent sector, then the next most frequent.

    This peeks at the label distribution, so it is an *oracle* floor: it is
    reported to show how much of the headline accuracy is explained by class
    imbalance alone, not as a legitimate unsupervised competitor.
    """

    needs_corpus = False

    def __init__(self, label_counts: Dict[str, int]):
        codes = sorted(load_taxonomy())
        self.order = sorted(codes, key=lambda c: (-label_counts.get(c, 0), c))

    def fit(self, corpus: Sequence[str]) -> None:
        return None

    def rank(self, text: str) -> List[Tuple[str, float]]:
        n = len(self.order)
        return [(c, float(n - i)) for i, c in enumerate(self.order)]


class TfidfRanker:
    """Classic lexical baseline: TF-IDF cosine against a per-sector pseudo-document.

    The sector pseudo-document is its NACE name, description and seed list —
    the same taxonomy text our method uses, so the comparison isolates
    *lexical matching vs. multilingual embeddings*, not access to information.
    """

    needs_corpus = True

    def __init__(self, use_seeds: bool = True, ngram_range: Tuple[int, int] = (1, 2)):
        self.use_seeds = use_seeds
        self.ngram_range = ngram_range
        self.taxonomy = load_taxonomy()
        self.codes = sorted(self.taxonomy)
        self.vectorizer = None
        self.stats = None
        self.sector_matrix = None

    def _sector_document(self, code: str) -> str:
        info = self.taxonomy[code]
        parts = [info.get("name", ""), info.get("description", "")]
        if self.use_seeds:
            parts.extend(info.get("seed_keywords", []))
        return " ".join(parts)

    def _new_vectorizer(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        return TfidfVectorizer(
            lowercase=True,
            ngram_range=self.ngram_range,
            min_df=2,
            max_df=0.6,
            sublinear_tf=True,
        )

    def fit(self, corpus: Sequence[str]) -> None:
        """Fit on the corpus, or fall back to the committed corpus statistics.

        The raw trade register records are not redistributed, so a clone has no
        corpus to fit on.  The vocabulary and IDF weights derived from it are
        committed instead, which is enough to reproduce this baseline exactly.
        """
        sector_docs = [self._sector_document(c) for c in self.codes]
        if corpus:
            self.vectorizer = self._new_vectorizer()
            # IDF statistics come from the unlabelled corpus only.
            self.vectorizer.fit(list(corpus) + sector_docs)
            self.stats = CorpusStatistics.from_vectorizer(
                self.vectorizer, n_documents=len(corpus)
            )
        else:
            self.vectorizer = None
            self.stats = CorpusStatistics.load()
        self.sector_matrix = np.vstack([self._vector(d) for d in sector_docs])

    def _vector(self, text: str) -> np.ndarray:
        if self.vectorizer is not None:
            return self.vectorizer.transform([text]).toarray()[0]
        return self.stats.transform(text)

    def rank(self, text: str) -> List[Tuple[str, float]]:
        from sklearn.metrics.pairwise import cosine_similarity

        vec = self._vector(text).reshape(1, -1)
        sims = cosine_similarity(vec, self.sector_matrix)[0]
        order = np.argsort(-sims)
        return [(self.codes[i], float(sims[i])) for i in order]

    def top_terms(self, text: str, top_n: int = 10) -> List[str]:
        """Highest-weighted TF-IDF terms of the document, as its keywords."""
        weights = self._vector(text)
        names = (
            self.vectorizer.get_feature_names_out()
            if self.vectorizer is not None
            else self.stats.feature_names
        )
        order = np.argsort(-weights)[:top_n]
        return [names[i] for i in order if weights[i] > 0]


class EmbeddingRanker:
    """Zero-shot cosine ranking in multilingual embedding space.

    ``use_description`` / ``use_seeds`` select which halves of the taxonomy go
    into the sector vector.  With both on this reproduces ``SectorClassifier``
    exactly; with seeds off it is the plain zero-shot baseline.
    """

    needs_corpus = False

    def __init__(
        self,
        use_description: bool = True,
        use_seeds: bool = True,
        model_name: str = EMBEDDING_MODEL,
        preprocess_input: bool = False,
    ):
        if not (use_description or use_seeds):
            raise ValueError("EmbeddingRanker needs at least one taxonomy source")
        self.use_description = use_description
        self.use_seeds = use_seeds
        self.model_name = model_name
        self.preprocess_input = preprocess_input
        self.embedder = get_embedder(model_name)
        self.taxonomy = load_taxonomy()
        self.codes = sorted(self.taxonomy)
        self.sector_vectors = self._build_sector_vectors()

    def _build_sector_vectors(self) -> np.ndarray:
        vectors = []
        for code in self.codes:
            info = self.taxonomy[code]
            parts = []
            if self.use_description:
                parts.append(
                    self.embedder.embed_text(f"{info.get('name', '')}. {info.get('description', '')}")
                )
            seeds = info.get("seed_keywords", [])
            if self.use_seeds and seeds:
                parts.append(self.embedder.embed_text(", ".join(seeds)))
            vectors.append(np.mean(np.vstack(parts), axis=0))
        return np.vstack(vectors)

    def rank(self, text: str) -> List[Tuple[str, float]]:
        if self.preprocess_input:
            text = get_preprocessor().clean_text(text) or text
        vec = np.asarray(self.embedder.embed_text(text), dtype=np.float64)
        sectors = np.asarray(self.sector_vectors, dtype=np.float64)
        # The model's forward pass leaves a floating-point flag set in BLAS,
        # which numpy then reports against the *next* matmul it sees — ours.
        # The inputs are checked below, so the flag is not about this code.
        with np.errstate(all="ignore"):
            norms = np.linalg.norm(sectors, axis=1) * np.linalg.norm(vec)
            sims = np.zeros(len(self.codes))
            nonzero = norms > 0
            sims[nonzero] = (sectors[nonzero] @ vec) / norms[nonzero]
        if not np.isfinite(sims).all():
            raise ValueError(f"Non-finite similarity for text: {text[:80]!r}")
        order = np.argsort(-sims)
        return [(self.codes[i], float(sims[i])) for i in order]

    def fit(self, corpus: Sequence[str]) -> None:
        return None


class LLMRanker:
    """Zero-shot sector assignment by a small LLM.

    Optional and off by default: it needs ``OPENAI_API_KEY`` and makes one
    paid call per document, so ``make reproduce`` never depends on it.
    """

    needs_corpus = False

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.taxonomy = load_taxonomy()
        self.codes = sorted(self.taxonomy)
        from openai import OpenAI

        self.client = OpenAI()
        self._menu = "\n".join(
            f"{c}: {self.taxonomy[c].get('name', c)}" for c in self.codes
        )

    def fit(self, corpus: Sequence[str]) -> None:
        return None

    def rank(self, text: str) -> List[Tuple[str, float]]:
        prompt = (
            "Classify the German business purpose into NACE Rev. 2 sections.\n"
            f"{self._menu}\n\n"
            f"Business purpose:\n{text}\n\n"
            "Answer with the three most likely section letters, best first, "
            "comma-separated, nothing else."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=16,
        )
        guessed = [
            tok.strip().upper()[:1]
            for tok in response.choices[0].message.content.split(",")
        ]
        ranked = [c for c in guessed if c in self.taxonomy]
        ranked += [c for c in self.codes if c not in ranked]
        n = len(ranked)
        return [(c, float(n - i)) for i, c in enumerate(ranked)]


# ─────────────────────────────────────────────────────────────────────────────
# Keyword extractors
# ─────────────────────────────────────────────────────────────────────────────

class KeywordStrategy(Protocol):
    def keywords(self, text: str, sector: str, top_n: int) -> List[str]: ...


class NoKeywords:
    """For classification-only baselines."""

    def keywords(self, text: str, sector: str, top_n: int) -> List[str]:
        return []


class KeyBERTStrategy:
    """KeyBERT, optionally guided by the predicted sector's seed keywords."""

    def __init__(
        self,
        guided: bool = True,
        use_filter: bool = True,
        model_name: str = EMBEDDING_MODEL,
        diversity: float = 0.7,
    ):
        self.guided = guided
        self.use_filter = use_filter
        self.model_name = model_name
        self.diversity = diversity

    def keywords(self, text: str, sector: str, top_n: int) -> List[str]:
        extractor = get_extractor(self.model_name)
        # Over-extract, because the filter stage drops candidates.
        raw_n = top_n * 2 if self.use_filter else top_n
        if self.guided:
            scored = extractor.extract_keywords_guided_by_sector(text, sector, top_n=raw_n)
        else:
            scored = extractor.extract_keywords(
                text, top_n=raw_n, seed_keywords=None, diversity=self.diversity
            )
        if self.use_filter:
            scored = get_filter().apply_all_filters(scored, sector_code=sector, top_n=top_n)
        return [kw for kw, _ in scored[:top_n]]


class TfidfKeywordStrategy:
    """Top TF-IDF terms of the document — the lexical keyword baseline."""

    def __init__(self, ranker: TfidfRanker):
        self.ranker = ranker

    def keywords(self, text: str, sector: str, top_n: int) -> List[str]:
        return self.ranker.top_terms(text, top_n=top_n)


# ─────────────────────────────────────────────────────────────────────────────
# A system = ranker + keyword strategy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class System:
    key: str
    label: str
    description: str
    ranker: object
    keywords: object
    is_oracle: bool = False

    def fit(self, corpus: Sequence[str]) -> "System":
        self.ranker.fit(corpus)
        return self

    def run(self, text: str, top_n_keywords: int = 10) -> Dict:
        ranked = self.ranker.rank(text)
        codes = [c for c, _ in ranked]
        top = codes[0] if codes else ""
        return {
            "ranked_sectors": codes,
            "scores": [s for _, s in ranked[:3]],
            "top_sector": top,
            "keywords": self.keywords.keywords(text, top, top_n_keywords),
        }


def build_baselines(label_counts: Optional[Dict[str, int]] = None) -> List[System]:
    """The comparison table: floors, lexical, zero-shot, and the full system."""
    tfidf = TfidfRanker()
    systems = [
        System(
            key="random",
            label="Random",
            description="Uniform random sector ranking (seeded).",
            ranker=RandomRanker(),
            keywords=NoKeywords(),
        ),
        System(
            key="tfidf-nace",
            label="TF-IDF → nearest NACE section",
            description="TF-IDF cosine between the text and a per-sector pseudo-document.",
            ranker=tfidf,
            keywords=TfidfKeywordStrategy(tfidf),
        ),
        System(
            key="embed-zeroshot",
            label="Zero-shot embeddings (no taxonomy guidance)",
            description=(
                "Multilingual sentence embeddings against the sector description only; "
                "no seed keywords anywhere in the pipeline."
            ),
            ranker=EmbeddingRanker(use_description=True, use_seeds=False),
            keywords=KeyBERTStrategy(guided=False, use_filter=False),
        ),
        System(
            key="keybert-unguided",
            label="Unguided KeyBERT",
            description=(
                "Our pipeline with the taxonomy removed from keyword extraction: "
                "description-only sector vectors, plain KeyBERT keywords."
            ),
            ranker=EmbeddingRanker(use_description=True, use_seeds=False),
            keywords=KeyBERTStrategy(guided=False, use_filter=True),
        ),
        System(
            key="full",
            label="Ours: taxonomy-guided",
            description=(
                "Sector vector = mean(description, seed list); KeyBERT guided by the "
                "predicted sector's seeds; six-stage filter."
            ),
            ranker=EmbeddingRanker(use_description=True, use_seeds=True),
            keywords=KeyBERTStrategy(guided=True, use_filter=True),
        ),
    ]
    if label_counts:
        systems.insert(
            1,
            System(
                key="majority",
                label="Majority class (oracle floor)",
                description="Always predicts the most frequent gold sector.",
                ranker=PriorRanker(label_counts),
                keywords=NoKeywords(),
                is_oracle=True,
            ),
        )
    return systems


def build_ablations(
    model_name: str = EMBEDDING_MODEL, extra: bool = False
) -> List[System]:
    """One row per removed component, all against the same ``full`` system.

    ``extra`` adds the two variants that need extra model downloads: the
    larger mpnet encoder and a German→English translation pivot.
    """
    systems = [
        System(
            key="full",
            label="Full system",
            description="Taxonomy-guided classification + guided extraction + filter.",
            ranker=EmbeddingRanker(model_name=model_name),
            keywords=KeyBERTStrategy(guided=True, use_filter=True, model_name=model_name),
        ),
        System(
            key="no-seed-vector",
            label="− seeds in sector vector",
            description="Sector vector from the description alone.",
            ranker=EmbeddingRanker(use_seeds=False, model_name=model_name),
            keywords=KeyBERTStrategy(guided=True, use_filter=True, model_name=model_name),
        ),
        System(
            key="no-desc-vector",
            label="− description in sector vector",
            description="Sector vector from the seed list alone.",
            ranker=EmbeddingRanker(use_description=False, model_name=model_name),
            keywords=KeyBERTStrategy(guided=True, use_filter=True, model_name=model_name),
        ),
        System(
            key="no-guided-extraction",
            label="− seed-guided extraction",
            description="Same classifier, plain KeyBERT instead of guided KeyBERT.",
            ranker=EmbeddingRanker(model_name=model_name),
            keywords=KeyBERTStrategy(guided=False, use_filter=True, model_name=model_name),
        ),
        System(
            key="no-filter",
            label="− six-stage keyword filter",
            description="Guided extraction, filter stage skipped.",
            ranker=EmbeddingRanker(model_name=model_name),
            keywords=KeyBERTStrategy(guided=True, use_filter=False, model_name=model_name),
        ),
        System(
            key="preprocessed-input",
            label="+ cleaned text into the classifier",
            description="Classify the preprocessed text instead of the raw purpose field.",
            ranker=EmbeddingRanker(preprocess_input=True, model_name=model_name),
            keywords=KeyBERTStrategy(guided=True, use_filter=True, model_name=model_name),
        ),
    ]
    if extra:
        systems += [
            System(
                key="mpnet",
                label="↔ mpnet-base-v2 encoder (768-dim)",
                description=(
                    "Same pipeline on paraphrase-multilingual-mpnet-base-v2 instead of "
                    "MiniLM-L12-v2. Tests the README's standing hypothesis that the "
                    "larger encoder fixes the Q/M boundary."
                ),
                ranker=EmbeddingRanker(model_name=MPNET_MODEL),
                keywords=KeyBERTStrategy(guided=True, use_filter=True, model_name=MPNET_MODEL),
            ),
            System(
                key="translated-en",
                label="↔ German translated to English first",
                description=(
                    "Marian de→en pivot, then the unchanged German-seeded pipeline. "
                    "Keyword P@K is not comparable here: the gold keywords are German."
                ),
                ranker=TranslatingRanker(EmbeddingRanker(model_name=model_name)),
                keywords=TranslatingKeywords(
                    KeyBERTStrategy(guided=True, use_filter=True, model_name=model_name)
                ),
            ),
        ]
    return systems

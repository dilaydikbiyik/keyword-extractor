# Sectoral Keyword Extraction Pipeline

An unsupervised NLP system based on the **NACE Rev. 2 taxonomy**, extracting keywords from German service description texts with **sectoral context awareness**.

**Data:** 9,993 German Trade Register (Handelsregister) records  | 
**Model:** `paraphrase-multilingual-MiniLM-L12-v2` (384-dim)  | 
**Language:** DE / TR / EN

---

## Evaluation Results

Based on a 30-sample stratified test set (covering 18 different NACE sectors):

| Metric | Result |
| --- | --- |
| Sector Top-1 Accuracy | **80.0%** |
| Sector Top-3 Accuracy | **83.3%** |
| F1-Macro (Sector) | **0.831** |
| Precision@3 | **0.389** |
| Precision@5 | **0.333** |
| Precision@10 | **0.203** |
| Tests Passed | **68 / 68 passed** |
| Processing Speed | ~150–250 ms / text (cached) |

> Evaluation report: `data/evaluation/evaluation_report.json`
> Methodology document: `docs/methodology.md`

---

## Pipeline Architecture

```
Raw Text (German)
      │
      ▼
① TextPreprocessor         Language detection · URL/Email cleaning · Stopword filtering
      │
      ▼
② EmbeddingService         paraphrase-multilingual-MiniLM-L12-v2 · MD5 disk cache
      │
      ▼
③ SectorClassifier         Zero-shot cosine similarity · 21 NACE sectors
                           vector = avg(description_emb, all_seeds_emb)
      │
      ▼
④ KeywordExtractor         Guided KeyBERT · seed_keywords = sector_keywords
                           score = α · doc_sim + β · seed_sim · MMR diversity
      │
      ▼
⑤ KeywordFilter            6 stages: Linguistic quality → Dedup → Sector relevance
                           → Information value → Min score → Top-K
      │
      ▼
⑥ LLMValidator (optional)  OpenAI API · Only for 0.3–0.6 confidence interval
      │
      ▼
Output JSON

```

---

## Installation

**Requirements:** Python 3.9+

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Turkish spaCy model
python -m spacy download tr_core_news_sm

# 4. Run demo
python quickstart.py

```

---

## Quick Usage

### Single Text

```python
from src.controllers.controller import ExtractionController
from src.services.embedder import EmbeddingService
from src.services.classifier import SectorClassifier
from src.services.extractor import KeywordExtractor
from src.services.filter import KeywordFilter
from src.utils.preprocessing import TextPreprocessor

embedder     = EmbeddingService()
classifier   = SectorClassifier(embedder)
extractor    = KeywordExtractor()
kw_filter    = KeywordFilter()
preprocessor = TextPreprocessor()

controller = ExtractionController(
    embedding_service=embedder,
    classifier=classifier,
    extractor=extractor,
    keyword_filter=kw_filter,
    preprocessor=preprocessor,
)

result = controller.extract(
    "Softwareentwicklung und API-Integration für Cloud-Lösungen.",
    top_n_keywords=10,
)

print(result["sector_classification"]["top_sector"])   # "J"
print(result["sector_classification"]["confidence"])   # 0.62
print([kw["keyword"] for kw in result["keywords"]])
# ['softwareentwicklung', 'api-integration', 'cloud', ...]

```

### Batch Processing

```python
texts = [
    "Handel mit Elektronik und E-Commerce-Plattform.",
    "Zahnklinik mit Implantaten und Prophylaxe.",
    "Unternehmensberatung und Managementberatung.",
]

results = controller.extract_batch(texts, top_n_keywords=10)

# Save timestamped JSON report
report_path = controller.save_batch_report(results)
# → output/batch_report_20260507_001234.json

stats = controller.get_extraction_stats(results)
print(stats["success_rate"])         # 1.0
print(stats["sector_distribution"])  # {"G": 1, "Q": 1, "M": 1}

```

### Iterative Seed Expansion

```python
corpus = [row["purpose"] for _, row in df.iterrows()]

new_seeds = extractor.iterative_expand(
    sector_code="J",
    corpus=corpus,
    max_iterations=5,
    quality_threshold=0.55,
    max_seed_size=80,
)
print(new_seeds)  # ['cloud-native', 'microservices', 'devops', ...]

```

### Taxonomy Management

```python
from src.models.taxonomy import TaxonomyManager

tm = TaxonomyManager()
print(tm.stats())
# {'total_sectors': 21, 'total_seeds': 340, 'avg_seeds_per_sector': 16.2}

tm.add_seed_keywords("Q", ["zahnprothetik", "parodontologie"])
tm.save()

```

---

## Project Structure

```
keyword-extractor/
├── src/
│   ├── controllers/
│   │   └── controller.py         ExtractionController — 5-step orchestration
│   ├── services/
│   │   ├── embedder.py           EmbeddingService — SentenceTransformer + cache
│   │   ├── classifier.py         SectorClassifier — zero-shot, 21 NACE sectors
│   │   ├── extractor.py          KeywordExtractor — guided KeyBERT + iterative_expand
│   │   ├── filter.py             KeywordFilter — 6-stage filtering
│   │   └── validator.py          LLMValidator — OpenAI (optional)
│   ├── models/
│   │   ├── taxonomy.py           TaxonomyManager — Sector taxonomy management
│   │   └── evaluation.py         EvaluationMetrics — Precision@K, F1, Kappa
│   └── utils/
│       └── preprocessing.py      TextPreprocessor — DE/TR/EN multilingual support
│
├── data/
│   ├── raw/
│   │   └── handelsregister_sample_10k.csv   9,993 German job/business descriptions
│   ├── taxonomy/
│   │   ├── sectors.json                    21 NACE sectors + seed keywords
│   │   └── sector_descriptions.txt
│   ├── cache/                              MD5 embedding cache (NPZ)
│   └── evaluation/
│       ├── human_labels.json               Ground truth for 30 samples
│       └── evaluation_report.json          Per-document metric report
│
├── docs/
│   ├── methodology.md            Academic literature, datasets, decision rationales
│   ├── project_results_summary.md  1-page summary of results
│   ├── ARCHITECTURE.md           System architecture and component diagrams
│   ├── PROJECT_SUMMARY.md        Comprehensive project summary
│   └── COMPLETION_CHECKLIST.md   Completion checklist
│
├── tests/
│   ├── test_preprocessing.py     16 unit tests
│   ├── test_extractor.py         13 mock-based tests
│   ├── test_evaluation.py        15 unit tests
│   └── test_pipeline_e2e.py      24 integration tests
│
├── notebooks/
│   ├── initial_analysis.ipynb    Data exploration
│   └── 04_evaluation.ipynb       Visual metric reports
│
├── config/
│   └── config.yaml               All hyperparameters
├── output/                       Batch reports (JSON, .gitignore)
├── quickstart.py                 Quick demo
├── main.py                       CSV batch processing
└── requirements.txt

```

---

## Configuration

All parameters are managed centrally via `config/config.yaml`:

| Parameter | Default | Description |
| --- | --- | --- |
| `embedding.model_name` | `paraphrase-multilingual-MiniLM-L12-v2` | SentenceTransformer model |
| `embedding.chunk_size` | `256` | Long text chunk size (words) |
| `embedding.chunk_overlap` | `50` | Chunk overlap size |
| `classification.confidence_threshold` | `0.30` | Sector confidence threshold |
| `classification.top_k` | `3` | Number of candidate sectors to return |
| `extraction.top_n_final` | `10` | Final keyword count in output |
| `extraction.alpha` | `0.6` | Weight for document similarity |
| `extraction.beta` | `0.4` | Weight for seed similarity |
| `extraction.diversity` | `0.7` | MMR diversity coefficient |
| `iteration.max_iterations` | `5` | Iterative seed expansion rounds |
| `iteration.quality_threshold` | `0.55` | Acceptance threshold for new seeds |

---

## Tests

```bash
# Run all tests (68/68)
pytest tests/ -v

# Run only unit tests
pytest tests/test_preprocessing.py tests/test_extractor.py tests/test_evaluation.py -v

# Run only integration tests
pytest tests/test_pipeline_e2e.py -v

# Lint check
python -m flake8 src/ main.py quickstart.py \
    --select=F401,F841,W293,E302,E303 --max-line-length=120

```

**Test Coverage:**

| Module | Test Count | Scope |
| --- | --- | --- |
| `test_preprocessing.py` | 16 | `clean_text`, `detect_language`, `generate_ngram_candidates`, `preprocess_pipeline` |
| `test_extractor.py` | 13 | `extract_keywords`, `extract_guided`, `iterative_expand` (mock-based) |
| `test_evaluation.py` | 15 | `precision_at_k`, `semantic_match_score`, `f1_macro`, `cohen_kappa`, `EvaluationMetrics` |
| `test_pipeline_e2e.py` | 24 | Full pipeline, batch processing, report files, timing fields |
| **Total** | **68** | **68/68 ✅** |

---

## Sector Taxonomy (NACE Rev. 2)

21 sectors, each with a custom list of seed keywords (average 16 seeds/sector, total 340):

| Code | Sector | Seed Count |
| --- | --- | --- |
| A | Agriculture & Forestry | 12 |
| B | Mining | 10 |
| C | Manufacturing | 18 |
| D | Electricity & Energy | 15 |
| E | Water & Waste Management | 12 |
| F | Construction | 18 |
| G | Wholesale & Retail Trade | 16 |
| H | Transport & Logistics | 14 |
| I | Accommodation & Food | 13 |
| J | Information & Communication Technology | 36 |
| K | Finance & Insurance | 16 |
| L | Real Estate | 13 |
| M | Professional & Scientific Services | 18 |
| N | Administrative & Support Services | 14 |
| O | Public Administration | 10 |
| P | Education | 13 |
| Q | Health & Social Services | 45 |
| R | Arts & Entertainment | 12 |
| S | Other Services | 13 |
| T | Household Employers | 8 |
| U | International Organizations | 8 |

---

## Related Literature

| Method | Source | Role |
| --- | --- | --- |
| **KeyBERT** | Grootendorst (2020) — Zenodo | Pipeline core |
| **YAKE!** | Campos et al. (2020) — *Information Sciences* | Comparison baseline |
| **PatternRank** | Schopf et al. (2022) — ICPRAM | N-gram candidate strategy |
| **PromptRank** | Kong et al. (2023) — ACL | LLM-based comparison |
| **Sentence-BERT** | Reimers & Gurevych (2019) — EMNLP | Embedding foundation |
| **Multilingual SBERT** | Reimers & Gurevych (2020) — EMNLP | Multilingual model selection |

For detailed literature review, dataset comparisons, and methodology decisions:

→ [`docs/methodology.md`](https://www.google.com/search?q=docs/methodology.md)

---

## Technology Stack

| Layer | Tool | Version |
| --- | --- | --- |
| Embedding | `sentence-transformers` | ≥ 2.6.0 |
| Keyword Extraction | `keybert` | ≥ 0.8.0 |
| NLP | `nltk`, `spacy` | ≥ 3.8, ≥ 3.6 |
| Language Detection | `langdetect` | ≥ 1.0.9 |
| Metrics | `scikit-learn` | ≥ 1.3.0 |
| Data | `pandas`, `numpy` | ≥ 2.0, ≥ 1.24 |
| LLM (optional) | `openai` | ≥ 1.0.0 |
| Test | `pytest` | ≥ 7.4.0 |

---

## Limitations

* **Q vs M Ambiguity:** `MiniLM-L12` (384-dim) has difficulty distinguishing dental/health from professional services at the boundary.
Solution: Switching to `paraphrase-multilingual-mpnet-base-v2` is expected to provide ~10% improvement.
* **Very Short Texts:** Embedding quality decreases for texts shorter than 50 characters.
* **Lack of Labeled Data:** The ground truth set was semi-manually produced (30 samples); true accuracy assessment requires domain expert labeling.

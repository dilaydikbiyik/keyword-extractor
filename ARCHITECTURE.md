# Architecture & Component Diagram

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User Application Layer                          │
│                                                                           │
│  main.py (CSV batch) | quickstart.py (demo) | test_integration.py      │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXTRACTION CONTROLLER                            │
│                    (src/controllers/controller.py)                      │
│                                                                           │
│  extract(text) → extract_batch(texts) → extract_from_dataframe(df)    │
│  get_extraction_stats() → configure() → get_config()                  │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐
    │ PREPROCESSING    │  │ SECTOR CLASSIFY  │  │ KEYWORD EXTRACT│
    │ (utils/)         │  │ (services/)      │  │ (services/)    │
    │                  │  │                  │  │                │
    │ • Language       │  │ • Embeddings     │  │ • KeyBERT      │
    │   Detection      │  │ • Zero-shot      │  │ • MMR diversity│
    │ • Text cleaning  │  │   classification │  │ • Seed keywords│
    │ • Stop-word      │  │ • Confidence     │  │ • Multi-sector │
    │   removal        │  │   threshold      │  │   mode         │
    │ • N-gram         │  │                  │  │                │
    │   generation     │  │                  │  │                │
    └──────────────────┘  └──────────────────┘  └────────────────┘
            │                      │                      │
            └──────────────────────┴──────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌─────────────────┐           ┌──────────────────┐
            │ KEYWORD FILTER  │           │ EMBEDDING SERVICE│
            │ (services/)     │           │ (services/)      │
            │                 │           │                  │
            │ • Sector        │           │ • SentenceXfmr   │
            │   relevance     │           │ • Text embedding │
            │ • Info value    │           │ • Sector embed   │
            │ • Linguistic    │           │ • Caching (NPZ)  │
            │   quality       │           │ • Similarity     │
            │ • Deduplication │           │ • Most similar   │
            │ • Min score     │           │                  │
            └────────┬────────┘           └──────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ LLM VALIDATOR   │
            │ (Optional)      │
            │ (services/)     │
            │                 │
            │ • OpenAI API    │
            │ • Confidence-   │
            │   range filter  │
            │ • Retry logic   │
            │ • Rate limiting │
            └────────┬────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ RESULTS          │
            │ JSON Output      │
            │ Statistics       │
            └──────────────────┘
```

## Data Flow Diagram

```
INPUT TEXT (German business description)
    │
    ├─→ [PREPROCESSING]
    │   ├─ Language Detection → German (0.85 confidence)
    │   ├─ URL/Email Removal
    │   ├─ Text Normalization
    │   ├─ Stop-word Filtering
    │   └─ N-gram Candidates Generation
    │       Output: {cleaned_text, language, candidates}
    │
    ├─→ [EMBEDDING]
    │   ├─ Text → 384-dim vector (paraphrase-multilingual-MiniLM-L12-v2)
    │   ├─ Cache lookup (MD5-based)
    │   └─ Sector embeddings (21 NACE)
    │       Output: embedding_vector
    │
    ├─→ [SECTOR CLASSIFICATION]
    │   ├─ Compute cosine similarity to sector embeddings
    │   ├─ Rank by confidence
    │   ├─ Apply threshold (default 0.5)
    │   └─ Return top-1 sector
    │       Output: sector_code, confidence, alternatives
    │
    ├─→ [KEYWORD EXTRACTION (Guided)]
    │   ├─ Load sector-specific seed keywords (10-25 per sector)
    │   ├─ KeyBERT extraction with seeds
    │   │   - Score = α * doc_sim + β * seed_sim
    │   ├─ MMR reranking (diversity=0.7)
    │   └─ Extract top 20 candidates
    │       Output: [(keyword, score), ...]
    │
    ├─→ [FILTERING (Multi-stage)]
    │   Stage 1: Linguistic Quality
    │   ├─ Remove short (<3 chars) / long (>50 chars)
    │   ├─ Remove numeric-heavy
    │   └─ Remove leading/trailing stopwords
    │
    │   Stage 2: Deduplication
    │   ├─ Lowercase normalization
    │   ├─ Keep highest-scoring duplicate
    │   └─ Stem-based grouping
    │
    │   Stage 3: Sector Relevance
    │   ├─ Check negative keywords (apply -0.3 penalty)
    │   └─ Boost positive keywords
    │
    │   Stage 4: Information Value
    │   ├─ Multi-word phrases: +1.15x multiplier
    │   ├─ Rare words: +0.8x boost
    │   └─ Domain specificity ranking
    │
    │   Stage 5: Min Score Threshold
    │   └─ Remove score < 0.1
    │
    │   Stage 6: Top-K Selection
    │   └─ Keep top 10 keywords
    │       Output: [(keyword, final_score), ...]
    │
    ├─→ [LLM VALIDATION] (optional)
    │   ├─ Filter by confidence range (0.3-0.6)
    │   ├─ Call OpenAI API for validation
    │   │   - Prompt: "Is '{keyword}' relevant for sector {sector}?"
    │   ├─ Exponential backoff retry (2^attempt)
    │   ├─ Rate limiting (1s between batches)
    │   └─ Merge LLM confidence with original score
    │       Output: validated keywords
    │
    └─→ FINAL OUTPUT
        {
          "input_text": "Softwareentwicklung...",
          "language": "de",
          "sector_classification": {
            "top_sector": "J",
            "confidence": 0.87,
            "alternatives": [("M", 0.42), ("C", 0.38)]
          },
          "keywords": [
            {"keyword": "softwareentwicklung", "score": 0.92},
            {"keyword": "api", "score": 0.88},
            ...
          ],
          "status": "success"
        }
```

## Component Dependencies

```
                    ┌─────────────────────────────┐
                    │   ExtractionController      │
                    └──────────────┬──────────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                  │
                ▼                  ▼                  ▼
        ┌─────────────┐     ┌──────────────┐    ┌─────────────┐
        │TextPreproc  │     │SectorClassif │    │KeywordExtra │
        └─────────────┘     │              │    │             │
                            └────────┬─────┘    └─────────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │EmbeddingService  │
                            │  (Foundation)    │
                            └──────────────────┘
                                     ▲
                ┌────────────────────┼────────────────────┐
                │                    │                    │
                ▼                    ▼                    ▼
        ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
        │KeywordFilter │    │LLMValidator  │    │TaxonomyMgr   │
        │              │    │  (Optional)  │    │ (Stub)       │
        └──────────────┘    └──────────────┘    └──────────────┘
```

## Service Characteristics

| Service | Input | Output | Dependencies | Caching |
|---------|-------|--------|--------------|---------|
| TextPreprocessor | Text | {cleaned_text, language, candidates} | NLTK, spaCy, langdetect | No |
| EmbeddingService | Text / Sectors | 384-dim vectors | SentenceTransformers | Yes (NPZ) |
| SectorClassifier | Text → Embeddings | [(sector, confidence)] | EmbeddingService | Yes (vectors) |
| KeywordExtractor | Text + Sector | [(keyword, score)] | KeyBERT | No |
| KeywordFilter | Keywords + Sector | [(keyword, final_score)] | sector JSON | No |
| LLMValidator | Keywords + Text + Sector | validated keywords | OpenAI API | Lazy init |
| ExtractionController | Text(s) | Structured results | All above | Coordinates |

## File Structure Reference

```
src/
├── __init__.py
├── controllers/
│   └── controller.py          ← ExtractionController (orchestration)
├── services/
│   ├── embedder.py            ← EmbeddingService (foundation)
│   ├── classifier.py          ← SectorClassifier
│   ├── extractor.py           ← KeywordExtractor
│   ├── filter.py              ← KeywordFilter
│   └── validator.py           ← LLMValidator (optional)
├── models/
│   ├── taxonomy.py            ← TaxonomyManager (TODO)
│   └── evaluation.py          ← EvaluationMetrics (TODO)
└── utils/
    └── preprocessing.py       ← TextPreprocessor

config/
└── config.yaml                ← Hyperparameters

data/
├── raw/
│   └── handelsregister_sample_10k.csv
├── taxonomy/
│   ├── sectors.json           ← 21 NACE sectors + seeds
│   └── sector_descriptions.txt
└── cache/
    └── embeddings_cache.npz

output/
└── results.json               ← CSV processing results

tests/
└── test_*.py

notebooks/
├── 01_data_exploration.ipynb
├── 02_taxonomy_building.ipynb
├── 03_extraction_experiments.ipynb
└── 04_evaluation.ipynb
```

## Performance Characteristics

```
Operation                    Time (approx)    Memory
─────────────────────────────────────────────────────
Text Preprocessing           5-10ms           <1MB
Embedding (cached)           1-2ms            384 bytes
Embedding (first-time)       200-300ms        384 bytes
Sector Classification        5-10ms           <1MB
Keyword Extraction           100-200ms        <5MB
Keyword Filtering            10-20ms          <1MB
LLM Validation (API)         1-2s             <10MB
───────────────────────────────────────────────────
Full Pipeline (cached)       ~150-250ms       ~10MB
Full Pipeline (first-time)   ~2-3s            ~10MB
Batch Processing (100 texts) ~15-25s          ~100MB
```

## Extension Points

Future enhancements can be added at:
1. **New Services**: Add to `src/services/` (e.g., `transliteration.py`)
2. **Models**: Implement `src/models/taxonomy.py` and `evaluation.py`
3. **Preprocessing**: Extend `TextPreprocessor` methods
4. **Configuration**: Add parameters to `config/config.yaml`
5. **Validation**: Create custom validators beyond LLM
6. **Output Formats**: Add export to CSV, Excel, database
7. **API Endpoint**: Flask/FastAPI wrapper around controller

---

**Last Updated**: 2025-01-XX  
**Architecture Version**: 1.0 (MVC with Service Orchestration)  
**Status**: ✅ Complete & Ready for Testing

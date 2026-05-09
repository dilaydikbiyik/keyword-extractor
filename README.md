# Sektörel Anahtar Kelime Çıkartma Pipeline'ı

Almanca hizmet tanımı metinlerinden **sektörel bağlamı anlayan** anahtar kelimeler çıkaran,
NACE Rev. 2 taksonomisine dayalı denetimsiz NLP sistemi.

**Veri:** 9.993 Alman Ticaret Sicili (Handelsregister) kaydı &nbsp;|&nbsp;
**Model:** `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) &nbsp;|&nbsp;
**Dil:** DE / TR / EN

---

## Değerlendirme Sonuçları

30 örneklik stratified test seti (18 farklı NACE sektörü) üzerinde:

| Metrik | Sonuç |
|---|---|
| Sektör Top-1 Accuracy | **%80.0** |
| Sektör Top-3 Accuracy | **%83.3** |
| F1-Macro (Sektör) | **0.831** |
| Precision@3 | **0.389** |
| Precision@5 | **0.333** |
| Precision@10 | **0.203** |
| Test Sayısı | **68 / 68 geçti** |
| İşleme Hızı | ~150–250 ms / metin (cache'li) |

> Değerlendirme raporu: `data/evaluation/evaluation_report.json`  
> Metodoloji belgesi: `docs/methodology.md`

---

## Pipeline Mimarisi

```
Ham Metin (Almanca)
      │
      ▼
① TextPreprocessor          dil tespiti · URL/e-posta temizleme · stopword filtreleme
      │
      ▼
② EmbeddingService          paraphrase-multilingual-MiniLM-L12-v2 · MD5 disk cache
      │
      ▼
③ SectorClassifier          zero-shot cosine similarity · 21 NACE sektörü
                             vektör = avg(description_emb, all_seeds_emb)
      │
      ▼
④ KeywordExtractor          guided KeyBERT · seed_keywords=sektor_kelimeleri
                             skor = α·doc_sim + β·seed_sim · MMR diversity
      │
      ▼
⑤ KeywordFilter             6 aşama: linguistic quality → dedup → sector relevance
                             → information value → min score → top-K
      │
      ▼
⑥ LLMValidator (opsiyonel)  OpenAI API · yalnızca 0.3–0.6 güven aralığı
      │
      ▼
Çıktı JSON
```

---

## Kurulum

**Gereksinimler:** Python 3.9+

```bash
# 1. Sanal ortam oluştur
python3 -m venv .venv
source .venv/bin/activate

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. (İsteğe bağlı) Türkçe spaCy modeli
python -m spacy download tr_core_news_sm

# 4. Demo çalıştır
python quickstart.py
```

---

## Hızlı Kullanım

### Tek Metin

```python
from src.controllers.controller import ExtractionController
from src.services.embedder import EmbeddingService
from src.services.classifier import SectorClassifier
from src.services.extractor import KeywordExtractor
from src.services.filter import KeywordFilter
from src.utils.preprocessing import TextPreprocessor

embedder    = EmbeddingService()
classifier  = SectorClassifier(embedder)
extractor   = KeywordExtractor()
kw_filter   = KeywordFilter()
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

### Batch İşleme

```python
texts = [
    "Handel mit Elektronik und E-Commerce-Plattform.",
    "Zahnklinik mit Implantaten und Prophylaxe.",
    "Unternehmensberatung und Managementberatung.",
]

results = controller.extract_batch(texts, top_n_keywords=10)

# Zaman damgalı JSON raporu kaydet
report_path = controller.save_batch_report(results)
# → output/batch_report_20260507_001234.json

stats = controller.get_extraction_stats(results)
print(stats["success_rate"])        # 1.0
print(stats["sector_distribution"]) # {"G": 1, "Q": 1, "M": 1}
```

### İteratif Seed Genişletme

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

### Taksonomi Yönetimi

```python
from src.models.taxonomy import TaxonomyManager

tm = TaxonomyManager()
print(tm.stats())
# {'total_sectors': 21, 'total_seeds': 340, 'avg_seeds_per_sector': 16.2}

tm.add_seed_keywords("Q", ["zahnprothetik", "parodontologie"])
tm.save()
```

---

## Proje Yapısı

```
keyword-extractor/
├── src/
│   ├── controllers/
│   │   └── controller.py         ExtractionController — 5 adımlı orkestrasyon
│   ├── services/
│   │   ├── embedder.py           EmbeddingService — SentenceTransformer + cache
│   │   ├── classifier.py         SectorClassifier — zero-shot, 21 NACE
│   │   ├── extractor.py          KeywordExtractor — guided KeyBERT + iterative_expand
│   │   ├── filter.py             KeywordFilter — 6 aşamalı filtreleme
│   │   └── validator.py          LLMValidator — OpenAI (opsiyonel)
│   ├── models/
│   │   ├── taxonomy.py           TaxonomyManager — sektör taksonomi yönetimi
│   │   └── evaluation.py         EvaluationMetrics — Precision@K, F1, Kappa
│   └── utils/
│       └── preprocessing.py      TextPreprocessor — DE/TR/EN çok dilli
│
├── data/
│   ├── raw/
│   │   └── handelsregister_sample_10k.csv   9.993 Almanca iş tanımı
│   ├── taxonomy/
│   │   ├── sectors.json                     21 NACE sektörü + seed keywords
│   │   └── sector_descriptions.txt
│   ├── cache/                               MD5 embedding cache (NPZ)
│   └── evaluation/
│       ├── human_labels.json                30 örneklik ground truth
│       └── evaluation_report.json           Per-document metrik raporu
│
├── docs/
│   ├── methodology.md            Akademik literatür, veri setleri, karar gerekçeleri
│   ├── project_results_summary.md  1 sayfalık sonuç özeti
│   ├── ARCHITECTURE.md           Sistem mimarisi ve bileşen diyagramları
│   ├── PROJECT_SUMMARY.md        Kapsamlı proje özeti
│   └── COMPLETION_CHECKLIST.md   Tamamlama kontrol listesi
│
├── tests/
│   ├── test_preprocessing.py     16 unit test
│   ├── test_extractor.py         13 mock-based test
│   ├── test_evaluation.py        15 unit test
│   └── test_pipeline_e2e.py      24 entegrasyon testi
│
├── notebooks/
│   ├── initial_analysis.ipynb    Veri keşfi
│   └── 04_evaluation.ipynb       Görsel metrik raporları
│
├── config/
│   └── config.yaml               Tüm hiperparametreler
├── output/                        Batch raporları (JSON, .gitignore)
├── quickstart.py                  Hızlı demo
├── main.py                        CSV batch işleme
└── requirements.txt
```

---

## Yapılandırma

`config/config.yaml` dosyasından tüm parametreler merkezi olarak yönetilir:

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `embedding.model_name` | `paraphrase-multilingual-MiniLM-L12-v2` | SentenceTransformer modeli |
| `embedding.chunk_size` | `256` | Uzun metin chunk boyutu (kelime) |
| `embedding.chunk_overlap` | `50` | Chunk örtüşme boyutu |
| `classification.confidence_threshold` | `0.30` | Sektör güven eşiği |
| `classification.top_k` | `3` | Döndürülecek aday sektör sayısı |
| `extraction.top_n_final` | `10` | Çıktı keyword sayısı |
| `extraction.alpha` | `0.6` | Doküman benzerliği ağırlığı |
| `extraction.beta` | `0.4` | Seed benzerliği ağırlığı |
| `extraction.diversity` | `0.7` | MMR çeşitlilik katsayısı |
| `iteration.max_iterations` | `5` | İteratif seed genişletme turu |
| `iteration.quality_threshold` | `0.55` | Yeni seed kabul eşiği |

---

## Testler

```bash
# Tüm testler (68/68)
pytest tests/ -v

# Sadece birim testler
pytest tests/test_preprocessing.py tests/test_extractor.py tests/test_evaluation.py -v

# Sadece entegrasyon testleri
pytest tests/test_pipeline_e2e.py -v

# Lint kontrolü
python -m flake8 src/ main.py quickstart.py \
    --select=F401,F841,W293,E302,E303 --max-line-length=120
```

**Test kapsamı:**

| Modül | Test Sayısı | Kapsam |
|---|---|---|
| `test_preprocessing.py` | 16 | `clean_text`, `detect_language`, `generate_ngram_candidates`, `preprocess_pipeline` |
| `test_extractor.py` | 13 | `extract_keywords`, `extract_guided`, `iterative_expand` (mock tabanlı) |
| `test_evaluation.py` | 15 | `precision_at_k`, `semantic_match_score`, `f1_macro`, `cohen_kappa`, `EvaluationMetrics` |
| `test_pipeline_e2e.py` | 24 | Tam pipeline, batch, rapor dosyası, zamanlama alanları |
| **Toplam** | **68** | **68/68 ✅** |

---

## Sektör Taksonomisi (NACE Rev. 2)

21 sektör, her biri için özel seed keyword listesi (ortalama 16 seed/sektör, toplam 340):

| Kod | Sektör | Seed Sayısı |
|---|---|---|
| A | Tarım & Ormancılık | 12 |
| B | Madencilik | 10 |
| C | İmalat | 18 |
| D | Elektrik & Enerji | 15 |
| E | Su & Atık Yönetimi | 12 |
| F | İnşaat | 18 |
| G | Toptan & Perakende Ticaret | 16 |
| H | Ulaşım & Lojistik | 14 |
| I | Konaklama & Yiyecek | 13 |
| J | Bilgi & İletişim Teknolojileri | 36 |
| K | Finans & Sigorta | 16 |
| L | Emlak | 13 |
| M | Profesyonel & Bilimsel Hizmetler | 18 |
| N | İdari & Destek Hizmetleri | 14 |
| O | Kamu Yönetimi | 10 |
| P | Eğitim | 13 |
| Q | Sağlık & Sosyal Hizmetler | 45 |
| R | Sanat & Eğlence | 12 |
| S | Diğer Hizmetler | 13 |
| T | Hane Halkı İşverenleri | 8 |
| U | Uluslararası Örgütler | 8 |

---

## İlgili Literatür

| Yöntem | Kaynak | Rol |
|---|---|---|
| **KeyBERT** | Grootendorst (2020) — Zenodo | Pipeline çekirdeği |
| **YAKE!** | Campos et al. (2020) — *Information Sciences* | Karşılaştırma baseline |
| **PatternRank** | Schopf et al. (2022) — ICPRAM | N-gram aday stratejisi |
| **PromptRank** | Kong et al. (2023) — ACL | LLM tabanlı karşılaştırma |
| **Sentence-BERT** | Reimers & Gurevych (2019) — EMNLP | Embedding temeli |
| **Multilingual SBERT** | Reimers & Gurevych (2020) — EMNLP | Çok dilli model seçimi |

Detaylı literatür taraması, veri seti karşılaştırması ve metodoloji kararları için:  
→ [`docs/methodology.md`](docs/methodology.md)

---

## Teknoloji Yığını

| Katman | Araç | Versiyon |
|---|---|---|
| Embedding | `sentence-transformers` | ≥ 2.6.0 |
| Keyword Çıkarımı | `keybert` | ≥ 0.8.0 |
| NLP | `nltk`, `spacy` | ≥ 3.8, ≥ 3.6 |
| Dil Tespiti | `langdetect` | ≥ 1.0.9 |
| Metrikler | `scikit-learn` | ≥ 1.3.0 |
| Veri | `pandas`, `numpy` | ≥ 2.0, ≥ 1.24 |
| LLM (opsiyonel) | `openai` | ≥ 1.0.0 |
| Test | `pytest` | ≥ 7.4.0 |

---

## Sınırlılıklar

- **Q vs M karışıklığı:** `MiniLM-L12` (384-dim) dental/sağlık ile profesyonel hizmetleri sınırda ayırt edemiyor.  
  Çözüm: `paraphrase-multilingual-mpnet-base-v2` ile ~%10 iyileşme beklenir.
- **Çok kısa metinler:** <50 karakter metinlerde embedding kalitesi düşüyor.
- **Etiketli veri yok:** Ground truth seti semi-manuel üretildi (30 örnek); gerçek doğruluk için domain uzmanı etiketlemesi gerekir.
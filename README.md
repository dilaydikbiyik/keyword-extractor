# Sektörel Anahtar Kelime Çıkartma Pipeline'ı

> Çok dilli (Almanca / Türkçe / İngilizce) iş metinlerinden sektöre özgü anahtar
> kelimeler çıkaran hibrit NLP pipeline'ı.  
> **Yaklaşım:** Guided KeyBERT + İteratif Seed Genişletme + VIB tarzı Filtreleme

---

## İçindekiler
- [Proje Amacı](#proje-amacı)
- [Pipeline Akışı](#pipeline-akışı)
- [Kurulum](#kurulum)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Proje Yapısı](#proje-yapısı)
- [Yapılandırma](#yapılandırma)
- [Testler](#testler)
- [Değerlendirme](#değerlendirme)

---

## Proje Amacı

Ticaret sicili / işletme kayıt metinlerinden (örn. Almanya Handelsregister)
otomatik olarak:
1. **Sektör tahmini** (NACE Rev. 2 kodları, A–U)
2. **Sektöre özgü anahtar kelimeler** (top-N, skor sıralamalı)

yapan bir NLP pipeline'ı sunmak.

---

## Pipeline Akışı

```
Ham hizmet metni
      │
      ▼
[1] Dil tespiti (langdetect)
      │
      ▼
[2] Metin temizleme + stop-word kaldırma
      │
      ▼
[3] Sektör sınıflandırması
    Cosine similarity (embedding ↔ sektör vektörü)
      │
      ▼
[4] Guided KeyBERT extraction
    seed_keywords = sektöre ait terimler
    use_mmr=True  (çeşitlilik)
      │
      ▼
[5] VIB tarzı filtreleme
    düşük bilgi yoğunluklu adayları at
      │
      ▼
[6] (Opsiyonel) LLM doğrulama
    sadece gri alan keyword'ler için
      │
      ▼
Çıktı: {sector, confidence, keywords[{keyword, score}]}
```

---

## Kurulum

**Gereksinimler:** Python 3.9+

```bash
# Sanal ortam oluştur
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## Hızlı Başlangıç

```bash
# 3 örnek metin ile pipeline'ı çalıştır
python quickstart.py
```

**Örnek çıktı:**

```
Sample 1: Softwareentwicklung und Programmierung für Web- und Mobileanwendungen...
Language:              de
Primary Sector:        J (confidence: 39.3%)

Extracted Keywords:
  1. mobileanwendungen          (score: 0.701)
  2. softwareentwicklung        (score: 0.688)
  3. web                        (score: 0.594)
  4. programmierung             (score: 0.592)
  5. cloud                      (score: 0.585)
```

---

### Python API

```python
import sys; sys.path.insert(0, 'src')

from utils.preprocessing import TextPreprocessor
from services.embedder import EmbeddingService
from services.classifier import SectorClassifier
from services.extractor import KeywordExtractor
from services.filter import KeywordFilter
from controllers.controller import ExtractionController

# Bileşenleri başlat
preprocessor = TextPreprocessor()
embedder     = EmbeddingService()
classifier   = SectorClassifier(embedder)
extractor    = KeywordExtractor()
kw_filter    = KeywordFilter()

controller = ExtractionController(
    embedding_service=embedder,
    classifier=classifier,
    extractor=extractor,
    keyword_filter=kw_filter,
    preprocessor=preprocessor,
)

# Tek metin
result = controller.extract(
    "Softwareentwicklung und Cloud-Lösungen.",
    top_n_keywords=10,
)
print(result["sector_classification"]["top_sector"])   # → "J"
print(result["keywords"][:3])

# Toplu işlem + JSON raporu kaydet
texts = ["metin1", "metin2", "metin3"]
results = controller.extract_batch(
    texts, top_n_keywords=10, save_report=True, report_dir="output"
)
```

---

### İteratif Seed Genişletme

```python
# Sektöre ait metin korpusu ile seed'leri otomatik genişlet
expanded_seeds = extractor.iterative_expand(
    texts=domain_texts,
    sector_code="Q",      # Sağlık sektörü
    n_iterations=3,
    quality_threshold=0.4,
    max_seed_size=50,
)
print(f"Expanded seeds: {len(expanded_seeds)}")
```

---

## Proje Yapısı

```
keyword-extractor/
├── config/
│   └── config.yaml              # Tüm hiperparametreler
├── data/
│   ├── raw/                     # Ham veri (git'e eklenmez)
│   ├── processed/               # Temizlenmiş metinler
│   └── taxonomy/
│       ├── sectors.json         # 21 NACE sektörü + seed keywords
│       └── generic_keywords.txt # Genel stop-word listesi
├── src/
│   ├── controllers/
│   │   └── controller.py        # ExtractionController (orkestrasyon)
│   ├── models/
│   │   ├── evaluation.py        # Precision@K, SemanticMatch, F1
│   │   └── taxonomy.py          # TaxonomyManager
│   ├── services/
│   │   ├── embedder.py          # EmbeddingService (+ chunking)
│   │   ├── extractor.py         # KeywordExtractor (+ iterative expand)
│   │   ├── classifier.py        # SectorClassifier
│   │   ├── filter.py            # KeywordFilter (VIB tarzı)
│   │   └── validator.py         # LLMValidator (opsiyonel)
│   └── utils/
│       └── preprocessing.py     # TextPreprocessor
├── tests/
│   ├── test_preprocessing.py
│   ├── test_extractor.py
│   ├── test_evaluation.py
│   └── test_pipeline_e2e.py
├── output/                      # Batch raporları (JSON)
├── quickstart.py                # Hızlı demo scripti
├── main.py                      # CSV işleme scripti
└── requirements.txt
```

---

## Yapılandırma

`config/config.yaml` dosyasından tüm hiperparametreler yönetilir:

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `embedding.model_name` | `paraphrase-multilingual-MiniLM-L12-v2` | SentenceTransformer modeli |
| `embedding.chunk_size` | `256` | Uzun metin chunk boyutu (kelime) |
| `classification.confidence_threshold` | `0.30` | Sektör güven eşiği |
| `extraction.top_n_final` | `10` | Döndürülecek keyword sayısı |
| `extraction.alpha` | `0.6` | Doküman benzerliği ağırlığı |
| `extraction.beta` | `0.4` | Seed benzerliği ağırlığı |
| `iteration.max_iterations` | `5` | İteratif seed genişletme turu |
| `iteration.quality_threshold` (kod içi) | `0.4` | Seed'e eklenecek min skor |

---

## Testler

```bash
# Tüm testler
source .venv/bin/activate
pytest tests/ -v

# Sadece unit testler (model gerektirmez)
pytest tests/test_preprocessing.py tests/test_evaluation.py tests/test_extractor.py -v

# E2E entegrasyon testi (~30s)
pytest tests/test_pipeline_e2e.py -v
```

---

## Değerlendirme

`src/models/evaluation.py` şu metrikleri uygular:

| Metrik | Fonksiyon |
|---|---|
| Precision@K | `precision_at_k(extracted, ground_truth, k)` |
| Semantik Eşleşme | `semantic_match_score(extracted, gold, model)` |
| Top-1 / Top-3 Accuracy | `top_k_accuracy(y_true, y_pred_top_k, k)` |
| F1-Macro | `f1_macro(y_true, y_pred)` |
| Cohen's Kappa | `cohen_kappa(y_true, y_pred)` |
| Toplu Değerlendirme | `EvaluationMetrics(k_values=[5,10,20])` |

```python
from src.models.evaluation import EvaluationMetrics

ev = EvaluationMetrics(k_values=[5, 10])
ev.add(extracted=["bulut", "yazılım", "api"],
       ground_truth=["yazılım", "api"],
       true_sector="J", predicted_sector="J", predicted_top3=["J","M","K"])
report = ev.compute()
# {'n_documents': 1, 'precision_at_5': 0.4, 'precision_at_10': 0.2,
#  'top1_accuracy': 1.0, 'top3_accuracy': 1.0, 'f1_macro': 1.0, ...}
```

---

## Başarı Kriterleri

| Metrik | Hedef |
|---|---|
| Sektör Top-1 Accuracy | ≥ %70 |
| Sektör Top-3 Accuracy | ≥ %85 |
| Precision@10 | ≥ %60 |
| Cosine Similarity Match | ≥ 0.65 |
| İşleme hızı (batch) | ≥ 10 metin/sn |
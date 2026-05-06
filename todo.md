# 📌 Sektörel Anahtar Kelime Çıkartma Projesi — Detaylı TODO

> **Hedef:** Hizmet metinlerinden, sektörel bağlamı anlayan bir hibrit NLP pipeline ile
> yüksek kaliteli anahtar kelimeler çıkartmak ve metin-sektör ataması yapmak.
> Yaklaşım: Guided KeyBERT + İteratif Seed Genişletme + VIB tarzı filtreleme + LLM doğrulama.

**✅ GÜNCELLEME NOTU (4 Mayıs 2026):** MVC benzeri temiz mimari uygulandı. Dosya yapısı yeniden organize edildi.

---

## ⚙️ FELSEFE & TASARIM KARARLARI

Bu projeye başlamadan önce aklında şu prensipleri tut:

- **Az etiketli veri varsayımı:** Milyonlarca labeled örnek yok. Bu yüzden seed-guided yaklaşım seçtik.
- **Anlam > Frekans:** TF-IDF değil, semantic embedding kullan. Her zaman daha iyi sonuç verir.
- **Pipeline sırası kritik:** Önce sektör sınıflandırması → sonra o sektöre özel keyword çıkarımı.
- **Güven eşiği koy:** Düşük confidence'lı tahminleri "belirsiz" işaretle, yanlış etiket verme.
- **Reproducibility:** Her `random_state` ve `seed` sabit olsun. Aynı veri → aynı sonuç.

---

## AŞAMA 0 — Proje Ortamı ve Altyapı

### 0.1 Klasör Yapısını Oluştur
```
project/
├── data/
│   ├── raw/                    # Ham veri (dokunma)
│   ├── processed/              # Temizlenmiş metinler
│   └── taxonomy/               # Sektör taksonomisi ve seed listeler
├── src/                        # ✅ MVC BENZERİ MİMARİ UYGULANDI
│   ├── __init__.py             # Ana paket import'ları
│   ├── controllers/            # Pipeline orkestrasyonu
│   │   ├── __init__.py
│   │   └── controller.py       # ExtractionController
│   ├── models/                 # Veri modelleri ve değerlendirme
│   │   ├── __init__.py
│   │   ├── taxonomy.py         # TaxonomyManager
│   │   └── evaluation.py       # EvaluationMetrics
│   ├── services/               # İş servisleri
│   │   ├── __init__.py
│   │   ├── embedder.py         # EmbeddingService
│   │   ├── extractor.py        # KeywordExtractor
│   │   ├── classifier.py       # SectorClassifier
│   │   ├── filter.py           # KeywordFilter
│   │   └── validator.py        # LLMValidator
│   └── utils/                  # Yardımcı fonksiyonlar
│       ├── __init__.py
│       └── preprocessing.py    # TextPreprocessor
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_taxonomy_building.ipynb
│   ├── 03_extraction_experiments.ipynb
│   └── 04_evaluation.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_extractor.py
│   └── test_evaluation.py
├── config/
│   └── config.yaml             # Tüm hiperparametreler burada
├── requirements.txt
├── README.md
└── todo.md
```

- [x] Klasör yapısını oluştur (`mkdir -p` ile)
- [x] Git repo başlat, `.gitignore` ekle (veri dosyaları, `__pycache__`, `.env`)
- [x] `config/config.yaml` dosyası oluştur — tüm parametreler (model adı, threshold, k değerleri) buradan okunacak

### 0.2 Bağımlılıklar (requirements.txt)
```txt
keybert>=0.8.0
sentence-transformers>=2.6.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
nltk>=3.8.0
spacy>=3.6.0
langdetect>=1.0.9
openai>=1.0.0          # LLM doğrulama için
transformers>=4.35.0
torch>=2.0.0
tqdm>=4.66.0
pyyaml>=6.0
mlflow>=2.8.0          # Deney takibi (opsiyonel ama önerilir)
pytest>=7.4.0
```

- [x] `requirements.txt` oluştur ve `pip install -r requirements.txt` ile yükle
- [x] `python -m spacy download tr_core_news_sm` (Türkçe)
- [x] `python -m spacy download de_core_news_sm` (Almanca)
- [x] `python -m nltk.downloader stopwords punkt` çalıştır

---

## AŞAMA 1 — Veri Hazırlığı

### 1.1 Mevcut Veriyi İncele
- [x] `data/raw/handelsregister_sample_10k.csv` dosyasını aç, sütunları listele
- [x] Hizmet/faaliyet metni içeren sütunu belirle (muhtemelen `Gegenstand` veya `description`)
- [x] Kaç satırın metin alanı dolu, kaçı boş? (`df['col'].isna().sum()`)
- [x] Dil dağılımını kontrol et (Almanca mı, karışık mı?)
- [x] Metin uzunluğu dağılımını histogram ile görselleştir (notebook'ta)
- [x] Mevcut sektör/WZ kodu etiketi var mı? Varsa bu altın standarttır — koru!

### 1.2 Ek Veri Kaynakları (İsteğe Bağlı ama Güçlendirir)
- [ ] **Türkiye için:** Ticaret Sicil Gazetesi PDF'lerinden metin çek (OCR gerekebilir)
- [ ] **Açık kaynak:** Hugging Face'te `multi-class text classification` dataseti ara (sektör etiketli)
- [ ] **Sentetik veri:** Az örnekli sektörler için ChatGPT ile paraphrase üret (augmentation)
- [ ] Tüm kaynakları `data/raw/` altında kaynak adıyla klasörle

### 1.3 Sektör Taksonomisi Oluştur (`data/taxonomy/sectors.json`)

**Kullanılacak standart:** NACE Rev. 2 (AB standardı) veya WZ 2008 (Almanya)
21 ana sektör + alt sınıflar.

```json
{
  "Bilgi_Teknolojileri": {
    "nace_code": "J62",
    "seed_keywords": ["yazılım", "uygulama", "bulut", "saas", "api", "veri tabanı",
                      "siber güvenlik", "mobil uygulama", "entegrasyon", "devops"],
    "negative_keywords": ["fabrika", "tarım", "hayvancılık"]
  },
  "Enerji": {
    "nace_code": "D35",
    "seed_keywords": ["elektrik", "doğalgaz", "yenilenebilir", "güneş", "rüzgar",
                      "enerji üretimi", "santral", "trafo", "şebeke", "emisyon"],
    "negative_keywords": ["yazılım", "turizm"]
  }
  // ... 19 sektör daha
}
```

- [x] 21 NACE sektörünü listele (A-U arası büyük gruplar)
- [x] Her sektör için minimum 10 Türkçe seed kelime belirle
- [x] Her sektör için minimum 5 Almanca seed kelime ekle (çok dilli destek)
- [x] Her sektör için 3-5 "negative keyword" belirle (karışıklığı önlemek için)
- [x] `data/taxonomy/sectors.json` dosyasına kaydet
- [x] `data/taxonomy/sector_descriptions.txt` — her sektörü 2-3 cümleyle tanımla (sektör vektörü için)

---

## AŞAMA 2 — Ön İşleme (`src/utils/preprocessing.py`)

### 2.1 Temel Temizleme
```python
def clean_text(text: str, lang: str = "auto") -> str:
    # 1. Küçük harf
    # 2. URL, email, özel karakter kaldır
    # 3. Noktalama temizle (ama tire ve kesme işareti koru: "e-ticaret", "it'nin")
    # 4. Fazla boşluk normalize et
    # 5. Sayıları kaldır (veya <NUM> token ile değiştir — dene ikisini de)
```

- [x] `clean_text()` fonksiyonunu yaz
- [x] URL regex pattern'i ekle (`https?://\S+`)
- [x] E-posta regex'i ekle
- [x] Özel karakter whitelist belirle (tire, kesme işareti korunabilir)

### 2.2 Çok Dilli Stop-Word Desteği
- [x] Türkçe stop-word listesi yükle (NLTK + özel eklentiler: "ve", "veya", "ile", "olan", "olan")
- [x] Almanca stop-word listesi yükle (NLTK)
- [x] İngilizce stop-word listesi yükle
- [x] **Sektöre özgü stop-word:** "hizmet", "şirket", "firma", "limited" gibi kelimeleri filtrele (bunlar anahtar kelime değil)
- [x] `remove_stopwords(tokens, lang)` fonksiyonu yaz

### 2.3 Dil Tespiti
- [x] `langdetect` ile her metni otomatik dilini tespit et
- [x] Güven skoru düşükse (< 0.8) metni "mixed" olarak etiketle
- [x] `detect_language(text) -> (lang_code, confidence)` fonksiyonu yaz

### 2.4 N-gram Aday Üretimi
```python
def generate_ngram_candidates(text: str, n_range=(1,3)) -> List[str]:
    # 1-gram, 2-gram, 3-gram adayları üret
    # Stopword ile başlayan/biten n-gramları filtrele
    # Minimum karakter sayısı filtresi (< 3 karakter olan n-gramları at)
```

- [x] `generate_ngram_candidates()` fonksiyonu yaz
- [x] Stopword ile başlayan n-gramları filtrele
- [x] Sadece harf içeren adaylar kabul et (sayı ağırlıklı n-gramları at)
- [x] Minimum uzunluk filtresi: unigram ≥ 3 karakter, bigram ≥ 5 karakter

### 2.5 Test Et
- [x] 20 farklı hizmet metni üzerinde preprocessing pipeline'ı çalıştır
- [x] Çıktıyı gözle incele: anlamsız kelimeler hâlâ var mı?
- [ ] Unit test yaz: `tests/test_preprocessing.py`

---

## AŞAMA 3 — Embedding ve Sektör Vektörleri (`src/services/embedder.py`)

### 3.1 Model Seçimi
**Kullan:** `paraphrase-multilingual-MiniLM-L12-v2`
- Türkçe, Almanca, İngilizce destekler
- Hızlı (MiniLM) ama kaliteli
- Alternatif (daha güçlü ama yavaş): `paraphrase-multilingual-mpnet-base-v2`

- [x] `SentenceTransformer` modeli yükle ve test et
- [x] GPU varsa `.to('cuda')` ekle, yoksa CPU ile devam et
- [x] `config.yaml`'da model adını parametre olarak tut

### 3.2 Sektör Vektörleri Oluştur
```python
def build_sector_vectors(taxonomy: dict, model) -> dict:
    # Her sektör için:
    # 1. seed_keywords'ü birleştir
    # 2. sector_description ile concat et
    # 3. Bu birleşik metni embed et → sektör vektörü
    # Sonuç: {"Bilgi_Teknolojileri": np.array([...]), ...}
```

- [x] Her sektör için seed keywords + sektör tanımını birleştir
- [x] Bu birleşik metni embed ederek "sektör vektörü" oluştur
- [x] Sektör vektörlerini `data/taxonomy/sector_vectors.npy` olarak kaydet (her seferinde yeniden hesaplama)
- [x] `load_sector_vectors()` fonksiyonu ekle (cache'den yükle)

### 3.3 Doküman Embedding
- [ ] Uzun metinler için chunking uygula (512 token sınırı — transformer limiti)
- [ ] Chunk embedding'lerini average pool ile birleştir
- [ ] `embed_document(text, chunk_size=256, overlap=32) -> np.array` fonksiyonu yaz

---

## AŞAMA 4 — Sektör Sınıflandırması (`src/services/classifier.py`)

> **ÖNEMLİ:** Bu adım keyword çıkarmadan ÖNCE gelir. Önce sektörü bul, sonra o sektöre özel keyword çıkar.

### 4.1 Zero-Shot Sektör Sınıflandırması
```python
def classify_sector(doc_vector, sector_vectors, top_k=3, threshold=0.3):
    # doc_vector ile her sector_vector arasındaki cosine similarity hesapla
    # En yüksek 3 sektörü döndür
    # Eğer en yüksek skor < threshold ise → "belirsiz" döndür
    return [
        {"sector": "Bilgi_Teknolojileri", "score": 0.85},
        {"sector": "Danışmanlık", "score": 0.10},
        {"sector": "Belirsiz", "score": None}  # threshold altı
    ]
```

- [x] Cosine similarity tabanlı sınıflandırıcı yaz
- [x] `top_k=3` ile en olası 3 sektörü döndür
- [x] `confidence_threshold=0.30` altındaki tahminleri "belirsiz" olarak işaretle
- [x] Çıktı formatı: `{"primary": sector, "confidence": float, "top3": [...]}`
- [x] **Negatif keyword filtresi:** Eğer metinde sektörün negative_keyword'ü varsa skoru cezalandır
- [x] `config.yaml`'da threshold parametresi tut

### 4.2 Değerlendirme (Sınıflandırıcı)
- [ ] Eğer ground truth etiket varsa: Accuracy, F1-macro hesapla
- [ ] Confusion matrix görselleştir (notebook'ta)
- [ ] Hangi sektörler en çok karıştırılıyor? → Seed listelerini güçlendir

---

## AŞAMA 5 — Guided KeyBERT Extraction (`src/services/extractor.py`)

### 5.1 Temel Guided Extraction
```python
def extract_keywords_guided(
    text: str,
    sector: str,
    seed_keywords: list,
    sector_vector: np.array,
    top_n: int = 10,
    n_gram_range: tuple = (1, 3),
    alpha: float = 0.6,   # doc relevance ağırlığı
    beta: float = 0.4,    # seed similarity ağırlığı
) -> list:
    # 1. N-gram adayları üret
    # 2. Her adayı embed et
    # 3. custom_score = alpha * cosine(aday, doc) + beta * cosine(aday, sector_vector)
    # 4. En yüksek top_n skoru döndür
```

- [x] N-gram aday üretimi entegre et (Aşama 2.4'ten)
- [x] Her aday için `custom_score` hesapla: `alpha * doc_similarity + beta * seed_similarity`
- [x] `alpha` ve `beta` parametrelerini `config.yaml`'a al, deneysel olarak ayarla
- [x] Çıktı formatı: `[{"keyword": str, "score": float, "doc_sim": float, "seed_sim": float}]`

### 5.2 MMR (Maximal Marginal Relevance) ile Çeşitlilik
- [x] Aynı anlama gelen kelimelerin hepsini döndürme (örn: "yazılım", "software", "uygulama" aynı cluster)
- [x] KeyBERT'in `use_mmr=True, diversity=0.5` parametresini kullan
- [x] Alternatif: kendi MMR implementasyonunu yaz (daha kontrollü)

### 5.3 İteratif Seed Genişletme
```python
def iterative_expand(
    texts: list,
    initial_seeds: list,
    sector_vector: np.array,
    n_iterations: int = 3,
    expand_top_n: int = 3,
    quality_threshold: float = 0.4
) -> list:
    current_seeds = initial_seeds.copy()
    for iteration in range(n_iterations):
        # 1. Mevcut seed'lerle batch extraction çalıştır
        # 2. Her metinden top-3 keyword çıkar
        # 3. Bunları quality_threshold üzerindeyse seed listesine ekle
        # 4. Duplicate'leri kaldır
        # 5. Yeni seed'lerle bir sonraki iterasyona geç
    return expanded_seed_list
```

- [ ] `iterative_expand()` fonksiyonu yaz
- [ ] Her iterasyonda kaç seed eklendiğini logla
- [ ] Quality threshold altındaki kelimeleri seed'e ekleme
- [ ] Maksimum seed listesi boyutu belirle (örn: 50) — sınırsız büyümeyi önle
- [ ] `n_iterations=3` başlangıç için yeterli, config'den ayarlanabilir olsun

## AŞAMA 6 — Gelişmiş Filtreleme (`src/services/filter.py`)

### 6.1 VIB Tarzı Bilgi Yoğunluğu Filtresi
Tam VIB matematiksel implementasyonu yerine pratik yaklaşım:

```python
def vib_style_filter(
    candidates: list,
    doc_vector: np.array,
    sector_vector: np.array,
    noise_threshold: float = 0.25
) -> list:
    filtered = []
    for candidate in candidates:
        cand_vector = embed(candidate["keyword"])
        # Bilgi yoğunluğu = sektörle ilgisi * dokümandaki varlığı
        information_score = cosine(cand_vector, sector_vector) * cosine(cand_vector, doc_vector)
        # Noise = sektörden uzaklık
        noise_score = 1 - cosine(cand_vector, sector_vector)
        if noise_score < noise_threshold:
            candidate["information_score"] = information_score
            filtered.append(candidate)
    return sorted(filtered, key=lambda x: x["information_score"], reverse=True)
```

- [x] `vib_style_filter()` fonksiyonu yaz
- [x] `noise_threshold=0.25` başlangıç değeri, config'den ayarlanabilir
- [x] Filtreleme öncesi/sonrası kelime sayısını logla (ne kadar filtrelendi?)

### 6.2 Genel Kelime Filtresi
- [x] Sektör-agnostik çok genel kelimeleri at: "hizmet", "şirket", "ürün", "müşteri", "kalite"
- [x] Bu "generic stopword" listesini `data/taxonomy/generic_keywords.txt`'e kaydet
- [x] Pipeline başında bu listeyi yükle

### 6.3 Minimum Skor Eşiği
- [x] Final skor < 0.2 olan adayları at
- [x] Eşiği config'den ayarlanabilir yap

---

## AŞAMA 7 — LLM Doğrulama (`src/services/validator.py`) [OPSİYONEL]

> Bu adım pahalı (API maliyeti) ve yavaştır. Sadece final kalite kontrol için veya belirsiz vakalar için kullan.

### 7.1 Doğrulama Stratejisi
- [x] Sadece confidence skoru 0.3-0.6 arasındaki "gri alan" keyword'leri LLM'e gönder
- [x] Yüksek confidence'lıları (> 0.6) doğrudan kabul et

### 7.2 Prompt Tasarımı
```python
VALIDATION_PROMPT = """
Görev: Aşağıdaki kelimenin verilen sektör için anlamlı bir anahtar kelime olup olmadığını değerlendir.

Sektör: {sector}
Hizmet metni: {text}
Anahtar kelime: {keyword}

Sadece şu formatta yanıt ver:
{{
  "is_relevant": true/false,
  "reason": "kısa açıklama"
}}
"""
```

- [x] `validate_with_llm(keyword, sector, context_text) -> bool` fonksiyonu yaz
- [x] Rate limiting ekle (dakikada max 60 istek)
- [x] API hata yönetimi ekle (retry logic)
- [x] Sonuçları cache'e al (aynı keyword + sektör kombinasyonunu tekrar sorma)
- [x] Batch validation: aynı anda 10 keyword gönder, maliyeti azalt

---

## AŞAMA 8 — Ana Pipeline Orkestrasyonu (`src/controllers/controller.py`)

### 8.1 Tam Pipeline Akışı
```
Girdi: Ham hizmet metni
  ↓
[1] Dil tespiti
  ↓
[2] Metin temizleme + stop-word kaldırma
  ↓
[3] N-gram aday üretimi
  ↓
[4] Doküman embedding
  ↓
[5] Sektör sınıflandırması (zero-shot cosine similarity)
  ↓
[6] Belirsiz mi? → "Belirsiz" etiketi ver ve dur (veya top-1 ile devam et)
  ↓
[7] Sektöre özel guided keyword extraction (alpha*doc + beta*seed)
  ↓
[8] MMR ile çeşitlilik sağla
  ↓
[9] VIB tarzı filtreleme
  ↓
[10] (Opsiyonel) LLM doğrulama (sadece gri alan vakalar)
  ↓
Çıktı: {"sector": "Bilgi_Teknolojileri", "confidence": 0.85,
         "keywords": ["bulut bilişim", "saas", "api entegrasyonu", ...]}
```

- [x] `Pipeline` sınıfı oluştur
- [x] `Pipeline.run(text: str) -> dict` ana metodu yaz
- [x] `Pipeline.run_batch(texts: list, parallel=True) -> list` ekle (hız için)
- [x] Parallel processing: `joblib.Parallel` veya `multiprocessing` kullan
- [x] Her adımda hata yakalanmasın, pipeline kırılmamalı → try/except + fallback
- [x] Her adımın süresini logla (hangi adım yavaş?)

### 8.2 Logging ve Deney Takibi
- [x] Python `logging` modülü ile structured log ekle
- [x] MLflow ile her run'ı kaydet:
  - Parametreler: alpha, beta, threshold, n_iterations
  - Metrikler: Precision@K, ortalama cosine similarity
  - Artifacts: çıktı JSON dosyaları
- [ ] Her batch run sonunda özet rapor üret (kaç metin işlendi, hata oranı, ortalama confidence)

---

## AŞAMA 9 — Değerlendirme (`src/models/evaluation.py`)

### 9.1 Precision@K
```python
def precision_at_k(extracted: list, ground_truth: list, k: int = 10) -> float:
    top_k = extracted[:k]
    hits = sum(1 for kw in top_k if kw in ground_truth)
    return hits / k
```

- [ ] `precision_at_k(extracted, ground_truth, k=10)` fonksiyonu yaz
- [ ] K=5, K=10, K=20 için ayrı ayrı hesapla
- [ ] Sektör bazında P@K hesapla (hangi sektör daha zor?)

### 9.2 Cosine Similarity Match
```python
def semantic_match_score(extracted_keywords, gold_standard_keywords, model) -> float:
    # Her extracted keyword için gold standard'a en yakın eşleşmeyi bul
    # Ortalama max cosine similarity döndür
```

- [ ] `semantic_match_score()` fonksiyonu yaz
- [ ] Bu metrik, exact match olmayan ama anlamsal olarak doğru kelimeleri de ödüllendirir

### 9.3 Sektör Sınıflandırma Metrikleri
- [ ] Accuracy (top-1)
- [ ] Top-3 Accuracy (doğru sektör top-3'te mi?)
- [ ] F1-Macro (sınıf dengesizliği varsa önemli)
- [ ] Cohen's Kappa (inter-annotator agreement için)

### 9.4 İnsan Değerlendirmesi (Manuel)
- [ ] 100 random örnek seç
- [ ] Her örnek için: hangi keyword'ler gerçekten sektöre uygun?
- [ ] Bu manuel etiketleri `data/evaluation/human_labels.json`'a kaydet
- [ ] Bu set ile final modeli değerlendir

### 9.5 Evaluation Notebook
- [ ] `notebooks/04_evaluation.ipynb` oluştur
- [ ] Metrik sonuçlarını tablo ve grafik olarak göster
- [ ] Hata analizi: en çok nerelerde yanılıyor?

---

## AŞAMA 10 — Test Altyapısı

- [ ] `tests/test_preprocessing.py` — temizleme fonksiyonları unit test
- [ ] `tests/test_extractor.py` — extraction çıktı format kontrolü
- [ ] `tests/test_evaluation.py` — metrik hesaplama doğruluğu
- [ ] `tests/test_pipeline_e2e.py` — baştan sona entegrasyon testi (5 örnek metin ile)
- [ ] `pytest` ile tüm testleri çalıştır, CI/CD'ye ekle (GitHub Actions opsiyonel)

---

## AŞAMA 11 — Dokümantasyon ve Sunum

- [ ] `README.md` güncelle:
  - Proje amacı
  - Kurulum adımları
  - Kullanım örneği (kod snippet)
  - Pipeline akış diyagramı
- [ ] Her fonksiyon için docstring yaz (Google style)
- [ ] Akademik rapor için kullanılan makaleleri ve metodoloji kararlarını `docs/methodology.md`'ye kaydet
- [ ] Hocaya göndermek için: sonuçları gösteren 1 sayfalık özet tablo hazırla

---

## 🔢 ÖNCELİK SIRASI (Hangi Sırayla Başla?)

| # | Görev | Neden Önce? | Tahmini Süre |
|---|-------|-------------|--------------|
| 1 | Proje klasör yapısı + requirements | Her şeyin temeli | 1 saat |
| 2 | Veri inceleme (1.1) | Neyle çalıştığını bilmeden devam edemezsin | 2 saat |
| 3 | Sektör taksonomisi (1.3) | Seed olmadan guided extraction olmaz | 3 saat |
| 4 | Preprocessing pipeline (Aşama 2) | Çöp girer → çöp çıkar | 2 saat |
| 5 | Embedder + sektör vektörleri (Aşama 3) | Temel representation | 2 saat |
| 6 | Sektör sınıflandırıcı (Aşama 4) | Pipeline sırası kritik | 2 saat |
| 7 | Guided extraction (Aşama 5.1) | Core algoritma | 3 saat |
| 8 | İteratif genişletme (Aşama 5.3) | Otonom öğrenme | 2 saat |
| 9 | VIB filtre (Aşama 6) | Çıktı kalitesi | 2 saat |
| 10 | Evaluation (Aşama 9) | Sonuçların anlamlı olması için | 3 saat |
| 11 | LLM doğrulama (Aşama 7) | İsteğe bağlı, son adım | 2 saat |
| 12 | Test + Dokümantasyon | Hocaya sunum için | 2 saat |

**Toplam tahmini süre: ~26 saat (düzenli çalışmayla 1-2 hafta)**

---

## ⚠️ BİLİNEN RİSKLER ve ÇÖZÜMLER

| Risk | Olasılık | Çözüm |
|------|----------|-------|
| Türkçe embedding kalitesi düşük | Orta | Çok dilli model kullan, gerekirse fine-tune et |
| Seed listesi yetersiz | Yüksek | İlk iterasyon sonucu manuel incele ve genişlet |
| LLM API maliyeti | Orta | Sadece gri alan vakalar için kullan, cache ekle |
| Veri etiketinin olmaması | Orta | Manuel 100 örnek etiketle, bunu altın standart kullan |
| Belirsiz sektör sınırları | Yüksek | Overlap'e izin ver (top-3 döndür), hard assignment yapma |
| Uzun metinlerde embedding sınırı | Düşük | Chunking + average pooling uygula |

---

## 📊 BAŞARI KRİTERLERİ

Projenin başarılı sayılması için minimum hedefler:

- [ ] Sektör sınıflandırma Top-1 Accuracy ≥ %70
- [ ] Sektör sınıflandırma Top-3 Accuracy ≥ %85
- [ ] Precision@10 ≥ %60 (10 keyword'den 6'sı doğru sektöre ait)
- [ ] Ortalama Cosine Similarity Match ≥ 0.65
- [ ] İşleme hızı: saniyede en az 10 metin (batch modda)

---

## 📝 NOTLAR

- `config.yaml`'ı düzenli güncelle — her hiperparametre oradan okunmalı
- Deney sonuçlarını MLflow veya basit bir CSV loguna kaydet (neyin işe yaradığını unutma)
- Her aşama bittikten sonra notebook'ta 5-10 örnekle görsel kontrol yap
- Hocan "sektörel tahmin" dedi — bunu hem sınıflandırma (hangi sektör?) hem de açıklama (neden?) olarak sun
- Sonuçları `{"sector": "X", "confidence": 0.85, "keywords": [...], "reasoning": "..."}` formatında sakla

---

*Son güncelleme: Proje başlangıcında oluşturuldu. Her tamamlanan görev `- [x]` ile işaretlenir.*
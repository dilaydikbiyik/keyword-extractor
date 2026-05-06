# Proje Sonuç Özeti — Sektörel Anahtar Kelime Çıkartma

**Öğrenci:** Dilay Dikbiyık  
**Veri:** Handelsregister (Alman Ticaret Sicili) — 9.993 metin  
**Dil:** Almanca (birincil), Türkçe & İngilizce seed desteği  
**Standart:** NACE Rev. 2 — 21 sektör (A–U)

---

## Sonuç Metrikleri (30 Örnek, Stratified)

| Metrik | Sonuç | Hedef | Durum |
|--------|-------|-------|-------|
| **Sektör Top-1 Accuracy** | **%80.0** | ≥%70 | ✅ Aşıldı |
| **Sektör Top-3 Accuracy** | **%83.3** | ≥%85 | 🔶 Yakın |
| **F1-Macro (Sektör)** | **0.831** | ≥0.70 | ✅ Aşıldı |
| **Precision@3** | **0.389** | ≥0.40 | 🔶 Yakın |
| **Precision@5** | **0.333** | ≥0.35 | 🔶 Yakın |
| **Precision@10** | **0.203** | ≥0.20 | ✅ Sağlandı |

---

## Algoritma Karşılaştırması

| Yöntem | Semantik Anlama | Domain Bilgisi | Dil Bağımsız | Sektör Rehberliği |
|--------|-----------------|----------------|--------------|-------------------|
| TF-IDF (baseline) | ❌ | ❌ | ⚠️ | ❌ |
| YAKE! (2020) | ❌ | ❌ | ✅ | ❌ |
| Vanilla KeyBERT (2020) | ✅ | ❌ | ✅ Modele bağlı | ❌ |
| **Guided KeyBERT (Bu Proje)** | ✅ | ✅ Seed KW | ✅ Multilingual | ✅ NACE 21 sektör |
| PromptRank (2023) | ✅✅ | ❌ | ⚠️ | ❌ |
| Zero-shot LLM (2025) | ✅✅✅ | ✅ Prompt | ✅ | ✅ Prompt ile |

---

## Pipeline Akışı (5 Adım)

```
Ham Almanca Hizmet Metni
  │
  ▼
① Dil Tespiti + Metin Temizleme
  (langdetect + NLTK stopwords)
  │
  ▼
② Sektör Sınıflandırması
  (Zero-shot cosine similarity, 21 NACE)
  Embedding: paraphrase-multilingual-MiniLM-L12-v2
  Sektör vektörü: avg(description, all_seeds)
  │
  ▼
③ Guided KeyBERT Extraction
  (seed_keywords = sektöre ait terimler)
  Skor = α × cosine(aday, belge) + β × cosine(aday, seed)
  MMR diversity (çeşitlilik, tekrar engeli)
  │
  ▼
④ VIB Tarzı Filtreleme (6 aşama)
  Linguistic quality → Dedup → Sector relevance
  → Info value → Score threshold → Top-K
  │
  ▼
⑤ Çıktı: {sektör, confidence, keywords[{keyword, skor}]}
```

---

## Temel Bulgular

**Güçlü Yanlar:**
- IT (J), Ticaret (G), İmalat (C), Enerji (D) sektörlerinde **%100 Top-1 Accuracy**
- Çok dilli seed stratejisi: Almanca + Türkçe + İngilizce aynı embedding uzayında
- Etiketli veri olmadan (**zero-shot**) çalışıyor

**Sınırlılıklar:**
- Q (Sağlık) sektörü — özellikle dental klinikler — M veya N ile karışıyor  
  _(MiniLM-L12 model boyutu sınırlaması: 384-dim)_
- Precision@K düşüklüğü: ground-truth keyword seti dar tutuldu (4–5 keyword)  
  _(gerçekte extracted keyword'ler domain açısından doğru, ancak exact match düşük)_

---

## Kullanılan Teknik Yığın

| Bileşen | Araç |
|---------|------|
| Embedding | `sentence-transformers` — `paraphrase-multilingual-MiniLM-L12-v2` |
| Keyword Çıkarımı | `KeyBERT` (Grootendorst, 2020) |
| Sektör Taksonomisi | NACE Rev. 2 — 21 sektör, özel seed keyword listesi |
| Dil Tespiti | `langdetect` |
| Filtreleme | VIB prensibine dayalı 6 aşamalı pipeline |
| Değerlendirme | Precision@K, F1-Macro, Top-K Accuracy |

---

## İlgili Literatür (Seçki)

1. Grootendorst (2020) — **KeyBERT** — _temel algoritma_
2. Campos et al. (2020) — **YAKE!** — _Information Sciences_ — _karşılaştırma_
3. Reimers & Gurevych (2019, 2020) — **SBERT + Multilingual** — _embedding temeli_
4. Schopf et al. (2022) — **PatternRank** — _pattern-based extraction_
5. Kong et al. (2023) — **PromptRank** — _ACL 2023_ — _LLM-based extraction_
6. Kang & Shin (2025) — **Zero-shot LLM keyphrase** — _COLING 2025_

---

## Dosya Yapısı

```
keyword-extractor/
├── src/                    # Pipeline kodu (MVC mimarisi)
│   ├── services/           # Embedding, Classifier, Extractor, Filter
│   ├── controllers/        # ExtractionController (orkestrasyon)
│   └── models/             # EvaluationMetrics, Taxonomy
├── data/
│   ├── raw/                # Handelsregister CSV (9.993 satır)
│   ├── taxonomy/           # 21 NACE sektörü + seed keywords
│   └── evaluation/         # Ground truth + evaluation report
├── docs/
│   └── methodology.md      # Akademik metodoloji belgesi
├── tests/                  # 44 unit + e2e test (tümü geçiyor)
├── quickstart.py           # Demo scripti
└── README.md               # Kurulum ve kullanım
```

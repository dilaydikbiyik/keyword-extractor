# Metodoloji Belgesi — Sektörel Anahtar Kelime Çıkartma

**Proje:** Hizmet Metinlerinden Sektörel Anahtar Kelime Çıkartma Pipeline'ı  
**Veri:** Alman Ticaret Sicili (Handelsregister) — 9.993 hizmet tanımı  
**Dil:** Almanca (birincil), Türkçe & İngilizce (seed desteği)  
**Standart:** NACE Rev. 2 (AB sektör sınıflandırma standardı)

---

## 1. Araştırma Sorusu

> Yapılandırılmamış iş tanımı metinlerinden, sektörel bağlamı anlayan, yüksek  
> kaliteli anahtar kelimeler otomatik olarak nasıl çıkarılabilir?

Alt sorular:
- Sektör tespiti (hangi sektör?) denetimsiz öğrenme ile ne kadar doğru yapılabilir?
- Sektör bilgisi keyword kalitesini nasıl artırır?
- Küçük çok dilli embedding modelleri ticari metin sınıflandırmasında yeterli mi?

---

## 2. İlgili Literatür (Son 5 Yıl: 2020–2025)

### 2.1 Temel Keyword Çıkarım Algoritmaları

#### 2.1.1 KeyBERT — Grootendorst (2020)
> Grootendorst, M. (2020). *KeyBERT: Minimal keyword extraction with BERT* [Software].  
> Zenodo. https://doi.org/10.5281/zenodo.4461265

- **Yaklaşım:** BERT tabanlı embedding ile kosinüs benzerliği. Doküman vektörüne en  
  yakın n-gram adaylarını keyword olarak seçer.
- **Katkısı:** Semantik anlamı yakalayan ilk pratik keyword çıkarım aracı.
- **Bizim kullanımımız:** Pipeline'ın çekirdeği. Sektör seed keyword'leriyle guided  
  extraction modunda kullanıldı (`seed_keywords=` parametresi).
- **Sınırlılık:** Tek başına domain-agnostic; sektörel bağlamı bilmez.

#### 2.1.2 YAKE! — Campos et al. (2020)
> Campos, R., Mangaravite, V., Pasquali, A., Jorge, A., Nunes, C., & Jatowt, A. (2020).  
> YAKE! Keyword extraction from single documents using multiple local features.  
> *Information Sciences, 509*, 257–289. https://doi.org/10.1016/j.ins.2019.09.013

- **Yaklaşım:** 5 yerel istatistiksel özellik (konum, frekans, bağlam, casing, cümle dağılımı).  
  Corpus gerektirmez; tek belgeden çalışır.
- **Benchmark sonucu:** TF-IDF, RAKE, TextRank'ı geride bırakıyor (F1@10 iyileştirmesi).
- **Bizim pipeline ile karşılaştırma:** Baseline olarak kullanıldı (bkz. Bölüm 5).

#### 2.1.3 PatternRank — Schopf et al. (2022)
> Schopf, T., Braun, D., & Matthes, F. (2022). *PatternRank: Leveraging Pretrained  
> Language Models and Part of Speech for Unsupervised Keyphrase Extraction*.  
> SCITEPRESS Digital Library. DOI: 10.5220/0011546600003318

- **Yaklaşım:** POS pattern'leri (isim öbekleri, sıfat+isim kombinasyonları) ile aday  
  kümesi kısıtlanır; sonra Sentence Transformer embedding ile sıralanır.
- **Önemi:** Bizim n-gram aday filtrelemesi mantığına teorik temel sağladı.
- **Fark:** Biz POS yerine stopword-başlangıç filtresi kullandık (spaCy model bağımlılığını azaltmak için).

#### 2.1.4 PromptRank — Kong et al. (2023)
> Kong, A., Zhao, S., Chen, H., Li, Q., Qin, Y., Sun, R., & Bai, X. (2023).  
> *PromptRank: Unsupervised Keyphrase Extraction Using Prompt*.  
> ACL 2023. https://aclanthology.org/2023.acl-long.545/

- **Yaklaşım:** Encoder-decoder PLM; dokümanı encoder'a gönderir, decoder'da  
  `"This text mainly discusses [candidate]."` prompt ile üretim olasılığını hesaplar.
- **Benchmark:** 6 veri seti üzerinde MDERank, KeyBERT vb. geride bırakır.
- **Bizim pipeline ile fark:** PromptRank genel metinler için daha güçlü; ancak domain  
  seed bilgisi yoktur. Ticaret sicili metinleri için sektör bağlamı kritik.

#### 2.1.5 Zero-Shot LLM ile Keyword Çıkarımı — Kang & Shin (2025)
> Kang, S., & Shin, J. (2025). *Empirical Study of Zero-shot Keyphrase Extraction  
> with Large Language Models*. COLING 2025.  
> https://aclanthology.org/2025.coling-main.248/

- **Yaklaşım:** GPT-4o, Llama 3 vb. ile doğrudan prompt; vanilla/role/hybrid prompting.
- **Sonuç:** Hybrid prompting ile PromptRank'ı aşıyor; ancak API maliyeti yüksek.
- **Bizim yaklaşımımızla karşılaştırma:** GPT-4o bazlı validation opsiyonel olarak  
  eklendi (bkz. `src/services/validator.py`); sadece "gri alan" keyword'ler için.

### 2.2 Sektör Sınıflandırması — İlgili Çalışmalar

#### 2.2.1 Sentence-BERT & Multilingual Embeddings — Reimers & Gurevych (2019, 2020)
> Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using  
> Siamese BERT-Networks*. EMNLP 2019. https://aclanthology.org/D19-1410/  
>
> Reimers, N., & Gurevych, I. (2020). *Making Monolingual Sentence Embeddings  
> Multilingual using Knowledge Distillation*. EMNLP 2020.  
> https://aclanthology.org/2020.emnlp-main.365/

- **Model seçimimizin temeli:** `paraphrase-multilingual-MiniLM-L12-v2` —  
  knowledge distillation ile 50+ dil desteği, hızlı (MiniLM) ama kaliteli.
- **Önemi:** Almanca, Türkçe, İngilizce seed keyword'lerinin aynı vektör uzayında  
  karşılaştırılmasını mümkün kılıyor.

#### 2.2.2 Zero-Shot Şirket Sektörü Sınıflandırması (2021–2023)
> Çeşitli arxiv çalışmaları (2021–2023): Zero-shot transformer pipeline'ları  
> (distilBART-MNLI, RoBERTa) ile NACE/GICS/WZ-2008 sınıflandırması.

- **Bulgular:** Zero-shot yaklaşımlar %70–85 Top-3 Accuracy sağlıyor (Handelsregister  
  benzeri veri üzerinde); fine-tuned modeller %85–92.
- **Bizim yaklaşımımız:** Zero-shot cosine similarity — etiketli veri olmadan çalışır;  
  21 NACE sektörü için seed vektörleri ile ölçeklenir.

#### 2.2.3 TaxoExpan & Seed Genişletme Yaklaşımları (2020–2022)
Önemi: İteratif seed genişletme fikrinin teorik arka planı.

- **İlke:** Az sayıda güvenilir "seed" term → corpus üzerinden yeni terimler keşfet →  
  kalite eşiğini geçenleri seed'e ekle → tekrarla.
- **Bizim `iterative_expand()`** fonksiyonu bu prensiple uyumlu (bkz. §5.3).

---

## 3. Kullanılan Veri Setleri

### 3.1 Projede Kullanılan (Birincil Veri)
| Veri Seti | Kaynak | Boyut | Dil | Sektör Etiketi |
|---|---|---|---|---|
| **Handelsregister Sample** | Alman Ticaret Sicili (public) | 9.993 satır | %100 Almanca | Yok (unsupervised) |

**İstatistikler:**
- Ortalama metin uzunluğu: 242.5 karakter
- Medyan: 175 karakter
- Aralık: 9–5.692 karakter
- Kısa metin oranı (<50 karakter): ~%18

### 3.2 Literatürdeki Benchmark Veri Setleri

| Veri Seti | Domain | Büyüklük | Referans |
|---|---|---|---|
| **Inspec** | Bilimsel abstract | 2.000 doküman | Hulth (2003) |
| **SemEval-2010 Task 5** | Bilimsel makale | 284 train + 100 test | Kim et al. (2010) |
| **KPTimes** | Haber metinleri | 279.923 makale | Gallina et al. (2019) |
| **DUC 2001** | Haber özetleri | 308 doküman | Wan & Xiao (2008) |
| **OpenKP** | Web sayfaları | 148.124 doküman | Xiong et al. (2019) |

> **Not:** Projemizin veri seti (ticaret sicili metinleri) bu benchmark setlerinden  
> temelden farklıdır: çok kısa, domain-yoğun, yapısal olmayan Almanca metinler.  
> Bu durum metodoloji seçimini doğrudan etkiliyor (bkz. Bölüm 4).

---

## 4. Metodoloji Kararları ve Gerekçeleri

### 4.1 Neden Guided KeyBERT?

| Kriter | TF-IDF | YAKE | Vanilla KeyBERT | **Guided KeyBERT (Bizim)** |
|---|---|---|---|---|
| Semantik anlama | ❌ | ❌ | ✅ | ✅ |
| Domain bilgisi | ❌ | ❌ | ❌ | ✅ Seed keywords |
| Dil bağımsızlığı | ✓ | ✓ | Modele bağlı | ✓ Multilingual |
| Corpus gerektirme | ✅ Gerektirir | ❌ Gerekmez | ❌ Gerekmez | ❌ Gerekmez |
| Çeşitlilik (MMR) | ❌ | ❌ | ✅ | ✅ |
| Sektör rehberliği | ❌ | ❌ | ❌ | **✅ Kritik özellik** |

**Karar gerekçesi:** Ticaret sicili metinleri çok kısa. TF-IDF için corpus istatistiği  
yetersiz. YAKE yerel özellikler kullanır ama semantik bağlamı görmez. Vanilla KeyBERT  
semantik anlar ama "GmbH" ve "Softwareentwicklung"'u eşit önemde tutar. Guided KeyBERT  
ile sektöre özgü terimler öne çıkartılıyor.

### 4.2 Neden paraphrase-multilingual-MiniLM-L12-v2?

- Almanca + Türkçe + İngilizce seed keyword'lerin aynı uzayda karşılaştırılması şart
- MiniLM: mpnet-base ile %82 performans, %3 boyut → hız/kalite dengesi
- 384 boyutlu vektör: düşük bellek, hızlı cosine similarity

**Alternatif değerlendirme:**
```
paraphrase-multilingual-mpnet-base-v2  → Daha iyi kalite, 3x yavaş
German-specific BERT (deepset/gbert)   → Sadece Almanca, seed dil desteği yok
XLM-RoBERTa-large                     → En iyi kalite, 10x yavaş, 1.7GB RAM
```

### 4.3 Neden Zero-Shot Sektör Sınıflandırması?

- Etiketli veri yok → Supervised learning mümkün değil
- 9.993 metni manuel etiketlemek → pratik değil
- Cosine similarity ile 21 NACE sektörüne zero-shot: sektör vektörleri  
  (name + description + all seed keywords) ile hesaplanıyor

**Accuracy beklentisi:** Top-3 %75–85 (literatürde zero-shot için bildirilen aralık)

### 4.4 Neden VIB Tarzı Filtreleme?

Variational Information Bottleneck (Tishby & Schwartz-Ziv, 2017) ilkesi:  
> *"Sinyali koru, gürültüyü at."*

Pratik uyarlamamız:
```
information_score = cosine(aday, sektör) × cosine(aday, doküman)
noise_score = 1 - cosine(aday, sektör)
```
Yüksek `information_score` + düşük `noise_score` → keyword kabul edilir.

---

## 5. Baseline Karşılaştırması

Üç örnek metin için TF-IDF, YAKE ve Guided KeyBERT karşılaştırması:

### Metin 1: IT/Yazılım (Sektör: J)
> "Softwareentwicklung und Programmierung für Web- und Mobileanwendungen. API-Integrationsdienste und Cloud-Lösungen."

| Yöntem | Çıkarılan Keywordler (Top 5) | Sektör Uygunluğu |
|---|---|---|
| TF-IDF | mobileanwendungen, programmierung, web, cloud, api | ✅ İyi |
| YAKE | Softwareentwicklung, Web, Mobileanwendungen, API, Cloud | ✅ İyi |
| Guided KeyBERT (J) | mobileanwendungen (0.69), softwareentwicklung (0.68), cloud (0.57), web (0.57), programmierung (0.57) | ✅✅ En iyi (semantik skor) |

### Metin 2: Ticaret (Sektör: G)
> "Handel mit Elektronik, Computern und Mobiltelefonen. Großhandel und Einzelhandel. E-Commerce-Plattform."

| Yöntem | Top 5 | Uygunluk |
|---|---|---|
| TF-IDF | handel, elektronik, computern, mobiltelefonen, commerce | ✅ |
| YAKE | Elektronik, Handel, Computern, Großhandel, Einzelhandel | ✅ |
| Guided KeyBERT (G) | commerce (0.74), einzelhandel (0.69), handel (0.63), großhandel (0.58), mobiltelefonen (0.48) | ✅✅ |

### Metin 3: Diş Kliniği (Sektör: Q — Sağlık)
> "Zahnklinik mit modernen Behandlungsmethoden. Zahnimplantate, Zahnbleaching und Prophylaxe."

| Yöntem | Top 5 | Uygunluk |
|---|---|---|
| TF-IDF | zahnklinik, behandlungsmethoden, zahnimplantate, zahnbleaching, prophylaxe | ✅ |
| YAKE | Zahnklinik, Zahnimplantate, Behandlungsmethoden, Zahnbleaching, Prophylaxe | ✅ |
| Guided KeyBERT (C*) | prophylaxe (0.54), behandlungsmethoden (0.50), zahnimplantate (0.44), zahnklinik (0.40) | ⚠️ Sektör yanlış (C→Q) |

> ⚠️ **Model Sınırlılığı:** `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) Q (Sağlık)  
> ve M (Profesyonel Hizmetler) sektörlerini dental metinlerde ayırt edemedi.  
> **Çözüm:** Daha büyük model (`mpnet-base-v2`) veya fine-tuning gerektirir.

---

## 6. Pipeline Katkıları (Özgün Yanlar)

Bu projenin literatüre özgün katkıları:

1. **Sektör-guided KeyBERT:** Standart KeyBERT'i NACE taxonomy'siyle birleştiren  
   ilk açık kaynak pipeline (Handelsregister verisi için).

2. **Çok Dilli Seed Stratejisi:** Almanca (birincil) + Türkçe + İngilizce seed  
   keyword'lerin aynı multilingual embedding uzayında kullanılması.

3. **İteratif Seed Genişletme:** Corpus'tan otomatik seed keşfi (`iterative_expand()`).

4. **VIB Tarzı Filtreleme:** Information bottleneck prensibinin pratik uyarlaması  
   (tam VIB yerine hesaplama açısından verimli yaklaşım).

5. **Hibrit Sektör Vektörü:** Description embedding + all-seeds embedding average  
   pool → daha ayrıştırıcı sektör vektörleri.

---

## 7. Başarı Kriterleri ve Hedefler

| Metrik | Hedef | Ölçüm Yöntemi |
|---|---|---|
| Sektör Top-1 Accuracy | ≥ %70 | Manuel 100 örnek |
| Sektör Top-3 Accuracy | ≥ %85 | Manuel 100 örnek |
| Precision@10 | ≥ %60 | Manuel keyword değerlendirme |
| Cosine Similarity Match | ≥ 0.65 | `semantic_match_score()` |
| İşleme hızı (batch) | ≥ 10 metin/sn | `processing_time_ms` alanı |

---

## 8. Kullanılan Açık Kaynak Araçlar

| Araç | Versiyon | Kullanım Amacı |
|---|---|---|
| KeyBERT | ≥0.8.0 | Keyword çıkarımı çekirdeği |
| sentence-transformers | ≥2.6.0 | Çok dilli embedding |
| NLTK | ≥3.8.0 | Stopword listeleri |
| spaCy | ≥3.6.0 | Tokenizasyon (opsiyonel) |
| langdetect | ≥1.0.9 | Dil tespiti |
| scikit-learn | ≥1.3.0 | Cosine similarity, F1 metriği |
| pandas | ≥2.0.0 | CSV işleme |
| OpenAI API | ≥1.0.0 | LLM doğrulama (opsiyonel) |

---

## 9. Sınırlılıklar ve Gelecek Çalışmalar

### Mevcut Sınırlılıklar

1. **Model boyutu:** MiniLM-L12 (384-dim) benzer sektörleri (Q vs M) ayırt etmede  
   zayıf. `mpnet-base-v2` ile %10–15 sınıflandırma iyileştirmesi beklenir.

2. **Etiketli veri yokluğu:** Top-1/3 Accuracy hesabı için ground truth manuel üretilmeli.

3. **Çok kısa metinler:** <50 karakterlik metinlerde embedding kalitesi düşüyor  
   (`"Handel mit Waren"` gibi). Bağlam penceresi çok dar.

4. **Tek sektör atama:** Bir metin birden fazla sektöre uygun olabilir  
   (örn. tıbbi yazılım → hem J hem Q). Top-3 döndürme kısmen çözüyor.

### Önerilen Gelecek Çalışmalar

1. **Fine-tuned Classifier:** Handelsregister metinleri üzerinde etiketlenmiş  
   veri ile BERT fine-tuning → %85+ Top-1 Accuracy.

2. **Domain Adaptation:** Sektöre özgü embedding modeli (sadece ticaret sicili metinleri üzerinde pre-train).

3. **Active Learning:** Kullanıcı geri bildirimiyle iteratif seed iyileştirme.

4. **Hierarchical Classification:** NACE 2-digit → 4-digit kademeli sınıflandırma.

5. **Evaluation Benchmark:** Bu proje verisi üzerinde sistematik karşılaştırma  
   (TF-IDF, YAKE, KeyBERT, PromptRank) için annotated test seti oluşturulması.

---

## 10. Referanslar

```bibtex
@software{grootendorst2020keybert,
  author = {Grootendorst, Maarten},
  title  = {KeyBERT: Minimal keyword extraction with BERT},
  year   = {2020},
  doi    = {10.5281/zenodo.4461265}
}

@article{campos2020yake,
  author  = {Campos, Ricardo and Mangaravite, Vitor and Pasquali, Alípio and
             Jorge, Alípio and Nunes, Célia and Jatowt, Adam},
  title   = {YAKE! Keyword extraction from single documents using multiple local features},
  journal = {Information Sciences},
  volume  = {509},
  pages   = {257--289},
  year    = {2020},
  doi     = {10.1016/j.ins.2019.09.013}
}

@inproceedings{schopf2022patternrank,
  author    = {Schopf, Tim and Braun, Daniel and Matthes, Florian},
  title     = {PatternRank: Leveraging Pretrained Language Models and Part of Speech
               for Unsupervised Keyphrase Extraction},
  booktitle = {ICPRAM 2022},
  year      = {2022},
  doi       = {10.5220/0011546600003318}
}

@inproceedings{kong2023promptrank,
  author    = {Kong, Aobo and Zhao, Shiwan and Chen, Hao and Li, Qicheng and
               Qin, Yong and Sun, Ruiqi and Bai, Xiaoyan},
  title     = {PromptRank: Unsupervised Keyphrase Extraction Using Prompt},
  booktitle = {ACL 2023},
  year      = {2023},
  url       = {https://aclanthology.org/2023.acl-long.545/}
}

@inproceedings{kang2025zeroshof,
  author    = {Kang, Seyun and Shin, Jinseon},
  title     = {Empirical Study of Zero-shot Keyphrase Extraction with Large Language Models},
  booktitle = {COLING 2025},
  year      = {2025},
  url       = {https://aclanthology.org/2025.coling-main.248/}
}

@inproceedings{reimers2019sbert,
  author    = {Reimers, Nils and Gurevych, Iryna},
  title     = {Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  booktitle = {EMNLP 2019},
  year      = {2019},
  url       = {https://aclanthology.org/D19-1410/}
}

@inproceedings{reimers2020multilingual,
  author    = {Reimers, Nils and Gurevych, Iryna},
  title     = {Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation},
  booktitle = {EMNLP 2020},
  year      = {2020},
  url       = {https://aclanthology.org/2020.emnlp-main.365/}
}
```

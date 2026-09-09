# Sectoral Keyword Extraction Pipeline

[![CI](https://github.com/dilaydikbiyik/keyword-extractor/actions/workflows/ci.yml/badge.svg)](https://github.com/dilaydikbiyik/keyword-extractor/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

An unsupervised pipeline that assigns German trade register business purposes to
**NACE Rev. 2** sections and extracts sector-aware keywords, using multilingual
sentence embeddings and taxonomy-guided KeyBERT — no labelled training data.

![Reproducing the reported results end to end](docs/assets/demo.gif)

---

## Results

30-document stratified evaluation set, 18 NACE sections.  Every figure is
produced by `make reproduce` and written to
[`results/metrics.json`](results/metrics.json).

| System | Top-1 | 95% CI | Top-3 | F1-macro | κ | P@5 | p vs. ours |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 6.7% | [0.0, 16.7] | 26.7% | 0.033 | 0.009 | 0.000 | 0.000 |
| Majority class (oracle floor) | 13.3% | [3.3, 26.7] | 33.3% | 0.013 | 0.000 | 0.000 | 0.000 |
| TF-IDF → nearest NACE section | 76.7% | [60.0, 90.0] | 80.0% | 0.657 | 0.749 | 0.140 | 1.000 |
| Zero-shot embeddings (no taxonomy) | 73.3% | [56.7, 86.7] | 86.7% | 0.744 | 0.714 | 0.327 | 0.500 |
| Unguided KeyBERT | 73.3% | [56.7, 86.7] | 86.7% | 0.744 | 0.714 | 0.327 | 0.500 |
| **Ours: taxonomy-guided** | **80.0%** | [63.3, 93.3] | **96.7%** | **0.831** | **0.786** | **0.333** | — |

`p` is an exact McNemar test against the full system on the same documents.

**Two caveats belong before the point estimates.** At n = 30 the intervals are
±15 points wide and none of the differences above is statistically significant;
the margin over TF-IDF is a single document. And the evaluation set is
hand-authored rather than sampled from the corpus — its documents are shorter
and far less legalistic than real register entries (median 116 vs. 175
characters, 3.3% vs. 24.5% carrying legal boilerplate), so this is not a corpus
accuracy. Both have the same fix, and it is in progress:
[`docs/paper_readiness.md`](docs/paper_readiness.md).

### Ablation

| Variant | Top-1 | Δ Top-1 | F1-macro | P@5 | Δ P@5 |
| --- | --- | --- | --- | --- | --- |
| Full system | 80.0% | +0.0 pp | 0.831 | 0.333 | +0.000 |
| − seeds in sector vector | 73.3% | −6.7 pp | 0.744 | 0.340 | +0.007 |
| − description in sector vector | 76.7% | −3.3 pp | 0.770 | 0.327 | −0.007 |
| − seed-guided extraction | 80.0% | +0.0 pp | 0.831 | 0.327 | −0.007 |
| − six-stage keyword filter | 80.0% | +0.0 pp | 0.831 | 0.347 | +0.013 |
| + cleaned text into the classifier | 86.7% | +6.7 pp | 0.833 | 0.333 | +0.000 |
| ↔ mpnet-base-v2 encoder (768-dim) | 73.3% | −6.7 pp | 0.668 | 0.347 | +0.013 |
| ↔ German translated to English first | 80.0% | +0.0 pp | 0.767 | 0.060 | −0.273 |

The last two rows need extra model downloads: `python run.py --extra-ablations`.
P@5 for the translation row is not comparable — the gold keywords are German.

Four things the table settles:

- **The seed-keyword vector is the component that carries the result** (−6.7 pp
  without it).  Everything else in the taxonomy story rests on this row.
- **Guided extraction and the six-stage filter show no measurable effect.**
  Removing the filter even nudges P@5 up.
- **The bigger encoder is worse, not better.**  This repository previously
  expected ~10% improvement from `mpnet-base-v2`; measured, it loses 6.7 points
  Top-1 and 0.163 F1-macro.  It does fix two of the three Q-sector confusions
  the smaller model gets wrong — and introduces four new errors elsewhere.
- **Translating to English first changes nothing at Top-1.**  Whatever the
  multilingual encoder is doing for German, an English pivot reproduces it.

Routing cleaned text into the classifier — which the pipeline computes but does
not use — is worth more than any of the modelling choices above.  It is
available as `classification.classify_preprocessed_text` in
[`config/config.yaml`](config/config.yaml) and **off by default**: on 30
documents the gain is two documents (p = 0.5), which is not enough evidence to
change the shipped behaviour.

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python quickstart.py
```

Three commands, no model downloads to arrange by hand and no optional language
models: the encoder is fetched on first use and everything else is pinned.
`make install` does the same and creates `.venv` if it is missing.

Every `make` target uses `.venv` automatically, so they work without activating
it. Bare `python …` commands need `source .venv/bin/activate` first.

## Reproduce every number above

```bash
make reproduce
```

Equivalent to `python run.py --config config/config.yaml`.  Runs the baseline
comparison, the ablation study and the error analysis with a fixed seed, and
writes:

| File | Contents |
| --- | --- |
| `results/metrics.json` | Headline metrics plus every system, with provenance |
| `results/baselines.json`, `results/ablation.json` | Full per-system results |
| `results/tables/*.md` | The markdown tables above |
| `results/error_analysis.csv` | Misclassified documents, ready for manual coding |

This works from a clean clone. The raw trade register records are **not**
redistributed, but the vocabulary and IDF weights the TF-IDF baseline needs are
committed under `data/derived/` and reproduce the fitted baseline exactly —
identical section rankings on all 30 documents. See
[`data/README.md`](data/README.md).

---

## Pipeline

![Pipeline architecture](docs/assets/architecture.svg)

Dashed stages are the ones the ablation finds no evidence for.

## Usage

```python
from src.controllers.controller import ExtractionController
from src.services.embedder import EmbeddingService
from src.services.classifier import SectorClassifier
from src.services.extractor import KeywordExtractor
from src.services.filter import KeywordFilter
from src.utils.preprocessing import TextPreprocessor

embedder = EmbeddingService()
controller = ExtractionController(
    embedding_service=embedder,
    classifier=SectorClassifier(embedder),
    extractor=KeywordExtractor(),
    keyword_filter=KeywordFilter(),
    preprocessor=TextPreprocessor(),
)

result = controller.extract(
    "Softwareentwicklung und API-Integration für Cloud-Lösungen.",
    top_n_keywords=10,
)
result["sector_classification"]["top_sector"]      # "J"
[kw["keyword"] for kw in result["keywords"]]       # ['softwareentwicklung', ...]
```

Batch processing, iterative seed expansion and taxonomy editing are documented
in [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md).

---

## Data

| Path | In git | What it is |
| --- | --- | --- |
| `data/taxonomy/sectors.json` | yes | 21 NACE sections, 340 hand-written seed keywords |
| `data/evaluation/human_labels.json` | yes | 30 gold documents: section + reference keywords |
| `data/derived/tfidf_corpus_stats.json` | yes | Vocabulary and IDF weights derived from the corpus |
| `data/raw/handelsregister_sample_10k.csv` | no | 9,993 German trade register purposes |

The corpus is not redistributed; the statistics derived from it are, which is
what makes `make reproduce` work from a clone.
[`data/README.md`](data/README.md) has the reasoning and the evidence that the
substitution is exact.

---

## Repository layout

```
src/                    The shipped pipeline
  controllers/          ExtractionController — 5-step orchestration
  services/             embedder · classifier · extractor · filter · validator
  models/               TaxonomyManager · evaluation metrics
  utils/                TextPreprocessor (DE / TR / EN)
experiments/            Paper-only code, kept out of src/
  systems.py            Baselines and ablations as compositions of components
  metrics.py            Bootstrap CIs and exact McNemar on top of src metrics
  error_analysis.py     Error codebook and automatic flags
  run_experiments.py    Baseline + ablation suites
  run_error_analysis.py Error report and annotation CSV
  build_annotation_queue.py  Stratified sampling for new labels
paper/                  Workshop paper skeleton; tables generated from results/
tools/                  annotate.html (offline labelling) · make_demo_gif.py
results/                Generated — every number cited anywhere
tests/                  125 tests
run.py                  make reproduce
```

## Tests

```bash
make test          # or: pytest tests/ -q
make lint
```

106 tests: 72 covering the pipeline, 34 covering the experiment harness.  One of
them asserts that the harness's "full system" predicts the same section as the
shipped `SectorClassifier`, so the ablation table measures the real pipeline
rather than a lookalike.

## Configuration

All tunables live in [`config/config.yaml`](config/config.yaml) and are read
through `src/utils/config.py`: embedding model and chunking, preprocessing
switches, classification threshold and top-k, extraction mode and MMR
diversity, filter cut-off, logging level.

Keys that change nothing have been removed rather than left as decoration, and
`tests/test_config.py::TestEveryConfigKeyIsHonoured` fails if one creeps back
in. Build the pipeline from a config with `pipeline.build_controller()`.

---

## Related literature

| Method | Source | Role |
| --- | --- | --- |
| KeyBERT | Grootendorst (2020) | Pipeline core |
| YAKE! | Campos et al. (2020), *Information Sciences* | Comparison baseline |
| PatternRank | Schopf et al. (2022), ICPRAM | N-gram candidate strategy |
| PromptRank | Kong et al. (2023), ACL | LLM-based comparison |
| Sentence-BERT | Reimers & Gurevych (2019), EMNLP | Embedding foundation |
| Multilingual SBERT | Reimers & Gurevych (2020), EMNLP | Model selection |

Full review and methodology decisions: [`docs/methodology.md`](docs/methodology.md).

---

## Limitations

- **The evaluation set is not drawn from the corpus.**  No document in it
  appears in `data/raw/`; three are edited corpus records and the rest were
  written by hand.  They are shorter (median 116 vs. 175 characters, 90th
  percentile 138 vs. 494) and far cleaner (3.3% vs. 24.5% carrying legal
  boilerplate) than real register entries.  Accuracy measured here should not
  be read as corpus accuracy.
- **It is also too small for the comparisons it is used for.**  Thirty
  documents, 95% CI ≈ ±14 points.  No result here is statistically significant,
  including the margin over TF-IDF.
- **One annotator, no measured agreement.**  Labels are single-pass; the κ
  reported above is classifier-vs-gold, not annotator-vs-annotator.
- **Two pipeline stages are unjustified by evidence.**  Guided extraction and
  the six-stage filter show no measurable effect in the ablation.
- **Q vs. M ambiguity.**  MiniLM-L12 (384-dim) struggles at the
  health/professional-services boundary.  `paraphrase-multilingual-mpnet-base-v2`
  fixes two of those three cases but is worse overall, so the boundary remains
  open — it is a taxonomy problem more than an encoder problem.
- **Very short texts.** Embedding quality drops below ~50 characters.
- **German only in practice.**  The model is multilingual and the preprocessor
  handles DE/TR/EN, but the corpus, the seeds and the evaluation are German.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).

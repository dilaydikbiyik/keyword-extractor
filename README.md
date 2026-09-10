# Zero-Shot NACE Classification of German Trade Register Texts

[![CI](https://github.com/dilaydikbiyik/keyword-extractor/actions/workflows/ci.yml/badge.svg)](https://github.com/dilaydikbiyik/keyword-extractor/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

Assigning **NACE Rev. 2** economic sections to German company purpose
statements with no labelled training data, by embedding the taxonomy's own
class descriptions and ranking them against the document.

**What this repository is really about:** those class descriptions turn out to
decide almost everything. Replacing category labels with definitions that
enumerate concrete activities is worth **+26.8 points** — while translating the
same descriptions into the language of the documents is worth nothing. The
keyword-extraction half the project was originally named after contributes no
measurable benefit at all.

![Reproducing the reported results end to end](docs/assets/demo.gif)

> **On the name.** The directory is still `keyword-extractor`, from when the
> project was framed around guided keyword extraction. Three separate
> measurements have now found that half of the pipeline has no effect, so the
> framing has moved to what the evidence supports. The directory name is
> historical.

---

## Results

299 documents sampled from the corpus, 18 NACE sections. Every figure is
produced by `make reproduce` and written to
[`results/metrics.json`](results/metrics.json).

| System | Top-1 | 95% CI | Top-3 | F1-macro | κ | p vs. ours |
| --- | --- | --- | --- | --- | --- | --- |
| Random | 5.0% | [2.7, 7.7] | 13.7% | 0.036 | −0.001 | <0.001 |
| Majority class (oracle floor) | 21.7% | [17.1, 26.4] | 46.8% | 0.020 | 0.000 | <0.001 |
| TF-IDF → nearest NACE section | 44.5% | [38.8, 50.2] | 65.6% | 0.297 | 0.385 | 0.037 |
| Zero-shot embeddings (no seed keywords) | 52.8% | [47.2, 58.5] | 81.9% | 0.371 | 0.475 | 1.000 |
| Unguided KeyBERT | 52.8% | [47.2, 58.5] | 81.9% | 0.371 | 0.475 | 1.000 |
| **Ours: taxonomy-guided** | **52.5%** | [46.8, 58.2] | **81.6%** | 0.356 | 0.474 | — |

**Top-3 at 81.6% is the operating point.** The correct section is among the
first three suggestions four times out of five, against 46.8% for an oracle
majority-class floor. This is a tool for proposing a code to a human coder, not
for assigning one unattended.

Almost all of that came from rewriting one line per class — see below.

### The finding

The taxonomy's class descriptions were category labels — *"Eğitim"*,
*"Emlak faaliyetleri"* — averaging ten words including the section name.
Rewriting them as NACE-style definitions that enumerate concrete activities is
worth **+26.8 points Top-1** (p = 4×10⁻¹⁵), 28.4% → 52.8%.

| Class descriptions | Top-1 | Top-3 | words/class |
| --- | --- | --- | --- |
| Original (16 of 21 in Turkish, terse) | 28.4% | 62.5% | 10 |
| Same content, rendered in German | 26.1% | 58.9% | 10 |
| **Rewritten as NACE-style definitions** | **52.8%** | **81.9%** | 19 |

**The middle row is the control that matters.** Sixteen of the descriptions
were in Turkish while every document is German, which looks like an obvious
explanation. It is not: translating them, with their content held constant,
changes nothing (−2.3 points, p = 0.14). What the description *says* dominates;
what language it says it in does not.

A crossed design agrees. Documents and descriptions were each varied between
German and machine-translated English: the richer descriptions win in both
document languages, and matching the languages is worth only +2.7 points on
average.

And it replicates on a second encoder — `mpnet-base-v2`, 768 dimensions:
language +0.0 points (p = 1.00), content **+13.7 points** (p = 1×10⁻⁵). Same
direction, same conclusion, different magnitude. `make study` runs all of it.

Validated on a held-out half never inspected during the rewrite: **+15.6 points
there (p = 0.0006)**, 34.0% → 49.7%. Section M, which holds both professional
services and every shell company whose only activity is managing another
company, went from 12% to 64% recall on those documents.

And the honest companion result: once the descriptions carry the information,
**the hand-written seed keyword lists add nothing measurable** (+0.3 points,
p = 1.000). They had been compensating for descriptions that named the class
instead of describing it.

[`docs/paper_readiness.md`](docs/paper_readiness.md) has the full account.

### Ablation

| Variant | Top-1 | Δ Top-1 | F1-macro | p vs. full |
| --- | --- | --- | --- | --- |
| Full system | 52.5% | — | 0.356 | — |
| − seeds in sector vector | 52.8% | +0.3 pp | 0.371 | 1.000 |
| − description in sector vector | 44.1% | −8.4 pp | 0.342 | 0.002 |
| − seed-guided extraction | 52.5% | 0.0 pp | 0.356 | 1.000 |
| − six-stage keyword filter | 52.5% | 0.0 pp | 0.356 | 1.000 |
| ↩ previous taxonomy (Turkish descriptions) | 34.8% | −17.7 pp | 0.293 | <0.001 |
| + cleaned text into the classifier | 41.8% | −10.7 pp | 0.352 | 0.001 |
| ↔ mpnet-base-v2 encoder (768-dim) | 42.1% | −10.4 pp | 0.303 | <0.001 |
| ↔ German translated to English first | 56.2% | +3.7 pp | 0.422 | 0.200 |

The last two rows need extra model downloads: `python run.py --extra-ablations`.
`--half dev` and `--half test` evaluate on either side of the split.

- **The description carries the sector vector; the seed list does not.**
- **Guided extraction and the six-stage filter show no effect**, now across
  three separate configurations of the system.
- **Seed leakage causes 34% of the remaining errors** — *Bodenbelagsarbeiten*
  goes to Mining on **Boden**, *Gebäudereinigung* to Construction on
  **Gebäude**. The ablation says the seed lists cost nothing to remove; the
  error analysis says they cause a third of the failures. Both point at cutting
  them, and [`docs/paper_readiness.md`](docs/paper_readiness.md) explains why
  they have not been.
- **The larger encoder is worse** by 10.4 points, not the ~10% better this
  repository used to expect.

### Evaluation labels

Model-assisted, produced by applying
[`docs/annotation_guidelines.md`](docs/annotation_guidelines.md) and validated
against a human pass: **80% agreement, κ = 0.772**, rising to **100%** on the
documents the labeller flagged as high-confidence. A blind pilot before the
guideline existed scored κ = 0.542; both figures belong in any write-up.

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
| `results/verification_sample.csv` | 50 documents for the human label check |

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
| `data/evaluation/human_labels.json` | yes | 299 corpus-sampled documents with NACE section labels |
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
tests/                  128 tests
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

- **52.5% Top-1 is a suggestion tool, not an automatic classifier.** Top-3 at
  81.6% is the usable figure; top-1 is not accurate enough to assign codes
  unattended.
- **The labels are model-assisted, not gold.** Validated on 50 documents
  (κ = 0.772); 50 of 299 carry human-verified labels. Any published number has
  to quote that agreement figure alongside it.
- **The seed-keyword result is inconclusive, not negative.** Seeds help by 4.6
  points on the development half and hurt by 5.4 on the held-out half, neither
  significant. The defensible claim is that they add nothing reliable, not that
  they are harmful.
- **One annotator on the validation pass.** No inter-annotator agreement has
  been measured; a second annotator on an overlapping subset is what reviewers
  ask for.
- **Sections R, S and T are barely represented** (0–4 documents each) and their
  per-section figures should not be read.
- **Keyword extraction is unevaluated on the current set.** It carries section
  labels only, and the keyword half of the pipeline has not shown a measurable
  effect in any configuration tested.
- **One corpus, one taxonomy.** The description-content finding holds across
  two encoders, two document languages and a held-out half, but it is measured
  on 299 German trade register documents against NACE Rev. 2. It has not been
  shown on a second dataset.
- **No LLM baseline.** An instruction-tuned model asked to pick a section
  directly is the obvious comparison in 2026. The adapter is implemented
  (`run.py --with-llm`) but needs an API key and has not been run.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).

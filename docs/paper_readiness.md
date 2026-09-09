# Paper readiness — Sectoral Keyword Extraction

Status of work package İP-1 after the baseline, ablation and error-analysis
runs.  Every number below comes from `results/`; regenerate with
`make reproduce`.

---

## 1. The contribution sentence

The guide asks for one sentence the whole paper is built around, and warns
that being unable to write it means the paper is not ready.  Here it is, in
the form the current evidence supports:

> We show that averaging a NACE Rev. 2 seed-keyword vector into the sector
> representation improves zero-shot section classification of German trade
> register texts by 6.7 points Top-1 and 10.0 points Top-3 over a
> description-only multilingual embedding baseline.

Note what that sentence is **not** about: keyword extraction.  See §3.

---

## 2. Baseline comparison

`results/tables/baselines.md`, 30 documents, 18 sections.

| System | Top-1 | 95% CI | Top-3 | F1-macro | κ | P@5 | p vs. full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 6.7% | [0.0, 16.7] | 26.7% | 0.033 | 0.009 | 0.000 | 0.000 |
| *Majority class (oracle floor) | 13.3% | [3.3, 26.7] | 33.3% | 0.013 | 0.000 | 0.000 | 0.000 |
| TF-IDF → nearest NACE section | 76.7% | [60.0, 90.0] | 80.0% | 0.657 | 0.749 | 0.140 | 1.000 |
| Zero-shot embeddings (no taxonomy guidance) | 73.3% | [56.7, 86.7] | 86.7% | 0.744 | 0.714 | 0.327 | 0.500 |
| Unguided KeyBERT | 73.3% | [56.7, 86.7] | 86.7% | 0.744 | 0.714 | 0.327 | 0.500 |
| Ours: taxonomy-guided | 80.0% | [63.3, 93.3] | 96.7% | 0.831 | 0.786 | 0.333 | — |

`*` uses the gold label distribution, so it is an oracle floor, not a competitor.
`p` is an exact McNemar test against the full system on the same documents.

### What this says

**The headline number survives contact with baselines — the claim built on it
does not, yet.**  80.0% Top-1 is real, and it beats every baseline.  But:

- **TF-IDF gets 76.7%.**  A 1970s lexical method lands 3.3 points behind a
  multilingual transformer pipeline.  On 30 documents that gap is one
  document, and McNemar puts it at p = 1.000.  A reviewer will find this in
  the first pass.  Right now the paper cannot claim embeddings beat TF-IDF.
- **Top-3 and F1-macro separate the systems more cleanly** than Top-1:
  96.7% vs. 80.0% Top-3, 0.831 vs. 0.657 F1-macro.  F1-macro is the better
  headline for an 18-class problem with this much imbalance, and it is where
  TF-IDF is genuinely weak.
- **Nothing is statistically significant at n = 30.**  Every CI is roughly
  ±15 points wide and overlaps every other CI.

---

## 3. Ablation

`results/tables/ablation.md`.

| Variant | Top-1 | Δ Top-1 | F1-macro | P@5 | Δ P@5 | p vs. full |
| --- | --- | --- | --- | --- | --- | --- |
| Full system | 80.0% | +0.0 pp | 0.831 | 0.333 | +0.000 | — |
| − seeds in sector vector | 73.3% | -6.7 pp | 0.744 | 0.340 | +0.007 | 0.500 |
| − description in sector vector | 76.7% | -3.3 pp | 0.770 | 0.327 | -0.007 | 1.000 |
| − seed-guided extraction | 80.0% | +0.0 pp | 0.831 | 0.327 | -0.007 | 1.000 |
| − six-stage keyword filter | 80.0% | +0.0 pp | 0.831 | 0.347 | +0.013 | 1.000 |
| + cleaned text into the classifier | 86.7% | +6.7 pp | 0.833 | 0.333 | +0.000 | 0.500 |
| ↔ mpnet-base-v2 encoder (768-dim) | 73.3% | −6.7 pp | 0.668 | 0.347 | +0.013 | 0.754 |
| ↔ German translated to English first | 80.0% | +0.0 pp | 0.767 | 0.060 | −0.273 | 1.000 |

The last two rows come from `python run.py --extra-ablations` and need two extra
model downloads.  P@5 on the translation row is not comparable: the gold
keywords are German and the extraction ran on English.

### What this says

Five findings, most of them uncomfortable and therefore worth keeping:

1. **The seed vector is the one component that earns its place.**  Removing
   it costs 6.7 points Top-1 and 0.087 F1-macro — the largest effect in the
   table, and the basis of the contribution sentence.
2. **Guided extraction and the six-stage filter do nothing measurable.**
   Removing either leaves Top-1 identical and moves P@5 by ±0.013.  Removing
   the filter *improves* P@5 slightly.  Two of the pipeline's six stages are
   currently unjustified by evidence.  A paper that describes them as
   contributions without this table is claiming something the data does not
   support.
3. **Feeding cleaned text to the classifier beats the shipped configuration
   by 6.7 points.**  The pipeline preprocessed text and then classified the
   *raw* string, so the cleaning step never reached the classifier.  It is now
   a configuration switch, `classification.classify_preprocessed_text`, **off
   by default** — two documents of evidence (p = 0.5) is not enough to change
   shipped behaviour, and this is one of the first things the enlarged
   evaluation set should settle.
4. **The bigger encoder is worse.**  This repository has carried the
   expectation that `paraphrase-multilingual-mpnet-base-v2` would deliver ~10%
   improvement.  Measured: 73.3% Top-1 (−6.7 pp) and 0.668 F1-macro (−0.163).
   The specific Q/M hypothesis is partly right and globally wrong — mpnet fixes
   samples 11 and 12 (Q misread as M and N) but breaks sample 13, and adds four
   new errors elsewhere (P→M, R→N, S→Q, F→M).  Reported as a negative result,
   this is a legitimate paper sentence and it closes off a planned direction.
5. **A German→English pivot changes nothing at Top-1.**  Translate every
   purpose with Marian de→en, keep the German-seeded taxonomy, and Top-1 is
   identical at 80.0% (F1 0.767).  Whatever cross-lingual work the multilingual
   encoder is doing here, an English pivot reproduces it — which weakens any
   claim that the method is specifically exploiting German representations.

---

## 4. Error analysis

`results/error_analysis.json`, `results/error_analysis.csv`.

Six errors out of 30.  Automatic flags: 5 recoverable in Top-3, 4 low-margin
(< 0.02 between first and second), 2 with a crowded Top-3, 1 outside Top-3
entirely.

Confusions are all adjacent-service-sector pairs — M↔K, M↔N, Q↔M, Q↔N, N↔S,
F↔D — i.e. the model is not confusing dentists with farms; it is failing on
the boundaries where NACE itself is ambiguous.  That is a publishable
observation, and `docs/methodology.md` already anticipates the Q/M case.

**Six errors is not an error analysis.**  The guide asks for 50 hand-inspected
errors, and reviewers weight that section heavily.  At the current error rate
that needs roughly 250 labelled documents.

---

## 5. What blocks submission

In priority order.

### Blocker 1 — the evaluation set is too small to support any comparison

Thirty documents give a 95% CI of about ±14 points at 80% accuracy.  Every
comparison in §2 is inside the noise.  No amount of writing fixes this.

| Labelled documents | 95% CI half-width at 80% |
| --- | --- |
| 30 | ±14.3 pp |
| 100 | ±7.8 pp |
| 200 | ±5.5 pp |
| 300 | ±4.5 pp |
| 500 | ±3.5 pp |

`results/annotation_queue.csv` holds 296 documents already selected for this,
stratified by predicted sector and split between low-margin and
representative cases.  Filling in `true_sector` is the highest-value work
available on this project — it converts a suggestive result into a claim, and
it produces enough errors for §4 as a side effect.

Estimated effort: at 30 seconds per document, roughly 2.5 hours.

### Blocker 2 — single annotator, no agreement measured

`human_labels.json` records `annotation_method: "semi_manual"` by one person.
Reviewers ask for inter-annotator agreement on any new evaluation set.  Have a
second annotator label an overlapping subset (100 documents is enough) and
report Cohen's κ between annotators, separately from the classifier κ already
in the tables.

### Blocker 3 — parts of the evaluation set are synthetic

Some entries (`TechSoft GmbH`, `WebPro UG`) are clean, invented descriptions,
not trade register text.  They are easier than real Handelsregister purposes,
which are long, legalistic and full of boilerplate.  Their presence inflates
accuracy.  Replace them with real records from the corpus when rebuilding the
evaluation set.

### Blocker 4 — corpus redistribution is unresolved

See [`data/README.md`](../data/README.md).  Needed for the artefact and for
the paper's data statement.

---

## 6. Suggested order of work

1. Label the annotation queue (Blocker 1).  Everything downstream depends on it.
2. Re-run `make reproduce`.  The comparison either becomes significant or it
   does not, and either answer is worth having before writing a word.
3. Hand-code the 50-error analysis in `results/error_analysis.csv` against the
   codebook in `experiments/error_analysis.py`.
4. Decide `classify_preprocessed_text` (§3.3) on the enlarged set and report
   it as a result, not as a silent bugfix.
5. Decide what to do about guided extraction and the filter (§3.2): either
   find a metric where they help, or describe them as engineering, not
   contributions.
6. Then write.  Method and experiments sections are largely determined by the
   tables above.

## 7. Still to do for the artefact standard (İP-4)

Done: one-command reproduction, `results/metrics.json`, pinned dependencies,
fixed seed, tests + CI, `LICENSE`, `CITATION.cff`, results-first README,
limitations section, data provenance, demo GIF
(`python tools/make_demo_gif.py`), rendered architecture diagram
(`docs/assets/architecture.svg`).

Nothing outstanding on İP-4 except what depends on Blocker 4: the repository
cannot honestly claim one-command reproduction from a fresh clone until the
corpus question is settled.

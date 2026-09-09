# Paper readiness — Sectoral Keyword Extraction

Status of the publication track after the baseline, ablation and error-analysis
runs. Every number comes from `results/`; regenerate with `make reproduce`.

**The headline finding is not the 80%.** It is that the evaluation set the 80%
is measured on does not resemble the corpus the system is for. Section 1 covers
that first, because it changes what every other number means.

---

## 1. The evaluation set does not represent the corpus

Not one of the 30 evaluation documents appears in
`data/raw/handelsregister_sample_10k.csv`. Three are shortened, edited versions
of corpus records; the rest were written by hand. The measured difference:

| | Evaluation set | Corpus |
| --- | --- | --- |
| Documents | 30 | 9,993 |
| Median length | 116 characters | 175 characters |
| 90th percentile length | 138 characters | 494 characters |
| Carrying legal boilerplate or `§` | 3.3% | 24.5% |

The evaluation text is roughly two-thirds the median length, has essentially
none of the long tail, and almost none of the legalistic register language that
a quarter of the real corpus carries. Real entries read like

> *Die für Wirtschaftsprüfungsgesellschaften gesetzlich und berufsrechtlich
> zulässigen Tätigkeiten gemäß § 2 WPO in Verbindung mit § 43 a Abs. 4 WPO.
> Handels- und Bankgeschäfte sind ausgeschlossen.*

and the evaluation set's version of that same company drops the second half and
appends a clean `Steuerberatung und Buchführung.`

**Consequence.** 80.0% is a real measurement of a real system, but it is
measured on text that is systematically easier than the input the system exists
to handle. It is not a corpus accuracy, and a reviewer who samples the released
evaluation file will see this immediately.

**This is not fatal and it is not slow to fix.**
`results/annotation_queue.csv` is drawn from the corpus itself, so labelling it
solves representativeness and sample size in the same pass. Section 5 has the
procedure.

---

## 2. The contribution sentence

One sentence the paper is built around. Being unable to write it means the
paper is not ready; here it is in the form the current evidence supports —
subject to §1, which the corpus-sampled evaluation set has to confirm.

> We show that averaging a NACE Rev. 2 seed-keyword vector into the sector
> representation improves zero-shot section classification of German trade
> register texts by 6.7 points Top-1 and 10.0 points Top-3 over a
> description-only multilingual embedding baseline.

Note what it is **not** about: keyword extraction. The ablation gives that half
of the pipeline no measurable credit.

---

## 3. Against baselines

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

- **TF-IDF gets 76.7%.** A 1970s lexical method lands 3.3 points behind a
  multilingual transformer pipeline. On 30 documents that gap is one document,
  and McNemar puts it at p = 1.000. The paper cannot currently claim that
  embeddings beat TF-IDF.
- **Top-3 and F1-macro separate the systems more cleanly:** 96.7% vs. 80.0%
  Top-3, 0.831 vs. 0.657 F1-macro. F1-macro is the better headline for an
  18-class problem this imbalanced, and it is where TF-IDF is genuinely weak.
- **Nothing is statistically significant at n = 30.** Every interval is about
  ±15 points wide and overlaps every other interval.

---

## 4. What each component contributes

`results/tables/ablation.md`.

| Variant | Top-1 | Δ Top-1 | F1-macro | P@5 | Δ P@5 | p vs. full |
| --- | --- | --- | --- | --- | --- | --- |
| Full system | 80.0% | +0.0 pp | 0.831 | 0.333 | +0.000 | — |
| − seeds in sector vector | 73.3% | −6.7 pp | 0.744 | 0.340 | +0.007 | 0.500 |
| − description in sector vector | 76.7% | −3.3 pp | 0.770 | 0.327 | −0.007 | 1.000 |
| − seed-guided extraction | 80.0% | +0.0 pp | 0.831 | 0.327 | −0.007 | 1.000 |
| − six-stage keyword filter | 80.0% | +0.0 pp | 0.831 | 0.347 | +0.013 | 1.000 |
| + cleaned text into the classifier | 86.7% | +6.7 pp | 0.833 | 0.333 | +0.000 | 0.500 |
| ↔ mpnet-base-v2 encoder (768-dim) | 73.3% | −6.7 pp | 0.668 | 0.347 | +0.013 | 0.754 |
| ↔ German translated to English first | 80.0% | +0.0 pp | 0.767 | 0.060 | −0.273 | 1.000 |

The last two rows come from `python run.py --extra-ablations` and need two extra
model downloads. P@5 on the translation row is not comparable: the gold keywords
are German and extraction ran on English.

### What this says

1. **The seed vector earns its place.** Removing it costs 6.7 points Top-1 and
   0.087 F1-macro — the largest effect in the table and the whole basis of the
   contribution sentence.
2. **Guided extraction and the six-stage filter do nothing measurable.**
   Removing either leaves Top-1 identical; P@5 moves ±0.013, and removing the
   filter nudges it *up*. Two of the six stages currently have no evidence
   behind them. Describe them as engineering, not as contributions.
3. **Cleaned text into the classifier is worth more than any modelling choice.**
   The pipeline preprocessed the text and then classified the raw string, so
   cleaning never reached the decision. It is now
   `classification.classify_preprocessed_text`, **off by default** — two
   documents of evidence (p = 0.5) is not enough to change shipped behaviour,
   and it is among the first things the new evaluation set should settle.
4. **The bigger encoder is worse.** This repository carried the expectation that
   `mpnet-base-v2` would give ~10% improvement. Measured: −6.7 points Top-1,
   −0.163 F1-macro. The specific Q/M hypothesis is partly right and globally
   wrong — mpnet fixes samples 11 and 12 but breaks 13 and adds four errors
   elsewhere (P→M, R→N, S→Q, F→M). A clean negative result that closes off a
   planned direction.
5. **A German→English pivot changes nothing at Top-1.** Translate every purpose
   with Marian de→en, keep the German-seeded taxonomy, and Top-1 is identical at
   80.0%. Whatever cross-lingual work the encoder is doing, an English pivot
   reproduces it — which weakens any claim that the method exploits German
   representations specifically.

---

## 5. Error analysis

Six errors in 30 documents: five recoverable in Top-3, four on a decision margin
under 0.02, one outside Top-3 entirely. `results/error_analysis.csv` is the
annotation sheet; the codebook is in `experiments/error_analysis.py`.

### First pass over the six — verify before citing

This is a reading of the six texts, not a completed manual pass. Each proposed
category needs confirming or overriding in the CSV.

| # | Gold → predicted | Margin | Proposed category | Reading |
| --- | --- | --- | --- | --- |
| 3 | M → K | 0.022 | `taxonomy_granularity` | *Wirtschaftsprüfung* and *Steuerberatung* are M69, but the vocabulary is financial. The prediction is defensible; NACE puts accountancy under professional services. |
| 5 | M → N | 0.0004 | `ambiguous_sector_definition` | Legal services are M69.1; the model is a rounding error away from correct. A near-tie, not a misunderstanding. |
| 10 | F → D | 0.129 | `seed_leakage` | *Elektroinstallationen* is building installation (F43), but section D's seeds own the word *Elektro*. The only confident error in the set, and the only one a seed-list edit would fix. |
| 11 | Q → M | 0.014 | `ambiguous_sector_definition` | Dental practice is Q86.23. "Klinik" and "Behandlungsmethoden" pull toward professional services. The known Q/M boundary. |
| 12 | Q → N | 0.0002 | `ambiguous_sector_definition` | Residential care is Q87. Effectively tied with N; "Betrieb einer Einrichtung" reads as facility management. |
| 27 | N → S | 0.015 | `taxonomy_granularity` | Security services are N80. Section S is the catch-all, and the model reaches for it. |

Two things follow, both worth a sentence in the paper. First, every confusion is
an adjacent service-sector pair (M/K, M/N, Q/M, Q/N, N/S, F/D) — the model is not
mistaking dentists for farms, it fails exactly where NACE itself is ambiguous.
Second, only one of the six (#10) is a confident error; the rest sit within
0.022 of correct, which suggests calibration and seed-list work rather than a
different model.

**Six errors is not an error analysis.** The target is 50 hand-inspected cases,
which at this error rate needs roughly 250 labelled documents.

---

## 6. What blocks submission

In priority order.

### Blocker 1 — the evaluation set is neither representative nor large enough

Both halves have the same fix, and §1 covers the representativeness half.
On size: 30 documents give a 95% interval of about ±14 points at 80% accuracy,
so every comparison in §3 sits inside the noise.

| Labelled documents | 95% CI half-width at 80% |
| --- | --- |
| 30 | ±14.3 pp |
| 100 | ±7.8 pp |
| 200 | ±5.5 pp |
| 300 | ±4.5 pp |
| 500 | ±3.5 pp |

**The procedure, end to end:**

1. **`make annotate`** — builds `results/annotation_queue.csv`, 299 documents
   drawn from the corpus, stratified by predicted section and split between
   low-margin and representative cases. No gold label is used in the selection.
   It then opens the labelling tool. Re-running it will not overwrite a queue
   that already has answers in it.
2. **Load that CSV in the tool.** One document at a time, the model's top three
   on keys `1` `2` `3`, everything else searchable, progress kept in the browser
   as you go. Roughly 30 seconds per document — about two and a half hours in
   total. Export when done.
3. **`make merge ARGS=--replace`.** The flag matters: the original 30 are
   hand-authored, so appending them to a corpus-sampled set reintroduces exactly
   the bias §1 describes. The merge backs up the old label file first.
4. **`make reproduce`.** Either the comparison becomes significant or it does
   not, and either answer is worth having before writing a word.

Every `make` target finds `.venv` on its own, so none of this needs the
environment activated first. Bare `python -m experiments...` commands do —
`source .venv/bin/activate`.

This is the highest-value work available on the project by a wide margin.

### Blocker 2 — one annotator, no agreement measured

Labels are single-pass and single-annotator. Reviewers ask for agreement figures
on any new evaluation set. Two options, in order of what they buy:

- **Inter-annotator (what reviewers want).** A second person labels an
  overlapping 100 documents; report Cohen's κ between annotators.
- **Intra-annotator (available alone).** Re-label a blind 100-document subset
  after at least a week and report agreement with your own earlier pass. Weaker,
  but honest, cheap, and far better than reporting nothing.

Either way, report it separately from the classifier κ already in the tables —
that one measures classifier against gold, not annotator against annotator.

### Blocker 3 — corpus licensing, for the paper rather than the artefact

Resolved for reproduction: the vocabulary and IDF weights the TF-IDF baseline
needs are committed under `data/derived/`, they reproduce the fitted baseline
exactly, and `make reproduce` therefore works from a clean clone without the raw
records. See [`../data/README.md`](../data/README.md).

What remains is the paper's data statement: say plainly how the sample was
collected, what is released, and what is not. Venues expect that section; they
do not expect every corpus to be redistributable.

---

## 7. Order of work

1. **Label the annotation queue** (Blocker 1). Everything downstream depends on it.
2. **`make reproduce`** on the new set. Expect the numbers to move — the new
   documents are longer and harder than the ones the 80% came from.
3. **Hand-code the 50-error analysis** in `results/error_analysis.csv`, starting
   from the first pass in §5.
4. **Settle `classify_preprocessed_text`** on the enlarged set and report it as
   a result, not a silent bugfix.
5. **Decide what guided extraction and the filter are.** Either find a metric
   where they help, or describe them as engineering.
6. **Fix the section-D seed list** so it stops claiming *Elektroinstallation*
   (error #10), and re-run — a taxonomy fix is a legitimate contribution in a
   taxonomy-guided method.
7. **Then write.** `paper/main.tex` is the skeleton; its tables regenerate from
   `results/` with `make paper-tables`, so no number is ever typed into the prose.

## 8. Decisions to make

Nobody else has to be consulted for these; they need choosing, not permission.

- **Authorship.** The work is yours. Decide whether anyone else has contributed
  enough to be named, and settle it before submission rather than after.
- **Venue.** [`../paper/venues.md`](../paper/venues.md) has the priority order
  and the deadlines to verify. The Student Research Workshop assigns a mentor,
  which is worth more on a first submission than a marginally stronger venue.
- **Scope.** Section-level NACE only, or a division-level evaluation as well?
  Section-level is defensible and already built.
- **Framing of the negative results.** A workshop paper that reports a falsified
  encoder hypothesis and two unsupported pipeline stages reads as careful, not
  weak — provided the contribution sentence stands on the component that does
  hold up.

## 9. Artefact standard

Done: one-command reproduction from a clean clone, `results/metrics.json`,
pinned dependencies, fixed seed, tests and CI, MIT licence, `CITATION.cff`,
results-first README, honest limitations, data provenance, demo GIF, rendered
architecture diagram, a configuration file every key of which the code actually
reads, and an annotation tool.

Nothing outstanding.

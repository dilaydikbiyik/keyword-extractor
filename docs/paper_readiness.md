# Paper readiness — Sectoral Keyword Extraction

Status of the publication track. Every number comes from `results/`;
regenerate with `make reproduce`.

**What changed.** The evaluation set used to be 30 hand-written documents, on
which the pipeline scored 80.0% and no comparison was significant. It is now
299 documents sampled from the corpus itself. On those the pipeline scores
**36.1%** — and for the first time the comparisons *are* significant. The
headline number got much worse and the paper got much stronger.

**Standing caveat.** The 299 labels are *silver*: produced by a language model
applying [`annotation_guidelines.md`](annotation_guidelines.md), not by a human
expert. Section 6 is the procedure that turns them into something citable, and
nothing here should be published before it is done.

---

## 1. Against baselines

`results/tables/baselines.md` · 299 corpus-sampled documents, 17 NACE sections.

| System | Top-1 | 95% CI | Top-3 | F1-macro | κ | p vs. ours |
| --- | --- | --- | --- | --- | --- | --- |
| Random | 6.0% | [3.3, 8.7] | 13.7% | 0.048 | 0.010 | <0.001 |
| *Majority class (oracle floor) | 22.4% | [17.7, 27.4] | 47.5% | 0.022 | 0.000 | 0.001 |
| TF-IDF → nearest NACE section | 26.1% | [21.4, 31.1] | 40.5% | 0.203 | 0.214 | 0.004 |
| Zero-shot embeddings (no taxonomy) | 29.8% | [24.7, 35.1] | 61.9% | 0.246 | 0.248 | 0.008 |
| Unguided KeyBERT | 29.8% | [24.7, 35.1] | 61.9% | 0.246 | 0.248 | 0.008 |
| **Ours: taxonomy-guided** | **36.1%** | [30.8, 41.8] | **67.9%** | **0.290** | **0.313** | — |

`*` uses the gold label distribution and is an oracle floor, not a competitor.
`p` is an exact McNemar test on the same documents.

### What this says

- **The contribution is now demonstrated, not suggested.** Taxonomy guidance
  beats the description-only embedding baseline by 6.4 points at p = 0.008, and
  TF-IDF by 10.0 points at p = 0.004. On the old 30-document set the same two
  comparisons sat at p = 0.500 and p = 1.000. Nothing about the method changed;
  the evaluation set did.
- **The real difficulty is now visible.** 36.1% over 17 imbalanced sections, on
  register text that is long, legalistic, and often describes a holding company
  rather than a business. The old 80% measured short, clean, hand-written
  descriptions.
- **Top-3 is 67.9%.** The correct section is in the top three two times out of
  three. For a system whose realistic use is suggesting a code to a human, that
  is the number worth arguing for.
- **The majority-class floor is 22.4%**, because the sample is dominated by
  professional services. Any headline that does not clear that floor by a wide
  margin is noise; 36.1% does, at p = 0.001.

---

## 2. The contribution sentence

> We show that averaging a NACE Rev. 2 seed-keyword vector into the sector
> representation improves zero-shot section classification of German trade
> register texts by 6.4 points Top-1 (p = 0.008) over a description-only
> multilingual embedding baseline, and by 10.0 points over TF-IDF.

Note what it is **not** about: keyword extraction. The ablation still gives that
half of the pipeline no measurable credit.

---

## 3. What each component contributes

`results/tables/ablation.md`.

| Variant | Top-1 | Δ Top-1 | F1-macro | p vs. full |
| --- | --- | --- | --- | --- |
| Full system | 36.1% | — | 0.290 | — |
| − seeds in sector vector | 29.8% | −6.4 pp | 0.246 | 0.008 |
| − description in sector vector | 33.8% | −2.3 pp | 0.290 | 0.450 |
| − seed-guided extraction | 36.1% | 0.0 pp | 0.290 | 1.000 |
| − six-stage keyword filter | 36.1% | 0.0 pp | 0.290 | 1.000 |
| + cleaned text into the classifier | 32.1% | −4.0 pp | 0.277 | 0.155 |
| ↔ mpnet-base-v2 encoder (768-dim) | 34.1% | −2.0 pp | 0.258 | 0.572 |
| ↔ German translated to English first | 41.8% | **+5.7 pp** | 0.315 | 0.050 |

Keyword Precision@K is not reported: the new evaluation set carries section
labels only. See §6, step 4.

### Five findings

1. **The seed vector is the contribution, and it now has a p-value.** −6.4
   points without it, p = 0.008. This is the paper.
2. **Guided extraction and the six-stage filter still do nothing.** Identical
   Top-1 with either removed, on ten times the data. Two of the pipeline's six
   stages have now failed to show an effect on two different evaluation sets.
   Describe them as engineering, not as contributions — or cut them.
3. **Translating to English first is the best variant: +5.7 points, p = 0.050.**
   This inverts the earlier reading. On the hand-written set the pivot changed
   nothing; on real register text it helps, presumably because the encoder's
   English representations are stronger than its German ones and legalistic
   German is where that gap shows. A genuine, publishable result that deserves
   its own paragraph.
4. **Feeding cleaned text to the classifier now *hurts*: −4.0 points.** On the
   old set it appeared to gain 6.7. Same code, opposite conclusion — a clean
   demonstration of what a 30-document evaluation set is worth. It stays off by
   default, and the reversal is worth a sentence as a caution about
   small-sample ablations.
5. **The bigger encoder is still worse** (−2.0 points, p = 0.572), though the
   gap narrowed. The original expectation of a ~10% gain from mpnet-base-v2
   remains falsified.

---

## 4. Where it fails

191 errors in 299 documents. Automatic flags: 117 on a decision margin under
0.02, 96 outside Top-3, 95 recoverable in Top-3, 61 with a crowded Top-3, 51
matching the holding-company boilerplate shape, 16 too short.

Top confusions, from `results/tables/confusions.md`:

| Gold → predicted | n |
| --- | --- |
| M → G | 11 |
| M → N | 10 |
| M → K | 9 |
| M → C | 8 |
| G → H | 6 |
| G → C | 5 |
| F → L | 5 |

**Section M is where the system breaks.** Per-section recall against the silver
labels: M 18%, N 18%, R and S 0%, against L 63% and H 67%. M is also the
largest class (67 of 299), because it absorbs both professional services and
every Komplementär-GmbH whose only activity is managing another company.

Two concrete, fixable causes:

- **The taxonomy has no vocabulary for shell companies.** 40 of 299 documents
  are holding or management shells; the pipeline gets 10% of them right and
  scatters them across G, K, N and C. Section M's seed list contains no
  *Beteiligung*, no *Komplementär*, no *Verwaltung eigenen Vermögens*.
- **Seed leakage across sections.** Section D's seeds own the word *Elektro*,
  so electrical installation firms (F) get pulled into energy supply.

**Excluding every shell company, agreement is still only 40.2%** — the shells
explain part of the picture, not all of it.

There are now enough errors for the 50-case manual analysis the write-up needs.
`results/error_analysis.csv` is the sheet; the codebook is in
`experiments/error_analysis.py`.

---

## 5. What the evaluation set is now

| | Old set | Current set |
| --- | --- | --- |
| Documents | 30 | 299 |
| Source | hand-written and edited | sampled from the corpus |
| Sections covered | 18 | 17 |
| 95% CI half-width | ±14 points | ±5.5 points |
| Labels | one human, semi-manual | model-assisted, guideline-driven |

The old set is preserved as `data/evaluation/human_labels.json.bak`.

Documents were selected by predicted section with a low-margin / representative
split, so the set is neither a random dump nor a pile of mined hard cases. No
gold label was used in the selection.

---

## 6. What still has to happen

### Step 1 — verify the labels (blocking)

The labels are model-produced. Until a human has checked a sample, no number in
this document can be published.

```bash
open tools/annotate.html      # load results/verification_sample.csv
make verify                   # score it once filled in
```

The sample holds 50 documents — 25 the labeller marked low-confidence and 25
drawn at random — with English translations, and the silver label hidden while
you decide. `make verify` reports raw agreement and Cohen's κ into
`results/verification_report.json`.

Roughly 25 minutes. What the result means:

- **κ ≥ 0.8** — report the labels as model-assisted with human validation, and
  move on.
- **κ 0.6–0.8** — usable, but the disagreements will show a systematic pattern.
  Fix the guideline, re-label the affected cases, re-verify.
- **κ < 0.6** — the guideline is wrong somewhere. Read the disagreements before
  changing anything else.

### Step 2 — say so in the paper

The data section has to state the procedure: *"Section labels were produced by
a language model applying a written annotation guideline (Appendix X), and
validated against a 50-document subset annotated by the author (κ = …)."* That
is an accepted method when declared. Presenting silver labels as gold is not.

### Step 3 — hand-code 50 errors

From `results/error_analysis.csv`, against the codebook. Start with the M
confusions: a third of all errors, with a named cause.

### Step 4 — decide about keyword evaluation

The new set has no keyword ground truth, so Precision@K is unmeasurable. Two
options:

- **Add keywords to ~60 documents** and keep P@K as a secondary metric.
- **Drop the keyword claim** and write a section-classification paper. The
  ablation has now twice found no measurable contribution from the keyword half
  of the pipeline, so this is the honest option and it makes a tighter paper.

Recommendation: the second. The contribution sentence never mentioned keywords.

### Step 5 — fix what the errors point at

- Add shell-company vocabulary to section M's seeds (*Beteiligung*,
  *Komplementär*, *Verwaltung eigenen Vermögens*, *Übernahme der
  Geschäftsführung*).
- Remove *Elektro*-prefixed installation terms from section D's seeds.
- Re-run. A taxonomy fix is a legitimate contribution in a taxonomy-guided
  method, and this one is motivated by error analysis rather than by fitting
  the test set.

### Step 6 — then write

`paper/main.tex` is the skeleton; `make paper-tables` regenerates its tables
from `results/`, so no number is typed into the prose.

---

## 7. Decisions to make

- **Authorship.** The work is yours. Settle it before submission.
- **Venue.** [`../paper/venues.md`](../paper/venues.md) has the priority order.
  The Student Research Workshop assigns a mentor, worth more on a first
  submission than a marginally stronger venue.
- **Framing.** With significant results, two falsified hypotheses and a
  documented small-sample reversal, this reads as a careful evaluation paper.
  That is a better story than a 90% claim nobody can reproduce.
- **The English-pivot result** needs a decision: one ablation row, or the
  paper's second contribution.

## 8. Artefact standard

Complete. One-command reproduction from a clean clone, `results/metrics.json`,
pinned dependencies, fixed seed, tests and CI, MIT licence, `CITATION.cff`,
results-first README, honest limitations, data provenance, demo GIF, rendered
architecture diagram, a configuration file every key of which the code reads,
an offline annotation tool with translations, written annotation guidelines,
and a verification procedure.

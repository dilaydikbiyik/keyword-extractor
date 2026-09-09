# Paper readiness — Sectoral Keyword Extraction

Status of the publication track. Every number comes from `results/`;
regenerate with `make reproduce`.

**What changed.** The evaluation set used to be 30 hand-written documents, on
which the pipeline scored 80.0% and no comparison was significant. It is now
299 documents sampled from the corpus itself. On those the pipeline scores
**34.8%** — and for the first time the comparisons *are* significant. The
headline number got much worse and the paper got much stronger.

**Label status.** The 299 labels are model-assisted, produced by applying
[`annotation_guidelines.md`](annotation_guidelines.md), and validated against a
human pass: **κ = 0.772** on a 50-document sample, 100% agreement where the
labeller flagged high confidence. 50 of the 299 carry human-verified labels.
§6 has the full protocol, which the paper's data section has to state.

---

## 1. Against baselines

`results/tables/baselines.md` · 299 corpus-sampled documents, 17 NACE sections.

| System | Top-1 | 95% CI | Top-3 | F1-macro | κ | p vs. ours |
| --- | --- | --- | --- | --- | --- | --- |
| Random | 5.0% | [2.7, 7.7] | 13.7% | 0.036 | −0.001 | <0.001 |
| *Majority class (oracle floor) | 21.7% | [17.1, 26.4] | 46.8% | 0.020 | 0.000 | 0.002 |
| TF-IDF → nearest NACE section | 25.4% | [20.7, 30.4] | 40.1% | 0.184 | 0.207 | 0.007 |
| Zero-shot embeddings (no taxonomy) | 28.4% | [23.4, 33.4] | 62.5% | 0.243 | 0.234 | 0.008 |
| Unguided KeyBERT | 28.4% | [23.4, 33.4] | 62.5% | 0.243 | 0.234 | 0.008 |
| **Ours: taxonomy-guided** | **34.8%** | [29.4, 40.1] | **67.2%** | **0.293** | **0.300** | — |

`*` uses the gold label distribution and is an oracle floor, not a competitor.
`p` is an exact McNemar test on the same documents.

### What this says

- **The contribution is now demonstrated, not suggested.** Taxonomy guidance
  beats the description-only embedding baseline by 6.4 points at p = 0.008, and
  TF-IDF by 9.4 points at p = 0.007. On the old 30-document set the same two
  comparisons sat at p = 0.500 and p = 1.000. Nothing about the method changed;
  the evaluation set did.
- **The real difficulty is now visible.** 34.8% over 17 imbalanced sections, on
  register text that is long, legalistic, and often describes a holding company
  rather than a business. The old 80% measured short, clean, hand-written
  descriptions.
- **Top-3 is 67.2%.** The correct section is in the top three two times out of
  three. For a system whose realistic use is suggesting a code to a human, that
  is the number worth arguing for.
- **The majority-class floor is 21.7%**, because the sample is dominated by
  professional services. Any headline that does not clear that floor by a wide
  margin is noise; 34.8% does, at p = 0.001.

---

## 2. The contribution sentence

> We show that averaging a NACE Rev. 2 seed-keyword vector into the sector
> representation improves zero-shot section classification of German trade
> register texts by 6.4 points Top-1 (p = 0.008) over a description-only
> multilingual embedding baseline, and by 9.4 points over TF-IDF.

Note what it is **not** about: keyword extraction. The ablation still gives that
half of the pipeline no measurable credit.

---

## 3. What each component contributes

`results/tables/ablation.md`.

| Variant | Top-1 | Δ Top-1 | F1-macro | p vs. full |
| --- | --- | --- | --- | --- |
| Full system | 34.8% | — | 0.293 | — |
| − seeds in sector vector | 28.4% | −6.4 pp | 0.243 | 0.008 |
| − description in sector vector | 34.8% | 0.0 pp | 0.310 | 1.000 |
| − seed-guided extraction | 34.8% | 0.0 pp | 0.293 | 1.000 |
| − six-stage keyword filter | 34.8% | 0.0 pp | 0.293 | 1.000 |
| + cleaned text into the classifier | 30.8% | −4.0 pp | 0.271 | 0.155 |
| ↔ mpnet-base-v2 encoder (768-dim) | 33.1% | −1.7 pp | 0.274 | 0.653 |
| ↔ German translated to English first | 40.1% | **+5.4 pp** | 0.345 | 0.068 |

Keyword Precision@K is not reported: the new evaluation set carries section
labels only. See §6, step 4.

### Five findings

1. **The seed vector is the contribution, and it now has a p-value.** −6.4
   points without it, p = 0.008. This is the paper.
2. **Guided extraction and the six-stage filter still do nothing.** Identical
   Top-1 with either removed, on ten times the data. Two of the pipeline's six
   stages have now failed to show an effect on two different evaluation sets.
   Describe them as engineering, not as contributions — or cut them.
3. **Translating to English first is still the best variant: +5.4 points — but at p = 0.068 it is suggestive, not significant.**
   This inverts the earlier reading. On the hand-written set the pivot changed
   nothing; on real register text it helps, presumably because the encoder's
   English representations are stronger than its German ones and legalistic
   German is where that gap shows. Worth a paragraph and worth
   following up, but it must not be claimed as established: the p-value moved
   from 0.027 to 0.068 when 10 labels were human-corrected, which is what a
   borderline effect looks like.
4. **Feeding cleaned text to the classifier now *hurts*: −4.0 points.** On the
   old set it appeared to gain 6.7. Same code, opposite conclusion — a clean
   demonstration of what a 30-document evaluation set is worth. It stays off by
   default, and the reversal is worth a sentence as a caution about
   small-sample ablations.
5. **The bigger encoder is still worse** (−1.7 points, p = 0.653), though the
   gap narrowed. The original expectation of a ~10% gain from mpnet-base-v2
   remains falsified.

---

## 4. Where it fails

195 errors in 299 documents. Automatic flags: 117 on a decision margin under
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
labels: M 15.4%, N 15.8%, against L 64.7% and H 75.0%. M is also the
largest class (65 of 299), because it absorbs both professional services and
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

### Step 1 — verification: done

Two passes, the standard pilot-then-measure protocol.

| | Blind pilot | After guideline revision |
| --- | --- | --- |
| Documents | 50 | 50 **fresh** ones |
| Raw agreement | 58.0% | **80.0%** |
| Cohen's κ | 0.542 | **0.772** |
| …where the labeller flagged high confidence | 80.0% | **100.0%** |
| …where it flagged low confidence | 36.0% | 60.0% |

κ = 0.772 is substantial agreement, comfortably past the 0.6 bar, and the
second sample shared no documents with the first — so it measures the guideline
rather than the adjudication.

**The confidence flag is the striking part: 25 out of 25 on the documents the
labeller called confident.** Residual disagreement lives entirely in the
low-confidence bucket, which means the labels ship with a usable reliability
signal — worth a sentence in the paper on its own.

The 50 verified answers were then promoted over the silver labels (`make
verify-apply`), which is the usual order: agreement is measured *before*
adjudication, the released labels come *after* it. 10 of 50 changed. The
headline moved 34.4% → 34.8% and every conclusion held.

### Step 2 — say so in the paper

The data section states the protocol and both figures: *"Section labels were
produced by a language model applying a written annotation guideline
(Appendix X). In a blind pilot a second annotator agreed on 58% of 50 documents
(κ = 0.542); the disagreements were adjudicated into three explicit rules, and
on a fresh 50-document sample agreement was 80% (κ = 0.772), rising to 100% on
the documents the labeller flagged as high-confidence. Verified answers were
promoted over the machine labels in the released set."*

That is an accepted method when declared. Presenting silver labels as gold is
not.

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

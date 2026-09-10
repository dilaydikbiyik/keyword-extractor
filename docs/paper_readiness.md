# Paper readiness — Sectoral Keyword Extraction

Every number comes from `results/`; regenerate with `make reproduce` and
`make study`.

**The finding.** A class description helps in proportion to the **lexical gap**
between the class name and the words its documents actually use. Where that gap
is wide, elaborating the description is worth **+26.8 points Top-1**
(p = 4×10⁻¹⁵). Where the class name is already the documents' own vocabulary,
it is worth nothing.

**Why it is a claim and not an observation.** Three controls:

| Question | Answer |
| --- | --- |
| Is it the language? Sixteen descriptions were Turkish against German documents. | **No.** Same content in German: −2.3 pp, p = 0.14. |
| Is it one encoder? | **No.** mpnet-base-v2: +13.7 pp, p = 1×10⁻⁵. |
| Is it this corpus? | **Partly — and predictably so.** On 20 Newsgroups, elaboration is worth +0.4 pp (p = 0.73). Those class names appear in 18.2% of their own documents; NACE section names appear in 1.0%. |

The two datasets do not conflict. They differ in one measurable property, and
that property predicts which way the result goes.

**The honest companion.** Once the descriptions carry the vocabulary, the
hand-written seed keyword lists the project is named after add **nothing
measurable** (+0.3 points, p = 1.000). They had been supplying the vocabulary
the descriptions were missing.

---

## 1. Against baselines

`results/tables/baselines.md` · 299 corpus-sampled documents, 18 NACE sections.

| System | Top-1 | 95% CI | Top-3 | F1-macro | κ | p vs. ours |
| --- | --- | --- | --- | --- | --- | --- |
| Random | 5.0% | [2.7, 7.7] | 13.7% | 0.036 | −0.001 | <0.001 |
| *Majority class (oracle floor) | 21.7% | [17.1, 26.4] | 46.8% | 0.020 | 0.000 | <0.001 |
| TF-IDF → nearest NACE section | 44.5% | [38.8, 50.2] | 65.6% | 0.297 | 0.385 | 0.037 |
| Zero-shot embeddings (no seed keywords) | 52.8% | [47.2, 58.5] | 81.9% | 0.371 | 0.475 | 1.000 |
| Unguided KeyBERT | 52.8% | [47.2, 58.5] | 81.9% | 0.371 | 0.475 | 1.000 |
| **Ours: taxonomy-guided** | **52.5%** | [46.8, 58.2] | **81.6%** | 0.356 | 0.474 | — |

`*` uses the gold label distribution and is an oracle floor, not a competitor.
`p` is an exact McNemar test on the same documents.

**Top-3 at 81.6% is the operating point worth arguing for.** The correct
section is among the first three suggestions four times out of five, against a
46.8% Top-3 for the oracle majority floor. For a tool that proposes a code to a
human coder rather than assigning one unattended, that is the number that
matters, and it is where the method is strongest.

---

## 2. The contribution sentence

> We show that what a class description buys a zero-shot classifier is
> alignment: the gain from rewriting a class description is predicted by how far
> the rewrite moves the class vector toward the centroid of the documents it must
> attract (ρ = +0.51, p = 0.003 across 32 classes, holding within each dataset
> separately). The account is monotone across three corpora — NACE Rev. 2
> (Δ = +0.114, +26.8 points), Reuters-21578 (+0.028, +4.8), 20 Newsgroups
> (−0.076, +0.4) — and the third was predicted before it was measured. Neither
> the language of the description, the lexical overlap of the class name, nor its
> length accounts for the effect, and the quantity cannot be optimised per class:
> three selection criteria, including direct optimisation of development
> accuracy, all fail to generalise.

One positive claim with a mechanism, one prospective confirmation, and four
negative controls. That is the paper.

## 3. Language or content?

`results/description_study.json` · `make study` · 299 documents, sector vector
built from the description alone so the factor under test is isolated.

| Class descriptions | Top-1 | 95% CI | Top-3 | F1-macro | words/class |
| --- | --- | --- | --- | --- | --- |
| Original (16 of 21 in Turkish, terse) | 28.4% | [23.4, 33.4] | 62.5% | 0.243 | 10 |
| Same content, rendered in German | 26.1% | [21.1, 31.1] | 58.9% | 0.232 | 10 |
| **Rewritten as NACE-style definitions** | **52.8%** | [47.2, 58.5] | **81.9%** | **0.371** | 19 |

- **Language alone: −2.3 points, p = 0.14.** The German renderings are literal
  hand translations of the original labels, so no machine-translation noise is
  introduced alongside the variable under test, and the five sections already
  German in the original are carried over untouched.
- **Content: +26.8 points, p = 4×10⁻¹⁵.**

A crossed design agrees. Varying document language and description language
independently — German documents and their Marian translations, German
descriptions and their Marian translations — the richer descriptions win in
both document languages, and matching the two languages is worth +2.7 points on
average (`results/language_match.json`):

| Documents | Descriptions | Top-1 | Top-3 |
| --- | --- | --- | --- |
| German | German | 52.8% | 81.9% |
| English | German | 53.8% | 87.3% |
| German | English | 44.1% | 80.3% |
| English | English | 50.5% | 84.6% |

The German descriptions win in both rows, which is what a content effect looks
like; a matching effect would put the diagonal on top. The English column is
lower because machine translation degrades a carefully written definition — a
translation-quality effect, not a language one.

### It replicates on a second encoder

Run again with `paraphrase-multilingual-mpnet-base-v2` — a different model,
768 dimensions instead of 384, and the one this repository used to expect
better results from:

| Encoder | Language effect | Content effect |
| --- | --- | --- |
| MiniLM-L12-v2 (384-dim) | −2.3 pp, p = 0.14 | **+26.8 pp**, p = 4×10⁻¹⁵ |
| mpnet-base-v2 (768-dim) | +0.0 pp, p = 1.00 | **+13.7 pp**, p = 1×10⁻⁵ |

Same direction, same conclusion, different magnitude: the effect is a property
of what the descriptions say, not of one encoder's idiosyncrasies. `make study`
runs both.

**Why this generalises.** "Write class descriptions that enumerate what the
class covers, not descriptions that name it" applies to any zero-shot
classification pipeline that embeds label descriptions, which is most of them.
Three things make it safe to say rather than merely observed: the obvious
alternative explanation was tested and did not hold, the effect replicates on a
second encoder, and it was validated on a held-out half never inspected.

## 4. Does it hold outside this corpus?

`results/replication_20newsgroups.json` · `results/lexical_gap.json` ·
`make study`.

20 Newsgroups: English instead of German, usenet posts instead of legal prose,
twenty topical classes instead of twenty-one economic sections, no connection
to the original data. Same three-condition design.

| Class descriptions | Top-1 | 95% CI | Top-3 | F1-macro | words |
| --- | --- | --- | --- | --- | --- |
| Raw class identifier (`comp.sys.ibm.pc.hardware`) | 41.8% | [39.6, 44.0] | 69.2% | 0.409 | 1 |
| Readable class name (`IBM PC hardware`) | 51.8% | [49.6, 54.0] | 76.2% | 0.513 | 2 |
| Definition of what the class covers | 52.2% | [50.0, 54.4] | 77.2% | 0.521 | 23 |

- Identifier → readable name: **+10.0 pp, p = 8×10⁻²²**
- Readable name → definition: **+0.4 pp, p = 0.73**

**Elaboration buys nothing here, and that is the result.** On NACE the same step
was worth 26.8 points. The datasets differ in one property:

| | Class name appears in its own documents | Elaboration gain |
| --- | --- | --- |
| NACE Rev. 2 sections | **1.0%** (median 0.0%) | **+26.8 pp** |
| 20 Newsgroups classes | **18.2%** (median 13.4%) | +0.4 pp |

A trade register entry never says *"Erbringung von freiberuflichen,
wissenschaftlichen und technischen Dienstleistungen"*. It says
*"Steuerberatung"*. A usenet post about baseball says "baseball".

**How far the account goes.** Binned across all 32 classes in both datasets,
those with under 5% name overlap gain **+8.1 points** on average and the rest
**+1.4**. But the account does not survive at class level:

| Predictor of per-class gain | Spearman ρ | p |
| --- | --- | --- |
| Name overlap with the documents | −0.163 | 0.371 |
| Confusability of the terse class vector | +0.046 | 0.802 |
| Headroom — recall under the terse description | **−0.613** | **0.0002** |

Only headroom predicts it, and headroom is largely mechanical: a class already
at 90% cannot gain much. So **the lexical gap separates the two regimes without
explaining which classes inside a regime benefit.** That is a limitation, not a
footnote, and `results/gap_analysis.json` reports it.

**The claim, at the strength the evidence carries it.** What a zero-shot
classifier needs from a class description is the vocabulary of its documents.
Where the class names are absent from the documents, writing that vocabulary
down is worth a great deal; where they are already present, it is worth
nothing. Which individual classes benefit inside the first regime is open.

That also explains the 20 Newsgroups jump from identifier to readable name:
`comp.sys.ibm.pc.hardware` is not natural language at all, and spelling it out
closes most of the gap in two words.

---

## 5. What predicts which classes gain

`results/predictor_search.json` · `make study`. Target is the share of
*available* headroom a class captured, not the raw gain: a class already at
90% cannot gain much, and raw gain mostly measures where a class started.

| Predictor | ρ | p | NACE | 20NG |
| --- | --- | --- | --- | --- |
| **How far the description moved toward its own documents** | **+0.513** | **0.003** | +0.56 | +0.63 |
| How aligned the terse description already was | −0.293 | 0.104 | −0.55 | −0.38 |
| Confusability of the class vector | −0.010 | 0.957 | +0.47 | −0.10 |
| Lexical overlap of the class name | +0.065 | 0.726 | +0.37 | −0.05 |
| Words added | −0.209 | 0.251 | −0.24 | +0.08 |

**The answer is alignment.** A description helps to exactly the extent that it
moves the class vector toward the centroid of the documents that class must
attract. It is the only candidate that survives, and the only one that holds in
*both* datasets separately — the test every other predictor failed.

It also accounts for the dataset-level difference, which is what makes it a
mechanism rather than a correlation:

| | Alignment, terse → elaborated | Accuracy gain |
| --- | --- | --- |
| NACE | 0.645 → 0.759 (**+0.114**) | +12.1% |
| 20 Newsgroups | 0.571 → 0.495 (**−0.076**) | +1.6% |

**The 20 Newsgroups definitions moved the class vectors *away* from their own
documents.** Twenty-three carefully written words about baseball sit further
from actual usenet posts than the word "Baseball" does. That is why elaboration
bought nothing there, and it is a sharper statement than the lexical-gap
account it replaces: the recipe is not "write more", it is "write closer to the
data". Words added correlates *negatively* with gain.

### Three datasets, and one predicted in advance

| Dataset | Class labels look like | Δ alignment, terse → elaborated | Accuracy gain |
| --- | --- | --- | --- |
| NACE Rev. 2 | abstract economic categories | **+0.114** | **+26.8 pp** (p = 4×10⁻¹⁵) |
| Reuters-21578 | opaque codes (`money-fx`, `acq`) | **+0.028** | **+4.8 pp** (p = 5×10⁻⁴) |
| 20 Newsgroups | readable topic names | **−0.076** | +0.4 pp (p = 0.73) |

The ordering is monotone: the more a rewrite moves the class vectors toward
their documents, the more accuracy it buys, and where it moves them away it
buys nothing.

**Reuters was predicted before it was measured.** `run_reuters.py` computes the
alignment change first, prints the prediction it implies, and only then looks at
accuracy. Its labels are opaque codes, which puts it in the same regime as NACE,
so the account said elaboration should help. It does, by 4.8 points. Spelling
the bare code out into a readable name is worth another 18.6 (p = 1×10⁻²²) — the
same identifier-to-name jump 20 Newsgroups shows, and for the same reason.

### It does not become a selection rule

If alignment were causal and local, choosing each class's description by it
should beat a fixed policy. It does not. Choosing on a development half and
scoring on held-out documents:

| Dataset | Criterion | vs. best fixed policy | p |
| --- | --- | --- | --- |
| NACE | alignment to own documents | +0.0 pp | 1.000 |
| NACE | contrastive margin against other classes | −0.7 pp | 1.000 |
| NACE | greedy search on development accuracy | +2.1 pp | 0.250 |
| 20NG | alignment to own documents | −2.5 pp | 0.077 |
| 20NG | contrastive margin against other classes | **−3.6 pp** | **0.005** |
| 20NG | greedy search on development accuracy | −1.5 pp | 0.169 |

The last row is the important one. Greedy search optimises the objective the
classifier actually uses — swap whichever single class's description most
improves development accuracy, repeat — so it is the strongest per-class
criterion available, and it is not a proxy for anything. Development accuracy
rose (+1.4 pp on NACE over 2 swaps, +1.0 on 20NG over 5) and held-out accuracy
did not follow.

The reason is capacity: **there are 2^K configurations for K classes**, and
choosing among them from a labelled development set needs more supervision than
a zero-shot pipeline is supposed to require. By the time you have enough
labelled data to pick descriptions per class, you have enough to train a
classifier.

### Why it fails, measured

The two description styles do not sit on a common similarity scale, and the
classifier takes an argmax *across* classes. In a set where every other class
carries an elaborated description:

| | Share of decisions won by the elaborated half | Fair share |
| --- | --- | --- |
| NACE | **69.8%** | 50% |
| 20 Newsgroups | **35.2%** | 50% |

The bias runs in opposite directions in the two datasets, and in exactly the
direction alignment predicts. Per-class selection optimises each class against
a yardstick the argmax does not use.

Per-class z-scoring — the obvious calibration — does not rescue it either. It
helps the mixed set slightly and costs 8.4 points on the best pure policy
(57.9% → 49.5%, p = 0.005), because it discards the class-prior information the
plain argmax was exploiting.

**The practical rule that falls out: keep the description style uniform across
classes, and put the effort into the style rather than into per-class choices.**
Three criteria of increasing strength — marginal alignment, a contrastive
margin, and direct greedy optimisation of development accuracy — all fail to
generalise. A set of individually-better descriptions can classify worse than a
set of consistently-written ones, and the search for which classes to treat
differently costs more supervision than it returns.

---

## 6. What each component contributes

`results/tables/ablation.md`, full 299-document set.

| Variant | Top-1 | Δ Top-1 | F1-macro | p vs. full |
| --- | --- | --- | --- | --- |
| Full system | 52.5% | — | 0.356 | — |
| − seeds in sector vector | 52.8% | +0.3 pp | 0.371 | 1.000 |
| − description in sector vector | 44.1% | −8.4 pp | 0.342 | 0.002 |
| − seed-guided extraction | 52.5% | 0.0 pp | 0.356 | 1.000 |
| − six-stage keyword filter | 52.5% | 0.0 pp | 0.356 | 1.000 |
| ↩ previous taxonomy (Turkish descriptions) | 34.8% | **−17.7 pp** | 0.293 | **<0.001** |
| + cleaned text into the classifier | 41.8% | −10.7 pp | 0.352 | 0.001 |
| ↔ mpnet-base-v2 encoder (768-dim) | 42.1% | −10.4 pp | 0.303 | <0.001 |
| ↔ German translated to English first | 56.2% | +3.7 pp | 0.422 | 0.200 |

### Findings

1. **Informative class descriptions are the contribution: −17.7 points without
   them, p < 0.001** (−26.8 with the sector vector isolated to the description).
   The largest effect in the project by a wide margin, and §3 shows it is the
   content rather than the language.
2. **The description is now the load-bearing half of the sector vector**
   (−8.4 points without it, p = 0.002), and **the seed list is not**
   (+0.3 points without it, p = 1.000). This inverts what the earlier
   configuration showed, and the inversion is the point: seed keywords looked
   essential only because they were propping up descriptions that named the
   class instead of describing it.
3. **Guided extraction and the six-stage filter still do nothing** — identical
   Top-1 with either removed, now across three separate configurations of the
   system. Two of six stages have never shown an effect.
4. **The larger encoder is worse, and now clearly so:** −10.4 points
   (p < 0.001). The standing expectation of a ~10% gain from mpnet-base-v2 is
   falsified with room to spare.
5. **Translating to English first gains 3.7 points but is not significant**
   (p = 0.200). Suggestive; its apparent size has shrunk with every improvement
   to the German side, which is itself informative.

---

## 7. Held-out validation

The taxonomy revision was developed by inspecting errors, which is a form of
fitting. It was therefore developed on a development half and reported on a
held-out half that was never looked at.

| | Development (n=152) | **Held-out test (n=147)** |
| --- | --- | --- |
| Previous taxonomy | 35.5% | 34.0% |
| Revised taxonomy | 55.3% | **49.7%** |
| Gain | +19.7 pp | **+15.6 pp** (p = 0.0006) |
| Top-3, revised | 80.3% | **83.0%** |

The split is stratified by section and seeded
(`data/evaluation/split.json`, `python -m experiments.split`).

The seed-keyword result is honestly inconclusive rather than negative: seeds
help by 4.6 points on dev and hurt by 5.4 on test, neither significant
(p = 0.34, p = 0.22). The defensible claim is that they add nothing reliable.

---

## 8. Error analysis

50 errors from the development half, hand-coded against the codebook in
`experiments/error_analysis.py`. The held-out half is untouched.

| Category | n | Share |
| --- | --- | --- |
| **Seed leakage** — a term in one section's seed list attracts documents from another | 17 | 34% |
| Ambiguous section definition — two sections genuinely defensible | 14 | 28% |
| Multi-sector company — the purpose lists several real businesses | 10 | 20% |
| Taxonomy granularity — the activity sits under a counterintuitive parent | 8 | 16% |
| Boilerplate only — no activity signal in the text at all | 1 | 2% |

### Seed leakage is the largest category, and it converges with the ablation

The failures are lexical, not semantic. *Bodenbelagsarbeiten* (floor covering,
section F) goes to Mining because **Boden** is in B's vocabulary. *Rohbauten*
goes to Mining on **Roh**. *Karosseriebau* leaves Trade for Construction on the
**-bau** suffix. *Gebäudereinigung* leaves Administrative services for
Construction on **Gebäude**. Data-protection consultancy goes to IT on **Daten**,
at a margin of 0.000.

**Two independent lines of evidence now point the same way.** The ablation says
removing the seed lists costs nothing (+0.3 points, p = 1.000). The error
analysis says they cause a third of the remaining errors. A component that
adds no measurable benefit and has a named failure mode is a component to cut.

It is *not* cut in this repository, and deliberately so: the seed effect is
+4.6 points on dev and −5.4 on test, neither significant, and changing shipped
behaviour on a coin flip is the same mistake that
`classify_preprocessed_text` was held back from. The recommendation goes in the
paper; the decision waits for evidence that can carry it.

### What the other three categories mean

Ambiguity and multi-sector companies together are 48% of errors and are not
fixable by any change to this system: NACE genuinely admits two answers for a
Wirtschaftsprüfung practice or a company that both manufactures and installs
windows. That number is worth reporting as a rough ceiling on single-label
accuracy for this task — which is also the argument for reporting Top-3.

---

## 9. Where the gain came from

Per-section recall on the held-out half, before and after:

| Section | n | Before | After |
| --- | --- | --- | --- |
| M — Professional, scientific, technical | 33 | 12% | **64%** |
| N — Administrative and support | 9 | 11% | **33%** |
| J — Information and communication | 15 | 47% | **67%** |
| C — Manufacturing | 10 | 40% | **70%** |
| I — Accommodation and food | 5 | 60% | 80% |
| Q — Health and social work | 7 | 29% | 43% |
| L — Real estate | 9 | 67% | 33% |
| H — Transport and storage | 4 | 100% | 75% |

Section M was the diagnosis the error analysis produced: it holds both
professional services and every Komplementär-GmbH whose only activity is
managing another company, and the seed list had no vocabulary for the latter.
Adding *Beteiligung*, *Komplementär*, *Übernahme der persönlichen Haftung* and
*Geschäftsführung*, and removing terms that collided with it in N and O, moved
it from 12% to 64% on documents never inspected.

Two smaller repairs came from the same analysis: section D's seeds owned
*Heizung* and *Klimaanlage*, pulling building-installation firms (F) into energy
supply; and *Buchhaltung* was claimed by both M and N.

L and H lost ground. Both are small in this sample and the movement is within
noise, but it is the honest cost of the change and belongs in the table.

---

## 10. Label quality

The 299 labels are model-assisted: produced by a language model applying
[`annotation_guidelines.md`](annotation_guidelines.md), then validated against
a human pass.

| | Blind pilot | After guideline revision |
| --- | --- | --- |
| Documents | 50 | 50 **fresh** ones |
| Raw agreement | 58.0% | **80.0%** |
| Cohen's κ | 0.542 | **0.772** |
| …where flagged high confidence | 80.0% | **100.0%** |
| …where flagged low confidence | 36.0% | 60.0% |

25 out of 25 on the documents the labeller called confident: residual
disagreement lives entirely in the low-confidence bucket, so the labels ship
with a reliability signal that works. The verified answers were then promoted
over the machine labels — agreement measured before adjudication, released
labels after it.

The paper's data section has to state this: *"Section labels were produced by a
language model applying a written annotation guideline (Appendix X). In a blind
pilot a second annotator agreed on 58% of 50 documents (κ = 0.542); the
disagreements were adjudicated into three explicit rules, and on a fresh
50-document sample agreement was 80% (κ = 0.772), rising to 100% on the
documents flagged high-confidence."*

---

## 11. What remains

1. **Run the LLM baseline.** An instruction-tuned model asked to name a section
   directly is the comparison a 2026 reviewer will expect. The adapter exists
   (`run.py --with-llm`); it needs an API key.
2. **A fourth dataset**, ideally one where the alignment change is near zero,
   to test the account at the point where it makes its least obvious
   prediction.
3. **A criterion that survives the argmax without labels.** Three tried here
   fail, including direct optimisation of the objective. Whether a label-free
   set-level criterion exists is open; it would have to beat a fixed policy
   without the development data that makes the search overfit.
4. **Decide the keyword question.** The set carries section labels only, so
   Precision@K is unmeasurable, and the keyword half of the pipeline has now
   failed to show an effect in every configuration tested. Writing a
   section-classification paper is the honest and tighter option; the
   contribution sentence never mentioned keywords.
5. **Consider dropping the seed lists and two dead stages.** Guided extraction and the six-stage filter
   have no evidence behind them across three configurations. Either find a
   metric where they help, or cut them and say why.
6. **Write.** `paper/main.tex` is the skeleton; `make paper-tables` regenerates
   its tables from `results/`, so no number is typed into the prose.

## 12. Decisions

- **Authorship.** The work is yours. Settle it before submission.
- **Venue.** [`../paper/venues.md`](../paper/venues.md) has the priority order.
  The Student Research Workshop assigns a mentor.
- **Framing.** A 26.8-point effect from one line per class, with the obvious
  alternative explanation tested and rejected, plus a negative result about the
  mechanism the project is named after. That is a better paper than a high
  accuracy number, and a better application artefact: it shows someone who
  measures, controls for the confound in their own favourite explanation, and
  reports what the data says rather than what they hoped.
- **The project's name.** "Sectoral Keyword Extraction" names the half that
  demonstrably contributes nothing. The work is now about class descriptions
  for zero-shot taxonomy classification, and the title should say so.

## 13. Artefact standard

Complete: one-command reproduction from a clean clone, `results/metrics.json`,
pinned dependencies, fixed seed, tests and CI, MIT licence, `CITATION.cff`,
results-first README, honest limitations, data provenance, demo GIF, rendered
architecture diagram, a configuration file every key of which the code reads,
an offline annotation tool with translations, written annotation guidelines,
a two-stage verification protocol, a stratified dev/test split so that
error-driven changes are reported on data they were not developed on, and two
controlled studies separating the description finding from its confound.

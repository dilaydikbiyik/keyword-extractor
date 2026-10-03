# TODO

What is done, what is open, and what is deliberately not being done. Updated as
work lands; every claim here points at the file that settles it.

**Where the project stands.** Five corpora, 11,141 documents, 108 classes. Eight
preregistered studies, 36 predictions committed to version control before they
were measured, 29 held and the seven failures published. 63 comparisons
corrected together with Holm (38 nominal, 27 surviving). 193 tests in CI, paper
body at the 8-page limit, submission artefacts built and checked anonymous.

---

## A. Open research gaps, ranked

Each item says what a reader could do afterwards that they cannot do now. That
is the test for whether it belongs in a paper rather than in a backlog.

### ~~A1. The quantity, estimated without labels~~ — **done, and it held**
All four registered predictions held. The estimate, which uses no label
anywhere, tracks the labelled quantity at ρ = +0.818 over 102 classes and
predicts the share of headroom captured at ρ = +0.564 (p = 6e-10, surviving
Holm) against +0.670 for the labelled version. It degrades where the
pseudo-assignment is poorest — on NACE, +0.201 against +0.759 — so the advice it
licenses is: where the terse classifier is already somewhat right, an unlabelled
sample tells you whether rewriting will pay. `make labelfree`,
`results/labelfree_predictor.json`, paper §*The quantity, without labels*.
**Still open from it:** a threshold rule with a measured false-positive rate,
i.e. "how large must the estimate be before rewriting is worth the effort".

### A2. Why alignment is the right quantity — *not started, no compute needed*
The account is empirical: a correlation that replicates. It is not derived. Under
nearest-prototype argmax the decision for a document depends on the margin
between its similarity to its own class vector and to the nearest competitor, so
moving a vector toward its class centroid raises that margin for its own
documents and lowers it for its neighbours' — which is exactly the shape of what
we observe, including why the per-class version holds while the corpus mean does
not. A short derivation plus a synthetic check (Gaussian clusters, prototypes
moved by hand) would show *when the account must fail*: classes that overlap, and
corpora where moving one prototype steals documents from another.
**Afterwards:** the paper explains rather than reports, and the failure modes are
predicted instead of discovered.

### A3. How much writing is enough — *not started, ~1 hour of compute*
Class texts here run from 1 word (`cs.CL`) to 50 (the independent model's
definitions), and the gains do not scale with length. Truncating each definition
to k words for k = 5, 10, 20, 40 and measuring the curve answers the question
every practitioner actually asks: how much description writing pays, and where it
stops paying.
**Afterwards:** a budget, not an exhortation.

### A4. What a wrong description costs — *not started, ~1 hour of compute*
Everything measured so far compares terse against good. Nobody has measured what
a *bad* description does: activities from a neighbouring class, plausible but
wrong vocabulary, a definition that describes the class's exclusions. Injecting
each kind at a known rate and measuring the degradation separates "descriptions
matter" from "any longer text helps".
**Afterwards:** a reader knows the risk of the advice, not only its upside.

### A5. A written recipe — *not started, no compute*
The evidence supports a short, concrete guide: enumerate the activities in the
documents' own vocabulary; do not translate terse labels; do not paste the
definitions into a prompt (that loses 35 points, `results/llm_baseline_definitions.json`);
check the estimate from A1 before paying for annotation. `docs/recipe.md`, every
line pointing at the result that supports it.
**Afterwards:** the work is usable by someone who will never read the paper.

### A6. Hierarchy — *not started, needs label work*
Every taxonomy tested here is flat, and NACE is not: sections divide into
divisions, groups and classes. Whether the effect compounds down a hierarchy, or
whether deeper levels are dominated by lexical overlap, is both open and
practically important — official coding happens at four digits, not at the
section.
**Afterwards:** the result applies to the task statistical offices actually run.

### A7. Abstention — *not started, ~2 hours*
48% of the coded errors are documents where NACE admits more than one defensible
answer. A system meant to propose candidates to a human coder should be able to
say "ask a human", and the precision/coverage trade-off of an abstention rule is
measurable on the existing predictions without new models.
**Afterwards:** the deployment story has a number attached.

### A8. Breadth: another language, another taxonomy — *not started, needs data*
German and English; NACE, Reuters, 20NG, Brown, arXiv. A second language with
gold labels (NACE is published in 24) or another gold-labelled taxonomy
(EuroVoc, DDC) would test whether the account is about taxonomies or about
German.
**Afterwards:** the claim is about zero-shot classification, not about one task.

---

## B. Mine to finish before submission

- [x] Anonymous archive: the anonymiser no longer carries the strings it removes
      (`private/identity.txt`, untracked, with a test that scans the archive).
- [x] Cost measured, and the ordering verified on a second device.
- [x] Two open LLM baselines from two families, with the harness defect that the
      first run of the second one measured instead of the model.
- [x] The research page republished with the current record (14 entries).
- [ ] Update the two stale published pages: "Alignment, Not Wording" and the
      Turkish defence notebook.
- [ ] A2 and A5 above, which need no compute and would land before the deadline.

## C. Yours, and nobody else can do them

- [ ] **OpenReview account**, institutional address as primary. Approval can take
      up to two weeks, so this is the long-lead item.
- [ ] **CV numbers**: 193 tests under CI; the word TF-IDF baseline is 44.5% and
      the character n-gram one 54.2% — they are different systems. Lead with
      five corpora and 11,141 documents rather than with 299.
- [ ] **Submit to ARR by 12 October**, then **register as a reviewer by 14
      October** — a submission whose authors are not registered can be desk
      rejected.
- [ ] **Read the paper end to end.** Every sentence has to be defensible by you;
      the defence notebook exists for this.
- [ ] **Optional, valuable:** code 50 errors blind (the coding tool is in your
      Downloads), and find a German reader for the 50-document blind sample
      (`make second-annotator`, instructions in `docs/second_annotator.md`).
- [ ] **Rename the repository to `class-description-effect`** — *after your
      internship applications resolve*, so no application points at a dead link.
      When you rename it on GitHub, the seven files that hardcode the old name,
      `CITATION.cff`, the README badges, the Pages URL and the research page's
      sentence about the repository keeping its original name all get fixed in
      one commit.

## D. Deliberately not doing, and why

- **Replacing the 299-document evaluation set to make it bigger.** Every
      preregistered prediction was committed against that exact set; swapping it
      would destroy the evidence that the predictions came first. The scale
      question is answered by adding corpora, which is what arXiv did.
- **Chasing 90%+ Top-1.** The labels themselves agree with a human on 80% of a
      blind sample (κ = 0.772), and 48% of coded errors are documents with more
      than one defensible answer, which puts the single-label ceiling near 77%.
      A system scoring 95% against these labels would be fitting the labeller.
      Top-3 (81.6%) is the number the intended use cares about.
- **Tuning prompts or hyperparameters until the numbers look better.** The
      Rocchio setting stayed untuned and got a sensitivity grid instead; the
      degenerate LLM run was diagnosed, not retried until it flattered us.
- **FastAPI, Docker, AWS, MLflow, monitoring.** Good for an engineering
      portfolio, irrelevant to a paper, and they would consume the days before
      the deadline. Revisit after submission.
- **A paid API baseline.** The adapter is released and unrun; it must not be a
      Claude model, which produced the labels and would be graded on its own
      answers.

## E. Dates

| | |
| --- | --- |
| ARR submission | **12 October 2026** |
| Reviewer registration, mandatory | **14 October 2026** |
| Reviews back | 16 November 2026 |
| Author response | 24–30 November 2026 |
| Commit to NAACL 2027 | 23 December 2026 |

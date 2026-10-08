# TODO

What is done, what is open, and what is deliberately not being done. Updated as
work lands; every claim here points at the file that settles it.

**Where the project stands.** Six evaluation sets, 12,341 documents, 108
classes. Nine preregistered studies, 40 predictions committed to version control
before they were measured, 33 held and the seven failures published. 70
comparisons corrected together with Holm (45 nominal, 32 surviving). The advice
the work supports is collected in `docs/recipe.md`, with a measured budget, a
measured price for getting it wrong, and an abstention curve. 199 tests in CI,
paper body at the 8-page limit, submission artefacts built and checked
anonymous.

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
sample tells you whether rewriting pays at all. It does not license an order to
work in; see the rewrite-priority simulation. `make labelfree`,
`results/labelfree_predictor.json`, paper §*The quantity, without labels*.
**Still open from it:** a threshold rule with a measured false-positive rate,
i.e. "how large must the estimate be before rewriting is worth the effort".

### ~~A2. Why alignment is the right quantity~~ — **done**
The derivation is in the paper: the decision is an argmax over cosines, so what
moves a document is the margin between its own class and the nearest competitor;
moving a class vector toward its centroid raises the first term for all of that
class's documents and lowers the second for its neighbours'. The simulation
confirms both halves — accuracy monotone in the distance moved at all four
separations, and moving one vector onto its own centroid gains that class +0.550
of recall while costing its nearest neighbour −0.012 and everyone else +0.001,
falling to +0.025 as the classes crowd together. `make synthetic`,
`results/synthetic.json`.
**Still open:** the derivation is informal. A bound rather than an argument —
expected margin change as a function of Δa and the neighbour distances — would
be stronger, and is a theory exercise rather than a measurement.

### ~~A3. How much writing is enough~~ — **done**
Twenty words, on both German sets: name only 29.4%, +10 words 47.5%, **+20
55.5%**, +40 52.8%. Ninety per cent of the gain arrives by twenty words and the
curve then turns slightly down, which matches the negative correlation between
words added and per-class gain. `make dose-response`.

### ~~A4. What a wrong description costs~~ — **done**
A quarter of the classes given another class's definition costs 4.7 points at
random and 12.7 when the swap happens between the classes already most similar;
a generic text of the same length as the definitions lands within three points of
the name-only floor. So length is not the active ingredient, and the pairs worth
checking are the ones that already look alike. `make description-noise`.

### ~~A5. A written recipe~~ — **done**
`docs/recipe.md`: seven sections, every claim pointing at the result that
carries it, including the three things it explicitly does not tell you (depth
beyond one hierarchy level, prompted models, and languages other than German and
English).

### A6. Hierarchy — *partly done; the real test still needs labels*
20 Newsgroups has two levels for free, and they behave the same way: the
identifier→name step pays at 20 classes (+10.0) and at the 6 groups they form
(+13.4), and the name→definition step at neither (+0.4, +1.3). So the quantity is
the distance between class text and document vocabulary rather than the number of
classes. `make hierarchy`.
**Still open:** NACE four digits down, which needs labels at that depth — the one
part of this that no amount of compute substitutes for.

### ~~A7. Abstention~~ — **done**
The margin the decision already computes prices it: Top-1 is 55.4% at 90%
coverage, 58.9% at 75%, 65.3% at 50% and 76.0% at 25%. Reaching 80% means handing
back all but 23% of the documents, which is the honest shape of the trade-off on
a task whose labels agree with a human 80% of the time. `make abstention`.

### A8. Breadth: another language, another taxonomy — *open, and the only one that needs data*
German and English; NACE, Reuters, 20NG, Brown, arXiv. A second language with
gold labels (NACE is published in 24) or another gold-labelled taxonomy
(EuroVoc, DDC, MASSIVE) would test whether the account is about taxonomies or
about German. Every other gap on this list was closed with the corpora already
here; this one cannot be, because it needs labelled documents in a language
nobody has labelled for us. It also needs a new dependency to fetch them, which
is a decision for after the submission rather than before it.
**Afterwards:** the claim is about zero-shot classification, not about one task.

---

## B. Mine to finish before submission

- [x] Anonymous archive: the anonymiser no longer carries the strings it removes
      (`private/identity.txt`, untracked, with a test that scans the archive).
- [x] Cost measured, and the ordering verified on a second device.
- [x] Two open LLM baselines from two families, with the harness defect that the
      first run of the second one measured instead of the model.
- [x] The research page republished with the current record (14 entries).
- [x] The two stale pages brought up to date, and a decision recorded about
      them: the GitHub Pages research record is the only one meant to be
      public. The readiness review was deleted on 4 October 2026 once every
      gap it listed had closed — `docs/paper_readiness.md` carries the same
      material, versioned. The Turkish defence notebook stays, private and
      unlinked; it is interview preparation and not part of the record.
- [x] A2 and A5, plus A3, A4, A6 and A7 — every gap that did not need new
      labels or a new dependency is closed.
- [x] The findings turned into something a stranger can run on their own
      taxonomy: `experiments/diagnose.py`, with every figure it quotes read out
      of `results/` at run time rather than typed into it, and
      `run_overlap_estimate` measuring what its regime check costs when it has
      no labels to group by. The guide in `docs/recipe.md` now opens with the
      procedure instead of ending with the findings.
- [x] The description effect tested on a modern encoder, after a review
      objected that the three in the paper are an older generation. `bge-m3`
      replicates the content effect at +17.7 points and the language contrast
      stays negative there too. Both new tests joined the hypothesis family and
      the whole family was corrected again, 72 tests with 32 surviving — the
      point is that they were corrected together, not that the old results
      happened to survive.
      *Still open:* this replicates the **effect**, not the mechanism. The
      per-class Δa-against-gain correlation was not computed on `bge-m3`, so
      the encoder-independence of the quantity itself rests on the three
      encoders already reported.
- [x] The label-free estimate tested as a decision rule, not only as a
      correlation, after a review pointed out that the paper promised the first
      and measured the second. Rewriting only the classes it ranks highest
      captures about what a random order captures; with labels the same
      quantity captures half the gain from a tenth of the classes. The paper,
      the guide and the diagnostic all now say this, and three tests hold the
      negative in place.
- [x] Body page count rechecked against the right boundary. The earlier "eight
      pages" was measured to `\appendix`, which falls after the bibliography;
      measured to the start of Limitations, which is what ARR counts, the body
      ran to nine. Related Work, the error analysis, the retired-pipeline
      ablation and the cost subsection were compressed; it now ends on page
      eight with every citation intact.

## C. Yours, and nobody else can do them

- [ ] **OpenReview account**, institutional address as primary. Approval can take
      up to two weeks, so this is the long-lead item.
- [x] **CV numbers** — done 8 October: the industry CV now reads 207 tests, nine
      preregistered studies, 33 of 40 predictions held, six evaluation sets and
      12,341 documents, and every project carries a repository link that was
      checked unauthenticated. The research variant still carries the old
      figures and a venue that has not been decided.
- [ ] **The research CV still carries the old figures**, and a target venue that
      has not been decided. It says three preregistered studies, 12 of 15, 46
      comparisons and 106 tests; the true numbers are nine, 33 of 40, 70 and 207.
      Its *Research* blurb also names ACL SRW as the target, which is not the
      decision. Two reminders when you update it: the word TF-IDF baseline is
      44.5% and the character n-gram one 54.2% — different systems, often
      conflated; and lead with six evaluation sets and 12,341 documents rather
      than with 299.
- [ ] **Pin the submitted version.** The order matters, because the tag has to
      mark the commit the uploaded files were built from, not the later commit
      that stores the record:

      1. `make submission` — build the PDF and archive from the current commit
      2. Upload that exact PDF; do not rebuild it afterwards
      3. `make submission-record VENUE="ARR October 2026" TAG=v1-arr-october-2026`
      4. `git tag -a v1-arr-october-2026 -m "Submitted to ARR, October 2026"`
         — at this commit, which is the one the record names
      5. Commit `docs/submission_v1.md` on top; this does not invalidate it,
         because the record says which commit it describes
      6. `python tools/submission_record.py --verify docs/submission_v1.md`
      7. `git push && git push --tags`

      Step 6 fails if the PDF was rebuilt after step 3, or if the tag points
      anywhere but the commit named in the record. Keep the OpenReview
      submission confirmation too: it is the one timestamp neither the record
      nor the tag can forge. Everything after it is
      post-submission work and gets its own preregistrations; nothing is
      written back into the ones already in `results/`.
- [ ] **Submit to ARR by 12 October**, then **register as a reviewer by 14
      October** — a submission whose authors are not registered can be desk
      rejected.
- [ ] **Read the paper end to end**, and look at the pages as well as the words.
      Three appendix tables were printing over each other until you noticed it in
      the PDF; a test now catches that class of defect, but only the kinds it was
      taught. Every sentence has to be defensible by you;
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

- **Replacing the 299-document evaluation set.** Every preregistered prediction
      was committed against that exact set; swapping its documents would destroy
      the evidence that the predictions came first. *Adding* sets is a different
      thing and has been done twice: 7,470 arXiv abstracts with gold labels, and
      a second German set of 1,200 documents with its own registered
      predictions. What stays true is that only 50 documents anywhere in this
      project carry a human-verified label.
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

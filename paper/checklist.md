# Responsible NLP Research checklist — draft answers

ACL Rolling Review asks every submission to answer this checklist
(<https://aclrollingreview.org/responsibleNLPresearch/>). These are draft
answers for this paper, written against the current `main.tex`. Where the honest
answer is "no" or "partly", the gap is listed at the end so it can be closed
before submitting rather than discovered by a reviewer.

## A. For every submission

- **A1. Limitations?** Yes — the unnumbered *Limitations* section.
- **A2. Potential risks?** Yes. The *Ethics Statement* covers the data
  (company records, owner names inside company names, redistribution) and the
  misuse risk: unattended automatic coding would propagate errors silently.

## B. Did you use or create scientific artifacts?

Yes: the trade register sample, NACE Rev. 2, 20 Newsgroups, Reuters-21578,
the sentence encoders, OPUS-MT, and the released evaluation set.

- **B1. Cited the creators?** Yes — encoders, OPUS-MT, NACE Rev. 2, the
  keyword-extraction methods, 20 Newsgroups (Lang 1995) and Reuters-21578
  (Lewis, UCI record).
- **B2. Licence or terms?** Partly. The *Ethics Statement* says the register
  sample's redistribution licence has not been established and it is therefore
  not released, and that both encoders are Apache-licensed. For 20 Newsgroups
  and Reuters-21578 it says they are used as distributed for research through
  scikit-learn and NLTK; their licence terms themselves are not stated. See gap 1.
- **B3. Intended use?** Partly, for the same reason.
- **B4. Personal information?** Partly. The *Ethics Statement* notes that
  company names can contain owners' names; no step removes them from the
  released evaluation set. See gap 2.
- **B5. Documentation of artifacts?** Yes — *Data* section, plus
  `data/README.md` and `docs/annotation_guidelines.md`.
- **B6. Statistics and splits?** Yes — *Data* section: set size, stratified
  development/test halves, corpus size.

## C. Did you run computational experiments?

Yes.

- **C1. Parameters and compute?** Partly. Both encoders are named with their
  parameter counts (118M and 278M) in the *Method* section; the runtime of the
  full reproduction is not reported. See gap 3.
- **C2. Hyperparameters?** Partly. There is no hyperparameter search; the few
  fixed settings (thresholds, top-k) are in the released configuration but not
  in the paper.
- **C3. Descriptive statistics?** Yes — bootstrap confidence intervals and exact
  McNemar tests throughout, with the test named in each table caption.
- **C4. Packages used?** Yes. sentence-transformers is covered by its papers;
  scikit-learn and SciPy are cited in the *Method* section, where the statistics
  are named.

## D. Did you use human annotators?

Yes — the author performed both verification passes. There were no hired or
recruited annotators.

- **D1. Full instructions?** Yes — `docs/annotation_guidelines.md` is
  released and referenced.
- **D2. Recruitment and pay?** Not applicable: the only annotator was the
  author, unpaid.
- **D3. Consent?** Not applicable for annotation. The texts are public company
  register records.
- **D4. Ethics review?** No board was consulted; no personal data about
  annotators was collected.
- **D5. Annotator characteristics?** Yes — the paper states the annotator does
  not read German and worked from OPUS-MT translations.

## E. AI assistants

- **E1. Information about their use?** Yes — the unnumbered section *Use of AI
  Assistants* describes what the assistant did (code, silver labels, class
  descriptions, error coding, analyses, drafting) and what the author did.

## Gaps to close before submitting

Closed: the misuse sentence, the dataset and software citations, and the
encoder parameter counts. Still open:

1. The licence terms of 20 Newsgroups and Reuters-21578. Their UCI and
   distribution records do not state one; check the original distributions.
2. Decide whether the released evaluation set keeps company names. If it does,
   say why; if not, strip or hash them before release.
3. Report the wall-clock time of `make reproduce` on the hardware used.
4. Before relying on the answers above, check each against the final PDF.

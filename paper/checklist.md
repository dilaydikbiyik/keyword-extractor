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
the sentence encoders, OPUS-MT, the released evaluation set, and the derived
TF-IDF statistics (vocabulary and IDF only) that stand in for the unreleased
corpus.

- **B1. Cited the creators?** Yes — encoders, OPUS-MT, NACE Rev. 2, the
  keyword-extraction methods, 20 Newsgroups (Lang 1995) and Reuters-21578
  (Lewis, UCI record).
- **B2. Licence or terms?** Partly. The *Ethics Statement* says the register
  sample's redistribution licence has not been established and it is therefore
  not released, that both encoders are Apache-licensed, and that Reuters-21578
  is distributed for research only by agreement of Reuters Ltd. and Carnegie
  Group (from the collection's own README). 20 Newsgroups' distribution states
  no licence, and the paper says so. See gap 1.
- **B3. Intended use?** Partly, for the same reason.
- **B4. Personal information?** Yes. The *Ethics Statement* notes that company
  names can contain owners' names, says that the evaluation set's records are
  released with them because they are part of a public register entry, and
  that the pipeline never reads them.
- **B5. Documentation of artifacts?** Yes — *Data* section, plus
  `data/README.md` (which lists every derived statistics file) and
  `docs/annotation_guidelines.md`.
- **B6. Statistics and splits?** Yes — *Data* section: set size, stratified
  development/test halves, corpus size.

## C. Did you run computational experiments?

Yes.

- **C1. Parameters and compute?** Yes. Both encoders are named with their
  parameter counts (118M and 278M) in the *Method* section, which also gives the
  wall-clock time of the complete reproduction and the hardware it ran on,
  recorded by `run.py` in `results/compute.json`.
- **C2. Hyperparameters?** Partly. There is no hyperparameter search; the few
  fixed settings (thresholds, top-k) are in the released configuration but not
  in the paper.
- **C3. Descriptive statistics?** Yes — bootstrap confidence intervals and exact
  McNemar tests throughout, with the test named in each table caption. Every
  comparison reported as a finding is corrected together with the Holm
  procedure and listed with its interval in the appendix; the text qualifies
  each claim that does not survive the correction.
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
  descriptions, error coding, analyses, drafting) and what the author did,
  including that the original pipeline was built by the author with AI coding
  tools.

## Gaps to close before submitting

Closed: the misuse sentence, the dataset and software citations, the encoder
parameter counts and the reproduction runtime. Still open:

1. ~~20 Newsgroups licence.~~ Decided: the sentence the paper has is enough. The
   collection states no licence anywhere in its distribution, has been a
   standard research benchmark for three decades, and is used here only for
   evaluation, as scikit-learn distributes it. Reuters-21578's terms are settled
   from its README.
2. ~~Company names in the released set.~~ Decided: kept, and the paper says
   why. The 299 records are public register entries, the names are part of the
   record, no classifier reads them, and removing them now would not remove them
   from the repository's history.
3. Before relying on the answers above, check each against the final PDF.

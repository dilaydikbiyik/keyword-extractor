# Venue tracker

Deadlines move every year and no date here is trustworthy until it has been
read off the venue's own call for papers. **Verify, then write the real date
into this table**, and set three calendar reminders per venue: six weeks, two
weeks and three days before.

## Priority order

| # | Venue | Why | Deadline (verify) | Reminders set | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | \*ACL **Student Research Workshop** (ACL / EACL / NAACL / EMNLP) | Built for a first publication; assigns a mentor before the deadline. Submit as a **long paper (8 pages)**: the body already runs past four pages with its tables and figure, measured with `make paper` | | ☐ | Check whether the mentorship round has an earlier deadline than the paper itself |
| 2 | Workshops on **low-resource / multilingual NLP** | The declared specialism; taxonomy-guided zero-shot fits the scope | | ☐ | Co-located workshop lists appear with each main conference's CFP |
| 3 | Workshops on **evaluation and benchmarking** | The contribution is as much about measurement as method | | ☐ | |
| 4 | **RANLP**, **LREC** | Mid-size, receptive to resource and evaluation papers; LREC especially for the evaluation set as a resource | | ☐ | LREC runs in even years |
| 5 | Turkish NLP workshops | A reasonable first experience of the review process | | ☐ | |

Starting points for verification: the ACL Anthology's venue pages, the main
conference site's "Calls" section, and each workshop's own page. Nothing else.

## What has to be true before submitting anywhere

From [`../docs/paper_readiness.md`](../docs/paper_readiness.md):

- [ ] Evaluation set enlarged past 30 documents — nothing is significant below that
- [ ] Error analysis hand-coded on roughly 50 errors
- [ ] Inter- or intra-annotator agreement measured and reported
- [ ] Synthetic entries replaced with real register records
- [ ] Data licence position stated in the paper's data section
- [ ] `make submission` passes: nothing in `dist/review.pdf` or
      `dist/anonymous_code.zip` identifies the author

## After submission

- [ ] arXiv preprint — **check the venue's anonymity policy first**; some
      forbid preprinting inside an embargo window
- [ ] Repository made public
- [ ] One post announcing it

A rejection is a normal first outcome: apply the reviews, send it to the next
workshop.

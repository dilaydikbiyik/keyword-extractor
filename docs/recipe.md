# How to write class descriptions for a zero-shot classifier

Everything here is a consequence of a measurement in this repository, and each
line points at the result that carries it. Nothing in this file is advice we
have not tested; where a question is open, it says so.

The setting is the common one: each class of a taxonomy is represented by a
short text, both the text and the document are embedded by a sentence encoder,
and the document goes to the nearest class. No labelled training data.

---

## The procedure, on your taxonomy rather than ours

None of what follows needs a label, so none of it needs to wait for an
annotation budget. Point the diagnostic at your own class list and a pile of
documents nobody has labelled:

```bash
python -m experiments.diagnose --taxonomy classes.json --documents pool.txt
```

`classes.json` maps each class to its name and description — `{"G": {"name":
"Retail trade", "description": "..."}}`, or just `{"Retail trade": "..."}` — and
the pool is one document per line, a JSON list, or a CSV column. The report says
which regime the taxonomy is in, which classes sit furthest from the documents
they already attract, which ones a careless rewrite would damage most, and what
abstaining would cost in coverage. Every figure it quotes is read out of
`results/` at run time, from the study that established it, with that study
named beside it.

Then write the descriptions for the classes it put first, and check the rewrite
before paying for anything:

```bash
python -m experiments.diagnose --taxonomy classes.json --documents pool.txt \
    --rewritten classes_v2.json
```

That second run reports, per class, whether the rewrite moved the class vector
toward the documents or away from it — the quantity that predicts the gain,
estimated without labels. Classes that moved away are the ones to look at again.

The rest of this file is what each number in that report means and how far it
can be trusted.

---

## 1. Write what the class covers, not what it is called

Rewriting a taxonomy's own terse labels into definitions that enumerate concrete
activities is worth **+26.8 points** of Top-1 accuracy on German trade-register
texts, and **+17.8** on a second, larger German set drawn afterwards
([`results/description_study.json`](../results/description_study.json),
[`results/extended_eval.json`](../results/extended_eval.json)).

It is the content that pays, not the language: rendering the same terse labels
in the documents' own language is worth **−2.3 points** and **−1.8** on the two
sets. A crossed design over document language and description language agrees.

## 2. Check whether it will pay, before you write anything

What a description buys is **alignment**: how far it moves the class vector
toward the centroid of the documents that class has to attract. Where the class
names are already the documents' own words, elaboration buys nothing — on 20
Newsgroups it is worth **+0.4 points**, and on arXiv, where the official
descriptions move the vectors by **+0.0013**, they are worth **+0.9**
([`results/arxiv.json`](../results/arxiv.json)).

You can estimate this without a single label. Assign your unlabelled documents
to their nearest class under the terse texts — the argmax the classifier already
computes — and take the centroid of each pseudo-class. Over five corpora and 102
classes that estimate tracks the labelled quantity at **ρ = +0.818** and still
predicts which classes gain at **ρ = +0.564**
([`results/labelfree_predictor.json`](../results/labelfree_predictor.json)).

The cheaper check comes first, though, and it is a word count rather than an
embedding: how often a class's name appears in the documents that class
attracts. Classes whose names are absent from their documents gained **+8.1
points** on average from elaboration; classes whose names are present gained
**+1.4** ([`results/gap_analysis.json`](../results/gap_analysis.json)). It
separates the two regimes and does not rank inside either (ρ = −0.16, p = 0.37),
which is all it is used for.

That check also needs labels in its published form — knowing which documents are
a class's own is the labelling nobody has done — so the diagnostic substitutes
the documents the terse vectors assign. **The substitution was measured rather
than assumed**: over 103 classes it tracks the labelled overlap at **ρ = +0.915**,
gives the same per-class verdict **88%** of the time, and puts **four of five**
corpora in the right regime
([`results/overlap_estimate.json`](../results/overlap_estimate.json)). Every
failure ran the same way — without labels the overlap reads up to **4.4×** high
in the low-overlap regime, because pseudo-assignment hands a class documents that
are not its own. So an estimate just above the threshold is not evidence against
rewriting; it is undecided, and the one corpus the check got wrong (Brown) sat in
exactly that band.

It is weaker where the terse classifier is itself poor: on the German task it
retains **+0.201** against the labelled version's **+0.759**. So the rule is:
*if the terse classifier is already somewhat right, the estimate is worth
trusting; if it is near chance, measure instead of estimating.*

## 3. About twenty words per class. More does not pay

Truncating each definition to its first *k* words gives the same curve on both
German sets ([`results/dose_response.json`](../results/dose_response.json)):

| Words per class | First set | Second set |
| --- | --- | --- |
| name only | 29.4% | 41.0% |
| + 3 | 30.1% | 42.8% |
| + 5 | 34.8% | 46.9% |
| + 10 | 47.5% | 57.8% |
| **+ 20** | **55.5%** | **61.7%** |
| + 40 | 52.8% | 60.3% |
| all (median 14) | 52.8% | 60.3% |

Ninety per cent of the gain arrives by twenty words, and past that the curve
turns slightly down. Across classes, *words added* correlates **negatively**
with the per-class gain ([`results/predictor_search.json`](../results/predictor_search.json)):
the recipe is not "write more" but "write closer to the documents".

## 4. Keep the style uniform across classes

Do not improve one class's description and leave its neighbours terse. Styles do
not share a similarity scale: in a set where half the classes carry elaborated
descriptions, that half wins **69.8%** of argmax decisions against a fair share
of 50%. Three per-class selection rules — marginal alignment, a contrastive
margin, and greedy search on development accuracy — all fail to generalise. A
set of individually better descriptions can classify worse than a set of
consistently written ones.

## 5. A wrong description costs more than a terse one

Measured by corrupting the paper's own definitions
([`results/description_noise.json`](../results/description_noise.json), first
set):

| Corruption | Cost |
| --- | --- |
| 10% of classes given another class's definition | −2.7 pp |
| 25% of classes | −4.7 pp |
| 25%, but swapped between the *most similar* classes | **−12.7 pp** |
| 50% of classes | −20.4 pp |
| every class the same generic text | −20.7 pp |
| name only (the floor) | −23.4 pp |

Two things to take from it. A mistake between **confusable** classes costs about
three times a random one at the same rate, so the pairs worth checking are the
ones that already look alike. And a generic text of the same length lands within
three points of the floor: length is not what helps.

## 6. Report Top-3, and let the classifier decline

On the German task the right section is in the top three **81.6%** of the time
against **52.5%** at Top-1, and nearly half of the coded errors are documents
where the taxonomy admits more than one defensible answer. If the system
proposes codes to a human, give it a threshold: abstaining below a margin
between the top two classes trades coverage for accuracy
([`results/abstention.json`](../results/abstention.json)):

| Coverage | Top-1 on what is kept |
| --- | --- |
| 100% | 52.5% |
| 90% | 55.4% |
| 75% | 58.9% |
| 50% | 65.3% |
| 25% | 76.0% |

Reaching 80% Top-1 means handing back three documents in four. That is the
honest shape of the trade-off on this task, and it is why the deployment story
is a tool for a human coder rather than an unattended one.

## 7. What this does not tell you

- **Depth.** The effect was measured on flat taxonomies. On 20 Newsgroups at two
  levels of one hierarchy the identifier→name step pays at both (+10.0 points
  over 20 classes, +13.4 over 6 groups) and the name→definition step at neither
  (+0.4, +1.3), so the quantity that matters is the distance between class text
  and document vocabulary rather than the number of classes
  ([`results/hierarchy.json`](../results/hierarchy.json)). Whether this holds
  four digits down a real taxonomy is untested.
- **Prompting.** Pasting the same definitions into a prompted language model
  does not transfer the gain: the same 7B model falls from 48.8% to 14.0%
  ([`results/llm_baseline_definitions.json`](../results/llm_baseline_definitions.json)).
  One prompt, untuned, so this bounds nothing about prompted models in general.
- **Who writes.** Definitions written by a model from a different family than
  the labeller gain in the same direction but recover only about a fifth of the
  gain ([`results/description_source.json`](../results/description_source.json)).
  Whether the rest is better writing or a shared bias is not separated.
- **Languages.** Three description languages and two document languages:
  sixteen of the twenty-one original NACE descriptions were in Turkish, the
  control rendered them in German, and the crossed design added English on both
  sides. No contrast among them approached the content effect, which is the
  scope of the claim and no wider — nothing here speaks to a language outside
  those three or to a script the encoder was not trained on.
- **The diagnostic's own limits.** It ranks; it does not forecast. The sign of
  the label-free estimate called the direction for only 58% of classes, so it
  says which classes gain more and not whether a given one gains at all. It
  cannot tell you the accuracy you will reach, because that needs labels and it
  reads none. And it rests on the pseudo-assignment being better than chance: on
  a taxonomy whose classes the encoder cannot separate at all, the input to
  every number above is noise, and the report will look exactly as confident.

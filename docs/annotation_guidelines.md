# Annotation guidelines — NACE Rev. 2 sections

The rules used to label `results/annotation_queue.csv`. They exist so a second
annotator produces comparable labels, and so a reviewer can see that the
recurring German trade register patterns were handled by a rule rather than
case by case.

**Provenance warning.** The labels in the queue were produced by a language
model (Claude, Anthropic) reading the German text and applying these rules, not by a human expert.
They are *silver* labels. Section 5 covers what has to happen before they are
reported as anything else.

---

## 1. The question

> Which NACE Rev. 2 **section** (A–U) best describes this company's **own**
> principal activity?

Section, not division: `M`, not `M69.20`. Principal, not exhaustive — a
Handelsregister purpose lists everything the company is permitted to do, and
most of that never happens.

## 2. Finding the principal activity

1. **Take the first substantive clause.** German purposes are ordered, and the
   first item is the actual business. Numbered lists put it in `(1)`.
2. **Ignore the closing boilerplate.** *"sowie alle damit
   zusammenhängenden Geschäfte"*, *"Die Gesellschaft ist zu allen Maßnahmen
   berechtigt…"*, *"Sie kann Zweigniederlassungen errichten"* — these appear in
   most entries and carry no information.
3. **Verb before noun.** *Herstellung* (manufacture) → C. *Handel* (trade) → G.
   *Beratung* (consulting) → M. *Vermittlung* (brokerage) → the section of what
   is being brokered, usually G or K.
4. **When manufacture and trade both appear**, the one named first wins.
   *"Entwicklung, Herstellung und Vertrieb"* → C. *"Handel und Vertrieb"* → G,
   even if the goods are manufactured products.

## 3. Recurring patterns and their fixed rulings

These cover roughly a third of the corpus. Apply them mechanically.

| Pattern in the German text | Section | Reasoning |
| --- | --- | --- |
| *Übernahme der persönlichen Haftung und Geschäftsführung* in a KG (Komplementär-GmbH) | **M** | Its own activity is managing a company — head office activities, NACE 70.10 |
| *Erwerb und Verwaltung von Beteiligungen* with no management duty | **K** | Pure holding, NACE 64.20 |
| *Verwaltung eigenen Vermögens* (no real estate named) | **K** | Own-asset management, NACE 64.20 |
| *Verwaltung eigenen Vermögens, insbesondere Grundvermögen* | **L** | Real estate, NACE 68.20 |
| *Hausverwaltung*, *Verwaltung von Wohn- und Gewerbeimmobilien* | **L** | NACE 68.32 |
| *Bauträger* — developing projects to sell | **F** | NACE 41.10, development of building projects |
| *An- und Verkauf von Grundstücken* for own account | **L** | NACE 68.10 |
| *Garten- und Landschaftsbau* | **N** | NACE 81.30, landscape service activities — not agriculture |
| *Baumschule*, *Ackerbau*, *Viehzucht*, *Forstarbeiten* | **A** | Primary production and support to it |
| *Elektroinstallation*, *Sanitär-, Heizungs-, Lüftungsbau* | **F** | Building installation, NACE 43.2 |
| *Gebäudereinigung*, *Glasreinigung*, *Winterdienst* | **N** | NACE 81.2 |
| *Facility Management*, *Gebäudemanagement* | **N** | NACE 81.10 |
| *Arbeitnehmerüberlassung* (AÜG) | **N** | NACE 78.20 |
| *Diskothek*, *Tanzlokal* | **I** | German WZ places these in 56.30, beverage serving |
| *Spedition*, *Güterkraftverkehr*, *Containerdienst mit Transport* | **H** | NACE 49.41 |
| *Sammlung und Sortierung von Wertstoffen*, *Recycling* | **E** | NACE 38 |
| *Handel mit Schrott* with no processing | **G** | NACE 46.77 |
| *Softwareentwicklung*, *IT-Beratung*, *Systemintegration* | **J** | NACE 62 |
| *Handel mit Computern und Software* | **G** | Trade, not IT services |
| *Verlag*, *Herausgabe von Zeitschriften*, *Film- und TV-Produktion* | **J** | NACE 58–60 |
| *Werbeagentur*, *Unternehmensberatung*, *Ingenieurbüro*, *Steuer- und Lohnbuchhaltung* | **M** | NACE 69–74 |
| *Versicherungsmakler*, *Anlagevermittlung*, *Finanzdienstleistungen* | **K** | NACE 66 |
| *Pflegedienst*, *Alten- und Pflegeheim*, *Arztpraxis*, *MVZ* | **Q** | NACE 86–88 |
| *Fahrschule*, *Schulungen und Seminare*, *Business School* | **P** | NACE 85 |
| *Fitness-, Wellness- und Kosmetikstudio* | **S** | NACE 96.04, physical well-being |
| *Religions- und Missionsgesellschaft* | **S** | NACE 94.91 |
| *Spielautomaten*, *Sportleragentur* | **R** | NACE 92, 93.19 |

## 4. The three boundaries that actually cause disagreement

A blind second-annotator pass over 50 documents produced κ = 0.542 — agreement
was 80% where the first pass was confident and 36% where it was not. Almost
every disagreement fell into one of three boundaries. These rules resolve them.
Read this section before annotating; the blind figure measures how intuitive
the task is, not how reproducible the guideline is.

### 4.1 Holding shells: K or M?

NACE draws the line at whether the unit *manages*:

- **64.20 Activities of holding companies (K)** — "holding assets… the units in
  this class do not provide any other service to the enterprises in which the
  equity is held, i.e. they do not administer or manage other units."
- **70.10 Activities of head offices (M)** — "overseeing and managing other
  units of the company".

A German Komplementär-GmbH states *Übernahme der persönlichen Haftung **und
Geschäftsführung***. It manages, by its own stated purpose. → **M**.

Pure *Erwerb und Verwaltung von Beteiligungen* with no management duty → **K**.
Being a *persönlich haftende Gesellschafterin* counts as managing: under German
law the Komplementär conducts the business.

This is a convention, not a fact — the other reading is defensible, and about
13% of the corpus hangs on it. Whichever is chosen must be stated in the paper.
This guideline chooses M.

### 4.2 IT and media: J or M?

**The deliverable decides, not the subject matter.**

- The thing sold is software, a system, a platform, or a media product → **J**
- The thing sold is advice, strategy, marketing or organisation — even about
  digital things → **M**

| Text | Section |
| --- | --- |
| *Vermarktung von Software* + sales and marketing services | M — the deliverable is marketing |
| *Strategieberatung im digitalen Bereich* | M — the deliverable is advice |
| Consulting **and** *Entwicklung und Vertrieb von Software* | J — software is a named deliverable |
| *Betrieb von Internetplattformen* | J — operating a platform |
| *Bewegtbild-/audiovisuelle Produktion* | J — a media product |
| *Datenschutzberatung*, external DPO | M — a professional service |
| *Herstellung von Druckerzeugnissen* (a Druckhaus) | C — printing is manufacturing |

### 4.3 Containerdienst: E or H?

- *Containerdienst* named first, or with *Entsorgung* / *Abfall* → **E** (38.11)
- Named alongside *Spedition*, *Fuhrbetrieb* or *Güterkraftverkehr* → **H**
- Named after construction work (*Erd- und Abbrucharbeiten*) → **F**

### 4.4 Remaining tie-breaks

- **Two sections equally defensible** → the one the *first* clause supports, and
  record the case as low confidence. The confidence flag is well calibrated:
  80% agreement where it says high, 36% where it says low.
- **Discotheque** → **I** when named with gastronomy (*gastronomische Betriebe*,
  *Erlebnisgastronomie*); **R** when named with events and entertainment.
- **Equestrian centre** (*Pferdepension und Reitschule*) → **R** (93.19), not P.
- **Kindergarten** — *Bau und Verwaltung* of the buildings only → **L**;
  *Bewirtschaftung* (operating them) → **Q** (88.91, child day-care).
- **Purpose field carries no activity at all** → still label it by whatever
  survives, and mark it low confidence. Skipping the hard cases is what makes
  an evaluation set easy.

## 5. Before these labels are reported as ground truth

They are model-produced. The protocol is the standard two-stage one:

1. **Pilot pass (done).** The author, who does not read German, labelled 50
   documents blind from OPUS-MT English translations, without this guideline.
   κ = 0.542, raw agreement 58%.
2. **Adjudication (done).** The disagreements were grouped, the three recurring
   boundaries above were written down, and 11 labels changed across the whole
   set — not only inside the pilot sample — so the guideline is applied
   uniformly. `adjudication_note` in the queue records each change and why.
3. **Measurement pass (done).** The author again, from translations: κ = 0.772
   and raw agreement 80% on a fresh sample (`results/verification_report.json`). A **fresh** sample, annotated after reading
   §4, is what produces the number to publish. Measuring again on the pilot
   sample would score the labels on the documents they were just tuned to.

```bash
make verify-new    # draw a fresh sample, excluding the pilot documents
make verify        # score it once filled in
```

4. **Report both figures in the paper**, in these terms: *"Section labels were
   produced by a language model applying a written annotation guideline
   (Appendix X). A blind human pass agreed on 58% of a 50-document pilot
   (κ = 0.542); after adjudication and guideline revision, agreement on a fresh
   50-document sample was N% (κ = …)."* The first number is honest about how
   hard the task is; the second is the one that describes the released labels.

If the second κ is still below 0.6, the guideline is still wrong somewhere —
fix it before touching the labels again.

5. **A second annotator who reads German (prepared, not done).** Agreement so
   far is human-versus-model, and the human worked from translations.
   `make second-annotator` writes the 50 measurement documents in German only,
   shuffled, with no translation, suggestion or earlier answer;
   `make second-annotator-score` reports human-versus-human agreement once they
   are filled in. Instructions: [`second_annotator.md`](second_annotator.md).

# Annotation guidelines — NACE Rev. 2 sections

The rules used to label `results/annotation_queue.csv`. They exist so a second
annotator produces comparable labels, and so a reviewer can see that the
recurring German trade register patterns were handled by a rule rather than
case by case.

**Provenance warning.** The labels in the queue were produced by a language
model reading the German text and applying these rules, not by a human expert.
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

## 4. Tie-breaking

- **Two sections equally defensible** → pick the one the *first* clause supports,
  and record the case as low confidence.
- **A shell company whose KG has a named business** → label the shell by its own
  activity (M or K per the table above), not by the KG's business. Statistical
  offices do the same; the alternative convention exists, and switching to it
  changes roughly 13% of this sample. Whichever is chosen has to be stated in
  the paper.
- **Purpose field carries no activity at all** (pure boilerplate) → still label
  it by whatever survives, and mark it low confidence. Do not skip it: skipping
  the hard cases is what makes an evaluation set easy.

## 5. Before these labels are reported as ground truth

They are model-produced. To use them in a paper:

1. **Verify a sample by hand.** `results/verification_sample.csv` holds 50
   documents — a random draw plus the cases the labeller marked low confidence
   — with English translations. Correct them in `tools/annotate.html`.
2. **Measure agreement** between the human pass and the silver labels
   (`python -m experiments.verify_labels`). Report Cohen's κ.
3. **Describe the procedure in the paper's data section**, in these terms:
   *"Section labels were produced by a language model applying a written
   annotation guideline, and validated on a randomly drawn subset of N
   documents annotated by the author (κ = X)."* That is an accepted method
   when it is declared. Presenting silver labels as gold is not.
4. If κ is low, the guideline is what gets fixed — not the labels.

#!/usr/bin/env python3
"""Does the description effect depend on how fine the taxonomy is?

    python -m experiments.run_hierarchy

Every taxonomy tested in this paper is flat, and the ones practitioners use are
not: NACE divides sections into divisions, groups and classes, and official
coding happens four digits down. Whether writing definitions pays the same at
every level is therefore a practical question, and 20 Newsgroups can answer part
of it without a single new label: its twenty classes group into six by
construction (``comp.*``, ``rec.*``, ``sci.*``, politics, religion, for sale).

The same three conditions run at both levels -- raw identifier, readable name,
definition -- on the same documents. If the effect is about the vocabulary a
class text carries rather than about the number of classes, the step from
identifier to readable name should pay at both levels, and the step to a
definition should pay where the names are furthest from the documents' own
words.

Not registered; it enters no corrected family.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

import numpy as np

from experiments.config import RESULTS_DIR, ROOT, ensure_dirs, set_seed
from experiments.metrics import mcnemar_exact
from experiments.run_dose_response import embed
from experiments.run_replication import DEFINITION, READABLE, load_newsgroups

RESULT = RESULTS_DIR / "hierarchy.json"

GROUP_OF = {
    "comp.graphics": "comp", "comp.os.ms-windows.misc": "comp",
    "comp.sys.ibm.pc.hardware": "comp", "comp.sys.mac.hardware": "comp",
    "comp.windows.x": "comp",
    "rec.autos": "rec", "rec.motorcycles": "rec",
    "rec.sport.baseball": "rec", "rec.sport.hockey": "rec",
    "sci.crypt": "sci", "sci.electronics": "sci", "sci.med": "sci", "sci.space": "sci",
    "talk.politics.guns": "politics", "talk.politics.mideast": "politics",
    "talk.politics.misc": "politics",
    "alt.atheism": "religion", "soc.religion.christian": "religion",
    "talk.religion.misc": "religion",
    "misc.forsale": "forsale",
}

GROUP_READABLE = {
    "comp": "Computing",
    "rec": "Recreation and sport",
    "sci": "Science",
    "politics": "Politics",
    "religion": "Religion",
    "forsale": "Items for sale",
}

GROUP_DEFINITION = {
    "comp": "Computing. Operating systems and drivers, PC and Macintosh hardware, graphics and image formats, windowing systems, networks and software tools.",
    "rec": "Recreation and sport. Cars and motorcycles, riding and maintenance, baseball and ice hockey, teams, players, games and scores.",
    "sci": "Science. Cryptography and privacy, electronics and circuits, medicine and treatment, space flight and astronomy, research and evidence.",
    "politics": "Politics. Gun control and firearms law, Middle East conflicts, government and elections, civil rights, foreign policy and public debate.",
    "religion": "Religion. God and faith, scripture and its interpretation, Christianity and the church, atheism and criticism of belief, morality.",
    "forsale": "Items for sale. Classified advertisements offering goods for sale or wanted, asking prices, shipping and condition, second-hand equipment.",
}


def conditions(level: str, names: List[str]) -> Dict[str, Dict[str, str]]:
    if level == "fine":
        return {"raw class identifier": {n: n for n in names},
                "readable class name": {n: READABLE[n] for n in names},
                "definition of what the class covers": {n: DEFINITION[n] for n in names}}
    return {"raw class identifier": {n: n for n in names},
            "readable class name": {n: GROUP_READABLE[n] for n in names},
            "definition of what the class covers": {n: GROUP_DEFINITION[n] for n in names}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=2000)
    args = parser.parse_args()
    set_seed()
    ensure_dirs()

    texts, fine_labels, fine_names = load_newsgroups("test", args.sample)
    docs = embed(texts)
    levels = {
        "fine (20 classes)": (fine_names, fine_labels),
        "coarse (6 groups)": (sorted(set(GROUP_OF.values())), [GROUP_OF[name] for name in fine_labels]),
    }

    report = {}
    for level, (names, gold) in levels.items():
        key = "fine" if level.startswith("fine") else "coarse"
        hits, accuracy = {}, {}
        for condition, mapping in conditions(key, names).items():
            vectors = embed([mapping[n] for n in names])
            predicted = [names[i] for i in np.argmax(docs @ vectors.T, axis=1)]
            hits[condition] = [g == p for g, p in zip(gold, predicted)]
            accuracy[condition] = float(np.mean(hits[condition]))
            print(f"  {level:20s} {condition:38s} {accuracy[condition]:.1%}", flush=True)
        order = list(accuracy)
        spelling = mcnemar_exact(hits[order[1]], hits[order[0]])
        spelling["gain_pp"] = 100 * (accuracy[order[1]] - accuracy[order[0]])
        defining = mcnemar_exact(hits[order[2]], hits[order[1]])
        defining["gain_pp"] = 100 * (accuracy[order[2]] - accuracy[order[1]])
        report[level] = {"classes": len(names), "n": len(gold), "top1_accuracy": accuracy,
                         "spelling_the_label_out": spelling, "defining_the_class": defining}

    payload = {
        "corpus": "20 Newsgroups, test split",
        "grouping": GROUP_OF,
        "note": "The same documents at two levels of one taxonomy. Not registered and no part of "
                "the corrected family.",
        "levels": report,
    }
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for level, r in report.items():
        print(f"  {level}: identifier -> name {r['spelling_the_label_out']['gain_pp']:+.1f} pp, "
              f"name -> definition {r['defining_the_class']['gain_pp']:+.1f} pp")
    print(f"Wrote {RESULT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

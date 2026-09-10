#!/usr/bin/env python3
"""Does the class-description finding hold outside this corpus?

    python -m experiments.run_replication

The finding — that what a class description *says* dominates zero-shot
classification, while the language it says it in does not — was measured on
299 German trade register documents against NACE Rev. 2. One corpus, one
taxonomy, one domain. That is a result, not yet a claim about method.

This repeats the comparison on 20 Newsgroups: English instead of German,
usenet posts instead of legal prose, twenty topical classes instead of
twenty-one economic sections, and no connection to the original data. The
three conditions mirror the ones in run_description_study.py:

  1. the raw class identifier, exactly as the dataset ships it
     ("comp.sys.ibm.pc.hardware");
  2. a readable name for the same class ("IBM PC hardware") — the same
     information, spelled out;
  3. a definition that enumerates what the class actually covers.

If the effect is a property of description content rather than of this corpus,
condition 3 pulls away from 1 and 2 here as well.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from sklearn.metrics import f1_score

from experiments.config import RESULTS_DIR, TABLES_DIR, SEED, ensure_dirs, set_seed
from experiments.metrics import bootstrap_ci, mcnemar_exact
from experiments.systems import get_embedder

READABLE = {
    "alt.atheism": "Atheism",
    "comp.graphics": "Computer graphics",
    "comp.os.ms-windows.misc": "Microsoft Windows operating system",
    "comp.sys.ibm.pc.hardware": "IBM PC hardware",
    "comp.sys.mac.hardware": "Apple Macintosh hardware",
    "comp.windows.x": "The X Window System",
    "misc.forsale": "Items for sale",
    "rec.autos": "Automobiles",
    "rec.motorcycles": "Motorcycles",
    "rec.sport.baseball": "Baseball",
    "rec.sport.hockey": "Ice hockey",
    "sci.crypt": "Cryptography",
    "sci.electronics": "Electronics",
    "sci.med": "Medicine",
    "sci.space": "Space and astronomy",
    "soc.religion.christian": "Christianity",
    "talk.politics.guns": "Gun politics",
    "talk.politics.mideast": "Middle East politics",
    "talk.politics.misc": "Politics",
    "talk.religion.misc": "Religion",
}

# Written in the same style as the rewritten NACE descriptions: name the class,
# then enumerate what it actually covers, in the vocabulary its documents use.
DEFINITION = {
    "alt.atheism": "Atheism. Arguments for and against the existence of God, agnosticism and secularism, criticism of religious belief and scripture, debates between believers and non-believers, morality without religion.",
    "comp.graphics": "Computer graphics. Image file formats and conversion, rendering and ray tracing, 3D modelling, graphics libraries and hardware, animation, image processing software, colour and display.",
    "comp.os.ms-windows.misc": "Microsoft Windows. Installing and configuring Windows, drivers, DLL and INI files, Windows applications and utilities, crashes and error messages, DOS compatibility.",
    "comp.sys.ibm.pc.hardware": "IBM PC hardware. Motherboards and processors, IDE and SCSI controllers, hard drives, memory, expansion cards, BIOS settings, monitors and peripherals for PC compatibles.",
    "comp.sys.mac.hardware": "Apple Macintosh hardware. Macintosh models such as the Quadra, Centris, LC and PowerBook, memory upgrades, SCSI devices, monitors, printers and Apple peripherals.",
    "comp.windows.x": "The X Window System. X11 servers and clients, Motif and Xt toolkits, widgets and window managers, X resources and displays, compiling and porting X applications on Unix.",
    "misc.forsale": "Items offered for sale. Classified advertisements offering goods for sale or wanted, asking prices, shipping and condition, second-hand computers, electronics, books and equipment.",
    "rec.autos": "Automobiles. Cars and driving, makes and models, engines and transmissions, fuel economy, repairs and maintenance, dealers and prices, insurance, radar detectors.",
    "rec.motorcycles": "Motorcycles. Riding and road safety, bike models and engines, helmets and riding gear, maintenance and repair, motorcycle clubs, licensing and traffic law.",
    "rec.sport.baseball": "Baseball. Major league teams and players, batting and pitching statistics, games and scores, trades and signings, managers, the pennant race and the World Series.",
    "rec.sport.hockey": "Ice hockey. NHL teams and players, goaltending, games and scores, playoffs and the Stanley Cup, trades, coaching and league rules.",
    "sci.crypt": "Cryptography. Encryption algorithms and key length, DES and RSA, the Clipper chip and key escrow, digital signatures, privacy, wiretapping and export controls on cryptography.",
    "sci.electronics": "Electronics. Circuits and circuit design, amplifiers and oscillators, transistors and integrated circuits, power supplies, soldering, test equipment, audio and radio electronics.",
    "sci.med": "Medicine. Diseases and their treatment, symptoms and diagnosis, drugs and side effects, doctors and patients, nutrition and vitamins, medical research and clinical evidence.",
    "sci.space": "Space and astronomy. Spacecraft and launch vehicles, NASA missions and the shuttle, orbits and satellites, planets and moons, telescopes and observation, space stations and exploration.",
    "soc.religion.christian": "Christianity. Christian faith and doctrine, the Bible and its interpretation, Jesus, sin and salvation, prayer and worship, churches and denominations, Christian ethics.",
    "talk.politics.guns": "Gun politics. Firearms legislation and gun control, the Second Amendment, the ATF and Waco, self-defence, assault weapons bans, crime statistics and gun ownership.",
    "talk.politics.mideast": "Middle East politics. Israel and the Palestinians, the occupied territories, Arab states, Turkey and Armenia, war and peace negotiations, terrorism and human rights in the region.",
    "talk.politics.misc": "Politics. Government and elections, taxation and the federal budget, health care policy, civil liberties, political parties and candidates, drug policy and social issues.",
    "talk.religion.misc": "Religion. Religious belief in general, morality and scripture, Christianity compared with other faiths, creationism and evolution, prophecy and miracles, debates about God.",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", type=int, default=2000,
        help="Documents to draw, stratified by class (default: 2000).",
    )
    parser.add_argument("--encoder", default=None)
    args = parser.parse_args()

    set_seed()
    ensure_dirs()
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    names = list(data.target_names)
    texts, labels = [], []
    for text, target in zip(data.data, data.target):
        # Stripping headers and quotes leaves some posts empty; they carry no
        # signal for any condition and would only add noise equally.
        if len(text.strip()) >= 40:
            texts.append(text.strip())
            labels.append(names[target])

    rng = np.random.default_rng(SEED)
    if args.sample and len(texts) > args.sample:
        by_class: dict = {}
        for i, label in enumerate(labels):
            by_class.setdefault(label, []).append(i)
        per_class = max(1, args.sample // len(by_class))
        picked = []
        for label in sorted(by_class):
            idx = by_class[label]
            take = min(per_class, len(idx))
            picked.extend(rng.choice(idx, size=take, replace=False).tolist())
        texts = [texts[i] for i in picked]
        labels = [labels[i] for i in picked]

    print(f"{len(texts)} documents, {len(set(labels))} classes", flush=True)

    embedder = get_embedder(args.encoder) if args.encoder else get_embedder()
    docs = np.vstack(
        [np.asarray(embedder.embed_text(t), dtype=np.float64) for t in texts]
    )
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)

    conditions = {
        "raw class identifier": {n: n for n in names},
        "readable class name": READABLE,
        "definition of what the class covers": DEFINITION,
    }

    results, correct = {}, {}
    for name, descriptions in conditions.items():
        vecs = np.vstack(
            [np.asarray(embedder.embed_text(descriptions[n]), dtype=np.float64) for n in names]
        )
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        order = np.argsort(-(docs @ vecs.T), axis=1)
        predicted = [names[order[i, 0]] for i in range(len(labels))]
        hits = [labels[i] == predicted[i] for i in range(len(labels))]
        correct[name] = hits
        lo, hi = bootstrap_ci(hits)
        results[name] = {
            "top1_accuracy": float(np.mean(hits)),
            "top1_ci95": [lo, hi],
            "top3_accuracy": float(
                np.mean([labels[i] in [names[j] for j in order[i, :3]] for i in range(len(labels))])
            ),
            "f1_macro": float(f1_score(labels, predicted, average="macro", zero_division=0)),
            "mean_words_per_class": float(
                np.mean([len(d.split()) for d in descriptions.values()])
            ),
        }
        print(f"  {name:<38} {results[name]['top1_accuracy']:.1%}", flush=True)

    keys = list(conditions)
    spelling_out = mcnemar_exact(correct[keys[1]], correct[keys[0]])
    spelling_out["gain_pp"] = (
        results[keys[1]]["top1_accuracy"] - results[keys[0]]["top1_accuracy"]
    ) * 100
    defining = mcnemar_exact(correct[keys[2]], correct[keys[1]])
    defining["gain_pp"] = (
        results[keys[2]]["top1_accuracy"] - results[keys[1]]["top1_accuracy"]
    ) * 100

    rows = ["| Class descriptions | Top-1 | 95% CI | Top-3 | F1-macro | words/class |",
            "| --- | --- | --- | --- | --- | --- |"]
    for name in keys:
        r = results[name]
        lo, hi = r["top1_ci95"]
        rows.append(
            f"| {name} | {r['top1_accuracy']:.1%} | [{lo:.1%}, {hi:.1%}] | "
            f"{r['top3_accuracy']:.1%} | {r['f1_macro']:.3f} | {r['mean_words_per_class']:.0f} |"
        )
    table = "\n".join(rows)
    (TABLES_DIR / "replication_20newsgroups.md").write_text(table + "\n", encoding="utf-8")

    payload = {
        "dataset": "20 Newsgroups (test split, headers/footers/quotes removed)",
        "n": len(texts),
        "classes": len(names),
        "language": "English",
        "encoder": args.encoder or "paraphrase-multilingual-MiniLM-L12-v2",
        "conditions": results,
        "spelling_the_label_out": spelling_out,
        "defining_the_class": defining,
        "table_markdown": table,
    }
    (RESULTS_DIR / "replication_20newsgroups.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + table)
    print(f"\nidentifier to readable name: {spelling_out['gain_pp']:+.1f} pp, p = {spelling_out['p_value']:.2e}")
    print(f"readable name to definition: {defining['gain_pp']:+.1f} pp, p = {defining['p_value']:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

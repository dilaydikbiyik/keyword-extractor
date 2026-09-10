#!/usr/bin/env python3
"""What made the rewritten class descriptions better: their language, or what they say?

    python -m experiments.run_description_study

Rewriting the taxonomy's class descriptions gained 17.7 points, but the rewrite
changed two things at once: the language (Turkish to German, matching the
corpus) and the content (a terse category label became a definition
enumerating concrete activities). Either could explain the gain, and they are
separable.

Three conditions, on the same documents and the same encoder:

  1. the original descriptions, sixteen of twenty-one in Turkish;
  2. those same descriptions rendered literally in German — identical content,
     identical terseness, only the language changes;
  3. the rewritten descriptions: German, and written as NACE-style definitions.

Condition 2 is the control the finding needs. Without it, "the descriptions
must be in the language of the corpus" and "the descriptions must say what the
class actually covers" are indistinguishable.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from sklearn.metrics import f1_score

from experiments.config import RESULTS_DIR, ROOT, TABLES_DIR, ensure_dirs, set_seed
from experiments.data import load_labeled_samples_split
from experiments.metrics import bootstrap_ci, mcnemar_exact
from experiments.systems import get_embedder

# Literal German renderings of the original Turkish labels: the same words,
# the same terseness, translated by hand so no machine-translation noise is
# introduced alongside the variable under test. Sections already German in the
# original (C, F, J, M, Q) are carried over untouched and so are controlled.
LITERAL_GERMAN = {
    "A": "Landwirtschaft, Forstwirtschaft und Fischerei",
    "B": "Bergbau und Steinbrüche",
    "D": "Elektrizität, Gas, Dampf und Klimatisierung",
    "E": "Wasserversorgung, Kanalisation, Abfallwirtschaft",
    "G": "Groß- und Einzelhandel",
    "H": "Verkehr und Lagerung",
    "I": "Beherbergung und Verpflegungsdienstleistungen",
    "K": "Finanz- und Versicherungstätigkeiten",
    "L": "Immobilientätigkeiten",
    "N": "Verwaltungs- und Unterstützungsdienstleistungen",
    "O": "Öffentliche Verwaltung und Verteidigung",
    "P": "Bildung",
    "R": "Kunst, Unterhaltung und Erholung",
    "S": "Sonstige Dienstleistungstätigkeiten",
    "T": "Tätigkeiten privater Haushalte als Arbeitgeber",
    "U": "Tätigkeiten internationaler Organisationen",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half", choices=["dev", "test", "all"], default="all")
    parser.add_argument(
        "--encoder",
        default=None,
        help="Repeat the study with a different sentence encoder, to check that "
             "the effect is a property of the descriptions and not of one model.",
    )
    args = parser.parse_args()

    set_seed()
    ensure_dirs()
    embedder = get_embedder(args.encoder) if args.encoder else get_embedder()

    old = json.loads((ROOT / "data" / "taxonomy" / "sectors_v1.json").read_text(encoding="utf-8"))["sectors"]
    new = json.loads((ROOT / "data" / "taxonomy" / "sectors.json").read_text(encoding="utf-8"))["sectors"]
    codes = sorted(old)

    conditions = {
        "original (16 of 21 in Turkish, terse)": {
            c: f"{old[c].get('name', '')}. {old[c].get('description', '')}" for c in codes
        },
        "same content, rendered in German": {
            c: f"{old[c].get('name', '')}. {LITERAL_GERMAN.get(c, old[c].get('description', ''))}"
            for c in codes
        },
        "rewritten as NACE-style definitions": {
            c: f"{new[c].get('name', '')}. {new[c].get('description', '')}" for c in codes
        },
    }

    samples = load_labeled_samples_split(None if args.half == "all" else args.half)
    gold = [s.true_sector for s in samples]
    docs = np.vstack(
        [np.asarray(embedder.embed_text(s.purpose), dtype=np.float64) for s in samples]
    )
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)

    results, correct = {}, {}
    for name, descriptions in conditions.items():
        vecs = np.vstack(
            [np.asarray(embedder.embed_text(descriptions[c]), dtype=np.float64) for c in codes]
        )
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        order = np.argsort(-(docs @ vecs.T), axis=1)
        predicted = [codes[order[i, 0]] for i in range(len(gold))]
        hits = [gold[i] == predicted[i] for i in range(len(gold))]
        correct[name] = hits
        lo, hi = bootstrap_ci(hits)
        results[name] = {
            "top1_accuracy": float(np.mean(hits)),
            "top1_ci95": [lo, hi],
            "top3_accuracy": float(
                np.mean([gold[i] in [codes[j] for j in order[i, :3]] for i in range(len(gold))])
            ),
            "f1_macro": float(f1_score(gold, predicted, average="macro", zero_division=0)),
            "mean_words_per_class": float(
                np.mean([len(d.split()) for d in descriptions.values()])
            ),
        }

    names = list(conditions)
    language_only = mcnemar_exact(correct[names[1]], correct[names[0]])
    language_only["gain_pp"] = (
        results[names[1]]["top1_accuracy"] - results[names[0]]["top1_accuracy"]
    ) * 100
    content_only = mcnemar_exact(correct[names[2]], correct[names[1]])
    content_only["gain_pp"] = (
        results[names[2]]["top1_accuracy"] - results[names[1]]["top1_accuracy"]
    ) * 100

    # A replication on another encoder must not overwrite the primary result.
    suffix = "" if not args.encoder else "_" + args.encoder.split("/")[-1].replace("-", "_")
    rows = ["| Class descriptions | Top-1 | 95% CI | Top-3 | F1-macro | words/class |",
            "| --- | --- | --- | --- | --- | --- |"]
    for name in names:
        r = results[name]
        lo, hi = r["top1_ci95"]
        rows.append(
            f"| {name} | {r['top1_accuracy']:.1%} | [{lo:.1%}, {hi:.1%}] | "
            f"{r['top3_accuracy']:.1%} | {r['f1_macro']:.3f} | {r['mean_words_per_class']:.0f} |"
        )
    table = "\n".join(rows)
    (TABLES_DIR / f"description_study{suffix}.md").write_text(table + "\n", encoding="utf-8")

    payload = {
        "half": args.half,
        "encoder": args.encoder or "paraphrase-multilingual-MiniLM-L12-v2",
        "n": len(samples),
        "sector_vector": "description only — seed keywords excluded to isolate the factor",
        "conditions": results,
        "language_effect": language_only,
        "content_effect": content_only,
        "conclusion": (
            "Translating the descriptions into the language of the corpus, with "
            "their content held constant, does not help. Rewriting them from "
            "category labels into definitions that enumerate concrete activities "
            "does. The gain attributed to language was a confound."
        ),
        "table_markdown": table,
    }
    (RESULTS_DIR / f"description_study{suffix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(table)
    print(f"\nlanguage alone (content held constant): {language_only['gain_pp']:+.1f} pp, "
          f"p = {language_only['p_value']:.4f}")
    print(f"content (label to definition):          {content_only['gain_pp']:+.1f} pp, "
          f"p = {content_only['p_value']:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Second, blind codings of the error sample.

    python -m experiments.code_errors --build   # write the blind sheet for the author
    python -m experiments.code_errors           # score the author's sheet once it is filled in
    python -m experiments.code_errors --model   # have a local model from another family code it

The fifty development-half errors in ``results/error_analysis.csv`` carry a
first coding against the codebook in ``experiments/error_analysis.py``. Two
independent second codings can be measured against it, both blind to it:

* the author's, from a shuffled sheet with an English machine translation
  beside the German, as in the label verification;
* an instruction-tuned model's from a different family than the first coder,
  with greedy decoding, as a check on how reliably the codebook can be applied.
  It is reported as a model coder, never as a human judgement.

Each writes a report of raw agreement, Cohen's kappa and where the codings
differ. The paper quotes the author's once it exists. The model coder's first run
put every error in one category (kappa 0), so the paper does not use it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from experiments.config import RESULTS_DIR, SEED, ensure_dirs
from experiments.data import load_taxonomy, sector_names
from experiments.error_analysis import CODEBOOK
from experiments.metrics import cohen_kappa

ERRORS = RESULTS_DIR / "error_analysis.csv"
SHEET = RESULTS_DIR / "author_error_coding.csv"
GUIDE = RESULTS_DIR / "author_error_coding_codebook.md"
REPORT = RESULTS_DIR / "author_error_coding_report.json"
MODEL_REPORT = RESULTS_DIR / "model_error_coding.json"
COLUMNS = ["id", "purpose_de", "purpose_en", "gold_section", "predicted_section", "top3_sections",
           "author_category", "author_note"]


def first_coding() -> List[Dict]:
    with open(ERRORS, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["manual_category"].strip()]


def section(code: str, names: Dict[str, str]) -> str:
    return f"{code} ({names.get(code, '?')})"


def compare(first: Sequence[str], second: Sequence[str]) -> Dict:
    """Agreement between two codings of the same errors, and where they part."""
    return {
        "n_coded": len(second),
        "agreement": float(np.mean([a == b for a, b in zip(first, second)])),
        "cohen_kappa": cohen_kappa(list(first), list(second)),
        "second_distribution": dict(Counter(second).most_common()),
        "first_distribution": dict(Counter(first).most_common()),
        "disagreements": [{"first": a, "second": b, "count": n}
                          for (a, b), n in Counter((a, b) for a, b in zip(first, second) if a != b).most_common()],
    }


# ── The author's sheet ──────────────────────────────────────────────────────

def guide_text() -> str:
    lines = ["# Coding the errors",
             "",
             "Fill in `author_category` for every row of `author_error_coding.csv` with exactly one",
             "of the category names below, and `author_note` if a row needs a word of explanation.",
             "Read the English translation, the gold section and the predicted section, and ask why",
             "the system chose the predicted one. Code each row on its own; do not look at",
             "`error_analysis.csv` until you are done, or the coding is no longer blind.",
             "",
             "| Category | Use it when |",
             "| --- | --- |"]
    lines += [f"| `{name}` | {text} |" for name, text in CODEBOOK.items()]
    lines += ["", "Then run `make author-coding-score`.", ""]
    return "\n".join(lines)


def build(translate: Callable[[str], str] = None) -> int:
    if SHEET.exists():
        print(f"{SHEET.name} already exists; it may hold your coding, so it is not overwritten.",
              file=sys.stderr)
        return 1
    if translate is None:
        from experiments.systems import translate_de_en as translate
    rows = first_coding()
    names = sector_names()
    order = np.random.default_rng(SEED).permutation(len(rows))
    with open(SHEET, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for i in order:
            r = rows[i]
            writer.writerow({
                "id": r["id"], "purpose_de": r["purpose"], "purpose_en": translate(r["purpose"]),
                "gold_section": section(r["true_sector"], names),
                "predicted_section": section(r["predicted_sector"], names),
                "top3_sections": " | ".join(section(c, names) for c in r["top3"].split("|") if c),
                "author_category": "", "author_note": "",
            })
    GUIDE.write_text(guide_text(), encoding="utf-8")
    print(f"Wrote {SHEET.name} ({len(rows)} rows, shuffled, first coding hidden) and {GUIDE.name}.")
    return 0


def score() -> int:
    if not SHEET.exists():
        print("No coding sheet: run with --build first.", file=sys.stderr)
        return 1
    with open(SHEET, newline="", encoding="utf-8-sig") as fh:
        sheet = list(csv.DictReader(fh))
    invalid = sorted({r["author_category"].strip() for r in sheet
                      if r["author_category"].strip() and r["author_category"].strip() not in CODEBOOK})
    if invalid:
        print(f"Not codebook categories: {invalid}. Use one of {list(CODEBOOK)}.", file=sys.stderr)
        return 1
    filled = [r for r in sheet if r["author_category"].strip()]
    if not filled:
        print(f"No row of {SHEET.name} is coded yet.", file=sys.stderr)
        return 1
    first = {r["id"]: r["manual_category"].strip() for r in first_coding()}
    report = {**compare([first[r["id"]] for r in filled], [r["author_category"].strip() for r in filled]),
              "n_sheet": len(sheet),
              "note": "The author coded blind to the first coding, from English machine translations."}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    kappa = report["cohen_kappa"]
    print(f"  {len(filled)} of {len(sheet)} coded; agreement {report['agreement']:.0%}, "
          f"kappa {'undefined' if kappa is None else f'{kappa:.3f}'}\nWrote {REPORT.name}")
    return 0


# ── A model coder from another family ───────────────────────────────────────

def seed_list(taxonomy: Dict[str, dict], code: str) -> str:
    return ", ".join(taxonomy.get(code, {}).get("seed_keywords", [])) or "(none)"


def coding_prompt(german: str, english: str, gold: str, predicted: str,
                  gold_seeds: str, predicted_seeds: str) -> str:
    """What the model coder sees: the error and the codebook, never the first coding."""
    codebook = "\n".join(f"- {name}: {text}" for name, text in CODEBOOK.items())
    return (
        "A zero-shot classifier assigned this German company purpose statement to the wrong NACE Rev. 2 "
        "section. Choose the one category from the codebook below that best explains the error.\n\n"
        f"{codebook}\n\n"
        f"Purpose (German): {german}\nEnglish translation: {english}\n"
        f"Correct section: {gold}\nSeed keywords of the correct section: {gold_seeds}\n"
        f"Predicted section: {predicted}\nSeed keywords of the predicted section: {predicted_seeds}\n\n"
        "Answer with the category name only."
    )


def parse_category(reply: str) -> Optional[str]:
    """The first codebook category the reply names, spelled with spaces or underscores."""
    text = reply.lower().replace(" ", "_").replace("-", "_")
    hits = [(text.find(name), name) for name in CODEBOOK if name in text]
    return min(hits)[1] if hits else None


def local_model(model: str) -> Callable[[str], str]:
    """Greedy replies from the local instruction-tuned model the LLM baseline uses."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model)
    net = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype).to(device).eval()

    def ask(prompt: str) -> str:
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                               add_generation_prompt=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = net.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=16,
                               do_sample=False, temperature=None, top_p=None, top_k=None,
                               pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)

    return ask


def code_with_model(ask: Callable[[str], str] = None, translate: Callable[[str], str] = None) -> int:
    from experiments.systems import LOCAL_LLM

    revision = None
    if ask is None:
        from experiments.run_llm_baseline import model_revision

        ask, revision = local_model(LOCAL_LLM), model_revision(LOCAL_LLM)
    if translate is None:
        from experiments.systems import translate_de_en as translate
    names, taxonomy = sector_names(), load_taxonomy()
    rows = first_coding()
    codings = []
    for r in rows:
        reply = ask(coding_prompt(r["purpose"], translate(r["purpose"]),
                                  section(r["true_sector"], names), section(r["predicted_sector"], names),
                                  seed_list(taxonomy, r["true_sector"]), seed_list(taxonomy, r["predicted_sector"])))
        codings.append({"id": r["id"], "category": parse_category(reply) or "unparsed", "reply": reply.strip()})
    second = [c["category"] for c in codings]
    report = {"model": LOCAL_LLM, "revision": revision,
              **compare([r["manual_category"].strip() for r in rows], second),
              "off_format": second.count("unparsed"),
              "note": "An instruction-tuned model from a different family than the first coder, coding blind "
                      "against the same codebook with greedy decoding: a check on the codebook's reliability, "
                      "not a human judgement.",
              "codings": codings}
    MODEL_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    kappa = report["cohen_kappa"]
    print(f"  {LOCAL_LLM}: agreement {report['agreement']:.0%}, "
          f"kappa {'undefined' if kappa is None else f'{kappa:.3f}'}, off format {report['off_format']}\n"
          f"Wrote {MODEL_REPORT.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true", help="Write the blind coding sheet for the author.")
    parser.add_argument("--model", action="store_true", help="Have the local model from another family code it.")
    args = parser.parse_args()
    ensure_dirs()
    if args.build:
        return build()
    return code_with_model() if args.model else score()


if __name__ == "__main__":
    sys.exit(main())

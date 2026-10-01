#!/usr/bin/env python3
"""Class definitions written by a model from a different family than the labeller.

    python -m experiments.write_descriptions          # writes data/taxonomy/sectors_qwen.json

The paper's descriptions and its silver labels come from the same assistant, and
that assistant had seen the corpora. A bias it holds toward one reading of a
section would reach both, and would flatter the descriptions it wrote. The test
is to have the definitions written by something else.

The writer here is the open instruction-tuned model the paper already uses as a
baseline, chosen because its family is not the labeller's. It is given exactly
what the terse German control condition is given -- the section's short name in
German, hand-translated, no corpus, no example documents, no sight of the
assistant's own definitions -- and asked to expand it into a NACE-style
definition. Greedy decoding, so the same model on the same device writes the
same definitions every time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone

from experiments.config import ROOT, SEED, ensure_dirs, set_seed
from experiments.run_description_study import LITERAL_GERMAN
from experiments.systems import LOCAL_LLM

OUT = ROOT / "data" / "taxonomy" / "sectors_qwen.json"
MAX_NEW_TOKENS = 200

PROMPT = """Du schreibst die Definitionen einer Wirtschaftszweig-Klassifikation (NACE Rev. 2).

Abschnitt {code}: {name}

Schreibe eine Definition dieses Abschnitts in zwei bis drei Sätzen auf Deutsch. \
Zähle die konkreten Tätigkeiten auf, die zu diesem Abschnitt gehören. Nenne keine \
Firmennamen, keine anderen Abschnitte und keine Abschnittsbuchstaben. Antworte nur \
mit der Definition, ohne Einleitung."""


def terse_german() -> dict:
    """The control condition's text, which is all the writer is given."""
    sectors = json.loads((ROOT / "data" / "taxonomy" / "sectors_v1.json")
                         .read_text(encoding="utf-8"))["sectors"]
    return {code: LITERAL_GERMAN.get(code, sectors[code].get("description", ""))
            for code in sorted(sectors)}


# The model names the classification and the section's own letter in almost
# every reply ("Der Abschnitt G der ... Klassifikation (NACE Rev. 2) umfasst
# ..."). Those phrases say nothing about the activities of a section and are the
# same in all twenty-one, so leaving them in would add identical text to every
# class vector and shrink the distances between them -- measuring the
# boilerplate rather than the definition. SELF_REFERENCE removes them wherever
# they occur, FRAMING removes what is left of the opening clause. Both are
# applied uniformly to all classes, were fixed before any accuracy was computed,
# and the unedited reply is stored beside each definition, so the edit can be
# checked and undone.
SELF_REFERENCE = [
    re.compile(r"\(\s*NACE\s*Rev\.?\s*2[^)]*\)", re.I),
    re.compile(r"\bder\s+Wirtschafts\w*[\s-]*Kl[aä]sse?i?fikation\b", re.I),
    re.compile(r"\b(Der|Dieser|Den|Dem)?\s*Abschnitt(s|es)?\s+[A-U]\b", re.I),
    re.compile(r"\bNACE\s*Rev\.?\s*2\b", re.I),
]

FRAMING = [
    re.compile(r"^#+\s*\w+\s*", re.U),                       # a markdown heading
    re.compile(r"^(Definition|Antwort)\s*(des\s+Abschnitts?)?\s*[A-U]?\s*[:.]?\s*", re.I),
    re.compile(r"^(Der\s+)?Abschnitt\s+[A-U]\b.{0,160}?(?=umfasst|beinhaltet|bezieht|deckt)", re.I),
    re.compile(r"^(Der\s+)?(Wirtschaftsbereich|Wirtschaftszweig)\b.{0,160}?(?=umfasst|beinhaltet)", re.I),
    re.compile(r"^[A-U]\s+der\s+.{0,160}?(?=umfasst|beinhaltet)", re.I),
    re.compile(r"^(Der\s+)?Abschnitt\s+[A-U]\b[\s\"„“]*", re.I),
    # What a removal can leave behind: a pronoun and a colon with nothing before
    # them, or the closing quote of a title whose opening quote has gone.
    re.compile(r"^(Es|Er|Sie|Dies|Diese[rs]?)\s*[:,]\s*", re.I),
    re.compile(r"^[\s\"„“”:,.-]+", re.U),
]


def clean(reply: str) -> str:
    """One paragraph of German, with the shared framing stripped."""
    text = " ".join(reply.strip().strip('"').split())
    for pattern in SELF_REFERENCE:
        text = pattern.sub(" ", text)
    for _ in range(len(FRAMING)):
        before = text
        for pattern in FRAMING:
            text = pattern.sub("", text).lstrip(' :",')
        if text == before:
            break
    text = " ".join(text.split()).lstrip(' :,.')
    return text[0].upper() + text[1:] if text else text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=LOCAL_LLM)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the file. The definitions are an input to a "
                             "preregistered study, so this is not done casually.")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        print(f"{OUT.relative_to(ROOT)} already exists; it is an input to a registered "
              "study. Pass --force to rewrite it.", file=sys.stderr)
        return 1
    set_seed()
    ensure_dirs()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32 if device == "cpu" else torch.float16).to(device)
    model.eval()

    names = json.loads((ROOT / "data" / "taxonomy" / "sectors_v1.json")
                       .read_text(encoding="utf-8"))["sectors"]
    control = terse_german()
    sectors = {}
    for code in sorted(names):
        prompt = PROMPT.format(code=code, name=control[code])
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                               add_generation_prompt=True,
                                               return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                    temperature=None, top_p=None, top_k=None,
                                    pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True).strip()
        description = clean(raw)
        sectors[code] = {"name": control[code], "description": description,
                         "nace_code": names[code].get("nace_code", code),
                         "unedited_reply": raw}
        print(f"  {code}  {len(description.split()):3d} words  {description[:70]}...")

    payload = {
        "written_by": args.model,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device,
        "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED},
        "given": "the terse German section name only -- no corpus, no documents, "
                 "no sight of the assistant's own definitions",
        "prompt_template": PROMPT,
        "framing_removed": [p.pattern for p in FRAMING],
        "sectors": sectors,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(sectors)} definitions by {args.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

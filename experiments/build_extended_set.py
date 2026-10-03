#!/usr/bin/env python3
"""A larger German evaluation set, drawn and labelled after the first one was fixed.

    python -m experiments.build_extended_set --draw         # the queue, from the corpus
    python -m experiments.build_extended_set --model-labels # the second labeller, local
    python -m experiments.build_extended_set --merge        # one file, with both labellers

The paper's German evaluation set is \u007f299\u007f documents, which is what a hand-checked
set costs and what every interval in the paper inherits. It cannot be
*replaced*: every preregistered prediction was committed against exactly those
documents, and swapping them would destroy the evidence that the predictions
came first. It can be *extended*, with new predictions registered against the
new documents, which is what this builds.

Two differences from the first set, both deliberate:

* **Two independent labellers, from different model families.** The first set
  had one (an assistant applying ``docs/annotation_guidelines.md``) plus a human
  check on fifty documents. Here the same guideline is applied by that assistant
  and, separately, by the open 7B model the paper uses as a baseline. Their
  agreement is reported, and the subset they agree on is marked, so the label
  noise of a silver set at this scale is bounded by a measurement rather than by
  an assumption.
* **No document from the first set.** The queue excludes them, so the two sets
  can be reported separately and the old one stays untouched.

Nothing here is a gold standard. It is a larger silver set whose noise is
measured, and the paper says so wherever it uses it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

from experiments.config import CORPUS_CSV, DATA_DIR, ROOT, SEED, ensure_dirs, set_seed

QUEUE = DATA_DIR / "evaluation" / "extended_queue.json"
MODEL_LABELS = DATA_DIR / "evaluation" / "extended_model_labels.json"
ASSISTANT_LABELS = DATA_DIR / "evaluation" / "extended_assistant_labels.json"
MERGED = DATA_DIR / "evaluation" / "extended_labels.json"
EXISTING = DATA_DIR / "evaluation" / "human_labels.json"
SIZE = 1200
MIN_WORDS = 8

SECTION_RULES = """Rules, applied in order:
- Take the first substantive clause; ignore closing boilerplate ("sowie alle damit
  zusammenhaengenden Geschaefte", "Zweigniederlassungen errichten").
- Verb before noun: Herstellung -> C, Handel/Vertrieb -> G, Beratung -> M,
  Vermittlung -> the section of what is brokered.
- Manufacture and trade both named: whichever comes first wins.
- Uebernahme der persoenlichen Haftung und Geschaeftsfuehrung (Komplementaer) -> M.
- Erwerb und Verwaltung von Beteiligungen, no management -> K. Verwaltung eigenen
  Vermoegens -> K, but with Grundvermoegen/Immobilien -> L. Hausverwaltung -> L.
- Bauträger, Elektroinstallation, Sanitaer-/Heizungsbau -> F. An- und Verkauf von
  Grundstuecken -> L. Garten- und Landschaftsbau, Gebaeudereinigung, Facility
  Management, Arbeitnehmerueberlassung -> N.
- Softwareentwicklung, IT-Beratung, Plattformbetrieb, Verlag, Film -> J;
  Handel mit Computern -> G; Beratung/Marketing about digital things -> M.
- Spedition, Gueterkraftverkehr -> H. Recycling, Entsorgung, Containerdienst with
  Abfall -> E. Pflegedienst, Arztpraxis -> Q. Fahrschule, Seminare -> P.
  Fitness-/Kosmetikstudio -> S. Spielautomaten -> R. Gastronomie, Diskothek -> I."""


def draw() -> int:
    """A fresh sample of the corpus, with the first set's documents held out."""
    import pandas as pd

    if not CORPUS_CSV.exists():
        print("The corpus is not redistributed; this step needs data/raw/. See data/README.md.",
              file=sys.stderr)
        return 1
    set_seed()
    corpus = pd.read_csv(CORPUS_CSV)
    taken = {s["purpose"].strip() for s in
             json.loads(EXISTING.read_text(encoding="utf-8"))["samples"]}
    retired = DATA_DIR / "evaluation" / "retired_handwritten.json"
    if retired.exists():
        payload = json.loads(retired.read_text(encoding="utf-8"))
        taken |= {s.get("purpose", "").strip() for s in payload.get("samples", payload)
                  if isinstance(s, dict)}

    rows = []
    for name, purpose in zip(corpus["legal_name"], corpus["purpose"]):
        text = str(purpose).strip()
        if text in taken or len(text.split()) < MIN_WORDS:
            continue
        rows.append({"legal_name": str(name), "purpose": text})
    rng = np.random.default_rng(SEED)
    picked = rng.choice(len(rows), size=min(SIZE, len(rows)), replace=False)
    queue = [{"id": f"x{i:04d}", **rows[int(j)]} for i, j in enumerate(sorted(picked))]
    QUEUE.write_text(json.dumps({
        "drawn_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": CORPUS_CSV.name, "seed": SEED, "size": len(queue),
        "excluded": "every document of the first evaluation set and of the retired "
                    "hand-written set, and purposes under eight words",
        "documents": queue,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {QUEUE.relative_to(ROOT)}: {len(queue)} documents, "
          f"{len(rows)} were eligible of {len(corpus)}")
    return 0


def model_labels(model: str, device: str, dtype: str) -> int:
    """The second labeller: an open model from a different family, same guideline."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.data import load_taxonomy

    queue = json.loads(QUEUE.read_text(encoding="utf-8"))["documents"]
    taxonomy = load_taxonomy()
    menu = "\n".join(f"{c}: {taxonomy[c].get('name', c)}" for c in sorted(taxonomy))
    tokenizer = AutoTokenizer.from_pretrained(model)
    net = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype)).to(device).eval()

    print(f"{model} on {device} in {dtype}: {len(queue)} documents", flush=True)
    out, off_format = {}, 0
    for n, row in enumerate(queue, 1):
        prompt = (f"Assign one NACE Rev. 2 section to this German company purpose.\n{menu}\n\n"
                  f"{SECTION_RULES}\n\nPurpose:\n{row['purpose'][:600]}\n\n"
                  "Answer with one section letter, nothing else.")
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                            add_generation_prompt=True,
                                            return_tensors="pt").to(device)
        with torch.no_grad():
            reply = net.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=4,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(reply[0, ids.shape[1]:], skip_special_tokens=True).strip()
        letters = [c for c in text.upper() if c in taxonomy]
        if not letters:
            off_format += 1
        out[row["id"]] = letters[0] if letters else None
        if n % 25 == 0 or n == len(queue):
            print(f"  {n}/{len(queue)}", flush=True)

    MODEL_LABELS.write_text(json.dumps({
        "labeller": model, "device": device, "dtype": dtype,
        "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decoding": "greedy, at most 4 new tokens",
        "prompt": "the section menu plus the compressed rules of docs/annotation_guidelines.md",
        "off_format_replies": off_format,
        "labels": out,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {MODEL_LABELS.relative_to(ROOT)}: {len(out)} labels, "
          f"{off_format} replies without a section letter")
    return 0


def merge() -> int:
    """One evaluation file, with both labellers and their agreement."""
    from sklearn.metrics import cohen_kappa_score

    queue = {r["id"]: r for r in json.loads(QUEUE.read_text(encoding="utf-8"))["documents"]}
    assistant: Dict[str, str] = json.loads(ASSISTANT_LABELS.read_text(encoding="utf-8"))["labels"]
    second = json.loads(MODEL_LABELS.read_text(encoding="utf-8"))
    model: Dict[str, str] = second["labels"]

    both = [i for i in queue if assistant.get(i) and model.get(i)]
    kappa = float(cohen_kappa_score([assistant[i] for i in both], [model[i] for i in both]))
    agree = float(np.mean([assistant[i] == model[i] for i in both]))
    samples: List[Dict] = []
    for i, row in queue.items():
        if not assistant.get(i):
            continue
        samples.append({
            "id": i, "legal_name": row["legal_name"], "purpose": row["purpose"],
            "true_sector": assistant[i],
            "second_labeller": model.get(i),
            "labellers_agree": bool(model.get(i) == assistant[i]),
            "annotation_method": "model_assisted",
            "provenance": "corpus_sample_extended",
        })
    distribution: Dict[str, int] = {}
    for s in samples:
        distribution[s["true_sector"]] = distribution.get(s["true_sector"], 0) + 1
    MERGED.write_text(json.dumps({
        "metadata": {
            "total": len(samples),
            "source": "corpus_sample_extended",
            "annotation": "two independent labellers applying docs/annotation_guidelines.md: an "
                          "assistant (Claude, Anthropic), as in the first set, and "
                          f"{second['labeller']}, which is a different model family. No human "
                          "verification pass has been run on this set.",
            "first_set": "data/evaluation/human_labels.json, 299 documents, untouched",
            "labeller_agreement": {"n": len(both), "raw": agree, "cohen_kappa": kappa},
            "sector_distribution": dict(sorted(distribution.items())),
            "date": datetime.now(timezone.utc).date().isoformat(),
        },
        "samples": samples,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {MERGED.relative_to(ROOT)}: {len(samples)} documents")
    print(f"  labellers agree on {agree:.1%} of {len(both)} documents, kappa = {kappa:.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draw", action="store_true", help="Draw the queue from the corpus.")
    parser.add_argument("--model-labels", action="store_true", help="Run the second labeller.")
    parser.add_argument("--merge", action="store_true", help="Merge both labellers into one file.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    ensure_dirs()
    if args.draw:
        return draw()
    if args.model_labels:
        return model_labels(args.model, args.device, args.dtype)
    if args.merge:
        return merge()
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

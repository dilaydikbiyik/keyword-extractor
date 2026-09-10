"""
Every number quoted in the README, the readiness report and the paper must
match results/. A figure that is correct in results/ and stale in the prose is
the failure mode this project has already hit twice; this makes it a test
failure instead of something a reader finds.

Run with: pytest tests/test_reported_numbers.py -v
"""

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name):
    path = RESULTS / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not generated yet — run `make reproduce` and `make study`")
    return json.loads(path.read_text(encoding="utf-8"))


def system(results, key):
    return next(s for s in results["systems"] if s["key"] == key)


def pct(value):
    return f"{value * 100:.1f}%"


DOCS = {
    "README.md": ROOT / "README.md",
    "docs/paper_readiness.md": ROOT / "docs" / "paper_readiness.md",
}


def claims():
    """(source, formatted figure) pairs the prose is required to agree with."""
    baselines = load("baselines")
    ablation = load("ablation")
    study = load("description_study")
    search = load("predictor_search")
    reuters = load("reuters")
    news = load("replication_20newsgroups")

    ours = system(baselines, "full")["sector"]
    tfidf = system(baselines, "tfidf-nace")["sector"]
    no_seeds = next(s for s in ablation["systems"] if s["key"] == "no-seed-vector")["sector"]
    old_taxonomy = next(s for s in ablation["systems"] if s["key"] == "taxonomy-v1")["sector"]

    return [
        ("headline Top-1", pct(ours["top1_accuracy"])),
        ("headline Top-3", pct(ours["top3_accuracy"])),
        ("TF-IDF Top-1", pct(tfidf["top1_accuracy"])),
        ("previous taxonomy Top-1", pct(old_taxonomy["top1_accuracy"])),
        ("no-seeds Top-1", pct(no_seeds["top1_accuracy"])),
        ("description study, terse", pct(list(study["conditions"].values())[0]["top1_accuracy"])),
        ("description study, translated", pct(list(study["conditions"].values())[1]["top1_accuracy"])),
        ("description study, rewritten", pct(list(study["conditions"].values())[2]["top1_accuracy"])),
        ("content effect", f"{study['content_effect']['gain_pp']:+.1f}".replace("+", "+")),
        ("alignment rho", f"{search['correlations']['alignment_gain']['rho']:+.3f}".replace("+", "+")),
        ("Reuters elaboration gain", f"{reuters['defining_the_class']['gain_pp']:+.1f}"),
        ("20NG elaboration gain", f"{news['defining_the_class']['gain_pp']:+.1f}"),
    ]


@pytest.mark.parametrize("doc", sorted(DOCS))
def test_prose_quotes_only_current_numbers(doc):
    """Each headline figure appears in the prose exactly as results/ has it."""
    text = DOCS[doc].read_text(encoding="utf-8")
    missing = []
    for label, figure in claims():
        # A figure may legitimately be absent from a given document; what must
        # never happen is the document quoting a *different* value for it.
        stripped = figure.lstrip("+")
        if stripped not in text:
            missing.append((label, figure))
    # The README carries every headline; the readiness report carries all of them too.
    assert not missing, (
        f"{doc} does not quote these current figures — it is probably still "
        f"showing an earlier run: {missing}"
    )


def test_no_superseded_headline_survives():
    """The two numbers this project has already had to correct in public."""
    for doc, path in DOCS.items():
        text = path.read_text(encoding="utf-8")
        for stale, why in (
            ("80.0%", "the accuracy measured on the retired hand-written evaluation set"),
            ("36.1%", "the headline before the taxonomy was rewritten"),
        ):
            for line in text.splitlines():
                low = line.lower()
                # The same digits appear as an inter-annotator agreement rate,
                # which is a different quantity entirely.
                if any(w in low for w in ("agreement", "κ", "kappa", "confidence", "pilot")):
                    continue
                if stale in line:
                    # Mentioning the old number while explaining it is fine;
                    # quoting it as a current result is not.
                    assert any(
                        marker in line.lower()
                        for marker in ("old", "previous", "earlier", "before", "was", "retired", "reported")
                    ), f"{doc} quotes {stale} ({why}) without marking it as superseded:\n  {line.strip()}"


def test_paper_hardcodes_no_figures():
    """The paper cites macros, never literal numbers."""
    import re

    tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    body = tex.split(r"\begin{document}", 1)[1]
    body = re.sub(r"%.*", "", body)
    body = re.sub(r"\\(?:label|ref|cite[pt]?|input|documentclass|usepackage)\{[^}]*\}", "", body)
    # Percentages and decimals that are not part of a macro name or a section number.
    literals = re.findall(r"(?<![\\{\w.])\d+\.\d+\\?%", body)
    allowed = {"0.005", "0.001"}  # p-value thresholds quoted as prose, not results
    offenders = [x for x in literals if x.rstrip("\\%") not in allowed]
    assert not offenders, (
        "paper/main.tex hardcodes figures instead of using macros from "
        f"tables/macros.tex: {sorted(set(offenders))}"
    )

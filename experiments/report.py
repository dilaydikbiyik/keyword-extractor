"""Markdown table rendering for the results files."""

from __future__ import annotations

from typing import Dict, List, Sequence


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def markdown_table(rows: Sequence[Dict], columns: Sequence[tuple]) -> str:
    """Render rows as a GitHub-flavoured markdown table.

    ``columns`` is a sequence of ``(header, key, formatter)`` triples.
    """
    header = "| " + " | ".join(h for h, _, _ in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, rule]
    for row in rows:
        lines.append(
            "| " + " | ".join(fmt(row.get(key)) for _, key, fmt in columns) + " |"
        )
    return "\n".join(lines)


def comparison_table(results: List[Dict]) -> str:
    rows = []
    for r in results:
        ci = r["sector"].get("top1_ci95")
        rows.append(
            {
                "label": ("*" if r.get("is_oracle") else "") + r["label"],
                "top1": r["sector"]["top1_accuracy"],
                "ci": f"[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]" if ci else None,
                "top3": r["sector"]["top3_accuracy"],
                "f1": r["sector"]["f1_macro"],
                "kappa": r["sector"]["cohen_kappa"],
                "p5": r["keywords"].get("precision_at_5"),
                "p": r.get("p_value_vs_full"),
            }
        )
    return markdown_table(
        rows,
        [
            ("System", "label", str),
            ("Top-1", "top1", _pct),
            ("95% CI", "ci", lambda v: v or "—"),
            ("Top-3", "top3", _pct),
            ("F1-macro", "f1", _fmt),
            ("κ", "kappa", _fmt),
            ("P@5", "p5", _fmt),
            ("p vs. full", "p", lambda v: "—" if v is None else f"{v:.3f}"),
        ],
    )


def ablation_table(results: List[Dict]) -> str:
    full = next((r for r in results if r["key"] == "full"), None)
    base_top1 = full["sector"]["top1_accuracy"] if full else 0.0
    base_p5 = (full["keywords"].get("precision_at_5") if full else None) or 0.0
    rows = []
    for r in results:
        p5 = r["keywords"].get("precision_at_5")
        rows.append(
            {
                "label": r["label"],
                "top1": r["sector"]["top1_accuracy"],
                "dtop1": r["sector"]["top1_accuracy"] - base_top1,
                "f1": r["sector"]["f1_macro"],
                "p5": p5,
                "dp5": None if p5 is None else p5 - base_p5,
                "p": r.get("p_value_vs_full"),
            }
        )
    return markdown_table(
        rows,
        [
            ("Variant", "label", str),
            ("Top-1", "top1", _pct),
            ("Δ Top-1", "dtop1", lambda v: "—" if v is None else f"{v * 100:+.1f} pp"),
            ("F1-macro", "f1", _fmt),
            ("P@5", "p5", _fmt),
            ("Δ P@5", "dp5", lambda v: "—" if v is None else f"{v:+.3f}"),
            ("p vs. full", "p", lambda v: "—" if v is None else f"{v:.3f}"),
        ],
    )

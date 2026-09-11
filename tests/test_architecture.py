"""The layering, enforced: each layer depends only on the ones below it."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def offenders(folder, forbidden):
    return [f"{path.relative_to(ROOT)} imports {name}"
            for path in (ROOT / folder).rglob("*.py")
            for name in imported_modules(path)
            if any(name == f or name.startswith(f + ".") for f in forbidden)]


def test_the_library_never_imports_the_research_code():
    assert not offenders("src", {"experiments"})


def test_services_do_not_reach_up_to_the_controller():
    assert not offenders("src/services", {"controllers", "pipeline"})


def test_the_domain_model_depends_on_nothing_above_it():
    assert not offenders("src/models", {"services", "controllers", "pipeline", "experiments"})

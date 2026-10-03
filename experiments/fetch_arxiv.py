#!/usr/bin/env python3
"""The fifth corpus: arXiv abstracts, author-assigned categories, arXiv's own descriptions.

    python -m experiments.fetch_arxiv --taxonomy   # arXiv's official category descriptions
    python -m experiments.fetch_arxiv              # the abstract sample, by committed id list

Everything the paper has measured so far rests on \u007fNEVAL\u007f German documents with
model-assisted labels, and on class descriptions written for the study. This
corpus removes all three limits at once:

* the labels are not ours and not a model's. A paper's primary arXiv category is
  chosen by its authors at submission;
* the elaborated class text is not ours either. arXiv publishes a description of
  every category, written by arXiv, and the study uses it verbatim;
* the sample is two orders of magnitude larger than the trade register set.

Only papers whose single Computer Science category is the primary one are kept,
the same single-label restriction the Reuters study uses: a cross-listed paper
cannot fairly score either category.

Abstracts are not redistributed here. What is committed is the list of paper
identifiers and their categories, so the sample is exactly reproducible with one
network call; the abstracts themselves are fetched into an ignored cache.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html import unescape
from typing import Dict, List

from experiments.config import DATA_DIR, ROOT, SEED, ensure_dirs, set_seed

TAXONOMY_URL = "https://arxiv.org/category_taxonomy"
API = "https://export.arxiv.org/api/query"
TAXONOMY = DATA_DIR / "taxonomy" / "arxiv_categories.json"
SAMPLE_IDS = DATA_DIR / "external" / "arxiv_sample.json"
CACHE = DATA_DIR / "cache" / "arxiv_abstracts.json"
ARCHIVE = "cs"
PER_CATEGORY = 200
PAGE = 100
PAUSE = 3.0          # arXiv asks for one request every three seconds
USER_AGENT = "keyword-extractor research experiment (one-off corpus sample)"


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_taxonomy(html: str) -> Dict[str, Dict[str, str]]:
    """Every category of one archive, with arXiv's own name and description."""
    out: Dict[str, Dict[str, str]] = {}
    pattern = re.compile(
        r"<h4>(?P<code>[a-z\-]+\.[A-Za-z\-]+)\s*<span>\((?P<name>[^)]*)\)</span></h4>"
        r".*?<div class=\"column\"><p>(?P<description>.*?)</p>",
        re.S)
    for match in pattern.finditer(html):
        code = match.group("code")
        if not code.startswith(ARCHIVE + "."):
            continue
        description = unescape(re.sub(r"<[^>]+>", " ", match.group("description")))
        out[code] = {"name": unescape(match.group("name")).strip(),
                     "description": " ".join(description.split())}
    return out


def api_page(category: str, start: int) -> List[Dict]:
    """One page of the oldest-first listing of a category, which is a stable order."""
    query = urllib.parse.urlencode({
        "search_query": f"cat:{category}",
        "start": start, "max_results": PAGE,
        "sortBy": "submittedDate", "sortOrder": "ascending",
    })
    feed = fetch(f"{API}?{query}")
    entries = []
    for entry in re.findall(r"<entry>(.*?)</entry>", feed, re.S):
        identifier = re.search(r"<id>http[s]?://arxiv\.org/abs/([^<]+)</id>", entry)
        summary = re.search(r"<summary>(.*?)</summary>", entry, re.S)
        primary = re.search(r'<arxiv:primary_category[^>]*term="([^"]+)"', entry)
        if not (identifier and summary and primary):
            continue
        categories = re.findall(r'<category[^>]*term="([^"]+)"', entry)
        entries.append({
            "id": identifier.group(1),
            "primary": primary.group(1),
            "categories": sorted(set(categories)),
            "abstract": " ".join(unescape(summary.group(1)).split()),
        })
    return entries


def single_label(entry: Dict, category: str) -> bool:
    """One Computer Science category, and it is the primary one."""
    within = [c for c in entry["categories"] if c.startswith(ARCHIVE + ".")]
    return entry["primary"] == category and within == [category]


def build_sample(categories: List[str]) -> None:
    set_seed()
    kept: Dict[str, List[Dict]] = {}
    for n, category in enumerate(sorted(categories), 1):
        rows: List[Dict] = []
        start = 0
        while len(rows) < PER_CATEGORY and start < 10 * PER_CATEGORY:
            try:
                page = api_page(category, start)
            except (urllib.error.URLError, TimeoutError) as error:
                print(f"  {category}: {error}; stopping this category", file=sys.stderr)
                break
            if not page:
                break
            rows += [e for e in page if single_label(e, category)
                     and len(e["abstract"].split()) >= 40]
            start += PAGE
            time.sleep(PAUSE)
        kept[category] = rows[:PER_CATEGORY]
        print(f"  [{n:2d}/{len(categories)}] {category:10s} {len(kept[category]):4d} kept", flush=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    SAMPLE_IDS.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_IDS.write_text(json.dumps({
        "source": "arXiv API, oldest first per category",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive": ARCHIVE, "per_category": PER_CATEGORY, "seed": SEED,
        "restriction": "the paper's only Computer Science category is its primary one, "
                       "and the abstract is at least 40 words",
        "note": "Abstracts are not redistributed; `python -m experiments.fetch_arxiv` "
                "fetches them into data/cache/ from these identifiers.",
        "documents": {c: [e["id"] for e in rows] for c, rows in kept.items()},
    }, indent=2) + "\n", encoding="utf-8")
    total = sum(len(v) for v in kept.values())
    print(f"Wrote {SAMPLE_IDS.relative_to(ROOT)} ({total} documents) and the abstract cache")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", action="store_true",
                        help="Refresh arXiv's official category descriptions and stop.")
    args = parser.parse_args()
    ensure_dirs()

    if args.taxonomy:
        categories = parse_taxonomy(fetch(TAXONOMY_URL))
        if len(categories) < 20:
            print(f"Parsed only {len(categories)} categories; the page layout changed.",
                  file=sys.stderr)
            return 1
        TAXONOMY.write_text(json.dumps({
            "source": TAXONOMY_URL,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "archive": ARCHIVE,
            "note": "Names and descriptions are arXiv's own text, quoted verbatim. They were "
                    "written by neither the authors of this paper nor the labeller.",
            "categories": categories,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for code, entry in sorted(categories.items()):
            print(f"  {code:10s} {entry['name'][:40]:42s} {len(entry['description'].split()):3d} words")
        print(f"Wrote {TAXONOMY.relative_to(ROOT)}: {len(categories)} categories")
        return 0

    if not TAXONOMY.exists():
        print("Fetch the taxonomy first: --taxonomy", file=sys.stderr)
        return 1
    categories = list(json.loads(TAXONOMY.read_text(encoding="utf-8"))["categories"])
    build_sample(categories)
    return 0


if __name__ == "__main__":
    sys.exit(main())

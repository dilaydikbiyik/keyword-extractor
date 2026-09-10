"""
TaxonomyManager — NACE Rev. 2 Sector Taxonomy Management

Provides a clean interface for loading, querying, and updating the
21-sector NACE taxonomy stored in data/taxonomy/sectors.json.
"""

from typing import List, Dict, Optional
import json
import os


class TaxonomyManager:
    """
    Manages the NACE Rev. 2 sector taxonomy (21 sectors, A–U).

    Responsibilities:
        - Load taxonomy from JSON
        - Query sector info, seeds, negative keywords
        - Add/update seeds at runtime
        - Validate sector codes
        - Export updated taxonomy back to JSON
    """

    def __init__(self, taxonomy_path: Optional[str] = None):
        """
        Load taxonomy from JSON file.

        Args:
            taxonomy_path: Path to sectors.json. Defaults to
                           data/taxonomy/sectors.json relative to project root.
        """
        if taxonomy_path is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)
            )))
            taxonomy_path = os.path.join(base, "data", "taxonomy", "sectors.json")

        self.taxonomy_path = taxonomy_path
        self._data: Dict = {}
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        """Load taxonomy from disk."""
        with open(self.taxonomy_path, encoding="utf-8") as f:
            self._data = json.load(f)

    # ── Query ─────────────────────────────────────────────────────────────────

    @property
    def sectors(self) -> Dict[str, Dict]:
        """Raw sector dict (code → info)."""
        return self._data.get("sectors", {})

    def list_codes(self) -> List[str]:
        """Sorted list of all sector codes."""
        return sorted(self.sectors.keys())

    def get_sector(self, code: str) -> Optional[Dict]:
        """
        Return full info dict for a sector code.

        Returns:
            Dict with name, description, seed_keywords, negative_keywords, …
            or None if the code does not exist.
        """
        return self.sectors.get(code)

    def get_seed_keywords(self, code: str) -> List[str]:
        """Seed keyword list for a sector."""
        s = self.get_sector(code)
        return list(s.get("seed_keywords", [])) if s else []

    def is_valid_code(self, code: str) -> bool:
        """Return True if the sector code exists in the taxonomy."""
        return code in self.sectors

    # ── Mutation ──────────────────────────────────────────────────────────────

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None):
        """
        Persist the current taxonomy back to disk.

        Args:
            path: Output path. Defaults to the original taxonomy_path.
        """
        out = path or self.taxonomy_path
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return a summary statistics dict."""
        counts = {
            code: len(self.get_seed_keywords(code))
            for code in self.list_codes()
        }
        return {
            "total_sectors": len(self.sectors),
            "seed_counts": counts,
            "total_seeds": sum(counts.values()),
            "avg_seeds_per_sector": round(sum(counts.values()) / max(len(counts), 1), 1),
            "min_seeds": min(counts.values()),
            "max_seeds": max(counts.values()),
        }

    def __repr__(self) -> str:
        return f"TaxonomyManager(sectors={len(self.sectors)}, path={self.taxonomy_path!r})"

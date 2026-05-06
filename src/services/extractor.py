"""
Keyword Extractor Service

Provides guided keyword extraction using KeyBERT with sector-specific seed keywords.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from keybert import KeyBERT
import json
from pathlib import Path


class KeywordExtractor:
    """
    Guided keyword extraction service using KeyBERT.
    
    Uses sector-specific seed keywords to guide extraction toward domain-relevant terms.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        sectors_file: str = "data/taxonomy/sectors.json",
        device: str = "cpu"
    ):
        """
        Initialize the keyword extractor.

        Args:
            model_name: Name of the embedding model
            sectors_file: Path to sectors configuration file
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device

        # Initialize KeyBERT
        print(f"Initializing KeyBERT with model: {model_name}")
        self.kw_model = KeyBERT(model=model_name)

        # Load sector keywords
        self.sector_keywords = self._load_sector_keywords(sectors_file)
        print(f"Loaded seed keywords for {len(self.sector_keywords)} sectors")

    def _load_sector_keywords(self, sectors_file: str) -> Dict[str, List[str]]:
        """
        Load seed keywords for each sector from JSON file.

        Args:
            sectors_file: Path to sectors.json

        Returns:
            Dictionary mapping sector codes to seed keyword lists
        """
        sector_keywords = {}

        try:
            with open(sectors_file, 'r', encoding='utf-8') as f:
                sectors_data = json.load(f)

            sectors = sectors_data.get('sectors', {})

            for code, sector_info in sectors.items():
                keywords = sector_info.get('seed_keywords', [])
                if keywords:
                    sector_keywords[code] = keywords

        except FileNotFoundError:
            print(f"Sectors file not found: {sectors_file}")
        except Exception as e:
            print(f"Error loading sectors: {e}")

        return sector_keywords

    def extract_keywords(
        self,
        text: str,
        top_n: int = 10,
        seed_keywords: Optional[List[str]] = None,
        diversity: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords from text using KeyBERT.

        Args:
            text: Input text
            top_n: Number of keywords to extract
            seed_keywords: List of seed keywords for guided extraction
            diversity: Diversity parameter (0-1, higher = more diverse)

        Returns:
            List of (keyword, score) tuples sorted by score
        """
        try:
            if seed_keywords:
                # Guided extraction with seed keywords
                keywords = self.kw_model.extract_keywords(
                    text,
                    seed_keywords=seed_keywords,
                    top_n=top_n,
                    use_mmr=True,  # Maximal Marginal Relevance for diversity
                    diversity=diversity
                )
            else:
                # Unguided extraction
                keywords = self.kw_model.extract_keywords(
                    text,
                    top_n=top_n,
                    use_mmr=True,
                    diversity=diversity
                )

            return keywords

        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []

    def extract_keywords_guided_by_sector(
        self,
        text: str,
        sector_code: str,
        top_n: int = 10,
        language: Optional[str] = None  # accepted for API compatibility; multilingual model handles language automatically
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords with guidance from sector-specific seed keywords.

        Args:
            text: Input text
            sector_code: Sector code (e.g., 'C' for Manufacturing)
            top_n: Number of keywords to extract
            language: Language of the text (informational; the multilingual model
                      handles all supported languages automatically)

        Returns:
            List of (keyword, score) tuples
        """
        # Get sector seed keywords
        seed_keywords = self.sector_keywords.get(sector_code, [])

        if not seed_keywords:
            print(f"Warning: No seed keywords found for sector {sector_code}")
            return self.extract_keywords(text, top_n=top_n)

        # Extract with sector guidance
        keywords = self.extract_keywords(
            text,
            top_n=top_n,
            seed_keywords=seed_keywords,
            diversity=0.7  # Higher diversity for sector-guided extraction
        )

        return keywords

    def extract_keywords_multi_sector(
        self,
        text: str,
        sector_codes: List[str],
        top_n: int = 10,
        language: Optional[str] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract keywords guided by multiple sectors.

        Useful for ambiguous texts that could belong to multiple sectors.

        Args:
            text: Input text
            sector_codes: List of sector codes
            top_n: Number of keywords to extract per sector
            language: Language of the text (informational; handled by multilingual model)

        Returns:
            Dictionary mapping sector codes to extracted keywords
        """
        results = {}

        for sector_code in sector_codes:
            keywords = self.extract_keywords_guided_by_sector(
                text,
                sector_code,
                top_n=top_n,
                language=language
            )
            results[sector_code] = keywords

        return results

    def extract_keyphrases(
        self,
        text: str,
        top_n: int = 5,
        seed_keywords: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Extract multi-word keyphrases (similar to extract_keywords but optimized for phrases).

        Args:
            text: Input text
            top_n: Number of keyphrases to extract
            seed_keywords: Optional seed keywords
            language: Language of the text

        Returns:
            List of (keyphrase, score) tuples
        """
        # Extract with emphasis on multi-word expressions
        keywords = self.extract_keywords(
            text,
            top_n=top_n * 2,  # Extract more to filter keyphrases
            seed_keywords=seed_keywords,
            diversity=0.8
        )

        # Filter for multi-word expressions
        keyphrases = [(kw, score) for kw, score in keywords if ' ' in kw]

        return keyphrases[:top_n]

    def batch_extract_keywords(
        self,
        texts: List[str],
        sector_codes: Optional[List[str]] = None,
        top_n: int = 10
    ) -> List[Dict]:
        """
        Extract keywords from multiple texts.

        Args:
            texts: List of texts
            sector_codes: Optional list of sector codes (one per text)
            top_n: Number of keywords per text

        Returns:
            List of dictionaries with extraction results
        """
        results = []

        for i, text in enumerate(texts):
            sector_code = sector_codes[i] if sector_codes and i < len(sector_codes) else None

            if sector_code:
                keywords = self.extract_keywords_guided_by_sector(
                    text,
                    sector_code,
                    top_n=top_n
                )
            else:
                keywords = self.extract_keywords(
                    text,
                    top_n=top_n
                )

            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'sector': sector_code,
                'keywords': keywords
            })

        return results

    def get_sector_seed_keywords(self, sector_code: str) -> List[str]:
        """
        Get seed keywords for a specific sector.

        Args:
            sector_code: Sector code

        Returns:
            List of seed keywords
        """
        return self.sector_keywords.get(sector_code, [])

    def list_available_sectors(self) -> List[str]:
        """
        List all available sector codes.

        Returns:
            List of sector codes
        """
        return sorted(list(self.sector_keywords.keys()))

    def iterative_expand(
        self,
        texts: List[str],
        sector_code: str,
        n_iterations: int = 3,
        expand_top_n: int = 3,
        quality_threshold: float = 0.4,
        max_seed_size: int = 50,
    ) -> List[str]:
        """
        Iteratively expand seed keywords for a sector using a corpus of texts.

        Runs guided extraction for n_iterations, each time adding newly
        discovered high-quality keywords back into the seed list.  This lets
        the pipeline discover domain terms that were absent from the original
        seed list (e.g. "Prophylaxe" from a dental corpus).

        Args:
            texts: Corpus of business descriptions for this sector
            sector_code: Target sector code (e.g. 'Q')
            n_iterations: Number of expansion rounds
            expand_top_n: Max new seeds to add per text per iteration
            quality_threshold: Minimum keyword score to be added as seed
            max_seed_size: Hard cap on seed list size (prevents unbounded growth)

        Returns:
            Expanded seed keyword list
        """
        import logging
        logger = logging.getLogger(__name__)

        # Start from current seeds for this sector
        current_seeds: List[str] = list(self.sector_keywords.get(sector_code, []))
        logger.info(
            f"[iterative_expand] sector={sector_code} "
            f"initial_seeds={len(current_seeds)} texts={len(texts)}"
        )

        for iteration in range(1, n_iterations + 1):
            new_seed_candidates: List[str] = []

            for text in texts:
                # Extract with current seeds
                keywords = self.extract_keywords(
                    text,
                    top_n=expand_top_n * 2,
                    seed_keywords=current_seeds if current_seeds else None,
                    diversity=0.6,
                )

                # Collect high-quality terms not yet in seeds
                for kw, score in keywords:
                    if (
                        score >= quality_threshold
                        and kw not in current_seeds
                        and kw not in new_seed_candidates
                    ):
                        new_seed_candidates.append(kw)

            # Limit candidates to expand_top_n per iteration (best first)
            new_seed_candidates = new_seed_candidates[:expand_top_n * len(texts)]

            # Merge, respecting max_seed_size cap
            slots_available = max_seed_size - len(current_seeds)
            added = new_seed_candidates[:max(0, slots_available)]
            current_seeds.extend(added)

            logger.info(
                f"  Iteration {iteration}/{n_iterations}: "
                f"+{len(added)} seeds "
                f"(candidates={len(new_seed_candidates)}, "
                f"total={len(current_seeds)})"
            )

            if not added:
                logger.info("  No new seeds — stopping early.")
                break

        # Persist the expanded seeds back into the in-memory store
        self.sector_keywords[sector_code] = current_seeds
        logger.info(
            f"[iterative_expand] Done. "
            f"Final seed count for {sector_code}: {len(current_seeds)}"
        )
        return current_seeds
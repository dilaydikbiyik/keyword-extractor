"""
Keyword Filter Service

Filters and ranks keywords based on multiple criteria including
information value, relevance, and sector specificity.
"""

from typing import List, Dict, Tuple, Optional
import json


class KeywordFilter:
    """
    Advanced keyword filtering based on information theory and domain relevance.

    Implements filtering based on:
    - Information value (uniqueness/rarity)
    - Sector relevance
    - Linguistic quality
    - Semantic similarity
    """

    def __init__(
        self,
        sectors_file: str = "data/taxonomy/sectors.json",
        negative_keywords_weight: float = -0.5
    ):
        """
        Initialize the keyword filter.

        Args:
            sectors_file: Path to sectors configuration file
            negative_keywords_weight: Weight for negative keywords (0-1)
        """
        self.sectors_file = sectors_file
        self.negative_keywords_weight = negative_keywords_weight

        # Load sector information
        self.sectors_info = self._load_sectors_info()
        self.negative_keywords_by_sector = self._load_negative_keywords()

    def _load_sectors_info(self) -> Dict[str, dict]:
        """Load sector information."""
        sectors_info = {}

        try:
            with open(self.sectors_file, 'r', encoding='utf-8') as f:
                sectors_data = json.load(f)
            sectors_info = sectors_data.get('sectors', {})
        except Exception as e:
            print(f"Error loading sectors: {e}")

        return sectors_info

    def _load_negative_keywords(self) -> Dict[str, set]:
        """Load negative keywords for each sector."""
        negative_keywords = {}

        for code, sector_info in self.sectors_info.items():
            neg_kws = sector_info.get('negative_keywords', [])
            negative_keywords[code] = set(kw.lower() for kw in neg_kws)

        return negative_keywords

    def filter_by_sector_relevance(
        self,
        keywords: List[Tuple[str, float]],
        sector_code: str,
        penalty: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Filter keywords based on sector-specific negative keywords.

        Args:
            keywords: List of (keyword, score) tuples
            sector_code: Sector code for negative keyword filtering
            penalty: Penalty multiplier for negative keywords

        Returns:
            Filtered and re-scored keywords
        """
        if sector_code not in self.negative_keywords_by_sector:
            return keywords

        negative_kws = self.negative_keywords_by_sector[sector_code]

        filtered = []

        for kw, score in keywords:
            kw_lower = kw.lower()

            # Check if keyword or its words are in negative list
            is_negative = kw_lower in negative_kws or any(
                word in negative_kws for word in kw_lower.split()
            )

            if is_negative:
                # Apply penalty but don't remove completely
                new_score = score * (1 - penalty)
            else:
                new_score = score

            if new_score > 0:  # Keep only positive scores
                filtered.append((kw, new_score))

        # Re-sort by score
        filtered.sort(key=lambda x: x[1], reverse=True)

        return filtered

    def filter_by_information_value(
        self,
        keywords: List[Tuple[str, float]],
        background_corpus_stats: Optional[Dict] = None,
        rare_weight: float = 0.8
    ) -> List[Tuple[str, float]]:
        """
        Filter based on information value (prefer rare/specific terms).

        Args:
            keywords: List of (keyword, score) tuples
            background_corpus_stats: Optional corpus statistics
            rare_weight: Weight for information value (0-1)

        Returns:
            Re-scored keywords based on information value
        """
        re_scored = []

        for kw, score in keywords:
            # Calculate linguistic specificity (prefer longer, more specific terms)
            word_count = len(kw.split())

            # Longer keywords are more specific
            specificity_bonus = min(0.3, (word_count - 1) * 0.1)

            # Adjust score based on specificity
            adjusted_score = score * (1 + specificity_bonus * rare_weight)

            re_scored.append((kw, adjusted_score))

        # Sort by adjusted score
        re_scored.sort(key=lambda x: x[1], reverse=True)

        return re_scored

    def filter_by_linguistic_quality(
        self,
        keywords: List[Tuple[str, float]],
        min_length: int = 3,
        max_length: int = 50,
        remove_numbers: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Filter keywords based on linguistic quality.

        Args:
            keywords: List of (keyword, score) tuples
            min_length: Minimum keyword length (characters)
            max_length: Maximum keyword length
            remove_numbers: Remove keywords with many numbers

        Returns:
            Filtered keywords
        """
        filtered = []

        for kw, score in keywords:
            # Length check
            if len(kw) < min_length or len(kw) > max_length:
                continue

            # Remove mostly numeric keywords
            if remove_numbers:
                num_count = sum(1 for c in kw if c.isdigit())
                if num_count / len(kw) > 0.3:  # More than 30% numbers
                    continue

            # Check for valid characters (mostly letters)
            valid_chars = sum(1 for c in kw if c.isalpha() or c.isspace() or c in '-\'')
            if valid_chars / len(kw) < 0.8:  # Less than 80% valid
                continue

            filtered.append((kw, score))

        return filtered

    def filter_duplicates_and_stems(
        self,
        keywords: List[Tuple[str, float]],
        keep_best: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Remove duplicate or near-duplicate keywords.

        Args:
            keywords: List of (keyword, score) tuples
            keep_best: Keep the keyword with highest score

        Returns:
            Deduplicated keywords
        """
        seen = {}

        for kw, score in keywords:
            kw_normalized = kw.lower().strip()

            if kw_normalized not in seen:
                seen[kw_normalized] = (kw, score)
            elif keep_best and score > seen[kw_normalized][1]:
                seen[kw_normalized] = (kw, score)

        return list(seen.values())

    def apply_all_filters(
        self,
        keywords: List[Tuple[str, float]],
        sector_code: Optional[str] = None,
        top_n: int = 10,
        min_score: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Apply all filters in optimal order.

        Args:
            keywords: List of (keyword, score) tuples
            sector_code: Optional sector code for sector-specific filtering
            top_n: Number of keywords to return
            min_score: Minimum score threshold

        Returns:
            Filtered and ranked keywords
        """
        # 1. Filter by linguistic quality
        keywords = self.filter_by_linguistic_quality(keywords)

        # 2. Remove duplicates
        keywords = self.filter_duplicates_and_stems(keywords)

        # 3. Filter by sector relevance
        if sector_code:
            keywords = self.filter_by_sector_relevance(keywords, sector_code)

        # 4. Filter by information value
        keywords = self.filter_by_information_value(keywords)

        # 5. Apply minimum score threshold
        keywords = [(kw, score) for kw, score in keywords if score >= min_score]

        # 6. Sort by score
        keywords.sort(key=lambda x: x[1], reverse=True)

        # 7. Return top-k
        return keywords[:top_n]

    def rank_keywords_by_domain_specificity(
        self,
        keywords: List[Tuple[str, float]],
        domain_keywords: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Re-rank keywords based on domain specificity.

        Args:
            keywords: List of (keyword, score) tuples
            domain_keywords: List of keywords known to be domain-specific

        Returns:
            Re-ranked keywords
        """
        domain_kws = set(kw.lower() for kw in domain_keywords)

        re_ranked = []

        for kw, score in keywords:
            kw_lower = kw.lower()

            # Check if keyword matches or contains domain keywords
            is_domain = kw_lower in domain_kws or any(
                dk in kw_lower for dk in domain_kws if len(dk) > 3
            )

            if is_domain:
                # Boost score for domain keywords
                new_score = score * 1.3
            else:
                new_score = score

            re_ranked.append((kw, new_score))

        # Re-sort
        re_ranked.sort(key=lambda x: x[1], reverse=True)

        return re_ranked

    def get_quality_score(
        self,
        keyword: str,
        extraction_score: float = 0.7,
        filter_severity: float = 0.5
    ) -> float:
        """
        Calculate overall quality score for a keyword.

        Args:
            keyword: Keyword to score
            extraction_score: Score from extraction model (0-1)
            filter_severity: How strict the filters are (0-1)

        Returns:
            Overall quality score (0-1)
        """
        quality_score = extraction_score

        # Apply linguistic penalties
        if len(keyword) < 4:
            quality_score *= (1 - 0.1 * filter_severity)

        if len(keyword) > 50:
            quality_score *= (1 - 0.2 * filter_severity)

        # Prefer multi-word phrases
        word_count = len(keyword.split())
        if word_count >= 2:
            quality_score *= (1 + 0.15 * filter_severity)

        return max(0, min(1, quality_score))  # Clamp to [0, 1]
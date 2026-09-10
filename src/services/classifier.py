"""
Sector Classifier Service

Classifies business descriptions into sectors using embedding similarity.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
import json

logger = logging.getLogger(__name__)


class SectorClassifier:
    """
    Sector classification service using embedding-based similarity.

    Predicts the most likely sector(s) for a business description.
    """

    def __init__(
        self,
        embedding_service,
        sectors_file: str = "data/taxonomy/sectors.json",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the sector classifier.

        Args:
            embedding_service: Initialized EmbeddingService instance
            sectors_file: Path to sectors configuration file
            confidence_threshold: Minimum confidence score (0-1)
        """
        self.embedding_service = embedding_service
        self.sectors_file = sectors_file
        self.confidence_threshold = confidence_threshold

        self.sectors_info = self._load_sectors_info()
        logger.info(f"Loaded {len(self.sectors_info)} sectors")

        self.sector_embeddings = self._build_sector_embeddings()
        logger.info(f"Built embeddings for {len(self.sector_embeddings)} sectors")

    def _load_sectors_info(self) -> Dict[str, dict]:
        """
        Load sector information from JSON file.

        Returns:
            Dictionary mapping sector codes to sector info
        """
        sectors_info = {}

        try:
            with open(self.sectors_file, 'r', encoding='utf-8') as f:
                sectors_data = json.load(f)

            sectors_info = sectors_data.get('sectors', {})

        except FileNotFoundError:
            logger.error(f"Sectors file not found: {self.sectors_file}")
        except Exception as e:
            logger.error(f"Error loading sectors: {e}")

        return sectors_info

    def _build_sector_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Build embeddings for sector descriptions.

        Each sector vector is the average of:
          1. The name + description embedding
          2. The seed-keywords-only embedding (all seeds, not just 10)

        Averaging the two gives more discriminative power to narrow sectors
        (e.g. dental within Health-Q) whose seeds are very specific.

        Returns:
            Dictionary mapping sector codes to embeddings
        """
        sector_embeddings = {}

        for code, sector_info in self.sectors_info.items():
            name = sector_info.get('name', '')
            description = sector_info.get('description', '')
            seed_keywords = sector_info.get('seed_keywords', [])

            # --- Embedding 1: name + description ---
            desc_text = f"{name}. {description}"
            desc_emb = self.embedding_service.embed_text(desc_text)

            if seed_keywords:
                # --- Embedding 2: all seeds joined as a sentence ---
                seed_text = ", ".join(seed_keywords)
                seed_emb = self.embedding_service.embed_text(seed_text)
                # Average the two for a balanced sector vector
                combined = np.mean(np.vstack([desc_emb, seed_emb]), axis=0)
            else:
                combined = desc_emb

            sector_embeddings[code] = combined

        return sector_embeddings

    def classify(
        self,
        text: str,
        top_k: int = 1,
    ) -> List[Tuple[str, float]]:
        """
        Classify a text into sector(s).

        Args:
            text: Input text to classify
            top_k: Number of top sectors to return

        Note:
            The threshold filter runs *before* the top-k cut, so this can return
            fewer than ``top_k`` sectors -- and for a document whose scores all
            sit below the threshold, none at all. Measuring top-k accuracy over
            this method therefore scores a shorter list than k and overstates
            the result; use :meth:`classify_with_details`, which always reports
            a best match, or read the unfiltered ranking directly.

        Returns:
            List of (sector_code, confidence_score) tuples, sorted by
            confidence; at most ``top_k`` and possibly fewer
        """
        text_embedding = self.embedding_service.embed_text(text)

        similarities = []

        for sector_code, sector_embedding in self.sector_embeddings.items():
            similarity = self.embedding_service.similarity(
                text_embedding,
                sector_embedding,
                metric="cosine"
            )
            similarities.append((sector_code, float(similarity)))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = [
            (code, score) for code, score in similarities
            if score >= self.confidence_threshold
        ]
        return results[:top_k]

    def classify_with_details(
        self,
        text: str,
        top_k: int = 3
    ) -> Dict:
        """
        Classify text with detailed information.

        Args:
            text: Input text
            top_k: Number of top sectors to return

        Returns:
            Dictionary with classification results and details
        """
        # Get all similarities (no threshold filtering) for top_sector selection
        text_embedding = self.embedding_service.embed_text(text)
        similarities = []
        for sector_code, sector_embedding in self.sector_embeddings.items():
            similarity = self.embedding_service.similarity(
                text_embedding, sector_embedding, metric="cosine"
            )
            similarities.append((sector_code, float(similarity)))
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Build detailed results; always populate top_sector with the best match
        results = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'classifications': [],
            'top_sector': similarities[0][0] if similarities else None
        }

        # Populate the classifications list (threshold-filtered, limited to top_k)
        count = 0
        for i, (sector_code, confidence) in enumerate(similarities):
            if confidence < self.confidence_threshold:
                continue
            sector_info = self.sectors_info.get(sector_code, {})
            results['classifications'].append({
                'rank': count + 1,
                'sector_code': sector_code,
                'sector_name': sector_info.get('name', 'Unknown'),
                'confidence': confidence
            })
            count += 1
            if count >= top_k:
                break

        # If no classification passed the threshold, include the best match anyway
        if not results['classifications'] and similarities:
            best_code, best_conf = similarities[0]
            sector_info = self.sectors_info.get(best_code, {})
            results['classifications'].append({
                'rank': 1,
                'sector_code': best_code,
                'sector_name': sector_info.get('name', 'Unknown'),
                'confidence': best_conf
            })

        return results

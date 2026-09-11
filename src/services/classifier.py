"""
Sector Classifier Service

Classifies business descriptions into sectors using embedding similarity.
"""

from models.taxonomy import load_sectors
import logging
from typing import Dict
import numpy as np

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
        return load_sectors(self.sectors_file)

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

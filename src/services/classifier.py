"""
Sector Classifier Service

Classifies business descriptions into sectors using embedding similarity.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import json
from pathlib import Path


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

        # Load sector information
        self.sectors_info = self._load_sectors_info()
        print(f"Loaded {len(self.sectors_info)} sectors")

        # Generate sector embeddings
        self.sector_embeddings = self._build_sector_embeddings()
        print(f"Built embeddings for {len(self.sector_embeddings)} sectors")

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
            print(f"Sectors file not found: {self.sectors_file}")
        except Exception as e:
            print(f"Error loading sectors: {e}")

        return sectors_info

    def _build_sector_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Build embeddings for sector descriptions.

        Returns:
            Dictionary mapping sector codes to embeddings
        """
        sector_embeddings = {}

        for code, sector_info in self.sectors_info.items():
            # Combine sector name, description, and seed keywords
            name = sector_info.get('name', '')
            description = sector_info.get('description', '')
            seed_keywords = sector_info.get('seed_keywords', [])

            # Create comprehensive sector text
            sector_text = f"{name}. {description}. Key terms: {', '.join(seed_keywords[:10])}"

            # Generate embedding
            embedding = self.embedding_service.embed_text(sector_text)
            sector_embeddings[code] = embedding

        return sector_embeddings

    def classify(
        self,
        text: str,
        top_k: int = 1,
        return_scores: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Classify a text into sector(s).

        Args:
            text: Input text to classify
            top_k: Number of top sectors to return
            return_scores: Whether to return confidence scores

        Returns:
            List of (sector_code, confidence_score) tuples, sorted by confidence
        """
        # Generate embedding for input text
        text_embedding = self.embedding_service.embed_text(text)

        # Calculate similarity with all sectors
        similarities = []

        for sector_code, sector_embedding in self.sector_embeddings.items():
            similarity = self.embedding_service.similarity(
                text_embedding,
                sector_embedding,
                metric="cosine"
            )
            similarities.append((sector_code, float(similarity)))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Filter by confidence threshold
        if return_scores:
            results = [
                (code, score) for code, score in similarities
                if score >= self.confidence_threshold
            ]
        else:
            results = [code for code, score in similarities if score >= self.confidence_threshold]

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
        # Get classifications
        classifications = self.classify(text, top_k=top_k, return_scores=True)

        # Build detailed results
        results = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'classifications': [],
            'top_sector': None
        }

        for i, (sector_code, confidence) in enumerate(classifications):
            sector_info = self.sectors_info.get(sector_code, {})
            classification_detail = {
                'rank': i + 1,
                'sector_code': sector_code,
                'sector_name': sector_info.get('name', 'Unknown'),
                'confidence': confidence
            }
            results['classifications'].append(classification_detail)

            if i == 0:
                results['top_sector'] = sector_code

        return results

    def batch_classify(
        self,
        texts: List[str],
        top_k: int = 1
    ) -> List[Dict]:
        """
        Classify multiple texts.

        Args:
            texts: List of texts to classify
            top_k: Number of top sectors per text

        Returns:
            List of classification results
        """
        results = []

        for text in texts:
            classification = self.classify_with_details(text, top_k=top_k)
            results.append(classification)

        return results

    def get_sector_info(self, sector_code: str) -> Dict:
        """
        Get detailed information about a sector.

        Args:
            sector_code: Sector code

        Returns:
            Dictionary with sector information
        """
        return self.sectors_info.get(sector_code, {})

    def list_sectors(self) -> List[Dict]:
        """
        List all available sectors with basic information.

        Returns:
            List of sectors with code, name, and description
        """
        sectors_list = []

        for code, info in self.sectors_info.items():
            sectors_list.append({
                'code': code,
                'name': info.get('name', 'Unknown'),
                'description': info.get('description', '')
            })

        return sorted(sectors_list, key=lambda x: x['code'])

    def set_confidence_threshold(self, threshold: float):
        """
        Set the confidence threshold for classifications.

        Args:
            threshold: Threshold value (0-1)
        """
        if 0 <= threshold <= 1:
            self.confidence_threshold = threshold
        else:
            print(f"Invalid threshold: {threshold}. Must be between 0 and 1.")
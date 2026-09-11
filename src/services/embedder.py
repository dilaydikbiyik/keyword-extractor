"""
Embedding Service for Keyword Extraction

Provides embeddings for texts and sector descriptions using
sentence-transformers for multilingual support.
"""

import logging
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating and managing embeddings using SentenceTransformers.

    Supports multilingual embeddings for German, Turkish, and English.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the SentenceTransformer model
            cache_dir: Directory to cache embeddings
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or "data/cache"
        self.device = device

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

        # Cache for embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._load_cache()

    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            use_cache: Whether to use cached embedding if available

        Returns:
            Embedding vector (ndarray)
        """
        cache_key = self._get_cache_key(text)

        if use_cache and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        embedding = self.model.encode(text, convert_to_numpy=True)

        # Store in cache
        if use_cache:
            self._embedding_cache[cache_key] = embedding

        return embedding

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        use_cache: bool = True,
        show_progress: bool = False
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            use_cache: Whether to use cached embeddings
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i, text in enumerate(texts):
            if show_progress and i % max(1, len(texts) // 10) == 0:
                logger.info(f"Embedding progress: {i}/{len(texts)}")

            embeddings.append(self.embed_text(text, use_cache=use_cache))

        return embeddings

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine', 'euclidean')

        Returns:
            Similarity score between 0 and 1
        """
        if metric == "cosine":
            # Cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(
                embedding1.reshape(1, -1),
                embedding2.reshape(1, -1)
            )[0][0]
            return float(similarity)

        elif metric == "euclidean":
            # Euclidean distance converted to similarity
            from sklearn.metrics.pairwise import euclidean_distances
            distance = euclidean_distances(
                embedding1.reshape(1, -1),
                embedding2.reshape(1, -1)
            )[0][0]
            # Convert to similarity (0-1 range)
            similarity = 1 / (1 + distance)
            return float(similarity)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text (using hash)."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()

    def _load_cache(self):
        """Load cached embeddings from disk if available."""
        cache_file = os.path.join(self.cache_dir, "embeddings.npz")
        try:
            if os.path.exists(cache_file):
                data = np.load(cache_file, allow_pickle=True)
                self._embedding_cache = dict(data['embeddings'].item())
                logger.info(f"Loaded {len(self._embedding_cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")

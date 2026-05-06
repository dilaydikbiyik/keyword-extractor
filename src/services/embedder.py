"""
Embedding Service for Keyword Extraction

Provides embeddings for texts and sector descriptions using
sentence-transformers for multilingual support.
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from pathlib import Path


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

        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Load model
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

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
        # Create cache key
        cache_key = self._get_cache_key(text)

        # Check cache
        if use_cache and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Store in cache
        if use_cache:
            self._embedding_cache[cache_key] = embedding

        return embedding

    def embed_document(
        self,
        text: str,
        chunk_size: int = 256,
        overlap: int = 32,
        max_length: int = 512,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Embed a (potentially long) document using overlapping chunking.

        Texts that fit within *max_length* tokens are encoded directly
        (fast path).  Longer texts are split into overlapping word-based
        chunks; each chunk is encoded and the results are averaged
        (average pooling), preventing silent truncation.

        Args:
            text: Document text to embed
            chunk_size: Target chunk size in words
            overlap: Overlap between consecutive chunks in words
            max_length: Token threshold below which chunking is skipped
            use_cache: Whether to use the embedding cache

        Returns:
            Averaged embedding vector (ndarray)
        """
        # Rough token estimate: words ≈ tokens (safe upper bound)
        words = text.split()
        if len(words) <= max_length:
            # Fast path — text fits in one encode call
            return self.embed_text(text, use_cache=use_cache)

        # Build overlapping chunks
        step = max(1, chunk_size - overlap)
        chunks: List[str] = []
        for start in range(0, len(words), step):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break

        # Encode all chunks and average-pool
        chunk_embeddings = [
            self.embed_text(chunk, use_cache=use_cache) for chunk in chunks
        ]
        averaged = np.mean(np.vstack(chunk_embeddings), axis=0)
        return averaged


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
                print(f"Embedding progress: {i}/{len(texts)}")

            embeddings.append(self.embed_text(text, use_cache=use_cache))

        return embeddings

    def embed_sector_descriptions(
        self,
        sector_file: str = "data/taxonomy/sector_descriptions.txt"
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for sector descriptions.

        Args:
            sector_file: Path to sector descriptions file

        Returns:
            Dictionary mapping sector codes to embeddings
        """
        sector_embeddings = {}

        try:
            with open(sector_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse sectors (format: ## A - Name)
            sectors = content.split('\n##')
            for sector_block in sectors:
                if not sector_block.strip():
                    continue

                lines = sector_block.strip().split('\n')
                if not lines:
                    continue

                # Extract sector code
                header = lines[0].strip()
                if not header.startswith(' '):
                    header = '##' + header

                # Extract sector code (first letter after ##)
                code_match = header.split('-')[0].strip().replace('#', '').strip()
                if len(code_match) == 1 and code_match.isalpha():
                    sector_code = code_match.upper()

                    # Get description text
                    description = ' '.join(lines[1:])

                    if description.strip():
                        # Generate embedding
                        embedding = self.embed_text(description)
                        sector_embeddings[sector_code] = embedding

        except FileNotFoundError:
            print(f"Sector file not found: {sector_file}")
        except Exception as e:
            print(f"Error loading sector descriptions: {e}")

        return sector_embeddings

    def load_sector_embeddings_from_json(
        self,
        sectors_file: str = "data/taxonomy/sectors.json"
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings from sector JSON file.

        Args:
            sectors_file: Path to sectors.json

        Returns:
            Dictionary mapping sector codes to embeddings
        """
        sector_embeddings = {}

        try:
            with open(sectors_file, 'r', encoding='utf-8') as f:
                sectors_data = json.load(f)

            sectors = sectors_data.get('sectors', {})

            for code, sector_info in sectors.items():
                # Use sector name and description for embedding
                name = sector_info.get('name', '')
                description = sector_info.get('description', '')
                text = f"{name}. {description}"

                if text.strip():
                    embedding = self.embed_text(text)
                    sector_embeddings[code] = embedding

        except FileNotFoundError:
            print(f"Sectors file not found: {sectors_file}")
        except Exception as e:
            print(f"Error loading sectors: {e}")

        return sector_embeddings

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

    def most_similar(
        self,
        embedding: np.ndarray,
        candidates: Dict[str, np.ndarray],
        top_k: int = 5,
        metric: str = "cosine"
    ) -> List[Tuple[str, float]]:
        """
        Find most similar candidates to an embedding.

        Args:
            embedding: Query embedding
            candidates: Dictionary of candidate embeddings {id: embedding}
            top_k: Number of top similar items to return
            metric: Similarity metric

        Returns:
            List of (candidate_id, similarity) tuples, sorted by similarity
        """
        similarities = []

        for candidate_id, candidate_embedding in candidates.items():
            sim = self.similarity(embedding, candidate_embedding, metric=metric)
            similarities.append((candidate_id, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

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
                print(f"Loaded {len(self._embedding_cache)} cached embeddings")
        except Exception as e:
            print(f"Could not load cache: {e}")

    def save_cache(self):
        """Save embeddings cache to disk."""
        cache_file = os.path.join(self.cache_dir, "embeddings.npz")
        try:
            np.savez_compressed(
                cache_file,
                embeddings=np.array(self._embedding_cache, dtype=object)
            )
            print(f"Saved {len(self._embedding_cache)} embeddings to cache")
        except Exception as e:
            print(f"Could not save cache: {e}")

    def save_sector_embeddings(
        self,
        sector_embeddings: Dict[str, np.ndarray],
        output_file: str = "data/taxonomy/sector_embeddings.npy"
    ):
        """
        Save sector embeddings to file.

        Args:
            sector_embeddings: Dictionary of sector embeddings
            output_file: Path to save embeddings
        """
        try:
            # Convert to numpy array for saving
            embeddings_array = {
                code: embedding.astype(np.float32)
                for code, embedding in sector_embeddings.items()
            }

            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            np.save(output_file, embeddings_array, allow_pickle=True)
            print(f"Saved sector embeddings to {output_file}")
        except Exception as e:
            print(f"Error saving sector embeddings: {e}")

    def load_sector_embeddings(
        self,
        input_file: str = "data/taxonomy/sector_embeddings.npy"
    ) -> Dict[str, np.ndarray]:
        """
        Load sector embeddings from file.

        Args:
            input_file: Path to embeddings file

        Returns:
            Dictionary of sector embeddings
        """
        try:
            embeddings_array = np.load(input_file, allow_pickle=True).item()
            print(f"Loaded {len(embeddings_array)} sector embeddings")
            return embeddings_array
        except FileNotFoundError:
            print(f"Embeddings file not found: {input_file}")
            return {}
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return {}
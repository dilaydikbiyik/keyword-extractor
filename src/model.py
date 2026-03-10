from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class KeywordModel:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        # Çok dilli BERT modelini yükle
        self.kw_model = KeyBERT(model_name)

    def get_embeddings(self, keywords):
        # Kelimelerin vektör karşılıklarını (embedding) al
        return self.kw_model.model.encode(keywords)

    def calculate_custom_scores(self, candidate_embeddings, seed_embeddings):
        """
        Makaledeki Two-Part Scoring sistemini uygular:
        1. Average Scoring: Adayın tüm tohumlara ortalama benzerliği.
        2. Max Scoring: Adayın en benzer olduğu tek tohuma benzerliği.
        """
        # Benzerlik matrisini hesapla
        similarities = cosine_similarity(candidate_embeddings, seed_embeddings)

        avg_scores = np.mean(similarities, axis=1)  # Ortalama skor
        max_scores = np.max(similarities, axis=1)  # Maksimum skor

        # Final skoru: Ortalama ve Maksimumun aritmetik ortalaması
        final_scores = (avg_scores + max_scores) / 2
        return final_scores
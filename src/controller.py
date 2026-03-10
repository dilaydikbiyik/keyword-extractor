import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .model import KeywordModel


class ExtractionController:
    def __init__(self):
        self.model = KeywordModel()

    def run_iteration(self, docs, current_seeds, n_new_seeds=3):
        """
        Makaledeki iki aşamalı skorlama şemasını kullanarak adayları reorder eder. [cite: 494, 527]
        """
        # 1. KeyBERT ile aday kelimeleri çıkar (Guided) [cite: 523, 524]
        # Bu aşamada tohum kelimelere odaklanıyoruz [cite: 525]
        keywords_with_scores = self.model.kw_model.extract_keywords(
            docs,
            seed_keywords=current_seeds,
            top_n=30,  # Değerlendirmek için geniş bir aday listesi alıyoruz
            keyphrase_ngram_range=(1, 1)  # Alman sektörel verileri için unigram verimlidir [cite: 578, 579, 580]
        )

        if not keywords_with_scores:
            return []

        # Adayları ve tohumları vektöre çevir
        candidates = [kw for kw, score in keywords_with_scores]
        candidate_embeddings = self.model.get_embeddings(candidates)
        seed_embeddings = self.model.get_embeddings(current_seeds)

        # 2. Two-Part Scoring (Average + Max) [cite: 528]
        similarities = cosine_similarity(candidate_embeddings, seed_embeddings)

        # Ortalama Benzerlik (Average Scoring) [cite: 529]
        avg_scores = np.mean(similarities, axis=1)
        # Maksimum Benzerlik (Max Scoring) [cite: 530]
        max_scores = np.max(similarities, axis=1)

        # Final Skoru: İkisinin ortalaması [cite: 531]
        final_scores = (avg_scores + max_scores) / 2

        # Skorlara göre adayları sırala
        ranked_candidates = sorted(zip(candidates, final_scores), key=lambda x: x[1], reverse=True)

        # En iyi n tanesini yeni tohum olarak döndür [cite: 533, 566]
        return ranked_candidates[:n_new_seeds]
import pandas as pd
from src.preprocessing import clean_text
from src.model import KeywordModel
from src.controller import ExtractionController


def main():
    # 1. Veriyi Yükle
    df = pd.read_csv('data/handelsregister_sample_10k.csv')
    # 2. Sektör Seçimi
    all_docs = df['purpose'].tolist()

    # 3. Metin Temizleme
    cleaned_docs = [clean_text(doc) for doc in all_docs[:100]]  # Test için ilk 100 döküman

    # 4. Makale Parametreleri
    # n_iterations: Kaç tur dönüleceği (Makale varsayılanı: 5)
    # n_new_seeds: Her turda eklenecek yeni kelime sayısı (Makale varsayılanı: 3)
    # percentile_newseed: Hangi başarı dilimindeki kelimeler alınacak (Makale varsayılanı: 99)
    n_iterations = 5
    n_new_seeds = 3

    # 5. Başlangıç Tohumları (Makale Tablo/Listing 2'den)
    current_seeds = [
        'Heizkraftwerke', 'Elektrizitaetserzeugung', 'Blockheizkraftwerk',
        'Waermeversorgung', 'Solarstromerzeugung', 'Energieversorgung'
    ]  #

    controller = ExtractionController()

    print(f"--- Starting Extraction for Sector: Energy ---")

    for i in range(n_iterations):  #
        print(f"\nIteration {i + 1} started...")

        # Yeni adayları çıkar ve tohum listesini genişlet
        new_candidates = controller.run_iteration(cleaned_docs, current_seeds, n_new_seeds)

        # Yeni bulunanları listeye ekle (Expand)
        current_seeds.extend([kw for kw, score in new_candidates])

        print(f"New seeds found: {[kw for kw, score in new_candidates]}")
        print(f"Total seeds in dictionary: {len(current_seeds)}")


if __name__ == "__main__":
    main()
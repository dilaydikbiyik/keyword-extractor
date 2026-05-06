#!/usr/bin/env python3
"""
Quick Start Guide - Multilingual Keyword Extraction Pipeline
"""

import sys
sys.path.insert(0, 'src')

from utils.preprocessing import TextPreprocessor
from services.embedder import EmbeddingService
from services.classifier import SectorClassifier
from services.extractor import KeywordExtractor
from services.filter import KeywordFilter
from controllers.controller import ExtractionController


def main():
    print("="*80)
    print("KEYWORD EXTRACTION PIPELINE - QUICK START")
    print("="*80)
    
    # Step 1: Initialize components
    print("\n[1/4] Initializing components...")
    
    preprocessor = TextPreprocessor()
    embedder = EmbeddingService()
    classifier = SectorClassifier(embedder)
    extractor = KeywordExtractor()
    keyword_filter = KeywordFilter()
    
    controller = ExtractionController(
        embedding_service=embedder,
        classifier=classifier,
        extractor=extractor,
        keyword_filter=keyword_filter,
        preprocessor=preprocessor
    )
    print("      ✓ Components initialized")
    
    # Step 2: Sample texts
    print("\n[2/4] Loading sample texts...")
    
    sample_texts = [
        "Softwareentwicklung und Programmierung für Web- und Mobileanwendungen. API-Integrationsdienste und Cloud-Lösungen.",
        "Handel mit Elektronik, Computern und Mobiltelefonen. Großhandel und Einzelhandel. E-Commerce-Plattform.",
        "Zahnklinik mit modernen Behandlungsmethoden. Zahnimplantate, Zahnbleaching und Prophylaxe.",
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"      {i}. {text[:60]}...")
    
    # Step 3: Extract keywords
    print("\n[3/4] Extracting keywords...")
    
    results = controller.extract_batch(sample_texts, top_n_keywords=5)
    
    # Step 4: Display results
    print("\n[4/4] Results:\n")
    
    for i, (text, result) in enumerate(zip(sample_texts, results), 1):
        print(f"{'='*80}")
        print(f"Sample {i}: {text[:70]}...")
        print(f"{'='*80}")
        
        if result['status'] == 'success':
            sector_info = result['sector_classification']
            print(f"Language:              {result['language']}")
            print(f"Primary Sector:        {sector_info['top_sector']} (confidence: {sector_info['classifications'][0]['confidence']:.1%})")
            print(f"\nExtracted Keywords:")
            for j, kw in enumerate(result['keywords'][:5], 1):
                print(f"  {j}. {kw['keyword']:<30} (score: {kw['score']:.3f})")
        else:
            print(f"Error: {result['error']}")
        
        print()
    
    # Step 5: Summary statistics
    print("="*80)
    stats = controller.get_extraction_stats(results)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total documents:       {stats['total_documents']}")
    print(f"Successful:            {stats['successful']} ({stats['success_rate']:.1%})")
    print(f"Failed:                {stats['failed']}")
    print(f"Total keywords:        {stats['total_keywords_extracted']}")
    print(f"Avg keywords/doc:      {stats['avg_keywords_per_document']:.1f}")
    print(f"Avg confidence:        {stats['avg_confidence']:.3f}")
    print(f"\nSector Distribution:")
    for sector, count in sorted(stats['sector_distribution'].items()):
        print(f"  {sector}: {count}")
    
    print("\n" + "="*80)
    print("✓ Quick Start Complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run: python test_integration.py")
    print("  2. Process your CSV: python main.py")
    print("  3. Analyze results: jupyter notebook notebooks/initial_analysis.ipynb")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python
"""
Integration test script for the keyword extraction pipeline.
Tests all components working together.
"""

import sys
sys.path.append('src')

from utils.preprocessing import TextPreprocessor
from services.embedder import EmbeddingService
from services.classifier import SectorClassifier
from services.extractor import KeywordExtractor
from services.filter import KeywordFilter
from controllers.controller import ExtractionController


def test_pipeline():
    """Test the complete extraction pipeline."""
    
    print("="*80)
    print("KEYWORD EXTRACTION PIPELINE - INTEGRATION TEST")
    print("="*80)
    
    # Sample German business descriptions
    test_texts = [
        "Softwareentwicklung, Programmierung und IT-Beratung für Unternehmenskunden. Wir bieten maßgeschneiderte Lösungen für Web- und Mobileanwendungen.",
        "Handel mit Elektronik und Computern. Großhandel und Einzelhandel mit Neuware und refurbished Produkten. Online-Verkauf und Ladengeschäfte.",
        "Medizinische Dienstleistungen, ärztliche Beratung und Therapie. Zahnklinik mit modernem Equipment.",
    ]
    
    try:
        # Initialize components
        print("\n1. Initializing components...")
        print("   - TextPreprocessor...")
        preprocessor = TextPreprocessor()
        
        print("   - EmbeddingService (this may take a moment on first run)...")
        embedder = EmbeddingService()
        
        print("   - SectorClassifier...")
        classifier = SectorClassifier(embedder)
        
        print("   - KeywordExtractor...")
        extractor = KeywordExtractor()
        
        print("   - KeywordFilter...")
        keyword_filter = KeywordFilter()
        
        # Note: Skipping LLMValidator as it requires API key
        
        print("   - ExtractionController...")
        controller = ExtractionController(
            embedding_service=embedder,
            classifier=classifier,
            extractor=extractor,
            keyword_filter=keyword_filter,
            preprocessor=preprocessor,
            validator=None
        )
        
        print("\n✓ All components initialized successfully!")
        
        # Test extraction
        print("\n2. Testing keyword extraction...")
        print("-" * 80)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text[:70]}...")
            
            result = controller.extract(
                text,
                top_n_keywords=5,
                use_validation=False,
                return_intermediate=False
            )
            
            print(f"  Status: {result['status']}")
            print(f"  Language: {result['language']}")
            
            if result['status'] == 'success':
                sector_info = result['sector_classification']
                print(f"  Primary Sector: {sector_info['top_sector']}")
                print(f"  Sector Confidence: {sector_info['classifications'][0]['confidence']:.3f}")
                
                keywords = result['keywords']
                print(f"  Keywords ({len(keywords)}):")
                for kw in keywords[:5]:
                    print(f"    - {kw['keyword']}: {kw['score']:.3f}")
        
        print("\n" + "="*80)
        print("✓ INTEGRATION TEST PASSED!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_pipeline()

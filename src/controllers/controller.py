"""
Main Extraction Controller

Orchestrates the complete keyword extraction pipeline:
1. Text preprocessing
2. Sector classification
3. Guided keyword extraction
4. Keyword filtering
5. Optional LLM validation
"""

from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime


class ExtractionController:
    """
    Main controller for the keyword extraction pipeline.
    
    Coordinates all services to extract keywords from business descriptions.
    """

    def __init__(
        self,
        embedding_service,
        classifier,
        extractor,
        keyword_filter,
        preprocessor,
        validator=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the extraction controller.

        Args:
            embedding_service: EmbeddingService instance
            classifier: SectorClassifier instance
            extractor: KeywordExtractor instance
            keyword_filter: KeywordFilter instance
            preprocessor: TextPreprocessor instance
            validator: Optional LLMValidator instance
            config: Configuration dictionary
        """
        self.embedding_service = embedding_service
        self.classifier = classifier
        self.extractor = extractor
        self.keyword_filter = keyword_filter
        self.preprocessor = preprocessor
        self.validator = validator
        self.config = config or {}

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def extract(
        self,
        text: str,
        top_n_keywords: int = 10,
        use_validation: bool = False,
        return_intermediate: bool = False
    ) -> Dict:
        """
        Extract keywords from a single business description.

        Args:
            text: Business description text
            top_n_keywords: Number of keywords to extract
            use_validation: Whether to use LLM validation
            return_intermediate: Return intermediate results for debugging

        Returns:
            Dictionary with extraction results
        """
        self.logger.info(f"Starting extraction for text: {text[:60]}...")

        result = {
            'input_text': text,
            'timestamp': datetime.now().isoformat(),
            'intermediate_steps': {} if return_intermediate else None
        }

        try:
            # Step 1: Preprocess text
            self.logger.info("Step 1: Preprocessing text...")
            preprocessing_result = self.preprocessor.preprocess_pipeline(text)
            result['language'] = preprocessing_result['detected_language']

            if return_intermediate:
                result['intermediate_steps']['preprocessing'] = preprocessing_result

            # Step 2: Classify sector
            self.logger.info("Step 2: Classifying sector...")
            sector_result = self.classifier.classify_with_details(text, top_k=3)
            result['sector_classification'] = sector_result
            primary_sector = sector_result.get('top_sector')

            if return_intermediate:
                result['intermediate_steps']['sector_classification'] = sector_result

            if not primary_sector:
                self.logger.warning("Could not determine sector")
                result['status'] = 'sector_classification_failed'
                return result

            # Step 3: Extract keywords (guided by sector)
            self.logger.info(f"Step 3: Extracting keywords for sector {primary_sector}...")
            keywords = self.extractor.extract_keywords_guided_by_sector(
                text,
                primary_sector,
                top_n=top_n_keywords * 2,  # Extract more to allow for filtering
                language=preprocessing_result['detected_language']
            )

            if return_intermediate:
                result['intermediate_steps']['keyword_extraction'] = keywords

            # Step 4: Filter keywords
            self.logger.info("Step 4: Filtering keywords...")
            filtered_keywords = self.keyword_filter.apply_all_filters(
                keywords,
                sector_code=primary_sector,
                top_n=top_n_keywords
            )

            if return_intermediate:
                result['intermediate_steps']['filtering'] = filtered_keywords

            # Step 5: LLM Validation (optional)
            if use_validation and self.validator and self.validator.is_available():
                self.logger.info("Step 5: LLM validation...")
                validated_keywords = self.validator.validate_keywords(
                    text,
                    filtered_keywords,
                    sector=primary_sector
                )
                result['keywords'] = validated_keywords
            else:
                result['keywords'] = [
                    {
                        'keyword': kw,
                        'score': score,
                        'validated': False
                    }
                    for kw, score in filtered_keywords
                ]

            result['status'] = 'success'
            self.logger.info("Extraction completed successfully")

        except Exception as e:
            self.logger.error(f"Error during extraction: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def extract_batch(
        self,
        texts: List[str],
        top_n_keywords: int = 10,
        use_validation: bool = False,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Extract keywords from multiple texts.

        Args:
            texts: List of business descriptions
            top_n_keywords: Number of keywords per text
            use_validation: Whether to use LLM validation
            show_progress: Whether to show progress

        Returns:
            List of extraction results
        """
        self.logger.info(f"Starting batch extraction for {len(texts)} texts...")

        results = []

        for i, text in enumerate(texts):
            if show_progress:
                print(f"Processing {i + 1}/{len(texts)}...")

            result = self.extract(
                text,
                top_n_keywords=top_n_keywords,
                use_validation=use_validation
            )

            results.append(result)

        self.logger.info(f"Batch extraction completed. Processed {len(texts)} texts.")

        return results

    def extract_from_dataframe(
        self,
        df,
        text_column: str,
        top_n_keywords: int = 10,
        use_validation: bool = False
    ):
        """
        Extract keywords from a pandas DataFrame.

        Args:
            df: DataFrame with business descriptions
            text_column: Name of the column containing texts
            top_n_keywords: Number of keywords per text
            use_validation: Whether to use LLM validation

        Yields:
            Extraction results with DataFrame row index
        """
        self.logger.info(f"Starting extraction from DataFrame ({len(df)} rows)...")

        for idx, row in df.iterrows():
            text = row[text_column]

            result = self.extract(
                text,
                top_n_keywords=top_n_keywords,
                use_validation=use_validation
            )

            result['row_index'] = idx
            yield result

    def get_extraction_stats(self, results: List[Dict]) -> Dict:
        """
        Get statistics from extraction results.

        Args:
            results: List of extraction results

        Returns:
            Statistics dictionary
        """
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = sum(1 for r in results if r.get('status') != 'success')

        sector_counts = {}
        total_keywords = 0
        avg_confidence = 0
        confidence_sum = 0

        for result in results:
            if result.get('status') == 'success':
                # Count sectors
                sector = result.get('sector_classification', {}).get('top_sector')
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

                # Count keywords
                keywords = result.get('keywords', [])
                total_keywords += len(keywords)
                confidence_sum += sum(kw.get('score', 0) for kw in keywords)

        if successful > 0 and total_keywords > 0:
            avg_confidence = confidence_sum / total_keywords

        return {
            'total_documents': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(results) if results else 0,
            'total_keywords_extracted': total_keywords,
            'avg_keywords_per_document': total_keywords / successful if successful > 0 else 0,
            'avg_confidence': avg_confidence,
            'sector_distribution': sector_counts
        }

    def configure(self, config: Dict):
        """
        Update configuration.

        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
        self.logger.info(f"Configuration updated: {config}")

    def get_config(self) -> Dict:
        """Get current configuration."""
        return self.config.copy()
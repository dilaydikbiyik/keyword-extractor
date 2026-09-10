"""
Main Extraction Controller

Orchestrates the complete keyword extraction pipeline:
1. Text preprocessing
2. Sector classification
3. Guided keyword extraction
4. Keyword filtering
5. Optional LLM validation
"""

from typing import List, Dict, Optional
import logging
import json
import os
import time
from datetime import datetime
from pathlib import Path


def _top_confidence(result: Dict) -> float:
    """Confidence of the best-ranked section, or 0.0 when none was reported."""
    classifications = result.get("sector_classification", {}).get("classifications")
    if not classifications:
        return 0.0
    return classifications[0].get("confidence") or 0.0


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
            config: Configuration dictionary. Recognised key:
                ``classify_preprocessed_text`` (bool) — send the cleaned text to
                the classifier instead of the raw purpose field.  Off by default;
                see ``classify_preprocessed_text`` below.
        """
        self.embedding_service = embedding_service
        self.classifier = classifier
        self.extractor = extractor
        self.keyword_filter = keyword_filter
        self.preprocessor = preprocessor
        self.validator = validator
        self.config = config or {}

        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    #: Package loggers the configured level applies to. The services log their
    #: own progress, so setting ``logging.level: INFO`` has to reach them too --
    #: configuring only this module's logger would silence everything below it.
    LOG_PACKAGES = ("controllers", "services", "models", "utils")

    def _setup_logging(self):
        """Attach one stream handler per package logger and set the level.

        The handler is added once per logger, not once per controller: building
        several controllers in one process used to multiply every log line.
        The default level is WARNING so the pipeline is quiet in normal use;
        set ``logging.level`` in the config to follow the per-step progress.
        """
        level = getattr(
            logging, str(self.config.get("log_level", "WARNING")).upper(), logging.WARNING
        )
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        for package in self.LOG_PACKAGES:
            package_logger = logging.getLogger(package)
            if not package_logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(formatter)
                package_logger.addHandler(handler)
            package_logger.setLevel(level)

    def extract(
        self,
        text: str,
        top_n_keywords: Optional[int] = None,
        use_validation: bool = False,
        return_intermediate: bool = False
    ) -> Dict:
        """
        Extract keywords from a single business description.

        Args:
            text: Business description text
            top_n_keywords: Number of keywords to extract; defaults to the
                configured ``extraction.top_n_final``
            use_validation: Whether to use LLM validation
            return_intermediate: Return intermediate results for debugging

        Returns:
            Dictionary with extraction results
        """
        if top_n_keywords is None:
            top_n_keywords = self.config.get("top_n_keywords", 10)
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
            sector_result = self.classifier.classify_with_details(
                self._classifier_input(text, preprocessing_result),
                top_k=self.config.get("top_k_sectors", 3),
            )
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
            # Extract more than requested: the filter stage drops candidates.
            over_extract = top_n_keywords * 2
            diversity = self.config.get("diversity", 0.7)
            if self.config.get("guided_mode", True):
                keywords = self.extractor.extract_keywords_guided_by_sector(
                    text,
                    primary_sector,
                    top_n=over_extract,
                    language=preprocessing_result['detected_language'],
                    diversity=diversity,
                )
            else:
                keywords = self.extractor.extract_keywords(
                    text, top_n=over_extract, diversity=diversity
                )

            if return_intermediate:
                result['intermediate_steps']['keyword_extraction'] = keywords

            # Step 4: Filter keywords
            self.logger.info("Step 4: Filtering keywords...")
            filtered_keywords = self.keyword_filter.apply_all_filters(
                keywords,
                sector_code=primary_sector,
                top_n=top_n_keywords,
                min_score=self.config.get("min_score", 0.1),
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
        top_n_keywords: Optional[int] = None,
        use_validation: bool = False,
        show_progress: bool = True,
        save_report: bool = False,
        report_dir: str = "output",
    ) -> List[Dict]:
        """
        Extract keywords from multiple texts.

        Args:
            texts: List of business descriptions
            top_n_keywords: Number of keywords per text
            use_validation: Whether to use LLM validation
            show_progress: Whether to show progress
            save_report: If True, persist a JSON summary to *report_dir*
            report_dir: Directory to write the report file

        Returns:
            List of extraction results
        """
        self.logger.info(f"Starting batch extraction for {len(texts)} texts...")
        batch_start = time.time()

        results = []

        for i, text in enumerate(texts):
            if show_progress:
                self.logger.info("Processing %d/%d...", i + 1, len(texts))

            t0 = time.time()
            result = self.extract(
                text,
                top_n_keywords=top_n_keywords,
                use_validation=use_validation
            )
            result["processing_time_ms"] = round((time.time() - t0) * 1000, 1)
            results.append(result)

        batch_elapsed = round(time.time() - batch_start, 2)
        self.logger.info(
            f"Batch extraction completed. Processed {len(texts)} texts "
            f"in {batch_elapsed}s."
        )

        if save_report:
            self.save_batch_report(results, output_dir=report_dir)

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

    def save_batch_report(
        self,
        results: List[Dict],
        output_dir: str = "output",
    ) -> str:
        """
        Save a full batch summary report to *output_dir* as a JSON file.

        The report contains:
          - Run metadata (timestamp, total documents, success rate)
          - Aggregate statistics from get_extraction_stats()
          - Per-document results (sector, language, keywords, timing)

        Args:
            results: List of extraction results from extract_batch()
            output_dir: Directory to write the report file

        Returns:
            Absolute path of the written report file
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        stats = self.get_extraction_stats(results)

        # Build per-document summary (keep it compact)
        docs_summary = []
        for r in results:
            doc = {
                "status": r.get("status"),
                "language": r.get("language"),
                "sector": r.get("sector_classification", {}).get("top_sector"),
                "confidence": _top_confidence(r),
                "keywords": [kw.get("keyword") for kw in r.get("keywords", [])],
                "processing_time_ms": r.get("processing_time_ms"),
                "error": r.get("error"),
            }
            docs_summary.append(doc)

        report = {
            "run_timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "documents": docs_summary,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"batch_report_{timestamp}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Batch report saved to {out_path}")
        return out_path

    def _classifier_input(self, text: str, preprocessing_result: Dict) -> str:
        """Choose which form of the text the sector classifier sees.

        The pipeline has always preprocessed the text and then classified the
        raw string.  Classifying the cleaned string instead scores 6.7 points
        higher on the current evaluation set (``results/tables/ablation.md``) —
        but that set holds 30 documents, so the gap is two documents and is not
        significant.  The behaviour is therefore a configuration switch, off by
        default, to be settled once the evaluation set is large enough to
        decide it.  See docs/paper_readiness.md.
        """
        if not self.config.get("classify_preprocessed_text", False):
            return text
        cleaned = (preprocessing_result or {}).get("cleaned_text", "")
        # Cleaning can empty out a very short or punctuation-only text.
        return cleaned if cleaned.strip() else text

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

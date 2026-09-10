import re
import os
import unicodedata
from typing import List, Tuple, Optional, Dict, Any
from langdetect import detect, LangDetectException
import nltk
from nltk.corpus import stopwords

# Register the project-local nltk_data directory (if present) so the project
# works without a system-wide NLTK installation.  The folder is intentionally
# excluded from git (see .gitignore); each developer downloads it once via
#   python -m nltk.downloader -d nltk_data stopwords punkt
_LOCAL_NLTK = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "nltk_data")
if os.path.isdir(_LOCAL_NLTK) and _LOCAL_NLTK not in nltk.data.path:
    nltk.data.path.insert(0, _LOCAL_NLTK)

# Tokenisation is deliberately regex-based rather than spaCy-based. spaCy was
# an optional dependency whose models silently failed to load on any
# environment where its compiled extensions disagreed with the installed numpy,
# so the same text tokenised differently from machine to machine and the
# fallback below is what actually ran. One tokeniser everywhere is worth more
# here than linguistic tokenisation on some machines only.
WORD_PATTERN = re.compile(r'\b\w+\b', re.UNICODE)

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')


class TextPreprocessor:
    """
    Comprehensive text preprocessor for multi-language business text processing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}

        # Behaviour switches, all read from the ``preprocessing`` config section.
        self.lowercase = self.config.get('lowercase', True)
        self.remove_urls = self.config.get('remove_urls', True)
        self.remove_emails = self.config.get('remove_emails', True)
        self.remove_punctuation = self.config.get('remove_punctuation', True)
        self.preserve_hyphens = self.config.get('preserve_hyphens', True)
        self.min_word_length = self.config.get('min_word_length', 3)
        self.max_ngram_length = self.config.get('max_ngram_length', 3)

        # URL and email patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

        # Special characters to preserve inside tokens
        self.preserve_chars = {'-', "'"} if self.preserve_hyphens else set()

        self.stopwords = self._load_stopwords()

        # Sector-specific stopwords (business jargon)
        self.sector_stopwords = {
            'de': {
                'gmbh', 'kg', 'ohg', 'ug', 'haftungsbeschränkt', 'gesellschaft', 'unternehmen',
                'firma', 'company', 'limited', 'ltd', 'inc', 'corporation', 'holding',
                'gruppe', 'konzern', 'betrieb', 'geschäft', 'dienstleistung', 'service',
                'beratung', 'consulting', 'entwicklung', 'development', 'produktion',
                'herstellung', 'handel', 'verkauf', 'vertrieb', 'und', 'mit', 'für',
                'von', 'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer',
                'eines', 'einem', 'einen', 'im', 'am', 'zum', 'zur', 'auf', 'aus',
                'bei', 'nach', 'vor', 'über', 'unter', 'zwischen', 'durch', 'gegen',
                'ohne', 'seit', 'bis', 'als', 'wie', 'so', 'da', 'weil', 'obwohl',
                'wenn', 'dann', 'aber', 'oder', 'sondern', 'denn', 'also'
            },
            'tr': {
                'ltd', 'şti', 'aş', 'anonim', 'şirketi', 'limited', 'şirket', 'firma',
                'şirketler', 'grup', 'holding', 'iş', 'ticaret', 'sanayi', 'hizmet',
                'danışmanlık', 'mühendislik', 'yazılım', 'teknoloji', 'üretim', 'imalat',
                'pazarlama', 'satış', 'dağıtım', 've', 'ile', 'için', 'üzerinde', 'altında',
                'önce', 'sonra', 'arasında', 'karşısında', 'nedeniyle', 'dolayısıyla',
                'ancak', 'fakat', 'ama', 'veya', 'ya da', 'yani', 'çünkü', 'zira',
                'eğer', 'ise', 'ki', 'de', 'da', 'mi', 'mı', 'mu', 'mü', 'mi', 'mı',
                'musun', 'musunuz', 'muyum', 'muyuz', 'midir', 'misin', 'misiniz',
                'miyim', 'miyiz', 'mış', 'mışsın', 'mışsınız', 'mışım', 'mışız',
                'mıştı', 'miştik', 'miştiniz', 'miştim', 'miştin', 'miştik'
            },
            'en': {
                'ltd', 'limited', 'company', 'corporation', 'inc', 'llc', 'corp',
                'business', 'enterprise', 'firm', 'organization', 'service', 'consulting',
                'development', 'production', 'manufacturing', 'trading', 'sales',
                'distribution', 'marketing', 'and', 'or', 'but', 'the', 'a', 'an',
                'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'into',
                'through', 'during', 'before', 'after', 'above', 'below', 'between',
                'among', 'within', 'without', 'against', 'along', 'around', 'beside',
                'besides', 'near', 'next', 'over', 'under', 'upon', 'via', 'while'
            }
        }

    def _load_stopwords(self) -> Dict[str, set]:
        """Load stopwords for supported languages."""
        stopwords_dict = {}

        # German stopwords
        try:
            stopwords_dict['de'] = set(stopwords.words('german'))
        except LookupError:
            stopwords_dict['de'] = set()

        # English stopwords
        try:
            stopwords_dict['en'] = set(stopwords.words('english'))
        except LookupError:
            stopwords_dict['en'] = set()

        # Turkish stopwords (NLTK)
        try:
            stopwords_dict['tr'] = set(stopwords.words('turkish'))
        except LookupError:
            stopwords_dict['tr'] = set()

        return stopwords_dict

    def clean_text(self, text: str, lang: str = "auto") -> str:
        """
        Clean and normalize text for keyword extraction.

        Args:
            text: Input text to clean
            lang: Language code ('de', 'tr', 'en', 'auto')

        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""

        if self.lowercase:
            text = text.lower()

        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)

        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)

        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

        if self.remove_punctuation:
            preserved = re.escape(''.join(sorted(self.preserve_chars)))
            text = re.sub(r'[^\w\s' + preserved + r']', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove isolated numbers (but keep numbers in words)
        text = re.sub(r'\b\d+\b', '', text)

        # Final whitespace cleanup
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of the text with confidence score.

        Args:
            text: Input text

        Returns:
            Tuple of (language_code, confidence_score)
        """
        if not text or len(text.strip()) < 10:
            return "unknown", 0.0

        try:
            # Use langdetect for language detection
            result = detect(text)
            # langdetect doesn't provide confidence, so we use a heuristic
            confidence = 0.8 if len(text) > 50 else 0.6
            return result, confidence
        except LangDetectException:
            return "unknown", 0.0

    def remove_stopwords(self, tokens: List[str], lang: str) -> List[str]:
        """
        Remove stopwords from token list.

        Args:
            tokens: List of tokens
            lang: Language code

        Returns:
            Filtered token list
        """
        if lang not in self.stopwords:
            lang = 'en'  # fallback to English

        # Combine general and sector-specific stopwords
        all_stopwords = self.stopwords.get(lang, set()) | self.sector_stopwords.get(lang, set())

        return [token for token in tokens if token.lower() not in all_stopwords]

    def tokenize_text(self, text: str, lang: str = "auto") -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text
            lang: Language code; accepted for API compatibility. The regex
                tokeniser is language-independent, so this does not change
                the result.

        Returns:
            List of tokens
        """
        if lang == "auto":
            lang, _ = self.detect_language(text)

        return WORD_PATTERN.findall(text)

    def generate_ngram_candidates(
        self, text: str, n_range: Optional[Tuple[int, int]] = None
    ) -> List[str]:
        """
        Generate n-gram candidates from text.

        Args:
            text: Input text
            n_range: Range of n-gram sizes (min_n, max_n); defaults to
                (1, the configured max_ngram_length)

        Returns:
            List of n-gram candidates
        """
        if n_range is None:
            n_range = (1, self.max_ngram_length)

        # Clean the text first
        cleaned_text = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize_text(cleaned_text)

        # Remove stopwords
        lang, _ = self.detect_language(text)
        filtered_tokens = self.remove_stopwords(tokens, lang)

        if not filtered_tokens:
            return []

        candidates = []
        seen = set()

        min_n, max_n = n_range

        for n in range(min_n, max_n + 1):
            for i in range(len(filtered_tokens) - n + 1):
                ngram = ' '.join(filtered_tokens[i:i + n])
                ngram_lower = ngram.lower()

                # Apply filters + deduplication
                if self._is_valid_ngram(ngram, n) and ngram_lower not in seen:
                    candidates.append(ngram)
                    seen.add(ngram_lower)

        return candidates

    def _is_valid_ngram(self, ngram: str, n: int) -> bool:
        """
        Check if n-gram is valid for keyword extraction.

        Args:
            ngram: N-gram candidate
            n: N-gram size

        Returns:
            True if valid, False otherwise
        """
        # Minimum character length grows with n; the unigram floor is the
        # configured min_word_length.
        floor = self.min_word_length
        if len(ngram) < floor + (n - 1) * 2:
            return False

        # Must contain only letters, spaces, hyphens, apostrophes
        if not re.match(r"^[a-zA-Z\s\-']+$", ngram):
            return False

        # Should not start or end with stopwords (for n>1)
        if n > 1:
            words = ngram.split()
            lang = 'en'  # default
            if any(word.lower() in self.stopwords.get(lang, set()) for word in [words[0], words[-1]]):
                return False

        # Should not be all uppercase (likely abbreviations)
        if ngram.isupper() and len(ngram) <= 5:
            return False

        return True

    def preprocess_pipeline(self, text: str) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.

        Args:
            text: Input text

        Returns:
            Dictionary with preprocessing results
        """
        # Detect language
        lang, confidence = self.detect_language(text)

        # Clean text
        cleaned_text = self.clean_text(text, lang)

        candidates = self.generate_ngram_candidates(text)

        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'detected_language': lang,
            'language_confidence': confidence,
            'ngram_candidates': candidates,
            'candidate_count': len(candidates)
        }

# Convenience functions for external use


def clean_text(text: str, lang: str = "auto") -> str:
    """Clean text using default preprocessor."""
    preprocessor = TextPreprocessor()
    return preprocessor.clean_text(text, lang)


def detect_language(text: str) -> Tuple[str, float]:
    """Detect language of text."""
    preprocessor = TextPreprocessor()
    return preprocessor.detect_language(text)


def generate_ngram_candidates(text: str, n_range: Tuple[int, int] = (1, 3)) -> List[str]:
    """Generate n-gram candidates from text."""
    preprocessor = TextPreprocessor()
    return preprocessor.generate_ngram_candidates(text, n_range)


def preprocess_pipeline(text: str) -> Dict[str, Any]:
    """Run complete preprocessing pipeline."""
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_pipeline(text)

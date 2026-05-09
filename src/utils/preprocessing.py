import re
import unicodedata
from typing import List, Tuple, Optional, Dict, Any
from langdetect import detect, LangDetectException
import nltk
from nltk.corpus import stopwords

# Lazy import for spacy (optional)
nlp_de = None
nlp_tr = None


def _load_spacy_models():
    """Lazy load spaCy models only when needed."""
    global nlp_de, nlp_tr
    try:
        import spacy
        nlp_de = spacy.load('de_core_news_sm')
        nlp_tr = spacy.load('tr_core_news_sm')
    except Exception as e:
        print(f"Warning: spaCy models not available: {e}")

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

        # URL and email patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

        # Special characters to preserve (hyphen, apostrophe)
        self.preserve_chars = {'-', "'"}

        # Load stopwords for different languages
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
        except:
            stopwords_dict['de'] = set()

        # English stopwords
        try:
            stopwords_dict['en'] = set(stopwords.words('english'))
        except:
            stopwords_dict['en'] = set()

        # Turkish stopwords (NLTK)
        try:
            stopwords_dict['tr'] = set(stopwords.words('turkish'))
        except:
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

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = self.url_pattern.sub(' ', text)

        # Remove emails
        text = self.email_pattern.sub(' ', text)

        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

        # Remove special characters but preserve hyphens and apostrophes
        text = re.sub(r'[^\w\s' + re.escape(''.join(self.preserve_chars)) + r']', ' ', text)

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
            lang: Language code

        Returns:
            List of tokens
        """
        if lang == "auto":
            lang, _ = self.detect_language(text)

        # Try to use spaCy for tokenization if available
        try:
            _load_spacy_models()
            if lang == 'de' and nlp_de:
                doc = nlp_de(text)
                tokens = [token.text for token in doc if token.is_alpha]
            elif lang == 'tr' and nlp_tr:
                doc = nlp_tr(text)
                tokens = [token.text for token in doc if token.is_alpha]
            else:
                # Fallback to simple whitespace tokenization
                tokens = re.findall(r'\b\w+\b', text)
        except:
            # Fallback to simple whitespace tokenization
            tokens = re.findall(r'\b\w+\b', text)

        return tokens

    def generate_ngram_candidates(self, text: str, n_range: Tuple[int, int] = (1, 3)) -> List[str]:
        """
        Generate n-gram candidates from text.

        Args:
            text: Input text
            n_range: Range of n-gram sizes (min_n, max_n)

        Returns:
            List of n-gram candidates
        """
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
        # Minimum length requirements
        min_lengths = {1: 3, 2: 5, 3: 7}  # characters
        if len(ngram) < min_lengths.get(n, 3):
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

        # Generate candidates
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
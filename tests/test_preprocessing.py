# Ön İşleme Testleri

import pytest
from src.utils.preprocessing import clean_text

def test_clean_text_basic():
    """Temel temizleme fonksiyonunu test eder"""
    text = "Bu bir TEST metni! 123"
    result = clean_text(text)
    assert result == "bu bir test metni"

def test_clean_text_urls():
    """URL temizleme test eder"""
    text = "Ziyaret edin: https://example.com"
    result = clean_text(text)
    assert "https" not in result
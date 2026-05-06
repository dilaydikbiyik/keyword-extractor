# Değerlendirme Testleri

import pytest
from src.models.evaluation import calculate_precision_at_k

def test_calculate_precision_at_k():
    """Precision@K metriğini test eder"""
    predicted = ["a", "b", "c", "d", "e"]
    ground_truth = ["a", "c", "f", "g"]
    precision = calculate_precision_at_k(predicted, ground_truth, k=3)
    assert precision == 2/3  # a ve c doğru, b yanlış
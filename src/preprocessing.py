import re

def clean_text(text):
    """
    İşletme tanımlarını temizler: Küçük harfe çevirir,
    URL ve alfanümerik olmayan karakterleri kaldırır. [cite: 1048, 1050, 1051]
    """
    text = str(text).lower()
    # URL temizleme
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Alfanümerik olmayan karakterleri boşlukla değiştir
    text = re.sub(r'\W', ' ', text)
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
from src.preprocess import clean_text


def test_clean_text_removes_links():
    text = "I love this http://example.com"
    cleaned = clean_text(text)
    assert "http" not in cleaned


def test_clean_text_lowercase():
    text = "HELLO WORLD"
    cleaned = clean_text(text)
    assert cleaned == "hello world"
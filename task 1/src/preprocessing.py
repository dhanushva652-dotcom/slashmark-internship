
import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Try to make the script self-sufficient.
# If the environment has no internet, the code still works with a regex fallback.
for resource, path in [
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4"),
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

_LEMMATIZER = WordNetLemmatizer()

@lru_cache(maxsize=1)
def _stop_words():
    try:
        return set(stopwords.words("english"))
    except Exception:
        return {
            "a", "an", "the", "and", "or", "but", "if", "while", "is", "am", "are",
            "was", "were", "be", "been", "being", "of", "to", "in", "for", "on",
            "with", "at", "by", "from", "as", "it", "this", "that", "these",
            "those", "i", "you", "he", "she", "we", "they", "me", "my", "your",
            "our", "their"
        }

def tokenize_text(text: str):
    """Tokenize text with NLTK when available, otherwise use a regex fallback."""
    text = text.lower()
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
    except Exception:
        tokens = re.findall(r"[a-zA-Z']+", text)
    return tokens

def clean_text(text):
    """Normalize, tokenize, remove stopwords and lemmatize."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z']", " ", text)
    tokens = tokenize_text(text)
    stop_words = _stop_words()
    cleaned = []
    for token in tokens:
        token = token.strip().lower()
        if len(token) < 3 or token in stop_words:
            continue
        try:
            token = _LEMMATIZER.lemmatize(token)
        except Exception:
            pass
        cleaned.append(token)
    return " ".join(cleaned)

def preprocess_data(df, text_column="Review"):
    """Add a cleaned text column to the dataframe."""
    if text_column not in df.columns:
        raise KeyError(f"Missing required column: {text_column}")
    df = df.copy()
    df["cleaned_review"] = df[text_column].fillna("").apply(clean_text)
    return df

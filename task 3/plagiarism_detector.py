from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from nltk.stem import PorterStemmer
except Exception:  # pragma: no cover
    PorterStemmer = None


DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
    "did", "do", "does", "doing", "for", "from", "had", "has", "have", "he", "her",
    "hers", "him", "his", "i", "if", "in", "into", "is", "it", "its", "just", "me",
    "more", "most", "my", "no", "not", "of", "on", "or", "our", "she", "so", "than",
    "that", "the", "their", "them", "there", "these", "they", "this", "to", "too",
    "up", "was", "we", "were", "what", "when", "where", "which", "who", "will",
    "with", "you", "your"
}


@dataclass
class Document:
    name: str
    raw_text: str
    normalized_text: str


class PlagiarismDetector:
    def __init__(
        self,
        threshold: float = 0.65,
        tfidf_weight: float = 0.75,
        fuzzy_weight: float = 0.25,
        use_stemming: bool = True,
    ) -> None:
        self.threshold = threshold
        self.tfidf_weight = tfidf_weight
        self.fuzzy_weight = fuzzy_weight
        self.use_stemming = use_stemming and PorterStemmer is not None
        self.stemmer = PorterStemmer() if self.use_stemming else None

    def load_documents(self, input_dir: Path) -> List[Document]:
        documents: List[Document] = []
        for file_path in sorted(input_dir.glob("*.txt")):
            raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
            documents.append(
                Document(
                    name=file_path.name,
                    raw_text=raw_text,
                    normalized_text=self.normalize_text(raw_text),
                )
            )
        return documents

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = []
        for token in text.split():
            if token in DEFAULT_STOPWORDS:
                continue
            if self.stemmer is not None:
                token = self.stemmer.stem(token)
            tokens.append(token)
        return " ".join(tokens)

    def _sentence_overlap(self, text_a: str, text_b: str) -> List[str]:
        sentences_a = [s.strip() for s in re.split(r"[.!?\n]+", text_a) if s.strip()]
        sentences_b = [s.strip() for s in re.split(r"[.!?\n]+", text_b) if s.strip()]
        overlaps = []

        normalized_b = {self.normalize_text(sentence) for sentence in sentences_b}
        for sentence in sentences_a:
            ns = self.normalize_text(sentence)
            if len(ns) < 20:
                continue
            if ns in normalized_b:
                overlaps.append(sentence)
        return overlaps[:5]

    def analyze(self, documents: List[Document]) -> Dict:
        names = [d.name for d in documents]
        raw_texts = [d.raw_text for d in documents]
        normalized_texts = [d.normalized_text for d in documents]

        tfidf = TfidfVectorizer(ngram_range=(1, 2), analyzer="word")
        tfidf_matrix = tfidf.fit_transform(normalized_texts)
        cosine_matrix = cosine_similarity(tfidf_matrix)

        scored_pairs = []
        for i, j in combinations(range(len(documents)), 2):
            tfidf_score = float(cosine_matrix[i, j])
            fuzzy_score = fuzz.token_sort_ratio(normalized_texts[i], normalized_texts[j]) / 100.0
            combined_score = (self.tfidf_weight * tfidf_score) + (self.fuzzy_weight * fuzzy_score)

            pair = {
                "doc_a": names[i],
                "doc_b": names[j],
                "tfidf_cosine": round(tfidf_score, 4),
                "fuzzy_score": round(fuzzy_score, 4),
                "combined_score": round(combined_score, 4),
                "flagged": combined_score >= self.threshold,
                "shared_snippets": self._sentence_overlap(raw_texts[i], raw_texts[j]),
            }
            scored_pairs.append(pair)

        scored_pairs.sort(key=lambda x: x["combined_score"], reverse=True)

        flagged_pairs = [pair for pair in scored_pairs if pair["flagged"]]
        return {
            "documents": [
                {
                    "name": d.name,
                    "raw_length": len(d.raw_text),
                    "normalized_length": len(d.normalized_text),
                }
                for d in documents
            ],
            "threshold": self.threshold,
            "pairs": scored_pairs,
            "flagged_pairs": flagged_pairs,
        }

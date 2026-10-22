"""Arabic-first Local RAG Engine using TF-IDF and Cosine Similarity."""

import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DATA_DIR, POLICY_CATALOG, RELEVANCE_THRESHOLD, DEFAULT_TOP_K
from src.safety import sanitize_prompt_injection


class RetrievedPassage(BaseModel):
    """Standardized representation of a retrieved policy passage."""
    source_id: str = Field(..., description="معرف مصدر السياسة الرسمي مثل HR-LEAVE-001")
    policy_title: str = Field(..., description="عنوان السياسة")
    section_title: str = Field(..., description="عنوان القسم أو الفقرة")
    passage_text: str = Field(..., description="نص الفقرة المسترجعة")
    similarity_score: float = Field(..., description="درجة التشابه المحسوبة عبر جيب التمام")


class RetrievalResult(BaseModel):
    """Container for RAG retrieval results with calibrated evidence gating."""
    passages: List[RetrievedPassage] = Field(default_factory=list)
    has_sufficient_evidence: bool = False
    top_source_id: Optional[str] = None
    insufficient_evidence_message: Optional[str] = None


# Arabic Diacritics Regex
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # Harakat and Tatweel

# Arabic Stopwords (normalized forms)
ARABIC_STOPWORDS = {
    "من", "في", "علي", "الي", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "ما", "ماذا", "هل", "هو", "هي", "هم", "هن", "ان", "كان", "كانت",
    "يكون", "تكون", "كل", "او", "ثم", "حيث", "كما", "قد", "لا", "لم",
    "لن", "لو", "غير", "بين", "فقط", "حتي", "اذا", "وهو", "وهي", "اي",
    "التي", "الذي", "الذين", "اللائي", "اللاتي"
}


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for accurate TF-IDF tokenization and matching."""
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # Remove Tashkeel and Tatweel
    text = ARABIC_DIACRITICS.sub("", text)

    # Normalize Alef forms
    text = re.sub(r"[إأآا]", "ا", text)

    # Normalize Taa Marbuta & Haa
    text = re.sub(r"ة\b", "ه", text)

    # Normalize Alef Maksura to Yaa
    text = re.sub(r"ى\b", "ي", text)

    # Lowercase Latin characters if any
    text = text.lower()

    # Remove extra punctuation but keep Arabic and Latin letters, digits, and spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


class PolicyKnowledgeBase:
    """Loads, indexes, and searches the fictional HR Markdown policy documents."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.passages: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._load_and_index()

    def _extract_source_id(self, content: str, filename: str) -> str:
        """Extract source ID from markdown metadata or filename fallback."""
        match = re.search(r"Source ID[\s\*\:\`]+(HR-[A-Z]+-\d{3})", content, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Fallback to catalog mapping
        for sid, meta in POLICY_CATALOG.items():
            if meta["file"] == filename:
                return sid
        return "HR-GEN-000"

    def _load_and_index(self) -> None:
        """Parse all markdown files into logical passages and build TF-IDF index."""
        self.passages.clear()

        if not self.data_dir.exists():
            return

        for md_file in sorted(self.data_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception:
                continue

            source_id = self._extract_source_id(content, md_file.name)
            meta = POLICY_CATALOG.get(source_id, {"title": md_file.stem, "file": md_file.name})
            policy_title = meta["title"]

            # Split document by markdown headings (H2: ## )
            sections = re.split(r"\n(?=##\s+)", content)
            for sec in sections:
                sec_text = sec.strip()
                if not sec_text:
                    continue

                # Extract section title if present
                first_line = sec_text.splitlines()[0].strip()
                if first_line.startswith("## "):
                    section_title = first_line.replace("## ", "").strip()
                elif first_line.startswith("# "):
                    section_title = first_line.replace("# ", "").strip()
                else:
                    section_title = "نظرة عامة والضوابط"

                # Defensive sanitation against prompt injection
                safe_text = sanitize_prompt_injection(sec_text)
                normalized_text = normalize_arabic(safe_text)

                # Append document header info to improve semantic matching
                enriched_text = f"{policy_title} {section_title} {normalized_text}"

                self.passages.append({
                    "source_id": source_id,
                    "policy_title": policy_title,
                    "section_title": section_title,
                    "raw_text": safe_text,
                    "normalized_text": enriched_text,
                })

        # Fit TF-IDF Vectorizer across all passages
        if self.passages:
            corpus = [p["normalized_text"] for p in self.passages]
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words=list(ARABIC_STOPWORDS),
                min_df=1,
                sublinear_tf=True,
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = RELEVANCE_THRESHOLD,
    ) -> RetrievalResult:
        """Retrieve the top matching passages with calibrated relevance gating."""
        if not self.passages or self.vectorizer is None or self.tfidf_matrix is None:
            return RetrievalResult(
                passages=[],
                has_sufficient_evidence=False,
                insufficient_evidence_message=(
                    "قاعدة المعرفة الخاصة بالسياسات غير متاحة حالياً."
                ),
            )

        norm_query = normalize_arabic(query)
        if not norm_query:
            return RetrievalResult(
                passages=[],
                has_sufficient_evidence=False,
                insufficient_evidence_message=(
                    "لم يتم العثور على شواهد أو سياسات موثوقة ذات صلة بهذا الاستفسار في سياسات الشركة التجريبية."
                ),
            )

        query_vec = self.vectorizer.transform([norm_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get indices sorted by highest similarity
        ranked_indices = np.argsort(similarities)[::-1]

        retrieved: List[RetrievedPassage] = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score <= 0.0:
                continue

            passage_data = self.passages[idx]
            retrieved.append(
                RetrievedPassage(
                    source_id=passage_data["source_id"],
                    policy_title=passage_data["policy_title"],
                    section_title=passage_data["section_title"],
                    passage_text=passage_data["raw_text"],
                    similarity_score=round(score, 4),
                )
            )
            if len(retrieved) >= top_k:
                break

        # Evidence Gating
        top_score = retrieved[0].similarity_score if retrieved else 0.0
        has_sufficient = bool(retrieved and top_score >= threshold)

        insufficient_msg = None
        if not has_sufficient:
            insufficient_msg = (
                "لم يتم العثور على شواهد أو سياسات موثوقة ذات صلة بهذا الاستفسار في سياسات الشركة التجريبية."
            )

        return RetrievalResult(
            passages=retrieved,
            has_sufficient_evidence=has_sufficient,
            top_source_id=retrieved[0].source_id if has_sufficient else None,
            insufficient_evidence_message=insufficient_msg,
        )


# Global singleton instance for efficient reuse across queries
_knowledge_base_instance: Optional[PolicyKnowledgeBase] = None

def get_knowledge_base() -> PolicyKnowledgeBase:
    """Return or initialize the singleton knowledge base instance."""
    global _knowledge_base_instance
    if _knowledge_base_instance is None:
        _knowledge_base_instance = PolicyKnowledgeBase()
    return _knowledge_base_instance

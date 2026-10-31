from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


ARTICLE_FOLDER_NAME = "article_txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

# Defaults used by the recorded development evaluation.
DEFAULT_CHUNK_SIZE_SENTENCES = 3
DEFAULT_OVERLAP_SENTENCES = 1
DEFAULT_TOP_K = 4
DEFAULT_CANDIDATE_K = 12
DEFAULT_MIN_SIMILARITY = 0.28
DEFAULT_MIN_ANSWER_KEYWORD_OVERLAP = 0.40
DEFAULT_CONTEXT_WINDOW_CHUNKS = 1
NOT_ENOUGH_INFO_MESSAGE = "I do not have enough information in the retrieved documents."

SAMPLE_QUESTIONS = [
    {
        "question": "What kind of food does aquaculture farm?",
        "gold_doc": "Aquaculture.txt",
        "gold_keywords": ["seafood"],
    },
    {
        "question": "What is archaeology the study of?",
        "gold_doc": "Archaeology.txt",
        "gold_keywords": ["human past"],
    },
    {
        "question": "Who do architects often work with to make structurally sound buildings?",
        "gold_doc": "Architecture.txt",
        "gold_keywords": ["structural engineers"],
    },
    {
        "question": "What is the capital of Argentina?",
        "gold_doc": "Argentina.txt",
        "gold_keywords": ["buenos aires"],
    },
    {
        "question": "In mathematics, what is the basic study of numbers called?",
        "gold_doc": "Arithmetic.txt",
        "gold_keywords": ["arithmetic"],
    },
    {
        "question": "What is the capital of Armenia?",
        "gold_doc": "Armenia.txt",
        "gold_keywords": ["yerevan"],
    },
    {
        "question": "What is the product of art called?",
        "gold_doc": "Art.txt",
        "gold_keywords": ["work of art"],
    },
    {
        "question": "What is the biggest object in the asteroid belt?",
        "gold_doc": "Asteroid.txt",
        "gold_keywords": ["ceres"],
    },
    {
        "question": "What is astronomy the scientific study of?",
        "gold_doc": "Astronomy.txt",
        "gold_keywords": ["celestial bodies"],
    },
    {
        "question": "What is in the middle of an atom?",
        "gold_doc": "Atom.txt",
        "gold_keywords": ["nucleus"],
    },
    {
        "question": "What kind of toe cover do work boots often have?",
        "gold_doc": "Boot.txt",
        "gold_keywords": ["steel toe cover"],
    },
    {
        "question": "What major historical change did the development of farming give rise to?",
        "gold_doc": "Farming.txt",
        "gold_keywords": ["neolithic revolution"],
    },
    {
        "question": "From what continent did the ancestors of Native Americans come?",
        "gold_doc": "Native_American.txt",
        "gold_keywords": ["asia", "siberia"],
    },
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "their",
    "them",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

# These words are useful for ranking but too generic to prove the document is about the
# question topic. The anchor check helps reject unrelated questions like "capital of France".
GENERIC_ANCHOR_TERMS = {
    "answer",
    "basic",
    "big",
    "biggest",
    "called",
    "capital",
    "change",
    "come",
    "continent",
    "development",
    "food",
    "gave",
    "historical",
    "kind",
    "major",
    "make",
    "middle",
    "name",
    "object",
    "often",
    "product",
    "rise",
    "scientific",
    "sound",
    "study",
    "thing",
    "toe",
    "work",
    "write",
    "writer",
    "written",
    "wrote",
}


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_extracted_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\xa0", " ")

    cutoff_patterns = [
        r"\nReferences\b",
        r"\nOther websites\b",
        r"\nExternal links\b",
        r"\nSee also\b",
        r"\nNotes\b",
        r"\nFurther reading\b",
        r"\nBibliography\b",
        r"\nSources\b",
        r"\nRelated pages\b",
        r"\nGallery\b",
    ]

    cut_index = len(text)
    for pattern in cutoff_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            cut_index = min(cut_index, match.start())

    text = text[:cut_index]

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\(https?://[^)]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)

    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        lower_line = line.lower()

        if not line:
            continue
        if len(line) < 3:
            continue
        if re.fullmatch(r"[\W_]+", line):
            continue
        if "archived" in lower_line:
            continue
        if "retrieved" in lower_line:
            continue
        if "wayback machine" in lower_line:
            continue
        if lower_line.startswith("website "):
            continue
        if lower_line.startswith("citation "):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def looks_like_section_heading(line: str) -> bool:
    words = re.findall(r"\b[A-Za-z0-9']+\b", line)
    if not words or len(words) > 6:
        return False
    if re.search(r"[.!?]$", line):
        return False
    if len(line) > 80:
        return False
    return True


def split_sentences_with_sections(text: str) -> List[Dict[str, str]]:
    cleaned = clean_extracted_text(text)
    sentence_items: List[Dict[str, str]] = []
    current_section = ""

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if looks_like_section_heading(line):
            current_section = line
            continue

        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            sentence = part.strip()
            if sentence:
                sentence_items.append({"text": sentence, "section": current_section})

    return sentence_items


def split_sentences(text: str) -> List[str]:
    return [item["text"] for item in split_sentences_with_sections(text)]


def build_sentence_chunks(
    sentence_items: Sequence[Dict[str, str]],
    chunk_size_sentences: int = DEFAULT_CHUNK_SIZE_SENTENCES,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> List[Dict[str, Any]]:
    if not sentence_items:
        return []

    chunk_size_sentences = max(1, chunk_size_sentences)
    overlap_sentences = max(0, min(overlap_sentences, chunk_size_sentences - 1))
    step = max(1, chunk_size_sentences - overlap_sentences)

    chunks: List[Dict[str, Any]] = []
    for start in range(0, len(sentence_items), step):
        end = min(start + chunk_size_sentences, len(sentence_items))
        chunk_sentences = sentence_items[start:end]
        chunk_text = " ".join(item["text"] for item in chunk_sentences).strip()
        section = next((item["section"] for item in chunk_sentences if item["section"]), "")

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "section": section,
                    "sentence_start": start,
                    "sentence_end": end,
                }
            )

        if end >= len(sentence_items):
            break

    return chunks


def chunk_text(
    text: str,
    chunk_size_sentences: int = DEFAULT_CHUNK_SIZE_SENTENCES,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> List[str]:
    sentence_items = split_sentences_with_sections(text)
    return [
        chunk["text"]
        for chunk in build_sentence_chunks(
            sentence_items,
            chunk_size_sentences=chunk_size_sentences,
            overlap_sentences=overlap_sentences,
        )
    ]


def is_good_sentence(sentence: str) -> bool:
    s = sentence.strip()
    sl = s.lower()

    if len(s) < 30:
        return False
    if "http" in sl:
        return False
    if "isbn" in sl:
        return False
    if "retrieved" in sl:
        return False
    if "archived" in sl:
        return False
    if "wayback machine" in sl:
        return False
    if sl.startswith("see also"):
        return False
    if sl.startswith("external links"):
        return False
    if sl.startswith("references"):
        return False
    if sl.startswith("sources"):
        return False

    words = re.findall(r"\b[a-zA-Z0-9']+\b", s)
    if len(words) < 6:
        return False

    return True


def simple_stem(token: str) -> str:
    token = token.lower().strip("'")
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def extract_keywords(text: str) -> List[str]:
    tokens = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
    keywords: List[str] = []
    for token in tokens:
        stemmed = simple_stem(token)
        if len(stemmed) < 2:
            continue
        if stemmed in STOPWORDS:
            continue
        if stemmed not in keywords:
            keywords.append(stemmed)
    return keywords


def keyword_overlap_score(query: str, text: str) -> float:
    query_terms = extract_keywords(query)
    if not query_terms:
        return 0.0

    text_terms = set(extract_keywords(text))
    normalized_text = " ".join(text_terms)
    matches = 0
    for term in query_terms:
        if term in text_terms:
            matches += 1
        elif len(term) >= 4 and term in normalized_text:
            matches += 1

    return matches / len(query_terms)


def query_anchor_terms(query: str) -> List[str]:
    anchors = [
        term
        for term in extract_keywords(query)
        if term not in GENERIC_ANCHOR_TERMS and len(term) > 2
    ]
    return anchors


def anchor_overlap_score(query: str, text: str) -> float:
    anchors = query_anchor_terms(query)
    if not anchors:
        return 1.0

    text_terms = set(extract_keywords(text))
    normalized_text = " ".join(text_terms)
    matches = 0
    for term in anchors:
        if term in text_terms:
            matches += 1
        elif len(term) >= 4 and term in normalized_text:
            matches += 1

    return matches / len(anchors)


def normalize_for_duplicate_check(text: str) -> set[str]:
    return set(extract_keywords(text))


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def source_label(source_path: str, chunk_id: int) -> str:
    return f"{Path(source_path).name}#chunk-{chunk_id}"


@dataclass
class RetrievedChunk:
    title: str
    source_path: str
    chunk_id: int
    text: str
    score: float
    section: str = ""
    sentence_start: int = 0
    sentence_end: int = 0
    query_used: str = ""
    keyword_score: float = 0.0
    title_score: float = 0.0
    anchor_score: float = 0.0
    rerank_score: float = 0.0
    expanded_text: str = ""
    expanded_chunk_ids: List[int] = field(default_factory=list)

    @property
    def label(self) -> str:
        return source_label(self.source_path, self.chunk_id)


class ArticleRAGChatbot:
    def __init__(
        self,
        chunk_size_sentences: int = DEFAULT_CHUNK_SIZE_SENTENCES,
        overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        small_to_big: bool = True,
        context_window_chunks: int = DEFAULT_CONTEXT_WINDOW_CHUNKS,
        keyword_weight: float = 0.25,
        title_weight: float = 0.15,
        min_answer_keyword_overlap: float = DEFAULT_MIN_ANSWER_KEYWORD_OVERLAP,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        local_files_only: bool = False,
        embedding_model_revision: str | None = None,
        model: Any = None,
    ):
        self.chunk_size_sentences = chunk_size_sentences
        self.overlap_sentences = overlap_sentences
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.min_similarity = min_similarity
        self.small_to_big = small_to_big
        self.context_window_chunks = context_window_chunks
        self.keyword_weight = keyword_weight
        self.title_weight = title_weight
        self.min_answer_keyword_overlap = min_answer_keyword_overlap
        self.embedding_model_name = embedding_model_name

        self.articles: List[Dict[str, Any]] = []
        self.chunks: List[Dict[str, Any]] = []
        self.article_chunks: Dict[int, List[Dict[str, Any]]] = {}

        if model is None:
            from sentence_transformers import SentenceTransformer

            if embedding_model_revision is None and self.embedding_model_name == EMBEDDING_MODEL_NAME:
                embedding_model_revision = EMBEDDING_MODEL_REVISION
            model = SentenceTransformer(
                self.embedding_model_name,
                revision=embedding_model_revision,
                local_files_only=local_files_only,
                device="cpu",
            )
        self.model = model
        self.chunk_embeddings: np.ndarray | None = None
        self.article_embeddings: np.ndarray | None = None

    def add_article(self, title: str, text: str, source_path: str = "") -> None:
        cleaned_text = clean_extracted_text(text)
        self.articles.append(
            {
                "title": title,
                "text": cleaned_text,
                "source_path": source_path,
                "sentence_items": split_sentences_with_sections(cleaned_text),
            }
        )

    def load_articles_from_folder(self, folder: str | Path) -> None:
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(
                f"Could not find article folder: {folder}. "
                f"Make sure your .txt article files are inside a folder named '{ARTICLE_FOLDER_NAME}'."
            )

        self.articles = []
        for path in sorted(folder.glob("*.txt")):
            title = path.stem.replace("_", " ")
            text = path.read_text(encoding="utf-8")
            self.add_article(title=title, text=text, source_path=str(path))

        if not self.articles:
            raise ValueError(f"No .txt files were found in {folder}")

    def build_index(self) -> None:
        self.chunks = []
        self.article_chunks = {}

        for article_index, article in enumerate(self.articles):
            article_chunks = build_sentence_chunks(
                article["sentence_items"],
                chunk_size_sentences=self.chunk_size_sentences,
                overlap_sentences=self.overlap_sentences,
            )
            self.article_chunks[article_index] = []

            for local_chunk_id, chunk in enumerate(article_chunks):
                # Embed the title and section with the text to retain document context.
                section_text = f" Section: {chunk['section']}." if chunk["section"] else ""
                chunk_text_for_index = f"Title: {article['title']}.{section_text} {chunk['text']}"
                stored_chunk = {
                    "global_chunk_id": len(self.chunks),
                    "article_index": article_index,
                    "title": article["title"],
                    "source_path": article["source_path"],
                    "chunk_id": local_chunk_id,
                    "section": chunk["section"],
                    "sentence_start": chunk["sentence_start"],
                    "sentence_end": chunk["sentence_end"],
                    "text": chunk["text"],
                    "text_for_index": chunk_text_for_index,
                }
                self.chunks.append(stored_chunk)
                self.article_chunks[article_index].append(stored_chunk)

        if not self.chunks:
            raise ValueError("No chunks were created from the articles.")

        chunk_texts = [chunk["text_for_index"] for chunk in self.chunks]
        article_texts = [
            f"Title: {article['title']}. {article['text'][:2000]}"
            for article in self.articles
        ]

        self.chunk_embeddings = self.model.encode(
            chunk_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        self.article_embeddings = self.model.encode(
            article_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self.model.encode(
            list(queries),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _encode_query(self, query: str) -> np.ndarray:
        return self._encode_queries([query])[0]

    def rewrite_query(self, query: str) -> str:
        """Build a keyword query alongside the original question."""
        cleaned = normalize_whitespace(query)
        keywords = extract_keywords(cleaned)
        if not keywords:
            return cleaned

        rewritten = " ".join(keywords)
        if rewritten.lower() == cleaned.lower():
            return cleaned
        return rewritten

    def build_retrieval_queries(self, query: str, use_query_rewrite: bool = True) -> List[str]:
        queries = [normalize_whitespace(query)]
        if use_query_rewrite:
            rewritten = self.rewrite_query(query)
            if rewritten and rewritten.lower() != queries[0].lower():
                queries.append(rewritten)
        return queries

    def _chunk_to_retrieved(
        self,
        chunk: Dict[str, Any],
        score: float,
        query_used: str,
        original_query: str,
    ) -> RetrievedChunk:
        keyword_score = max(
            keyword_overlap_score(query, chunk["text_for_index"])
            for query in self.build_retrieval_queries(original_query)
        )
        title_score = keyword_overlap_score(original_query, chunk["title"])
        anchor_score = anchor_overlap_score(original_query, chunk["text_for_index"])
        rerank_score = score + (self.keyword_weight * keyword_score) + (self.title_weight * title_score)

        retrieved = RetrievedChunk(
            title=chunk["title"],
            source_path=chunk["source_path"],
            chunk_id=chunk["chunk_id"],
            text=chunk["text"],
            score=float(score),
            section=chunk["section"],
            sentence_start=chunk["sentence_start"],
            sentence_end=chunk["sentence_end"],
            query_used=query_used,
            keyword_score=float(keyword_score),
            title_score=float(title_score),
            anchor_score=float(anchor_score),
            rerank_score=float(rerank_score),
        )
        return self._expand_retrieved_chunk(retrieved, chunk)

    def _expand_retrieved_chunk(
        self,
        retrieved: RetrievedChunk,
        chunk: Dict[str, Any],
    ) -> RetrievedChunk:
        if not self.small_to_big:
            retrieved.expanded_text = retrieved.text
            retrieved.expanded_chunk_ids = [retrieved.chunk_id]
            return retrieved

        article_index = chunk["article_index"]
        article = self.articles[article_index]
        article_chunks = self.article_chunks[article_index]
        start_chunk_id = max(0, chunk["chunk_id"] - self.context_window_chunks)
        end_chunk_id = min(len(article_chunks) - 1, chunk["chunk_id"] + self.context_window_chunks)

        sentence_start = article_chunks[start_chunk_id]["sentence_start"]
        sentence_end = article_chunks[end_chunk_id]["sentence_end"]
        expanded_sentences = article["sentence_items"][sentence_start:sentence_end]
        retrieved.expanded_text = " ".join(item["text"] for item in expanded_sentences).strip()
        retrieved.expanded_chunk_ids = list(range(start_chunk_id, end_chunk_id + 1))
        return retrieved

    def _passes_metadata_filter(
        self,
        chunk: Dict[str, Any],
        metadata_filter: Dict[str, str] | None,
    ) -> bool:
        if not metadata_filter:
            return True

        for key, expected in metadata_filter.items():
            actual = str(chunk.get(key, "")).lower()
            if str(expected).lower() not in actual:
                return False
        return True

    def _deduplicate_retrieved(self, retrieved: Sequence[RetrievedChunk]) -> List[RetrievedChunk]:
        kept: List[RetrievedChunk] = []
        kept_token_sets: List[set[str]] = []

        for item in retrieved:
            text_for_dupe_check = item.expanded_text or item.text
            token_set = normalize_for_duplicate_check(text_for_dupe_check)
            if any(jaccard_similarity(token_set, existing) >= 0.88 for existing in kept_token_sets):
                continue

            kept.append(item)
            kept_token_sets.append(token_set)

        return kept

    def retrieve_naive(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[RetrievedChunk]:
        """Old Naive RAG retrieval: top-k chunks by cosine similarity only."""
        if self.chunk_embeddings is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")

        query_embedding = self._encode_query(query)

        # Because embeddings are normalized, dot product == cosine similarity.
        scores = self.chunk_embeddings @ query_embedding
        ranked_ids = np.argsort(scores)[::-1][:top_k]

        results: List[RetrievedChunk] = []
        for idx in ranked_ids:
            chunk = self.chunks[int(idx)]
            results.append(
                RetrievedChunk(
                    title=chunk["title"],
                    source_path=chunk["source_path"],
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    score=float(scores[int(idx)]),
                    section=chunk["section"],
                    sentence_start=chunk["sentence_start"],
                    sentence_end=chunk["sentence_end"],
                    query_used=query,
                    rerank_score=float(scores[int(idx)]),
                    expanded_text=chunk["text"],
                    expanded_chunk_ids=[chunk["chunk_id"]],
                )
            )
        return results

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        min_similarity: float | None = None,
        use_query_rewrite: bool = True,
        metadata_filter: Dict[str, str] | None = None,
    ) -> List[RetrievedChunk]:
        """Rank rewritten-query candidates, then filter and deduplicate context."""
        if self.chunk_embeddings is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")

        top_k = top_k or self.top_k
        candidate_k = max(candidate_k or self.candidate_k, top_k)
        min_similarity = self.min_similarity if min_similarity is None else min_similarity

        retrieval_queries = self.build_retrieval_queries(query, use_query_rewrite=use_query_rewrite)
        query_embeddings = self._encode_queries(retrieval_queries)

        # Cosine similarity/vector search: normalized embeddings make dot product cosine.
        score_matrix = self.chunk_embeddings @ query_embeddings.T
        best_scores = np.max(score_matrix, axis=1)
        best_query_ids = np.argmax(score_matrix, axis=1)
        ranked_ids = np.argsort(best_scores)[::-1]

        candidates: List[RetrievedChunk] = []
        for idx in ranked_ids:
            if len(candidates) >= candidate_k:
                break

            score = float(best_scores[int(idx)])
            if score < min_similarity:
                continue

            chunk = self.chunks[int(idx)]
            if not self._passes_metadata_filter(chunk, metadata_filter):
                continue

            query_used = retrieval_queries[int(best_query_ids[int(idx)])]
            candidates.append(
                self._chunk_to_retrieved(
                    chunk=chunk,
                    score=score,
                    query_used=query_used,
                    original_query=query,
                )
            )

        candidates.sort(key=lambda item: item.rerank_score, reverse=True)
        deduped = self._deduplicate_retrieved(candidates)

        # Negative rejection: if no retrieved context contains the question's anchor terms,
        # the system should not guess from a merely similar article.
        anchors = query_anchor_terms(query)
        if anchors:
            deduped = [
                item
                for item in deduped
                if anchor_overlap_score(query, f"{item.title} {item.expanded_text or item.text}") > 0
            ]

        return deduped[:top_k]

    def answer_baseline(self, query: str) -> Dict[str, Any]:
        """Article-level baseline: choose one article, then its best sentence."""
        if self.article_embeddings is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")

        query_embedding = self._encode_query(query)
        scores = self.article_embeddings @ query_embedding
        best_article_idx = int(np.argmax(scores))
        best_article = self.articles[best_article_idx]

        sentences = [s for s in split_sentences(best_article["text"]) if is_good_sentence(s)]
        if not sentences:
            return {
                "answer": best_article["text"][:300],
                "source_title": best_article["title"],
                "score": float(scores[best_article_idx]),
            }

        sentence_embeddings = self.model.encode(
            sentences,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        sentence_scores = sentence_embeddings @ query_embedding
        best_sentence_idx = int(np.argmax(sentence_scores))

        return {
            "answer": sentences[best_sentence_idx],
            "source_title": best_article["title"],
            "score": float(scores[best_article_idx]),
        }

    def _select_best_sentence(
        self,
        query: str,
        retrieved: Sequence[RetrievedChunk],
        retrieval_queries: Sequence[str],
    ) -> Dict[str, Any] | None:
        candidate_sentences: List[Dict[str, Any]] = []
        seen_sentences: set[str] = set()

        for item in retrieved:
            context_text = item.expanded_text or item.text
            for sentence in split_sentences(context_text):
                if not is_good_sentence(sentence):
                    continue

                normalized_sentence = normalize_whitespace(sentence.lower())
                if normalized_sentence in seen_sentences:
                    continue
                seen_sentences.add(normalized_sentence)

                # Expanded context may answer from a neighbor, not the retrieval anchor.
                supporting_chunk = next(
                    (chunk for chunk in self.chunks
                     if chunk["source_path"] == item.source_path
                     and chunk["chunk_id"] in item.expanded_chunk_ids
                     and sentence in chunk["text"]),
                    None,
                )
                candidate_sentences.append(
                    {
                        "title": item.title,
                        "source_path": item.source_path,
                        "chunk_id": supporting_chunk["chunk_id"] if supporting_chunk else item.chunk_id,
                        "section": supporting_chunk["section"] if supporting_chunk else item.section,
                        "sentence": sentence,
                        "retrieval_score": item.score,
                        "rerank_score": item.rerank_score,
                    }
                )

        if not candidate_sentences:
            return None

        sentence_texts = [item["sentence"] for item in candidate_sentences]
        sentence_embeddings = self.model.encode(
            sentence_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_embeddings = self._encode_queries(retrieval_queries)
        sentence_score_matrix = sentence_embeddings @ query_embeddings.T
        sentence_scores = np.max(sentence_score_matrix, axis=1)

        combined_scores = []
        for i, item in enumerate(candidate_sentences):
            sentence_keyword_score = max(
                keyword_overlap_score(retrieval_query, item["sentence"])
                for retrieval_query in retrieval_queries
            )
            sentence_anchor_score = anchor_overlap_score(query, item["sentence"])
            item["sentence_keyword_score"] = float(sentence_keyword_score)
            item["sentence_anchor_score"] = float(sentence_anchor_score)
            combined = (
                float(sentence_scores[i])
                + (0.35 * sentence_keyword_score)
                + (0.10 * item["rerank_score"])
                + (0.10 * sentence_anchor_score)
            )
            combined_scores.append(combined)

        best_id = int(np.argmax(combined_scores))
        best_item = candidate_sentences[best_id]
        best_item["sentence_score"] = float(sentence_scores[best_id])
        best_item["combined_score"] = float(combined_scores[best_id])
        return best_item

    def answer_naive_rag(self, query: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """Old RAG answer behavior preserved for before-vs-after evaluation."""
        retrieved = self.retrieve_naive(query, top_k=top_k)

        if not retrieved:
            return {
                "answer": "I could not find supporting evidence.",
                "citations": [],
                "sources": [],
                "retrieved": [],
            }

        query_embedding = self._encode_query(query)

        candidate_sentences: List[Dict[str, Any]] = []
        for item in retrieved:
            for sentence in split_sentences(item.text):
                if is_good_sentence(sentence):
                    candidate_sentences.append(
                        {
                            "title": item.title,
                            "source_path": item.source_path,
                            "chunk_id": item.chunk_id,
                            "section": item.section,
                            "sentence": sentence,
                            "retrieval_score": item.score,
                        }
                    )

        if not candidate_sentences:
            first = retrieved[0]
            return {
                "answer": first.text,
                "citations": [
                    {
                        "title": first.title,
                        "chunk_id": first.chunk_id,
                        "source_path": first.source_path,
                        "section": first.section,
                    }
                ],
                "sources": [first.label],
                "retrieved": [asdict(item) for item in retrieved],
            }

        sentence_texts = [item["sentence"] for item in candidate_sentences]
        sentence_embeddings = self.model.encode(
            sentence_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        sentence_scores = sentence_embeddings @ query_embedding

        # Old behavior: small blend of semantic sentence match + semantic chunk match.
        combined_scores = []
        for i, item in enumerate(candidate_sentences):
            combined = float(sentence_scores[i]) + (0.15 * item["retrieval_score"])
            combined_scores.append(combined)

        best_id = int(np.argmax(combined_scores))
        best_item = candidate_sentences[best_id]
        label = source_label(best_item["source_path"], best_item["chunk_id"])

        return {
            "answer": best_item["sentence"],
            "citations": [
                {
                    "title": best_item["title"],
                    "chunk_id": best_item["chunk_id"],
                    "source_path": best_item["source_path"],
                    "section": best_item["section"],
                }
            ],
            "sources": [label],
            "retrieved": [asdict(item) for item in retrieved],
        }

    def build_grounded_prompt(self, query: str, retrieved: Sequence[RetrievedChunk]) -> str:
        context_blocks = []
        for item in retrieved:
            section = f", section: {item.section}" if item.section else ""
            context = item.expanded_text or item.text
            context_blocks.append(f"[{item.label}{section}]\n{context}")

        context_text = "\n\n".join(context_blocks) if context_blocks else "(no retrieved context)"
        return (
            "You are a grounded RAG answerer.\n"
            "Use only the provided retrieved context.\n"
            "Do not make up facts or use outside knowledge.\n"
            "When possible, mention the source file and chunk id that support the answer.\n"
            f'If the answer is not supported, say "{NOT_ENOUGH_INFO_MESSAGE}"\n\n'
            f"Question: {query}\n\n"
            f"Retrieved context:\n{context_text}\n\n"
            "Concise grounded answer:"
        )

    def answer_with_rag(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        min_similarity: float | None = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        retrieval_queries = self.build_retrieval_queries(query)
        retrieved = self.retrieve(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            min_similarity=min_similarity,
        )
        prompt = self.build_grounded_prompt(query, retrieved)

        if not retrieved:
            result = {
                "answer": NOT_ENOUGH_INFO_MESSAGE,
                "citations": [],
                "sources": [],
                "retrieved": [],
                "original_query": query,
                "rewritten_query": retrieval_queries[-1] if len(retrieval_queries) > 1 else query,
                "prompt": prompt,
            }
            if debug:
                result["debug"] = format_debug_retrieval(result)
            return result

        best_item = self._select_best_sentence(query, retrieved, retrieval_queries)

        if best_item is None:
            result = {
                "answer": NOT_ENOUGH_INFO_MESSAGE,
                "citations": [],
                "sources": [],
                "retrieved": [asdict(item) for item in retrieved],
                "original_query": query,
                "rewritten_query": retrieval_queries[-1] if len(retrieval_queries) > 1 else query,
                "prompt": prompt,
            }
            if debug:
                result["debug"] = format_debug_retrieval(result)
            return result

        # Reject weakly supported sentences even when retrieval found related text.
        if not self._answer_sentence_is_supported(
            query,
            best_item["sentence"],
            best_item.get("sentence_keyword_score", 0.0),
        ):
            result = {
                "answer": NOT_ENOUGH_INFO_MESSAGE,
                "citations": [],
                "sources": [],
                "retrieved": [asdict(item) for item in retrieved],
                "original_query": query,
                "rewritten_query": retrieval_queries[-1] if len(retrieval_queries) > 1 else query,
                "prompt": prompt,
            }
            if debug:
                result["debug"] = format_debug_retrieval(result)
            return result

        label = source_label(best_item["source_path"], best_item["chunk_id"])
        result = {
            "answer": best_item["sentence"],
            "citations": [
                {
                    "title": best_item["title"],
                    "chunk_id": best_item["chunk_id"],
                    "source_path": best_item["source_path"],
                    "section": best_item["section"],
                }
            ],
            "sources": [label],
            "retrieved": [asdict(item) for item in retrieved],
            "original_query": query,
            "rewritten_query": retrieval_queries[-1] if len(retrieval_queries) > 1 else query,
            "prompt": prompt,
        }
        if debug:
            result["debug"] = format_debug_retrieval(result)
        return result

    def _answer_contains_gold(self, answer: str, gold_keywords: Sequence[str]) -> int:
        answer_lower = answer.lower()
        return int(any(keyword.lower() in answer_lower for keyword in gold_keywords))

    def _answer_is_grounded(self, answer: str, retrieved: Sequence[Dict[str, Any]]) -> int:
        if answer == NOT_ENOUGH_INFO_MESSAGE:
            return 1
        context_text = " ".join(
            item.get("expanded_text") or item.get("text", "")
            for item in retrieved
        ).lower()
        return int(answer.lower() in context_text)

    def _answer_sentence_is_supported(
        self,
        query: str,
        answer_sentence: str,
        sentence_keyword_score: float,
    ) -> bool:
        if sentence_keyword_score < self.min_answer_keyword_overlap:
            return False

        query_terms = set(extract_keywords(query))
        answer_anchor_score = anchor_overlap_score(query, answer_sentence)
        if query_anchor_terms(query) and answer_anchor_score == 0:
            return False

        # Simple relation checks for common class-demo questions. They prevent
        # false positives such as "capital of France" matching any sentence with
        # France but no capital relation.
        if "capital" in query_terms and keyword_overlap_score("capital", answer_sentence) == 0:
            return False

        writer_terms = {"author", "authored", "write", "writer", "written", "wrote"}
        if query_terms & writer_terms and query_anchor_terms(query) and answer_anchor_score < 1.0:
            return False

        return True

    def evaluate(self, questions: List[Dict[str, Any]], top_k: int | None = None) -> Dict[str, Any]:
        top_k = top_k or self.top_k
        supported_questions = [item for item in questions if not item.get("should_reject")]
        negative_questions = [item for item in questions if item.get("should_reject")]
        supported_total = len(supported_questions) or 1
        negative_total = len(negative_questions) or 1

        metrics = {
            "article_baseline": {
                "answer_hits": 0,
                "negative_rejections": 0,
                "answer_fidelity": 0,
            },
            "naive_rag": {
                "answer_hits": 0,
                "retrieval_top1": 0,
                "retrieval_topk": 0,
                "context_relevance": 0,
                "negative_rejections": 0,
                "answer_fidelity": 0,
            },
            "improved_rag": {
                "answer_hits": 0,
                "retrieval_top1": 0,
                "retrieval_topk": 0,
                "context_relevance": 0,
                "negative_rejections": 0,
                "answer_fidelity": 0,
            },
        }
        details = []

        for item in questions:
            question = item["question"]
            gold_doc = item.get("gold_doc", "")
            gold_keywords = [keyword.lower() for keyword in item.get("gold_keywords", [])]
            should_reject = bool(item.get("should_reject"))

            baseline_result = self.answer_baseline(question)
            naive_retrieved = self.retrieve_naive(question, top_k=top_k)
            naive_result = self.answer_naive_rag(question, top_k=top_k)
            improved_retrieved = self.retrieve(question, top_k=top_k)
            improved_result = self.answer_with_rag(question, top_k=top_k)

            naive_doc_names = [Path(chunk.source_path).name for chunk in naive_retrieved]
            improved_doc_names = [Path(chunk.source_path).name for chunk in improved_retrieved]

            if should_reject:
                metrics["article_baseline"]["negative_rejections"] += int(
                    baseline_result["answer"] == NOT_ENOUGH_INFO_MESSAGE
                )
                metrics["naive_rag"]["negative_rejections"] += int(
                    naive_result["answer"] == NOT_ENOUGH_INFO_MESSAGE
                )
                metrics["improved_rag"]["negative_rejections"] += int(
                    improved_result["answer"] == NOT_ENOUGH_INFO_MESSAGE
                )
            else:
                metrics["article_baseline"]["answer_hits"] += self._answer_contains_gold(
                    baseline_result["answer"],
                    gold_keywords,
                )
                metrics["naive_rag"]["answer_hits"] += self._answer_contains_gold(
                    naive_result["answer"],
                    gold_keywords,
                )
                metrics["improved_rag"]["answer_hits"] += self._answer_contains_gold(
                    improved_result["answer"],
                    gold_keywords,
                )

                metrics["naive_rag"]["retrieval_top1"] += int(
                    bool(naive_doc_names) and naive_doc_names[0] == gold_doc
                )
                metrics["naive_rag"]["retrieval_topk"] += int(gold_doc in naive_doc_names)
                metrics["improved_rag"]["retrieval_top1"] += int(
                    bool(improved_doc_names) and improved_doc_names[0] == gold_doc
                )
                metrics["improved_rag"]["retrieval_topk"] += int(gold_doc in improved_doc_names)

                metrics["naive_rag"]["context_relevance"] += int(gold_doc in naive_doc_names)
                metrics["improved_rag"]["context_relevance"] += int(gold_doc in improved_doc_names)

                metrics["article_baseline"]["answer_fidelity"] += 1
                metrics["naive_rag"]["answer_fidelity"] += self._answer_is_grounded(
                    naive_result["answer"],
                    naive_result["retrieved"],
                )
                metrics["improved_rag"]["answer_fidelity"] += self._answer_is_grounded(
                    improved_result["answer"],
                    improved_result["retrieved"],
                )

            details.append(
                {
                    "question": question,
                    "gold_doc": gold_doc,
                    "should_reject": should_reject,
                    "naive_retrieved_docs": naive_doc_names,
                    "improved_retrieved_docs": improved_doc_names,
                    "baseline_answer": baseline_result["answer"],
                    "naive_rag_answer": naive_result["answer"],
                    "improved_rag_answer": improved_result["answer"],
                    "improved_sources": improved_result["sources"],
                    "baseline_hit": self._answer_contains_gold(baseline_result["answer"], gold_keywords)
                    if not should_reject
                    else 0,
                    "naive_hit": self._answer_contains_gold(naive_result["answer"], gold_keywords)
                    if not should_reject
                    else 0,
                    "improved_hit": self._answer_contains_gold(improved_result["answer"], gold_keywords)
                    if not should_reject
                    else 0,
                    "improved_rejected": int(improved_result["answer"] == NOT_ENOUGH_INFO_MESSAGE),
                }
            )

        summary = {
            "total_questions": len(questions),
            "supported_questions": len(supported_questions),
            "negative_questions": len(negative_questions),
            "article_baseline": {
                "answer_accuracy": metrics["article_baseline"]["answer_hits"] / supported_total,
                "answer_fidelity": metrics["article_baseline"]["answer_fidelity"] / supported_total,
                "negative_rejection_rate": metrics["article_baseline"]["negative_rejections"] / negative_total
                if negative_questions
                else None,
            },
            "naive_rag": {
                "answer_accuracy": metrics["naive_rag"]["answer_hits"] / supported_total,
                "retrieval_top1_accuracy": metrics["naive_rag"]["retrieval_top1"] / supported_total,
                "retrieval_topk_hit_rate": metrics["naive_rag"]["retrieval_topk"] / supported_total,
                "context_relevance": metrics["naive_rag"]["context_relevance"] / supported_total,
                "answer_fidelity": metrics["naive_rag"]["answer_fidelity"] / supported_total,
                "negative_rejection_rate": metrics["naive_rag"]["negative_rejections"] / negative_total
                if negative_questions
                else None,
            },
            "improved_rag": {
                "answer_accuracy": metrics["improved_rag"]["answer_hits"] / supported_total,
                "retrieval_top1_accuracy": metrics["improved_rag"]["retrieval_top1"] / supported_total,
                "retrieval_topk_hit_rate": metrics["improved_rag"]["retrieval_topk"] / supported_total,
                "context_relevance": metrics["improved_rag"]["context_relevance"] / supported_total,
                "answer_fidelity": metrics["improved_rag"]["answer_fidelity"] / supported_total,
                "negative_rejection_rate": metrics["improved_rag"]["negative_rejections"] / negative_total
                if negative_questions
                else None,
            },
            "details": details,
        }

        # Backward-compatible names used by the original script output.
        summary["baseline_answer_match_rate"] = summary["article_baseline"]["answer_accuracy"]
        summary["retrieval_top1_accuracy"] = summary["improved_rag"]["retrieval_top1_accuracy"]
        summary["retrieval_topk_hit_rate"] = summary["improved_rag"]["retrieval_topk_hit_rate"]
        summary["grounded_answer_match_rate"] = summary["improved_rag"]["answer_accuracy"]
        return summary


def load_evaluation_questions(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def format_debug_retrieval(result: Dict[str, Any]) -> str:
    lines = [
        f"Original query: {result.get('original_query', '')}",
        f"Rewritten query: {result.get('rewritten_query', '')}",
        "Retrieved chunks:",
    ]

    retrieved = result.get("retrieved", [])
    if not retrieved:
        lines.append("  (none above the similarity/support threshold)")
        return "\n".join(lines)

    for idx, item in enumerate(retrieved, start=1):
        section = f", section={item.get('section')}" if item.get("section") else ""
        lines.append(
            "  "
            f"{idx}. {source_label(item['source_path'], item['chunk_id'])}{section} | "
            f"cosine={item['score']:.3f}, "
            f"keyword={item.get('keyword_score', 0.0):.3f}, "
            f"title={item.get('title_score', 0.0):.3f}, "
            f"rerank={item.get('rerank_score', item['score']):.3f}"
        )
        preview = normalize_whitespace(item.get("expanded_text") or item.get("text", ""))[:220]
        lines.append(f"     {preview}")

    return "\n".join(lines)


def interactive_qa(bot: ArticleRAGChatbot) -> None:
    print("\n=== Interactive Question Mode ===")
    print("Type 'quit' or 'exit' to stop. Prefix a question with 'debug:' to show retrieval scores.\n")

    while True:
        query = input("Ask a question: ").strip()

        if query.lower() in {"quit", "exit"}:
            print("Exiting interactive mode.")
            break

        if not query:
            continue

        debug = query.lower().startswith("debug:")
        if debug:
            query = query.split(":", 1)[1].strip()

        result = bot.answer_with_rag(query, debug=debug)

        print("\nAnswer:", result["answer"])
        if result["sources"]:
            print("Sources:", ", ".join(result["sources"]))
        if debug:
            print("\n" + result["debug"])
        print()


def print_main_summary(results: Dict[str, Any]) -> None:
    print(f"Total questions: {results['total_questions']}")
    print(f"Supported questions: {results['supported_questions']}")
    print(f"Negative questions: {results['negative_questions']}")
    print(f"Article baseline answer keyword match: {results['article_baseline']['answer_accuracy']:.3f}")
    print(f"Old naive RAG answer keyword match: {results['naive_rag']['answer_accuracy']:.3f}")
    print(f"Improved RAG answer keyword match: {results['improved_rag']['answer_accuracy']:.3f}")
    print(f"Improved retrieval top-1 accuracy: {results['improved_rag']['retrieval_top1_accuracy']:.3f}")
    print(f"Improved retrieval top-{DEFAULT_TOP_K} hit rate: {results['improved_rag']['retrieval_topk_hit_rate']:.3f}")
    if results["improved_rag"]["negative_rejection_rate"] is not None:
        print(f"Improved negative rejection rate: {results['improved_rag']['negative_rejection_rate']:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic retrieval with cited extractive answers.")
    parser.add_argument("--documents", type=Path, default=Path("examples/documents"))
    parser.add_argument("--questions", type=Path, help="Optional development evaluation before interactive mode")
    parser.add_argument("--local-files-only", action="store_true", help="Require a cached model; do not download")
    args = parser.parse_args()
    article_dir = args.documents
    evaluation_path = args.questions

    bot = ArticleRAGChatbot(local_files_only=args.local_files_only)
    bot.load_articles_from_folder(article_dir)
    bot.build_index()

    if evaluation_path is not None:
        questions = load_evaluation_questions(evaluation_path)
        results = bot.evaluate(questions)
        print_main_summary(results)

    interactive_qa(bot)

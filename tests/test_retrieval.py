from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from rag_chatbot import (
    ArticleRAGChatbot, EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_REVISION,
    NOT_ENOUGH_INFO_MESSAGE, RetrievedChunk, build_sentence_chunks,
    clean_extracted_text, extract_keywords, format_debug_retrieval,
    load_evaluation_questions, source_label, split_sentences_with_sections,
)


class KeywordEncoder:
    """A normalized test double, not a substitute benchmark for MiniLM."""
    vocabulary = ["harbor", "willow", "servic", "store", "deployment", "record", "sqlite",
                  "database", "publish", "index", "health", "check", "application", "log",
                  "plain", "text", "file", "rotate", "previous", "available"]

    def encode(self, texts, **options):
        vectors = np.array([[extract_keywords(text).count(term) for term in self.vocabulary]
                            for text in texts], dtype=float)
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(lengths == 0, 1, lengths)


class TextProcessingTests(unittest.TestCase):
    def test_cleaning_stops_before_references(self):
        text = "A useful statement about deployment records. [12]\nReferences\nUnrelated citation."
        self.assertEqual(clean_extracted_text(text), "A useful statement about deployment records.")

    def test_sections_follow_their_sentences(self):
        result = split_sentences_with_sections(
            "Storage\nThe service stores records in a database. Another sentence follows here.\n"
            "Recovery\nThe service restores the previous index after failure.")
        self.assertEqual([row["section"] for row in result], ["Storage", "Storage", "Recovery"])

    def test_chunks_overlap_without_losing_the_last_sentence(self):
        sentences = [{"text": f"Sentence {i}.", "section": "Storage"} for i in range(6)]
        chunks = build_sentence_chunks(sentences, 3, 1)
        self.assertEqual([(c["sentence_start"], c["sentence_end"]) for c in chunks],
                         [(0, 3), (2, 5), (4, 6)])
        self.assertIn("Sentence 5.", chunks[-1]["text"])

    def test_chunk_parameters_are_clamped_to_make_progress(self):
        sentences = [{"text": "One sentence.", "section": ""}] * 3
        self.assertEqual(len(build_sentence_chunks(sentences, 0, 100)), 3)
        self.assertEqual(build_sentence_chunks([]), [])

    def test_source_labels_do_not_expose_parent_paths(self):
        self.assertEqual(source_label("private/location/Harbor.txt", 3), "Harbor.txt#chunk-3")

    def test_evaluation_questions_can_be_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "questions.json"
            path.write_text('[{"question": "A question?", "gold_doc": "Harbor.txt"}]')
            self.assertEqual(load_evaluation_questions(path)[0]["gold_doc"], "Harbor.txt")


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.bot = ArticleRAGChatbot(model=KeywordEncoder())
        self.bot.load_articles_from_folder(Path(__file__).resolve().parents[1] / "examples/documents")
        self.bot.build_index()
        self.query = "Where does the Harbor service store deployment records?"

    def test_index_preserves_document_and_chunk_metadata(self):
        self.assertEqual([a["title"] for a in self.bot.articles], ["Harbor", "Willow"])
        self.assertEqual(self.bot.chunk_embeddings.shape[0], len(self.bot.chunks))
        np.testing.assert_allclose(np.linalg.norm(self.bot.chunk_embeddings, axis=1), 1)
        self.assertTrue(all("Title:" in c["text_for_index"] for c in self.bot.chunks))

    def test_missing_or_empty_corpus_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                self.bot.load_articles_from_folder(folder)
            with self.assertRaises(FileNotFoundError):
                self.bot.load_articles_from_folder(Path(folder) / "missing")

    def test_empty_index_fails_explicitly(self):
        with self.assertRaises(ValueError):
            ArticleRAGChatbot(model=KeywordEncoder()).build_index()

    def test_unbuilt_index_cannot_be_queried(self):
        bot = ArticleRAGChatbot(model=KeywordEncoder())
        for method in [bot.retrieve, bot.retrieve_naive, bot.answer_baseline]:
            with self.subTest(method=method.__name__), self.assertRaises(RuntimeError):
                method(self.query)

    def test_naive_scores_match_normalized_dot_products(self):
        expected = self.bot.chunk_embeddings @ self.bot._encode_query(self.query)
        result = self.bot.retrieve_naive(self.query, top_k=2)
        self.assertEqual([r.score for r in result], sorted(expected, reverse=True)[:2])

    def test_reranking_retrieves_the_relevant_document(self):
        result = self.bot.retrieve(self.query)
        self.assertTrue(result)
        self.assertEqual(result[0].title, "Harbor")
        self.assertTrue(all(r.rerank_score >= r.score for r in result))

    def test_metadata_filter_excludes_other_documents(self):
        result = self.bot.retrieve("Where does the Willow service store application logs?",
                                   metadata_filter={"title": "willow"})
        self.assertTrue(result)
        self.assertTrue(all(r.title == "Willow" for r in result))

    def test_similarity_threshold_can_reject_every_candidate(self):
        self.assertEqual(self.bot.retrieve(self.query, min_similarity=1.01), [])

    def test_unknown_topic_is_rejected_without_citations(self):
        result = self.bot.answer_with_rag("What was Apollo 11's lunar module called?")
        self.assertEqual(result["answer"], NOT_ENOUGH_INFO_MESSAGE)
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["sources"], [])

    def test_supported_answer_has_a_valid_source(self):
        result = self.bot.answer_with_rag(self.query, debug=True)
        self.assertIn("SQLite", result["answer"])
        citation = result["citations"][0]
        matching = [c for c in self.bot.chunks if c["source_path"] == citation["source_path"]
                    and c["chunk_id"] == citation["chunk_id"]]
        self.assertEqual(len(matching), 1)
        self.assertIn(result["answer"], matching[0]["text"])
        self.assertIn(result["sources"][0], format_debug_retrieval(result))

    def test_related_context_cannot_answer_an_unsupported_relation(self):
        result = self.bot.answer_with_rag("What is the capital of Harbor?")
        self.assertEqual(result["answer"], NOT_ENOUGH_INFO_MESSAGE)

    def test_expansion_uses_only_neighbors_from_the_same_document(self):
        bot = ArticleRAGChatbot(model=KeywordEncoder(), chunk_size_sentences=1,
                                overlap_sentences=0, context_window_chunks=1)
        bot.load_articles_from_folder(Path(__file__).resolve().parents[1] / "examples/documents")
        bot.build_index()
        chunk = bot.article_chunks[0][1]
        result = bot._chunk_to_retrieved(chunk, 0.9, self.query, self.query)
        self.assertEqual(result.expanded_chunk_ids, [0, 1, 2])
        self.assertIn("SQLite", result.expanded_text)
        self.assertNotIn("Willow", result.expanded_text)

    def test_answer_from_expanded_neighbor_cites_the_chunk_containing_it(self):
        bot = ArticleRAGChatbot(model=KeywordEncoder(), chunk_size_sentences=1,
                                overlap_sentences=0, context_window_chunks=1)
        bot.load_articles_from_folder(Path(__file__).resolve().parents[1] / "examples/documents")
        bot.build_index()
        neighbor = bot._chunk_to_retrieved(bot.article_chunks[0][1], 0.9, self.query, self.query)
        with patch.object(bot, "retrieve", return_value=[neighbor]):
            result = bot.answer_with_rag(self.query)
        self.assertIn("SQLite", result["answer"])
        citation = result["citations"][0]
        source_chunk = bot.article_chunks[0][citation["chunk_id"]]
        self.assertIn(result["answer"], source_chunk["text"])

    def test_duplicate_context_is_suppressed(self):
        item = RetrievedChunk("Harbor", "Harbor.txt", 0,
                              "The Harbor service stores records in a local SQLite database.", 0.8)
        self.assertEqual(self.bot._deduplicate_retrieved([item, item]), [item])

    def test_original_query_is_retained_alongside_keyword_rewrite(self):
        queries = self.bot.build_retrieval_queries(self.query)
        self.assertEqual(queries[0], self.query)
        self.assertEqual(len(set(queries)), len(queries))
        self.assertIn("harbor", queries[-1].lower())

    def test_evaluation_counts_supported_and_negative_cases_separately(self):
        result = self.bot.evaluate([
            {"question": self.query, "gold_doc": "Harbor.txt", "gold_keywords": ["sqlite"]},
            {"question": "Who wrote Hamlet?", "should_reject": True},
        ])
        self.assertEqual((result["supported_questions"], result["negative_questions"]), (1, 1))
        self.assertEqual(result["improved_rag"]["answer_accuracy"], 1)
        self.assertEqual(result["improved_rag"]["negative_rejection_rate"], 1)


class ModelBoundaryTests(unittest.TestCase):
    def test_default_model_loading_uses_the_pinned_revision_and_cpu(self):
        fake_module = types.ModuleType("sentence_transformers")
        calls = []
        fake_module.SentenceTransformer = lambda *a, **k: calls.append((a, k)) or KeywordEncoder()
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            ArticleRAGChatbot(local_files_only=True)
        self.assertEqual(calls, [((EMBEDDING_MODEL_NAME,), {
            "revision": EMBEDDING_MODEL_REVISION, "local_files_only": True, "device": "cpu"})])

    def test_import_does_not_load_the_model_library(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys, rag_chatbot; assert 'sentence_transformers' not in sys.modules"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_custom_model_does_not_receive_the_default_models_revision(self):
        fake_module = types.ModuleType("sentence_transformers")
        calls = []
        fake_module.SentenceTransformer = lambda *a, **k: calls.append((a, k)) or KeywordEncoder()
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            ArticleRAGChatbot(embedding_model_name="local-model", local_files_only=True)
        self.assertIsNone(calls[0][1]["revision"])

    def test_entrypoint_help_requires_no_model(self):
        for script in ["rag_chatbot.py", "demo.py", "evaluate_rag.py", "download_articles.py"]:
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, script, "--help"],
                    cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()

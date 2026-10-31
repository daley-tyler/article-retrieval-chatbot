"""Opt-in integration checks using real, cached MiniLM weights on CPU."""
import os
from pathlib import Path
import unittest

from rag_chatbot import ArticleRAGChatbot, NOT_ENOUGH_INFO_MESSAGE


@unittest.skipUnless(os.environ.get("RAG_RUN_MODEL_TESTS") == "1", "Set RAG_RUN_MODEL_TESTS=1 for cached-model tests")
class RealModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = ArticleRAGChatbot(local_files_only=True)
        cls.bot.load_articles_from_folder(Path(__file__).resolve().parents[1] / "examples/documents")
        cls.bot.build_index()

    def test_embeddings_have_the_expected_dimension(self):
        self.assertEqual(self.bot.chunk_embeddings.shape[1], 384)

    def test_real_model_answers_with_source_text(self):
        answer = self.bot.answer_with_rag("Where does the Harbor service store deployment records?")
        self.assertIn("SQLite", answer["answer"])
        self.assertTrue(answer["sources"][0].startswith("Harbor.txt#chunk-"))

    def test_real_model_refuses_an_out_of_corpus_question(self):
        answer = self.bot.answer_with_rag("What was the name of Apollo 11's lunar module?")
        self.assertEqual(answer["answer"], NOT_ENOUGH_INFO_MESSAGE)
        self.assertEqual(answer["sources"], [])

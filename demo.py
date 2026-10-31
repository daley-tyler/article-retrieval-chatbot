"""Run cited retrieval against the included fictional service notes."""
import argparse
from pathlib import Path

from rag_chatbot import ArticleRAGChatbot, format_debug_retrieval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=Path(__file__).parent / "examples/documents")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    bot = ArticleRAGChatbot(local_files_only=args.local_files_only)
    bot.load_articles_from_folder(args.documents)
    bot.build_index()
    for query in [
        "Where does the Harbor service store deployment records?",
        "What was the name of Apollo 11's lunar module?",
    ]:
        result = bot.answer_with_rag(query, debug=True)
        print(f"QUESTION: {query}\nANSWER: {result['answer']}")
        print("Sources:", ", ".join(result["sources"]) or "none")
        print(format_debug_retrieval(result))
        print()


if __name__ == "__main__":
    main()

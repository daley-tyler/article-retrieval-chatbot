from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from rag_chatbot import (
    ArticleRAGChatbot,
    load_evaluation_questions,
)


def format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def print_table(headers: List[str], rows: Iterable[List[str]]) -> None:
    table = [headers, *rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def build_summary_rows(results: Dict[str, Any]) -> List[List[str]]:
    baseline = results["article_baseline"]
    naive = results["naive_rag"]
    improved = results["improved_rag"]

    return [
        [
            "Answer keyword match",
            format_rate(baseline["answer_accuracy"]),
            format_rate(naive["answer_accuracy"]),
            format_rate(improved["answer_accuracy"]),
        ],
        [
            "Retrieval top-1 accuracy",
            "n/a",
            format_rate(naive["retrieval_top1_accuracy"]),
            format_rate(improved["retrieval_top1_accuracy"]),
        ],
        [
            "Retrieval top-k hit rate",
            "n/a",
            format_rate(naive["retrieval_topk_hit_rate"]),
            format_rate(improved["retrieval_topk_hit_rate"]),
        ],
        [
            "Gold document present (same as top-k)",
            "n/a",
            format_rate(naive["context_relevance"]),
            format_rate(improved["context_relevance"]),
        ],
        [
            "Source containment / refusal",
            format_rate(baseline["answer_fidelity"]),
            format_rate(naive["answer_fidelity"]),
            format_rate(improved["answer_fidelity"]),
        ],
        [
            "Negative rejection",
            format_rate(baseline["negative_rejection_rate"]),
            format_rate(naive["negative_rejection_rate"]),
            format_rate(improved["negative_rejection_rate"]),
        ],
    ]


def compare_old_and_new_pipelines(
    questions: List[Dict[str, Any]],
    article_dir: Path,
    *,
    local_files_only: bool = False,
) -> tuple[Dict[str, Any], ArticleRAGChatbot, ArticleRAGChatbot]:
    old_bot = ArticleRAGChatbot(
        chunk_size_sentences=4,
        overlap_sentences=1,
        top_k=3,
        candidate_k=3,
        small_to_big=False,
        min_similarity=0.0,
        local_files_only=local_files_only,
    )
    old_bot.load_articles_from_folder(article_dir)
    old_bot.build_index()
    old_results = old_bot.evaluate(questions, top_k=3)

    improved_bot = ArticleRAGChatbot(local_files_only=local_files_only)
    improved_bot.load_articles_from_folder(article_dir)
    improved_bot.build_index()
    improved_results = improved_bot.evaluate(questions)

    combined_details = []
    for old_row, new_row in zip(old_results["details"], improved_results["details"]):
        combined_details.append(
            {
                "question": new_row["question"],
                "gold_doc": new_row["gold_doc"],
                "should_reject": new_row["should_reject"],
                "naive_retrieved_docs": old_row["naive_retrieved_docs"],
                "improved_retrieved_docs": new_row["improved_retrieved_docs"],
                "naive_rag_answer": old_row["naive_rag_answer"],
                "improved_rag_answer": new_row["improved_rag_answer"],
                "improved_sources": new_row["improved_sources"],
            }
        )

    combined_results = {
        "total_questions": improved_results["total_questions"],
        "supported_questions": improved_results["supported_questions"],
        "negative_questions": improved_results["negative_questions"],
        "article_baseline": old_results["article_baseline"],
        "naive_rag": old_results["naive_rag"],
        "improved_rag": improved_results["improved_rag"],
        "details": combined_details,
    }
    return combined_results, old_bot, improved_bot


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare retrieval pipelines on development questions.")
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("evaluation_questions.json"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    article_dir = args.documents
    eval_path = args.questions

    questions = load_evaluation_questions(eval_path)

    results, old_bot, improved_bot = compare_old_and_new_pipelines(questions, article_dir, local_files_only=args.local_files_only)

    print("=== Before vs After RAG Evaluation ===")
    print(f"Evaluation file: {eval_path}")
    print(f"Articles: {article_dir}")
    print(f"Supported questions: {results['supported_questions']}")
    print(f"Negative questions: {results['negative_questions']}")
    print(
        "Config: "
        f"old(chunk_size={old_bot.chunk_size_sentences}, top_k={old_bot.top_k}) -> "
        f"new(chunk_size={improved_bot.chunk_size_sentences}, "
        f"overlap={improved_bot.overlap_sentences}, "
        f"candidate_k={improved_bot.candidate_k}, "
        f"top_k={improved_bot.top_k}, "
        f"min_similarity={improved_bot.min_similarity}, "
        f"min_answer_keyword_overlap={improved_bot.min_answer_keyword_overlap})"
    )
    print()

    print_table(
        ["Metric", "Article baseline", "Old naive RAG", "Improved RAG"],
        build_summary_rows(results),
    )

    print("\nKeyword matches and containment are heuristics, not independent factual judgments.")
    print("Baseline containment is 1 by construction; context relevance duplicates top-k hit rate.")
    print("\n=== Question Details ===")
    for row in results["details"]:
        status = "REJECT" if row["should_reject"] else "ANSWER"
        print(f"\n[{status}] {row['question']}")
        print(f"Gold doc: {row['gold_doc'] or '(not in corpus)'}")
        print(f"Old docs: {row['naive_retrieved_docs']}")
        print(f"New docs: {row['improved_retrieved_docs']}")
        print(f"Old answer: {row['naive_rag_answer']}")
        print(f"New answer: {row['improved_rag_answer']}")
        if row["improved_sources"]:
            print(f"New sources: {', '.join(row['improved_sources'])}")


if __name__ == "__main__":
    main()

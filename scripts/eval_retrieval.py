import json
from pathlib import Path

from app.services.retriever import retrieve_chunks
from app.services.reranker import rerank_chunks

EVAL_PATH = Path("eval/retrieval_eval.json")


def reciprocal_rank(results: list[dict], expected_papers: list[str]) -> float:
    for idx, row in enumerate(results, start=1):
        if row["paper_id"] in expected_papers:
            return 1.0 / idx
    return 0.0


def hit_at_k(results: list[dict], expected_papers: list[str]) -> int:
    return int(any(row["paper_id"] in expected_papers for row in results))


def main():
    with EVAL_PATH.open("r", encoding="utf-8") as f:
        examples = json.load(f)

    total = len(examples)
    hits = 0
    rr_sum = 0.0

    for item in examples:
        question = item["question"]
        expected_papers = item["expected_papers"]

        retrieval = retrieve_chunks(question, k=5)
        
        results = rerank_chunks(question,retrieval,top_n=5)

        hit = hit_at_k(results, expected_papers)
        rr = reciprocal_rank(results, expected_papers)

        hits += hit
        rr_sum += rr

        print("=" * 80)
        print("Q:", question)
        print("Expected:", expected_papers)
        print("Hit@5:", hit, "| RR:", rr)

        for r in results:
            print(
                f"- {r['paper_id']} | {r['chunk_id']} | "
                f"sim={float(r['similarity']):.4f}"
            )

    hit_rate = hits / total if total else 0.0
    mrr = rr_sum / total if total else 0.0

    print("\n" + "=" * 80)
    print(f"Total questions: {total}")
    print(f"Hit@5: {hit_rate:.4f}")
    print(f"MRR:   {mrr:.4f}")


if __name__ == "__main__":
    main()
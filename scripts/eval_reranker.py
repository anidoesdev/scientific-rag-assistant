import json
from pathlib import Path

from app.services.retriever import retrieve_chunks
from app.services.reranker import rerank_chunks

EVAL_PATH = Path("eval/retrieval_eval.json")
CANDIDATE_K = 10
FINAL_K = 5


def reciprocal_rank(results: list[dict], expected_papers: list[str]) -> float:
    for idx, row in enumerate(results, start=1):
        if row["paper_id"] in expected_papers:
            return 1.0 / idx
    return 0.0


def hit_at_k(results: list[dict], expected_papers: list[str]) -> int:
    return int(any(row["paper_id"] in expected_papers for row in results))


def summarize(name: str, total: int, hits: int, rr_sum: float):
    hit_rate = hits / total if total else 0.0
    mrr = rr_sum / total if total else 0.0
    print(f"{name} -> Hit@{FINAL_K}: {hit_rate:.4f} | MRR: {mrr:.4f}")


def main():
    with EVAL_PATH.open("r", encoding="utf-8") as f:
        examples = json.load(f)

    base_hits = 0
    base_rr = 0.0

    rerank_hits = 0
    rerank_rr = 0.0

    for item in examples:
        question = item["question"]
        expected_papers = item["expected_papers"]

        retrieved = retrieve_chunks(question, k=CANDIDATE_K)

        baseline = retrieved[:FINAL_K]
        reranked = rerank_chunks(question, retrieved, top_n=FINAL_K)

        base_hits += hit_at_k(baseline, expected_papers)
        base_rr += reciprocal_rank(baseline, expected_papers)

        rerank_hits += hit_at_k(reranked, expected_papers)
        rerank_rr += reciprocal_rank(reranked, expected_papers)

        print("=" * 100)
        print("Q:", question)
        print("Expected:", expected_papers)

        print("\nBaseline top results:")
        for i, r in enumerate(baseline, start=1):
            print(
                f"{i}. {r['paper_id']} | {r['chunk_id']} | "
                f"sim={float(r.get('similarity', 0.0)):.4f}"
            )

        print("\nReranked top results:")
        for i, r in enumerate(reranked, start=1):
            print(
                f"{i}. {r['paper_id']} | {r['chunk_id']} | "
                f"sim={float(r.get('similarity', 0.0)):.4f} | "
                f"rerank={int(r.get('rerank_score', 0))}"
            )

    print("\n" + "=" * 100)
    summarize("Baseline", len(examples), base_hits, base_rr)
    summarize("Reranked", len(examples), rerank_hits, rerank_rr)


if __name__ == "__main__":
    main()
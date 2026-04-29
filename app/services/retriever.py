import ollama
from sqlalchemy import text as sql_text
from app.db.session import SessionLocal

OLLAMA_EMBED_MODEL = "nomic-embed-text"

def retrieve_chunks(question:str,k:int):
    response = ollama.embed(
        model=OLLAMA_EMBED_MODEL,
        input=question
    )
    query_embeddings = response.embeddings[0]
    stmt = sql_text(
        """
        SELECT
        chunk_id,
        paper_id,
        text,
        1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
        FROM chunks
        ORDER BY embedding <=> CAST(:query_embedding AS vector)
        LIMIT :k;
        """
    )
    with SessionLocal() as session:
        rows = session.execute(
            stmt,
            {
                "query_embedding": f"[{','.join(map(str, query_embeddings))}]",
                "k": k
            }
        ).mappings().all()
    return [dict(row) for row in rows]


# results = retrieve_chunks("What are unidirectional error correcting codes?", k=3)

# for row in results:
#     print(row["chunk_id"], row["similarity"])
#     print(row["text"][:300])
#     print("-" * 80)
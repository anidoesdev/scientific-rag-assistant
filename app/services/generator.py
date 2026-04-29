import ollama

OLLAMA_CHAT_MODEL = "llama3.1"


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i,chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}]\n"
            f"chunk_id: {chunk['chunk_id']}\n"
            f"paper_id: {chunk['paper_id']}\n"
            f"similarity: {chunk['similarity']}\n"
            f"text:\n{chunk['text']}\n"
        )
    return "\n\n".join(parts)


def build_prompt(question: str,chunks: list[dict]) -> str:
    context = build_context(chunks)

    return f"""
        You are a scientific research assistant.

        Answer the user's question using only the provided retrieved context.
        If the answer is not supported by the context, say that clearly.
        Do not make up facts.
        Cite sources inline using [Source 1], [Source 2], etc. based on the provided context.

        User question:
        {question}

        Retrieved context:
        {context}

        Return a concise but helpful answer with inline citations.
        """.strip()


def generate_answer(question: str,chunks: list[dict]) -> dict:
    prompt = build_prompt(question, chunks)
    
    response = ollama.chat(
        model=OLLAMA_CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.2
        }
    )
    
    answer_text = response["messages"]["content"].strip()
    citations = [
        {
            "chunk_id": chunk["chunk_id"],
            "paper_id": chunk["paper_id"],
            "similarity": float(chunk["similarity"]),
        }
        for chunk in chunks
    ]
    
    retrieved_chunks = [
        {
            "chunk_id": chunk["chunk_id"],
            "paper_id": chunk["paper_id"],
            "text": chunk["text"],
            "similarity": float(chunk["similarity"])
        }
        for chunk in chunks
    ]
    
    return {
        "answer":answer_text,
        "citations": citations,
        "retrieved_chunks": retrieved_chunks
    }
    
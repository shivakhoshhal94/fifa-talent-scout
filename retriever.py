# retriever.py
import openai
import numpy as np

PROMPT_TEMPLATE = """
You are a scouting assistant. Use the retrieved player information to answer the question.
Question: {query}
Relevant players: {docs}
Provide a short, clear scouting-style answer.
"""

def embed_query(query: str, model="text-embedding-3-small"):
    resp = openai.Embedding.create(model=model, input=[query])
    vec = np.array(resp["data"][0]["embedding"], dtype=np.float32)
    return vec.reshape(1, -1)

def build_qa_chain(index, metadata):
    """
    Returns a function query(string) -> text answer
    """
    def qa(query: str):
        from indexer_chunked import search_index_faiss
        vec = embed_query(query)
        results = search_index_faiss(index, metadata, vec, k=5)

        docs_text = ""
        for meta, score in results:
            docs_text += f"[{score:.2f}] {meta}\n"

        prompt = PROMPT_TEMPLATE.format(query=query, docs=docs_text)

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response["choices"][0]["message"]["content"]

    return qa

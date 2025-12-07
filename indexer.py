
""

import os
import pickle
import time
from typing import List, Tuple, Dict, Any
import numpy as np
import faiss
import openai

# config
EMBED_CACHE = "embeddings_cache_no_lc.pkl"
FAISS_INDEX_FILE = "faiss_index_no_lc.index"
METADATA_FILE = "faiss_metadata_no_lc.pkl"
OPENAI_EMBED_MODEL = "text-embedding-3-small" 
BATCH_SIZE = 64


if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Please set OPENAI_API_KEY in environment before using indexer_no_langchain")

openai.api_key = os.getenv("OPENAI_API_KEY")

def _get_page_text(doc_or_row) -> str:
    
    if hasattr(doc_or_row, "page_content"):
        return doc_or_row.page_content
    if isinstance(doc_or_row, dict):
        return doc_or_row.get("page_content") or doc_or_row.get("text") or ""
    return str(doc_or_row)

def _get_row_hash(doc_or_row) -> str:
    
    if hasattr(doc_or_row, "metadata"):
        return str(doc_or_row.metadata.get("row_hash", ""))
    if isinstance(doc_or_row, dict):
        return str(doc_or_row.get("metadata", {}).get("row_hash", ""))
    return ""

def _get_metadata(doc_or_row) -> Dict[str, Any]:
    if hasattr(doc_or_row, "metadata"):
        return dict(doc_or_row.metadata)
    if isinstance(doc_or_row, dict):
        return dict(doc_or_row.get("metadata", {}))
    return {"text": _get_page_text(doc_or_row)}

def embed_texts(texts: List[str], model: str = OPENAI_EMBED_MODEL) -> List[List[float]]:
    """Batch embeddings via OpenAI API."""
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        resp = openai.Embedding.create(model=model, input=batch)
        for item in resp["data"]:
            vectors.append(item["embedding"])
    return vectors

def build_index(docs: List, rebuild: bool = False) -> None:
    """
    Build or load a FAISS index (file-based). docs: list of Document-like objects.
    Produces files: FAISS_INDEX_FILE, EMBED_CACHE, METADATA_FILE.
    """
    # load caches if available
    embed_cache: Dict[str, np.ndarray] = {}
    metadata_map: List[Dict] = []
    if os.path.exists(EMBED_CACHE) and not rebuild:
        with open(EMBED_CACHE, "rb") as f:
            embed_cache = pickle.load(f)
        print(f"[indexer_no_lc] loaded embed cache: {len(embed_cache)} entries")

    # create list of texts to embed and track order
    texts_to_embed = []
    hashes_to_embed = []
    docs_order = []
    for doc in docs:
        h = _get_row_hash(doc)
        docs_order.append(h)
        if h and h in embed_cache and not rebuild:
            continue
        texts_to_embed.append(_get_page_text(doc))
        hashes_to_embed.append(h)

    # embed new texts
    if texts_to_embed:
        print(f"[indexer_no_lc] embedding {len(texts_to_embed)} docs via OpenAI...")
        start = time.time()
        new_vectors = embed_texts(texts_to_embed)
        for h, vec in zip(hashes_to_embed, new_vectors):
            embed_cache[h] = np.asarray(vec, dtype=np.float32)
        # persist cache
        with open(EMBED_CACHE, "wb") as f:
            pickle.dump(embed_cache, f)
        print(f"[indexer_no_lc] persisted embed cache ({len(embed_cache)} vectors) in {EMBED_CACHE} ({time.time()-start:.1f}s)")

    # Build vector matrix in order of docs
    vectors = []
    metadata_map = []
    for doc, h in zip(docs, docs_order):
        if h in embed_cache:
            vec = embed_cache[h]
        else:
            # if missing, compute on the fly (rare)
            vec = np.asarray(embed_texts([_get_page_text(doc)])[0], dtype=np.float32)
            embed_cache[h] = vec
        vectors.append(vec)
        metadata_map.append(_get_metadata(doc))

    if len(vectors) == 0:
        raise RuntimeError("No vectors to build index from")

    vectors_np = np.vstack(vectors).astype("float32")
    dim = vectors_np.shape[1]

    # create or overwrite index
    index = faiss.IndexFlatIP(dim) 
    faiss.normalize_L2(vectors_np)
    index.add(vectors_np)

    # save index and metadata
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata_map, f)
    with open(EMBED_CACHE, "wb") as f:
        pickle.dump(embed_cache, f)
    print(f"[indexer_no_lc] Saved FAISS index ({index.ntotal} vectors) to {FAISS_INDEX_FILE}")

def load_index() -> Tuple[faiss.Index, List[Dict]]:
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("Index or metadata file not found; build_index first.")
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata_map = pickle.load(f)
    return index, metadata_map

def search_index(query: str, k: int = 5, model: str = OPENAI_EMBED_MODEL) -> List[Tuple[Dict, float]]:
    """Embed query, search FAISS, return top-k (metadata, score)."""
    index, metadata_map = load_index()
    q_emb = np.asarray(embed_texts([query], model=model)[0], dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)  # D: scores, I: indices
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata_map):
            continue
        results.append((metadata_map[idx], float(score)))
    return results

if __name__ == "__main__":
    print("indexer_no_langchain module loaded. Call build_index(docs) with your docs list to create index.")

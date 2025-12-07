
import os
import time
import pickle
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

import numpy as np
import pandas as pd

from typing import List, Dict, Any, Tuple


OPENAI_ENV_NAME = "OPENAI_API_KEY"

CSV_PATH = "data/male_players.csv"


CANONICAL_COLS = [
    "sofifa_id", "long_name", "age", "club_name", "league_name",
    "nationality_name", "player_pos", "preferred_foot", "weak_foot",
    "overall", "potential", "pace", "shooting", "passing",
    "dribbling", "defending", "physic"
]

# common alternative names observed in your CSV (keeps mapping robust)
CANONICAL_TO_POSSIBLE = {
    "sofifa_id":         ["player_id", "sofifa_id", "id"],
    "long_name":         ["long_name", "long name", "name", "short_name"],
    "age":               ["age", "player_age"],
    "club_name":         ["club_name", "club name", "club"],
    "league_name":       ["league_name", "league name", "league"],
    "nationality_name":  ["nationality_name", "nationality name", "nationality"],
    "player_pos":        ["player_pos", "player_positions", "player_position", "club_position"],
    "preferred_foot":    ["preferred_foot", "preferred foot"],
    "weak_foot":         ["weak_foot", "weak foot"],
    "overall":           ["overall", "ovr"],
    "potential":         ["potential", "pot"],
    "value_eur":         ["value_eur", "value_eur"],
    "wage_eur":          ["wage_eur", "wage_eur"],
    "pace":              ["pace"],
    "shooting":          ["shooting"],
    "passing":           ["passing", "attacking_short_passing", "skill_long_passing"],
    "dribbling":         ["dribbling", "skill_dribbling"],
    "defending":         ["defending", "defending_standing_tackle"],
    "physic":            ["physic", "power_stamina", "power_strength"]
}

# Index / cache files
FAISS_INDEX_FILE = "faiss_index_chunked.index"
METADATA_FILE = "faiss_metadata_chunked.pkl"
CACHE_FILE = "embeddings_cache_chunked.pkl"

# Embedding settings 
EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 128

def _detect_column_map(csv_columns: List[str]) -> Dict[str, str]:
    csv_lower = {c.lower(): c for c in csv_columns}
    mapping: Dict[str, str] = {}
    for canon in CANONICAL_COLS:
        chosen = None
        possibilities = CANONICAL_TO_POSSIBLE.get(canon, [canon])
        for p in possibilities:
            lp = p.lower()
            if lp in csv_lower:
                chosen = csv_lower[lp]
                break
        if chosen is None and canon in csv_lower:
            chosen = csv_lower[canon]
        mapping[canon] = chosen
    return mapping

def _rename_and_fill(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    rename_map = {}
    for canon, actual in mapping.items():
        if actual and actual in df.columns and actual != canon:
            rename_map[actual] = canon
    df2 = df.rename(columns=rename_map).copy()
    for canon in CANONICAL_COLS:
        if canon not in df2.columns:
            df2[canon] = ""
    # ensure canonical order
    df2 = df2[CANONICAL_COLS]
    return df2

def row_to_text(row: pd.Series) -> str:
    return (
        f"{row.get('long_name','')} | Age:{row.get('age','')} | "
        f"Pos:{row.get('player_pos','')} | OVR:{row.get('overall','')} | POT:{row.get('potential','')}"
    )

def _ensure_openai_key():
    key = os.getenv(OPENAI_ENV_NAME)
    if not key:
        raise RuntimeError(f"Set {OPENAI_ENV_NAME} in environment before building index")
    openai.api_key = key
    return key

def embed_texts_openai(texts: List[str], model: str = EMBED_MODEL, batch: int = EMBED_BATCH) -> List[List[float]]:
    vectors = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i+batch]
        resp = openai.Embedding.create(model=model, input=batch_texts)
        for item in resp["data"]:
            vectors.append(item["embedding"])
    return vectors

def load_cache() -> Dict[str, np.ndarray]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            raw = pickle.load(f)
        return {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
    return {}

def save_cache(cache: Dict[str, np.ndarray]):
    out = {k: v.tolist() for k, v in cache.items()}
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(out, f)

INDEX_DIR = "faiss_langchain_index"  # folder name to store index


def build_index_chunked(
    csv_path: str = CSV_PATH,
    chunk_rows: int = 15000,
    progress_callback=None,
):
    """
    Build a FAISS index using LangChain instead of manual faiss/openai calls.
    Saves it to disk.
    """
    _ensure_openai_key()  # ensures API key exists

    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=os.getenv(OPENAI_ENV_NAME),
    )

    vectorstore = None
    total_rows = 0

    reader = pd.read_csv(
        csv_path,
        chunksize=chunk_rows,
        low_memory=True,
        encoding="utf-8",
    )

    for chunk_i, raw_df in enumerate(reader):
        df = raw_df.fillna("")
        mapping = _detect_column_map(list(df.columns))
        df_can = _rename_and_fill(df, mapping)

        texts = []
        metadatas = []
        for _, row in df_can.iterrows():
            texts.append(row_to_text(row))
            metadatas.append(
                {
                    "id": row.get("sofifa_id", None),
                    "name": row.get("long_name", ""),
                    "club": row.get("club_name", ""),
                    "position": row.get("player_pos", ""),
                    "overall": row.get("overall", ""),
                    "potential": row.get("potential", ""),
                }
            )

        if vectorstore is None:
            vectorstore = FAISS.from_texts(
                texts=texts, embedding=embeddings, metadatas=metadatas
            )
        else:
            vectorstore.add_texts(texts=texts, metadatas=metadatas)

        vectorstore.save_local(INDEX_DIR)

        total_rows += len(df_can)
        if callable(progress_callback):
            progress_callback(chunk_i, total_rows, 0)

    return vectorstore



    for chunk_i, raw_df in enumerate(reader):
        t0 = time.time()
        df = raw_df.fillna("")
        mapping = _detect_column_map(list(df.columns))
        df_can = _rename_and_fill(df, mapping)

        texts = [row_to_text(r) for _, r in df_can.iterrows()]
        keys = [str(r["sofifa_id"]) if str(r["sofifa_id"]) else f"row_{total_rows + i}" for i, (_, r) in enumerate(df_can.iterrows())]

        need_idx = [i for i, k in enumerate(keys) if k not in cache]
        if need_idx:
            to_embed = [texts[i] for i in need_idx]
            new_vectors = embed_texts_openai(to_embed, batch=embed_batch)
            for i_pos, vec in zip(need_idx, new_vectors):
                cache[keys[i_pos]] = np.asarray(vec, dtype=np.float32)
            save_cache(cache)

        chunk_vecs = np.vstack([cache[k] for k in keys]).astype(np.float32)
        faiss.normalize_L2(chunk_vecs)
        if index is None:
            dim = chunk_vecs.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(chunk_vecs)
        else:
            index.add(chunk_vecs)

        for _, row in df_can.iterrows():
            metadata.append({
                "id": row.get("sofifa_id", None),
                "name": row.get("long_name", ""),
                "club": row.get("club_name", ""),
                "position": row.get("player_pos", ""),
                "overall": row.get("overall", ""),
                "potential": row.get("potential", "")
            })

        total_rows += len(df_can)
        faiss.write_index(index, FAISS_INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(metadata, f)

        elapsed = time.time() - t0
        # call progress callback if provided
        if callable(progress_callback):
            try:
                progress_callback(chunk_i, total_rows, elapsed)
            except Exception:
                pass

    if index is None:
        raise RuntimeError("No data processed; check CSV path/headers")
    return index, metadata


def load_vectorstore() -> FAISS:
    """Load LangChain-FAISS index from disk."""
    _ensure_openai_key()
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=os.getenv(OPENAI_ENV_NAME),
    )
    return FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )


def search_players(query: str, k: int = 5):
    """Search players semantically via LangChain FAISS."""
    vs = load_vectorstore()
    results = vs.similarity_search_with_score(query, k=k)

    formatted = []
    for doc, score in results:
        meta = dict(doc.metadata)
        meta["text"] = doc.page_content
        formatted.append((meta, score))

    return formatted



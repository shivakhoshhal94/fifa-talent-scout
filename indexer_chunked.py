# indexer_chunked.py (fixed)
import os
import time
import pickle
import faiss
import numpy as np
import pandas as pd
import openai
from typing import List, Dict, Any, Tuple

# NOTE: do NOT raise on import if OPENAI_API_KEY missing. We'll check lazily in build_index_chunked.
OPENAI_ENV_NAME = "OPENAI_API_KEY"

CSV_PATH = "data/male_players.csv"

# canonical columns our app/indexer expects
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

# Embedding settings (tune EMBED_BATCH down if memory/rate-limits)
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

def build_index_chunked(csv_path: str = CSV_PATH, chunk_rows: int = 15000, embed_batch: int = EMBED_BATCH, progress_callback=None):
    """
    progress_callback: optional function called as progress_callback(chunk_i, total_rows, time_taken_seconds)
    """
    _ensure_openai_key()

    cache = load_cache()
    metadata: List[Dict[str, Any]] = []
    index = None
    total_rows = 0

    reader = pd.read_csv(csv_path, chunksize=chunk_rows, low_memory=True)

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


def load_index_and_meta() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError("Index not built")
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("Metadata not found")
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def search_index_faiss(index: faiss.Index, metadata: List[Dict[str, Any]], query_emb: np.ndarray, k: int = 5):
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx >= 0 and idx < len(metadata):
            results.append((metadata[idx], float(dist)))
    return results

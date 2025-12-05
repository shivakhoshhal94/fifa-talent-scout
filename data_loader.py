# data_loader.py (REPLACE your existing file with this exact content)
from typing import Tuple, List, Dict, Any
import pandas as pd
import os
import hashlib
import json

# canonical columns our app expects
CANONICAL_COLS = [
    "sofifa_id", "long_name", "age", "club_name", "league_name",
    "nationality_name", "player_pos", "preferred_foot", "weak_foot",
    "skill_moves", "overall", "potential", "value_eur", "wage_eur",
    "pace", "shooting", "passing", "dribbling", "defending", "physic"
]

# common alternative names found in CSVs (observed in your file)
CANONICAL_TO_POSSIBLE = {
    "sofifa_id":         ["player_id", "sofifa_id", "id"],
    "long_name":         ["long_name", "long name", "name", "short_name"],
    "age":               ["age", "player_age"],
    "club_name":         ["club_name", "club name", "club_name"],
    "league_name":       ["league_name", "league name", "league"],
    "nationality_name":  ["nationality_name", "nationality name", "nationality"],
    "player_pos":        ["player_pos", "player_positions", "player_position", "club_position", "player_pos"],
    "preferred_foot":    ["preferred_foot", "preferred foot"],
    "weak_foot":         ["weak_foot", "weak foot"],
    "skill_moves":       ["skill_moves", "skill moves"],
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

def _detect_column_map(csv_columns: List[str]) -> Dict[str, str]:
    """
    Return mapping canonical_name -> actual_csv_column_name (or None if not found).
    """
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
        # final fallback: if canonical itself present (exact case)
        if chosen is None and canon in csv_lower:
            chosen = csv_lower[canon]
        mapping[canon] = chosen
    return mapping

def _make_row_text(row: pd.Series) -> str:
    # row expected to contain canonical columns (we will ensure that)
    parts = [
        f"Name: {row.get('long_name','')}",
        f"Age: {row.get('age','')}",
        f"Club: {row.get('club_name','')}",
        f"League: {row.get('league_name','')}",
        f"Nationality: {row.get('nationality_name','')}",
        f"Position: {row.get('player_pos','')}",
        f"Preferred foot: {row.get('preferred_foot','')}",
        f"Overall: {row.get('overall','')}",
        f"Potential: {row.get('potential','')}",
        f"Pace: {row.get('pace','')}, Shooting: {row.get('shooting','')}, Passing: {row.get('passing','')}, Dribbling: {row.get('dribbling','')}, Defending: {row.get('defending','')}, Physic: {row.get('physic','')}"
    ]
    return ", ".join([p for p in parts if p and not p.endswith(": ")])

def _row_hash_from_row(row: pd.Series) -> str:
    d = {c: str(row.get(c, "")) for c in CANONICAL_COLS}
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _rename_and_fill(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Create a DataFrame that has canonical column names:
      - rename actual -> canonical where mapping points to an actual column
      - create any missing canonical columns (filled with "")
    Returns a new DataFrame with columns CANONICAL_COLS in that order.
    """
    rename_map = {}
    for canon, actual in mapping.items():
        if actual and actual in df.columns and actual != canon:
            rename_map[actual] = canon
    df2 = df.rename(columns=rename_map).copy()
    # ensure all canonical columns exist
    for canon in CANONICAL_COLS:
        if canon not in df2.columns:
            df2[canon] = ""
    # Keep only canonical columns in canonical order
    df2 = df2[CANONICAL_COLS]
    return df2

def load_preview(csv_path: str = "data/male_players.csv", nrows: int = 200) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Read a small preview from csv_path and return:
      - preview_df with canonical columns (safe placeholders added)
      - mapping canonical->actual (None if not found)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
    mapping = _detect_column_map(list(df.columns))
    df_canonical = _rename_and_fill(df, mapping)
    return df_canonical, mapping

def load_full_and_docs(csv_path: str = "data/male_players.csv", chunksize: int = None) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Load the full dataframe (or concatenated chunks) and produce docs list.
    Note: this loads full file into memory unless chunksize-based processing is used.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    if chunksize:
        df_iter = pd.read_csv(csv_path, chunksize=chunksize, low_memory=True)
        df = pd.concat(df_iter, ignore_index=True)
    else:
        df = pd.read_csv(csv_path, low_memory=False)

    mapping = _detect_column_map(list(df.columns))
    df_canonical = _rename_and_fill(df, mapping)

    docs: List[Dict[str, Any]] = []
    for _, row in df_canonical.iterrows():
        text = _make_row_text(row)
        metadata = {
            "player_id": row.get("sofifa_id", None),
            "name": row.get("long_name", ""),
            "age": row.get("age", ""),
            "club": row.get("club_name", ""),
            "position": row.get("player_pos", ""),
            "overall": row.get("overall", ""),
            "potential": row.get("potential", ""),
            "row_hash": _row_hash_from_row(row)
        }
        docs.append({"page_content": text, "metadata": metadata})
    return df_canonical, docs

# compatibility wrapper expected by existing app.py
def load_data(csv_path: str = "data/male_players.csv") -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    preview_df, _ = load_preview(csv_path, nrows=200)
    # lightweight docs from preview
    docs = []
    for _, row in preview_df.iterrows():
        text = _make_row_text(row)
        metadata = {
            "player_id": row.get("sofifa_id", None),
            "name": row.get("long_name", ""),
            "age": row.get("age", ""),
            "club": row.get("club_name", ""),
            "position": row.get("player_pos", ""),
            "overall": row.get("overall", ""),
            "potential": row.get("potential", ""),
            "row_hash": _row_hash_from_row(row)
        }
        docs.append({"page_content": text, "metadata": metadata})
    return preview_df, docs

if __name__ == "__main__":
    # quick CLI test
    import sys
    path = "data/male_players.csv" if len(sys.argv) < 2 else sys.argv[1]
    df_preview, mapping = load_preview(path, nrows=5)
    print("Preview columns (canonical):", list(df_preview.columns))
    print("Mapping canonical->actual (None if none):")
    for k, v in mapping.items():
        print("  ", k, "->", v)
    print(df_preview.head(3).to_string(index=False))

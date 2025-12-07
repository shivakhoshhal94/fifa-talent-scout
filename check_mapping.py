import pandas as pd
from indexer_chunked import _detect_column_map, _rename_and_fill, row_to_text

CSV_PATH = "data/male_players.csv"  # adjust if your file is elsewhere

df = pd.read_csv(CSV_PATH, nrows=5, encoding="utf-8")
mapping = _detect_column_map(list(df.columns))
print("COLUMN MAP:")
for k, v in mapping.items():
    print(f"  {k:15s} -> {v}")

df_can = _rename_and_fill(df.fillna(""), mapping)
print("\nCANONICAL VIEW (first 5 rows):")
print(df_can[["sofifa_id", "long_name", "age", "player_pos", "overall", "potential"]])

print("\nTEXTS THAT WILL BE EMBEDDED:")
for _, r in df_can.iterrows():
    print("  ", row_to_text(r))

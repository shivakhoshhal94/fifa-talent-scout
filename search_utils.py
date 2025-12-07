# search_utils.py
import re
from typing import List, Tuple, Optional

import pandas as pd
from rapidfuzz import fuzz

# ----------------------------
# Dictionaries / helpers
# ----------------------------

NATIONALITY_ADJECTIVES = {
    "dutch": ["netherlands", "holland", "the netherlands"],
    "english": ["england", "united kingdom", "uk", "great britain"],
    "scottish": ["scotland"],
    "welsh": ["wales"],
    "irish": ["ireland"],
    "french": ["france"],
    "german": ["germany"],
    "spanish": ["spain"],
    "italian": ["italy"],
    "portuguese": ["portugal"],
    "brazilian": ["brazil"],
    "argentinian": ["argentina"],
    "belgian": ["belgium"],
    "danish": ["denmark"],
    "swedish": ["sweden"],
    "norwegian": ["norway"],
    "finnish": ["finland"],
    "turkish": ["turkey"],
    "polish": ["poland"],
    "croatian": ["croatia"],
    "serbian": ["serbia"],
    "chilean": ["chile"],
    "mexican": ["mexico"],
    "american": ["united states", "usa", "united states of america"],
    "canadian": ["canada"],
}

POSITION_SYNONYMS = {
    "forward": ["st", "cf", "rw", "lw", "lf", "rf"],
    "striker": ["st"],
    "winger": ["rw", "lw", "rm", "lm"],
    "midfielder": ["cm", "cam", "cdm", "lcm", "rcm", "rm", "lm"],
    "defender": ["cb", "lcb", "rcb", "lb", "rb", "lwb", "rwb"],
    "centre-back": ["cb", "lcb", "rcb"],
    "fullback": ["lb", "rb", "lwb", "rwb"],
    "right-back": ["rb", "rwb"],
    "left-back": ["lb", "lwb"],
    "goalkeeper": ["gk"],
}


def _expand_position_token(tok: str) -> Optional[List[str]]:
    t = tok.lower().strip()
    if t in POSITION_SYNONYMS:
        return POSITION_SYNONYMS[t]
    # if looks like a position code (2–3 letters) itself (cb, rw, cdm, …)
    if 1 <= len(t) <= 3 and t.isalpha():
        return [t]
    return None


def _expand_nationality_token(tok: str) -> Optional[List[str]]:
    t = tok.lower().strip()
    if t in NATIONALITY_ADJECTIVES:
        return NATIONALITY_ADJECTIVES[t]

    # generic adjective → country mapping for common ones
    mapping = {
        "english": "england",
        "german": "germany",
        "french": "france",
        "spanish": "spain",
        "italian": "italy",
        "portuguese": "portugal",
        "brazilian": "brazil",
        "argentinian": "argentina",
        "danish": "denmark",
    }
    if t in mapping:
        return [mapping[t]]

    return None


def _parse_age_constraint(q: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Return (min_age, max_age) based on 'young', 'under 23', 'u23', 'over 30', etc.
    """
    q_low = q.lower()
    min_age = None
    max_age = None

    # 'young' heuristic: treat as <= 23
    if re.search(r"\byoung\b", q_low):
        max_age = 23

    # under / below / u23
    m = re.search(r"\bunder\s+(\d{1,2})\b", q_low)
    if not m:
        m = re.search(r"\bbelow\s+(\d{1,2})\b", q_low)
    if not m:
        m = re.search(r"\bu(\d{1,2})\b", q_low)  # u23, u21...
    if m:
        max_age = int(m.group(1))

    # over / above
    m2 = re.search(r"\bover\s+(\d{1,2})\b", q_low)
    if not m2:
        m2 = re.search(r"\babove\s+(\d{1,2})\b", q_low)
    if m2:
        min_age = int(m2.group(1))

    return min_age, max_age


def _parse_overall_constraint(q: str) -> Optional[int]:
    """
    Understand 'overall 85', 'overall 85+', 'ovr 82+' etc.
    Returns minimum overall rating.
    """
    q_low = q.lower()
    m = re.search(r"\b(overall|ovr)\s*(\d{1,3})\+?\b", q_low)
    if m:
        return int(m.group(2))
    return None


# ----------------------------
# Main search function
# ----------------------------

def df_search(df: pd.DataFrame, query: str, top_n: int = 50) -> pd.DataFrame:
    """
    Deterministic + fuzzy search over the dataframe.

    Hard filters:
        - nationality (if we can detect)
        - preferred foot (left/right)
        - overall >= N (if specified)
        - age bounds (young, under 23, over 30, ...)
    Soft filters:
        - positions, clubs, names, etc. via token search / fuzzy scoring.

    If strict filters return no rows, constraints are gradually relaxed and
    RapidFuzz similarity is used to rank candidates.
    """
    q = str(query or "").strip()
    if not q:
        return df.head(0)

    q_low = q.lower()

    # figure out which columns exist
    nat_col = "nationality_name" if "nationality_name" in df.columns else None
    pos_col = None
    for c in ["player_pos", "player_positions"]:
        if c in df.columns:
            pos_col = c
            break

    # hard constraints from full query text
    min_age, max_age = _parse_age_constraint(q_low)
    min_ovr = _parse_overall_constraint(q_low)

    # foot constraint
    foot_constraint = None
    if re.search(r"\bleft\s+foot(ed)?\b", q_low) or re.search(r"\bleft\b", q_low):
        foot_constraint = "left"
    elif re.search(r"\bright\s+foot(ed)?\b", q_low) or re.search(r"\bright\b", q_low):
        foot_constraint = "right"

    # tokenization
    tokens = [t for t in re.split(r"\s+|[,;]", q_low) if t]

    # tokens we already used structurally and should NOT match literally
    DROP_TOKENS = {
        "left", "right", "foot", "footed", "overall", "ovr",
        "young", "under", "over", "below", "above",
    }

    cleaned_tokens: List[str] = []
    for t in tokens:
        if t in DROP_TOKENS:
            continue
        if re.fullmatch(r"\d{1,3}", t):      # pure number, usually for age/overall
            continue
        if re.fullmatch(r"\d{1,3}\+?", t):   # 85+
            continue
        if re.fullmatch(r"u\d{1,2}", t):     # u23, u21...
            continue
        cleaned_tokens.append(t)

    # ------------------------
    # 1. build hard mask
    # ------------------------
    mask = pd.Series(True, index=df.index)

    # nationality: from tokens only (we keep original tokens for this)
    if nat_col:
        nat_mask_total = pd.Series(True, index=df.index)
        any_nat = False
        for t in tokens:
            nat_list = _expand_nationality_token(t)
            if not nat_list:
                continue
            any_nat = True
            nat_mask = pd.Series(False, index=df.index)
            for country in nat_list:
                nat_mask |= df[nat_col].astype(str).str.lower().str.contains(re.escape(country), na=False)
            nat_mask_total &= nat_mask
        if any_nat:
            mask &= nat_mask_total

    # preferred foot
    if foot_constraint and "preferred_foot" in df.columns:
        foot_mask = df["preferred_foot"].astype(str).str.lower().str.contains(foot_constraint, na=False)
        mask &= foot_mask

    # overall
    if min_ovr is not None and "overall" in df.columns:
        mask &= (pd.to_numeric(df["overall"], errors="coerce") >= min_ovr)

    # age bounds
    if (min_age is not None or max_age is not None) and "age" in df.columns:
        age_series = pd.to_numeric(df["age"], errors="coerce")
        if min_age is not None:
            mask &= (age_series >= min_age)
        if max_age is not None:
            mask &= (age_series <= max_age)

    # ------------------------
    # 2. apply soft token filters (positions / generic text)
    # ------------------------
    if cleaned_tokens:
        for tok in cleaned_tokens:
            tok = tok.strip()
            if not tok:
                continue

            # positions
            pos_list = _expand_position_token(tok)
            pos_mask = pd.Series(False, index=df.index)
            if pos_list and pos_col:
                pos_vals = df[pos_col].astype(str).str.lower()
                for code in pos_list:
                    pos_mask |= pos_vals.str.contains(r"\b" + re.escape(code.lower()) + r"\b", na=False)

            # generic text across a few columns
            col_mask = pd.Series(False, index=df.index)
            for c in ["long_name", "short_name", pos_col, "club_name", "league_name"]:
                if c and c in df.columns:
                    col_mask |= df[c].astype(str).str.lower().str.contains(re.escape(tok), na=False)

            token_mask = pos_mask | col_mask
            mask &= token_mask

    # ------------------------
    # 3. collect strict results
    # ------------------------
    strict_results = df[mask].copy()

    # sort strict results by overall & potential if available
    sort_cols = [c for c in ["overall", "potential"] if c in strict_results.columns]
    if sort_cols and not strict_results.empty:
        strict_results = strict_results.sort_values(by=sort_cols, ascending=False)

    if not strict_results.empty:
        return strict_results.head(top_n)

    # ----------------------------------------------------
    # 4. If nothing matched strictly, RELAX & use fuzzy
    # ----------------------------------------------------
    # Relax age and overall, but keep nationality + foot if specified.
    relax_mask = pd.Series(True, index=df.index)

    # keep nationality
    if nat_col:
        nat_mask_total = pd.Series(True, index=df.index)
        any_nat = False
        for t in tokens:
            nat_list = _expand_nationality_token(t)
            if not nat_list:
                continue
            any_nat = True
            nat_mask = pd.Series(False, index=df.index)
            for country in nat_list:
                nat_mask |= df[nat_col].astype(str).str.lower().str.contains(re.escape(country), na=False)
            nat_mask_total &= nat_mask
        if any_nat:
            relax_mask &= nat_mask_total

    # keep preferred foot
    if foot_constraint and "preferred_foot" in df.columns:
        relax_mask &= df["preferred_foot"].astype(str).str.lower().str.contains(foot_constraint, na=False)

    # candidate pool after relaxed hard constraints
    pool = df[relax_mask].copy()
    if pool.empty:
        # as a last resort, search full df
        pool = df.copy()

    # fuzzy score on a combined text field
    q_for_fuzzy = q_low
    scores = []
    for idx, row in pool.iterrows():
        parts = []
        for c in ["long_name", "short_name", pos_col, "club_name", "league_name", nat_col]:
            if c and c in pool.columns:
                parts.append(str(row[c]))
        candidate_text = " ".join(parts).lower()
        score = fuzz.token_set_ratio(q_for_fuzzy, candidate_text)
        scores.append((idx, score))

    if not scores:
        return df.head(0)

    # sort by score desc and keep rows with reasonable similarity
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    best_indices = [idx for idx, s in scores_sorted[: max(top_n, 50)] if s >= 30]

    fuzzy_results = pool.loc[best_indices].copy()

    # final sort by score then overall
    fuzzy_results["__score"] = [dict(scores_sorted)[i] for i in fuzzy_results.index]
    sort_by = ["__score"]
    if "overall" in fuzzy_results.columns:
        sort_by.append("overall")
    fuzzy_results = fuzzy_results.sort_values(by=sort_by, ascending=False)
    return fuzzy_results.head(top_n).drop(columns=["__score"], errors="ignore")

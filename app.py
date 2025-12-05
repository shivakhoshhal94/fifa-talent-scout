# app.py — Streamlit UI with search box + deterministic dataframe search + optional RAG
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
import pandas as pd
import numpy as np

# Use the compatibility loader you have in the repo
from data_loader import load_data  # returns (df_preview, docs_preview)

# Optional chunked indexer & retriever (only used if you built index)
try:
    from indexer_chunked import load_index_and_meta, search_index_faiss
    INDEXER_AVAILABLE = True
except Exception:
    INDEXER_AVAILABLE = False

# Optional retriever/LLM helper if index + openai available
try:
    from retriever import build_qa_chain  # this expects index+metadata or works with indexer style
    RETRIEVER_AVAILABLE = True
except Exception:
    RETRIEVER_AVAILABLE = False

st.set_page_config(page_title="⚽ FIFA 23 Talent Scout — Search UI", layout="wide")
st.title("⚽ FIFA 23 Talent Scout — Search & RAG")

# Load preview (fast)
try:
    df_preview, docs_preview = load_data("data/male_players.csv")
except Exception as e:
    st.error(f"Could not load preview: {e}")
    st.stop()

# Left column: Index controls and Search box
col1, col2 = st.columns([1, 3])

with col1:
    st.sidebar.header("Index Controls")
    if INDEXER_AVAILABLE:
        st.sidebar.success("Chunked indexer available")
        if st.sidebar.button("Load index & metadata"):
            try:
                index, metadata = load_index_and_meta()
                st.sidebar.success("Index loaded")
            except Exception as e:
                st.sidebar.error(f"Failed to load index: {e}")
                index, metadata = None, None
    else:
        st.sidebar.info("No chunked indexer module found")

    st.markdown("---")
    st.sidebar.header("Quick actions")
    build_btn = st.sidebar.button("Build FAISS Index (chunked)")

with col2:
    st.subheader("Data preview")
    st.dataframe(df_preview.head(20), use_container_width=True)

# Search UI (big centered input)
st.markdown("## 🔎 Search players or ask (name, club, position, or natural language)")
user_query = st.text_input("Type a player name or description (e.g., 'young left-footed Dutch defenders')", value="")

# Which columns to search (use canonical columns that exist in preview)
SEARCH_COLS = [c for c in ["long_name", "player_pos", "club_name", "league_name", "nationality_name"] if c in df_preview.columns]

import re
import pandas as pd

# add nationality adjective -> country name mapping (expand if needed)
NATIONALITY_ADJECTIVES = {
    "dutch": ["netherlands","holland","the netherlands"],
    "english": ["england","united kingdom","uk","great britain"],
    "scottish": ["scotland"],
    "french": ["france"],
    "german": ["germany"],
    # add more as needed
}

POSITION_SYNONYMS = {
    "forward": ["st","cf","rw","lw","lf","rf"],
    "striker": ["st"],
    "winger": ["rw","lw","rm","lm"],
    "midfielder": ["cm","cam","cdm","lcm","rcm","rm","lm"],
    "defender": ["cb","lcb","rcb","lb","rb","lwb","rwb"],
    "goalkeeper": ["gk"],
}

def _expand_token_pos(token):
    t = token.lower().strip()
    return POSITION_SYNONYMS.get(t, None)

def _expand_nationality(token):
    t = token.lower().strip()
    # direct adjective match
    if t in NATIONALITY_ADJECTIVES:
        return NATIONALITY_ADJECTIVES[t]
    # common conversions (e.g., dutch->netherlands) using simple heuristics
    if t.endswith("ish") or t.endswith("an") or t.endswith("ese"):
        # fallback: return token itself so substring match might succeed
        return [t]
    return None

def _parse_age_constraint(tokens):
    """
    Recognize phrases like 'young', 'under 23', 'u23', 'over 25', 'below 21'
    Returns tuple (min_age, max_age) where None = no bound.
    """
    min_age = None
    max_age = None
    tok_str = " ".join(tokens).lower()
    # 'young' heuristic -> under 23
    if re.search(r"\byoung\b", tok_str):
        max_age = 23
    # explicit patterns
    m = re.search(r"\bunder\s+(\d{1,2})\b", tok_str)
    if not m:
        m = re.search(r"\bu(\d{1,2})\b", tok_str)  # u23
    if m:
        max_age = int(m.group(1))
    m2 = re.search(r"\bover\s+(\d{1,2})\b", tok_str)
    if m2:
        min_age = int(m2.group(1))
    m3 = re.search(r"\b(\d{1,2})\s*-\s*yr\b", tok_str)
    # you can add more patterns as needed
    return min_age, max_age

def df_search(df: pd.DataFrame, query: str, search_cols=None, top_n=50):
    """
    Improved DF search:
      - tokenizes query,
      - interprets 'young', 'under N', 'left-footed',
      - maps nationality adjectives, and position synonyms,
      - returns rows that satisfy all token constraints (AND across tokens).
    """
    if search_cols is None:
        search_cols = [c for c in ["long_name", "player_pos", "club_name", "league_name", "nationality_name", "preferred_foot"] if c in df.columns]

    q = str(query or "").strip().lower()
    if not q:
        return df.head(0)

    # normalize hyphens etc.
    q = q.replace("-", " ").replace("_", " ")
    tokens = [t for t in re.split(r"\s+|[,;]", q) if t]

    # parse age constraints early
    min_age, max_age = _parse_age_constraint(tokens)

    mask = pd.Series(True, index=df.index)

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue

        # handle preferred foot phrases
        foot_mask = pd.Series(False, index=df.index)
        if "left" in tok:
            if "preferred_foot" in df.columns:
                foot_mask = df["preferred_foot"].astype(str).str.lower().str.contains("left", na=False)
        elif "right" in tok:
            if "preferred_foot" in df.columns:
                foot_mask = df["preferred_foot"].astype(str).str.lower().str.contains("right", na=False)

        # nationality expansion
        nat_list = _expand_nationality(tok)
        nat_mask = pd.Series(False, index=df.index)
        if nat_list and "nationality_name" in df.columns:
            for n in nat_list:
                nat_mask = nat_mask | df["nationality_name"].astype(str).str.lower().str.contains(re.escape(n), na=False)

        # position expansion
        pos_list = _expand_token_pos(tok)
        pos_mask = pd.Series(False, index=df.index)
        if pos_list and "player_pos" in df.columns:
            for code in pos_list:
                # many player_pos fields are like "CB, LB" or "CM, CAM"
                pos_mask = pos_mask | df["player_pos"].astype(str).str.lower().str.contains(r"\b" + re.escape(code) + r"\b", na=False)

        # generic column matching (search tokens in provided columns)
        col_mask = pd.Series(False, index=df.index)
        for col in search_cols:
            col_mask = col_mask | df[col].astype(str).str.lower().str.contains(re.escape(tok), na=False)

        # token is satisfied if any of these masks match
        token_mask = col_mask | pos_mask | nat_mask | foot_mask

        # Special-case numeric tokens: check if token is an age number -> match age
        if re.fullmatch(r"\d{1,2}", tok) and "age" in df.columns:
            token_mask = token_mask | (df["age"].astype(str).str.contains(r"\b" + re.escape(tok) + r"\b", na=False))

        mask = mask & token_mask

    # Apply age constraints if present
    if min_age is not None or max_age is not None:
        age_mask = pd.Series(True, index=df.index)
        if "age" in df.columns:
            if min_age is not None:
                age_mask = age_mask & (pd.to_numeric(df["age"], errors="coerce") >= min_age)
            if max_age is not None:
                age_mask = age_mask & (pd.to_numeric(df["age"], errors="coerce") <= max_age)
            mask = mask & age_mask

    results = df[mask].copy()

    # sort by overall/potential if available
    sort_cols = [c for c in ["overall", "potential"] if c in results.columns]
    if len(sort_cols) and not results.empty:
        results = results.sort_values(by=sort_cols, ascending=False)

    return results.head(top_n)


# Button: Run deterministic DF search
if st.button("Run Data Search"):
    if not user_query:
        st.info("Type something in the search box and press 'Run Data Search'.")
    else:
        df_results = df_search(df_preview, user_query)
        if df_results.empty:
            st.warning("No players matched the query in preview. Try broader terms or 'Build Index' for RAG search.")
        else:
            st.success(f"Found {len(df_results)} matching rows (preview)")
            # Show results and allow selecting one row for details
            st.dataframe(df_results.reset_index(drop=True).head(50), use_container_width=True)

            # Selection box for detailed view
            sel_idx = st.selectbox("Select a result to view details (index in result table)", options=list(df_results.index), format_func=lambda x: f"{x} — {df_results.loc[x,'long_name']}" if 'long_name' in df_results.columns else str(x))
            if sel_idx is not None and sel_idx in df_results.index:
                row = df_results.loc[sel_idx]
                st.markdown("### Player detail")
                st.write(row.to_dict())

# If index + retriever available, run RAG when user presses this button
if INDEXER_AVAILABLE and RETRIEVER_AVAILABLE:
    st.markdown("---")
    st.markdown("### 🧠 RAG / LLM Search (requires index & model)")
    colA, colB = st.columns([2,1])
    with colA:
        long_question = st.text_area("Ask a scouting question to the RAG system (will use vector retrieval + LLM)", height=140)
    with colB:
        run_rag = st.button("Run RAG Query")

    if run_rag:
        if not long_question.strip():
            st.info("Write a question for the RAG system.")
        else:
            # Load index if not already loaded (attempt)
            try:
                try:
                    index, metadata
                except NameError:
                    index, metadata = load_index_and_meta()
                # build a simple QA chain using the retriever module
                qa = build_qa_chain(index, metadata)
                with st.spinner("Retrieving and running LLM..."):
                    answer = qa(long_question)
                st.markdown("#### RAG Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"RAG query failed: {e}")

# Show hint if no retriever available
if not (INDEXER_AVAILABLE and RETRIEVER_AVAILABLE):
    st.info("If you want LLM answers, add the chunked indexer and retriever modules, build the index and then use the RAG panel above.")

# Optional: let user type an exact name quick-search box
st.markdown("---")
st.subheader("Quick find by exact name")
name_query = st.text_input("Exact (or partial) name to jump to", key="name_query")
if name_query:
    close = df_preview[df_preview["long_name"].astype(str).str.contains(name_query, case=False, na=False)]
    if close.empty:
        st.warning("No exact/partial name match found in preview.")
    else:
        st.dataframe(close.head(10), use_container_width=True)

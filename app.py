# app.py — Streamlit UI (thin) + search_utils for logic
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

from data_loader import load_data  # your existing loader
from search_utils import df_search  # new search logic

# Optional chunked indexer & retriever (RAG)
try:
    from indexer_chunked import load_index_and_meta
    INDEXER_AVAILABLE = True
except Exception:
    INDEXER_AVAILABLE = False

try:
    from retriever import build_qa_chain
    RETRIEVER_AVAILABLE = True
except Exception:
    RETRIEVER_AVAILABLE = False

st.set_page_config(page_title="FIFA 23 Talent Scout — Search UI", layout="wide")
st.title("FIFA 23 Talent Scout — Search & RAG")

# ----- Load data preview -----
try:
    df_preview, docs_preview = load_data("data/male_players.csv")
except Exception as e:
    st.error(f"Could not load preview: {e}")
    st.stop()

st.subheader("Data preview")
st.dataframe(df_preview.head(20), use_container_width=True)

# ----- Search UI -----
st.markdown("## 🔎 Search players")
default_example = "spanish right defender left foot young overall 85+"
user_query = st.text_input(
    "Type a description (e.g., 'spanish right defender left foot young overall 85+')",
    value=default_example,
)

if st.button("Run Data Search"):
    if not user_query.strip():
        st.info("Type something in the search box and press 'Run Data Search'.")
    else:
        df_results = df_search(df_preview, user_query, top_n=50)
        if df_results.empty:
            st.warning("No players matched the query in preview. Try changing nationality/age or make the description softer.")
        else:
            st.success(f"Found {len(df_results)} matching rows (preview)")
            st.dataframe(df_results.reset_index(drop=True), use_container_width=True)

            # Player detail selection
            if "long_name" in df_results.columns:
                options = list(df_results.index)
                sel_idx = st.selectbox(
                    "Select a result to view details (index in result table)",
                    options=options,
                    format_func=lambda x: f"{x} — {df_results.loc[x, 'long_name']}",
                )
            else:
                sel_idx = st.selectbox(
                    "Select a result to view details (index in result table)",
                    options=list(df_results.index),
                )

            if sel_idx is not None and sel_idx in df_results.index:
                st.markdown("### Player detail")
                st.write(df_results.loc[sel_idx].to_dict())

# ----- Optional RAG / LLM panel -----
if INDEXER_AVAILABLE and RETRIEVER_AVAILABLE:
    st.markdown("---")
    st.markdown("### RAG / LLM Search (requires index & model)")
    colA, colB = st.columns([2, 1])
    with colA:
        long_question = st.text_area("", height=140)
    with colB:
        run_rag = st.button("Run RAG Query")

    if run_rag:
        if not long_question.strip():
            st.info("Write a question for the RAG system.")
        else:
            try:
                try:
                    index, metadata
                except NameError:
                    index, metadata = load_index_and_meta()
                qa = build_qa_chain(index, metadata)
                with st.spinner("Retrieving and running LLM..."):
                    answer = qa(long_question)
                st.markdown("#### RAG Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"RAG query failed: {e}")
else:
    st.info("If you want LLM answers, add the chunked indexer and retriever modules, build the index and then use the RAG panel.")

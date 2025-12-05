# ⚽ FIFA Talent Scout — RAG Document Q&A

A Retrieval-Augmented Generation (RAG) system for exploring and querying FIFA player, coach, and team data (FIFA 15–23). Use natural-language search, keyword filters, and a FAISS-backed semantic index to discover players and build scouting-style summaries.

## 🎯 Features

- Hybrid search: deterministic DataFrame filters + semantic search via FAISS
- RAG answers powered by OpenAI (optional) using retrieved player context
- Dataset support for male & female players, coaches, and teams across FIFA versions
- Streamlit UI for interactive exploration and Q&A
- Intelligent data loader with canonical column mapping

## 📁 Project structure

```
fifa-talent-scout/
├── app.py                # Streamlit UI
├── data/                 # Data (NOT included in repo - see data/README.md)
├── data_loader.py        # CSV loading & normalization
├── indexer.py            # FAISS index builder (basic)
├── indexer_chunked.py    # Chunked index builder + caching
├── retriever.py          # RAG chain and LLM integration
├── faiss_index/          # (ignored) prebuilt FAISS indices
├── requirements.txt
├── SETUP_GUIDE.md
├── CONTRIBUTING.md
├── README.md             # This file
└── .gitignore
```

## 🚀 Quick start

Prerequisites:
- Python 3.9+
- An OpenAI API key (optional — required only for embeddings/LLM features)

Steps:

1. Clone the repository:

```powershell
git clone https://github.com/shivakhoshhal94/fifa-talent-scout.git
cd fifa-talent-scout
```

2. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Download the dataset (NOT included in this repo):

- Dataset: "FIFA 23 Complete Player Dataset" on Kaggle
- Link: https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset?resource=download
- Extract CSV files into the `data/` directory (see `data/README.md` for details)

5. Copy environment template and add your OpenAI key:

```powershell
copy .env.example .env
# edit .env and set OPENAI_API_KEY
```

6. (Optional) Build a FAISS index for semantic search:

```powershell
python indexer_chunked.py
```

7. Run the Streamlit app:

```powershell
streamlit run app.py
```

Open the UI at http://localhost:8501

## 📚 Dataset information & attribution

Source dataset: "FIFA 23 Complete Player Dataset" by Stefano Leone on Kaggle.
- Original scraping source: sofifa.com (see dataset page for details and terms).

Dataset highlights:
- Coverage: FIFA 15–23 (multiple updates between 2015–2023)
- Players: 110+ attributes (ratings, skills, physicals, personal info)
- Coaches: 8 attributes
- Teams: 54 attributes

Important: The raw CSV files are large and are intentionally excluded from this Git repository. See `data/README.md` for download instructions.

## 🔧 How it works (brief)

1. `data_loader.py` normalizes CSV column names to a canonical set and produces preview rows and documents.
2. `indexer_chunked.py` builds embeddings (OpenAI) in chunks, caches vectors, and writes a FAISS index.
3. `retriever.py` embeds user queries, retrieves top-k matches from the FAISS index, and formats a prompt for the LLM.
4. `app.py` provides a Streamlit UI with deterministic DataFrame search and optional RAG/LLM search.

## ⚙️ Configuration

- `OPENAI_API_KEY` — required for embeddings/LLM. Place in a local `.env` file (not committed).
- Index files and caches are ignored by `.gitignore` (faiss_index/, *.pkl, embeddings cache, etc.).

## 🧰 Development notes

- The project uses pandas for CSV handling and FAISS for vector search.
- To customize prompts, edit `retriever.py` and update `PROMPT_TEMPLATE`.

## 🛡️ Data privacy & licensing

- Do not commit dataset CSVs or API keys. `.gitignore` protects these by default.
- This project is for educational and research purposes. FIFA, EA Sports and related trademarks are owned by their respective holders.

## 🤝 Contributing

See `CONTRIBUTING.md` for guidelines on issues, pull requests, and coding style.

## ✅ Final notes

If you want, I can also:
- run a quick syntax check across the Python files
- restore any additional documentation you prefer
- open a small test script that validates `load_data` with a local sample CSV

---

Built for scouting — enjoy exploring FIFA data! ⚽


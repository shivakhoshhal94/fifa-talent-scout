# Setup Guide - fifa-talent-scout

## 📥 Getting Started

This guide walks you through setting up the project locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/shivakhoshhal94/fifa-talent-scout.git
cd fifa-talent-scout
```

### Step 2: Create Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

The project requires FIFA player data. Download from Kaggle:

1. Go to [FIFA 23 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset?resource=download)
2. Click "Download" (requires Kaggle account)
3. Extract the CSV files into the `data/` folder

**Expected structure:**
```
data/
├── male_players.csv
├── female_players.csv
├── male_coaches.csv
├── female_coaches.csv
├── male_teams.csv
└── female_teams.csv
```

### Step 5: Configure Environment Variables

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

Get your key from: https://platform.openai.com/api-keys

### Step 6: (Optional) Build FAISS Index

If you want to use semantic search, build the vector index:

```bash
python indexer_chunked.py
```

This creates embeddings and saves them to `faiss_index/index.faiss`.

### Step 7: Run the Application

```bash
streamlit run app.py
```

The app will open at: `http://localhost:8501`

## ✅ Verify Installation

Try a simple search:
1. Open the app
2. Type a player name (e.g., "Haaland")
3. You should see results instantly

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `OPENAI_API_KEY not found` | Check `.env` file exists and has your key |
| `FileNotFoundError: data/` | Download datasets from Kaggle |
| Slow search | Run `python indexer_chunked.py` to build FAISS index |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |

## 📚 Next Steps

- Read [README.md](README.md) for full documentation
- Explore the codebase: `app.py`, `indexer_chunked.py`, `retriever.py`
- Customize prompts in `retriever.py`
- Add new data sources

## 🤝 Need Help?

Open an issue on GitHub or check the troubleshooting section in README.md!

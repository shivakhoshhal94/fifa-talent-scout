# Data Directory

## 📥 Download Datasets

The FIFA player, coach, and team datasets are **NOT included** in this repository due to their large file size (1GB+).

### Getting the Data

1. **Visit Kaggle**: [FIFA 23 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset?resource=download)

2. **Download** the dataset (requires free Kaggle account)

3. **Extract CSV files** into this folder:
   - `male_players.csv`
   - `female_players.csv`
   - `male_coaches.csv`
   - `female_coaches.csv`
   - `male_teams.csv`
   - `female_teams.csv`

### Expected Structure

```
data/
├── male_players.csv          (~1.2 GB)
├── female_players.csv         (~90 MB)
├── male_coaches.csv
├── female_coaches.csv
├── male_teams.csv
├── female_teams.csv
└── README.md                 (this file)
```

### Dataset Info

- **Source**: [sofifa.com](https://sofifa.com/)
- **Coverage**: FIFA 15-23, all updates
- **Players**: 110+ attributes each
- **Coaches**: 8 attributes each
- **Teams**: 54 attributes each

### Why Not in Git?

Large files are excluded from git to:
- ✅ Keep repository size small (<1MB)
- ✅ Avoid GitHub's 100MB file size limit
- ✅ Allow faster clones and pulls
- ✅ Protect bandwidth

### Using the Data

Once downloaded, the app will automatically load and process the CSVs:

```bash
streamlit run app.py
```

The app will:
1. Load CSVs from `data/` folder
2. Normalize column names
3. Create preview DataFrames
4. Build FAISS indices for search

---

**Questions?** See `SETUP_GUIDE.md` for detailed setup instructions.

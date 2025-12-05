# Contributing to fifa-talent-scout

Thanks for your interest in contributing! Here's how you can help.

## 🐛 Report Bugs

Found a bug? Open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (OS, Python version, etc.)

## 💡 Suggest Features

Have an idea? Open an issue with:
- Feature description
- Use case / why it's useful
- Possible implementation approach

## 🔧 Contributing Code

### 1. Fork the Repository
```bash
git clone https://github.com/YOUR_USERNAME/fifa-talent-scout.git
cd fifa-talent-scout
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes
- Keep changes focused on one feature
- Follow existing code style
- Add comments for complex logic
- Update README if needed

### 4. Test Your Changes
```bash
# Run the app to test UI
streamlit run app.py

# Verify no syntax errors
python -m py_compile *.py
```

### 5. Commit and Push
```bash
git add .
git commit -m "Add: Brief description of changes"
git push origin feature/your-feature-name
```

### 6. Open a Pull Request
- Describe what you changed and why
- Link related issues
- Wait for review

## 📋 Code Style Guidelines

- Use **PEP 8** style guide
- Add docstrings to functions
- Use type hints where possible
- Keep functions focused and small
- Use meaningful variable names

### Example:
```python
def search_index(query: str, k: int = 5) -> List[Tuple[str, float]]:
    """
    Search FAISS index for top-k similar documents.
    
    Args:
        query: Search query string
        k: Number of results to return
        
    Returns:
        List of (document, similarity_score) tuples
    """
    vec = embed_query(query)
    results = search_index_faiss(index, metadata, vec, k=k)
    return results
```

## 🚀 Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows

# Install dependencies
pip install -r requirements.txt

# Install dev tools (optional)
pip install black flake8 pytest
```

## 📝 Commit Message Format

```
Type: Brief description (50 chars max)

Longer explanation if needed. Explain what and why,
not how. Reference issues: Fixes #123
```

Types:
- `Add:` New feature
- `Fix:` Bug fix
- `Refactor:` Code cleanup (no logic change)
- `Docs:` Documentation updates
- `Test:` Add/update tests

## 🔒 Security

- Never commit API keys or secrets
- Check `.gitignore` before pushing
- Don't include large data files
- Report security issues privately

## ❓ Questions?

Open an issue with label `question` or start a discussion!

---

Thank you for contributing! 🎉

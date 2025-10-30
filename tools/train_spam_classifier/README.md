# tools/train_spam_classifier

Baseline training helper for spam/SMS classification.

What it does
- Downloads an example SMS spam dataset (public CSV) and trains a TF-IDF + LogisticRegression baseline.
- Prints evaluation metrics and saves model and vectorizer to `tools/train_spam_classifier/models/`.

Quick start (Windows PowerShell)

```powershell
# create a venv (optional)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# install deps
pip install -r tools/train_spam_classifier/requirements.txt

# run training
python tools/train_spam_classifier/train.py
```

Notes
- This script is a minimal reproducible baseline for offline evaluation. It is safe to run locally. Do not commit any raw payloads or PII into the repo.

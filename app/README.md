# Streamlit app for Spam Email Classification

This folder contains a simple Streamlit app to demo the baseline spam classifier.

Run locally (Windows PowerShell):

```powershell
# create and activate venv (optional)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# install deps
pip install -r app/requirements.txt

# run streamlit app
streamlit run app/streamlit_app.py
```

Notes
- The app will attempt to load model artifacts from one of these locations:
  - `tools/train_spam_classifier/models/`
  - `tools/models/`
  - `models/`

- If artifacts are not present, you can upload both the model and vectorizer via the sidebar.
- To create the baseline model, run `python tools/train_spam_classifier/train.py`.

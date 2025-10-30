import streamlit as st
from pathlib import Path
import joblib
import tempfile
import os


MODEL_CANDIDATES = [
    Path("tools/train_spam_classifier/models/baseline_logreg.joblib"),
    Path("tools/models/baseline_logreg.joblib"),
    Path("models/baseline_logreg.joblib"),
]
VECT_CANDIDATES = [
    Path("tools/train_spam_classifier/models/tfidf_vectorizer.joblib"),
    Path("tools/models/tfidf_vectorizer.joblib"),
    Path("models/tfidf_vectorizer.joblib"),
]


def load_joblib_from_uploaded(uploaded_file):
    # Save uploaded file to a temp file and load with joblib
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        obj = joblib.load(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return obj


def try_load_candidates(candidates):
    for p in candidates:
        if p.exists():
            try:
                return joblib.load(p), p
            except Exception as e:
                st.warning(f"Found file {p} but failed to load: {e}")
    return None, None


def main():
    st.set_page_config(page_title="Spam Email Classifier", layout="centered")
    st.title("Spam Email Classification Demo")

    st.markdown(
        "This demo loads a baseline TF-IDF + LogisticRegression model and classifies input text as `spam` or `not spam`."
    )

    st.sidebar.header("Settings & model")
    threshold = st.sidebar.slider("Spam threshold (probability)", 0.0, 1.0, 0.5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Upload model and vectorizer (optional)")
    uploaded_model = st.sidebar.file_uploader("Upload model (.joblib)", type=["joblib"], key="model")
    uploaded_vect = st.sidebar.file_uploader("Upload vectorizer (.joblib)", type=["joblib"], key="vect")

    clf = None
    clf_path = None
    vect = None
    vect_path = None

    if uploaded_model and uploaded_vect:
        try:
            clf = load_joblib_from_uploaded(uploaded_model)
            vect = load_joblib_from_uploaded(uploaded_vect)
            st.sidebar.success("Loaded model and vectorizer from upload")
        except Exception as e:
            st.sidebar.error(f"Failed to load uploaded files: {e}")

    if clf is None:
        clf, clf_path = try_load_candidates(MODEL_CANDIDATES)
        if clf is not None:
            st.sidebar.success(f"Loaded model from {clf_path}")

    if vect is None:
        vect, vect_path = try_load_candidates(VECT_CANDIDATES)
        if vect is not None:
            st.sidebar.success(f"Loaded vectorizer from {vect_path}")

    if clf is None or vect is None:
        st.warning(
            "Model or vectorizer not found. Upload both files in the sidebar or place them under `tools/train_spam_classifier/models/` or `tools/models/` and refresh."
        )

    st.header("Classify text")
    text = st.text_area("Paste email or SMS text here", height=200)
    if st.button("Predict"):
        if not text or text.strip() == "":
            st.info("Please enter some text to classify.")
        elif clf is None or vect is None:
            st.error("Model or vectorizer not available. Upload or place the artifacts and retry.")
        else:
            X = vect.transform([text])
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X)[0][1]
            else:
                # Some classifiers don't implement predict_proba; fall back to decision_function
                try:
                    score = clf.decision_function(X)[0]
                    # map score to 0..1 via logistic
                    import math

                    proba = 1 / (1 + math.exp(-score))
                except Exception:
                    proba = None

            label = "Spam" if (proba is not None and proba >= threshold) else "Not spam"
            st.subheader("Result")
            st.write(f"**Prediction:** {label}")
            if proba is not None:
                st.write(f"**Spam probability:** {proba:.4f}")

    st.markdown("---")
    st.markdown(
        "If you don't have the model artifacts locally, run the baseline trainer in `tools/train_spam_classifier/train.py` to create them."
    )


if __name__ == "__main__":
    main()

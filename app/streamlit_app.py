import os
import json
import time
from collections import Counter
from typing import List, Tuple
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    f1_score,
)

# Text normalization for inference
import re
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\-\s]{7,}\d)\b")

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


def normalize_text(text: str, keep_numbers: bool = False) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.lower()
    t = URL_RE.sub("<URL>", t)
    t = EMAIL_RE.sub("<EMAIL>", t)
    t = PHONE_RE.sub("<PHONE>", t)
    if not keep_numbers:
        t = re.sub(r"\d+", "<NUM>", t)
    t = re.sub(r"[^\w\s<>]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def try_load_candidates(candidates):
    for p in candidates:
        if p.exists():
            try:
                return joblib.load(p), p
            except Exception as e:
                st.warning(f"Found file {p} but failed to load: {e}")
    return None, None

def token_topn(series: pd.Series, topn: int) -> List[Tuple[str, int]]:
    counter: Counter = Counter()
    for s in series.astype(str):
        counter.update(s.split())
    return counter.most_common(topn)

def main():
    st.set_page_config(page_title="Spam Email Classifier", layout="wide")
    st.title("Spam/Ham Email Classification Demo")
    st.caption("Interactive dashboard for testing the spam classifier with visualizations")

    # Sidebar settings
    with st.sidebar:
        st.header("Settings & Model")
        threshold = st.slider("Decision threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.01)

    # Load model and vectorizer
    clf, clf_path = try_load_candidates(MODEL_CANDIDATES)
    vect, vect_path = try_load_candidates(VECT_CANDIDATES)

    if clf is None or vect is None:
        st.warning(
            "Model or vectorizer not found. Please run the baseline trainer first:\n"
            "```\npython tools/train_spam_classifier/train.py\n```"
        )
        st.stop()

    st.sidebar.success(f"Loaded model from {clf_path}")
    st.sidebar.success(f"Loaded vectorizer from {vect_path}")

    # Live Inference section
    st.header("Test the classifier")

    # Example messages
    ex_spam = "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."
    ex_ham = "Ok lar... Joking wif u oni..."
    
    c_ex1, c_ex2 = st.columns(2)
    with c_ex1:
        if st.button("Try spam example"):
            st.session_state["input_text"] = ex_spam
    with c_ex2:
        if st.button("Try ham example"):
            st.session_state["input_text"] = ex_ham

    # Text input with session state
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
    
    user_text = st.text_area(
        "Paste email/SMS text to classify",
        key="input_text",
        height=100
    )

    if st.button("Classify"):
        if not user_text.strip():
            st.info("Please enter some text to classify.")
        else:
            # Normalize and classify
            cleaned = normalize_text(user_text)
            with st.expander("Show normalized text", expanded=False):
                st.code(cleaned)
            
            X = vect.transform([cleaned])
            prob = float(clf.predict_proba(X)[:, 1][0])
            pred_label = "Spam" if prob >= threshold else "Not spam"

            # Show prediction with probability bar
            st.success(f"Prediction: {pred_label}  |  Spam probability = {prob:.4f}  (threshold = {threshold:.2f})")

            # Probability visualization
            fig_g, ax_g = plt.subplots(figsize=(8, 0.8))
            ax_g.barh([0], [prob], color="#d62728" if pred_label == "Spam" else "#1f77b4")
            ax_g.axvline(threshold, color="black", linestyle="--", linewidth=1)
            ax_g.set_xlim(0, 1)
            ax_g.set_yticks([])
            ax_g.set_xlabel("spam probability")
            ax_g.text(min(prob + 0.02, 0.98), 0, f"{prob:.2f}", va="center")
            st.pyplot(fig_g)

    # Token replacement statistics for the current input
    if user_text.strip():
        st.subheader("Text analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Token replacements
            repl = {
                "<URL>": len(URL_RE.findall(user_text)),
                "<EMAIL>": len(EMAIL_RE.findall(user_text)),
                "<PHONE>": len(PHONE_RE.findall(user_text)),
                "<NUM>": len(re.findall(r"\d+", user_text))
            }
            st.write("Token replacements found:")
            st.table(pd.DataFrame.from_dict(repl, orient="index", columns=["count"]))
        
        with col2:
            # Word frequency
            words = normalize_text(user_text).split()
            if words:
                word_freq = pd.Series(Counter(words)).sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(8, max(4, min(len(word_freq), 10) * 0.4)))
                word_freq.tail(10).plot(kind='barh', ax=ax)
                ax.set_title("Top 10 tokens in input")
                st.pyplot(fig)
            else:
                st.info("No tokens found after normalization")

    # Model Performance section
    st.header("Model Performance Visualization")
    st.caption("Evaluation metrics and curves from the test set")

    # Prepare test data
    df = pd.read_csv("https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv", 
                     names=["label", "text"])
    df["y"] = (df["label"].str.lower() == "spam").astype(int)
    X = df["text"].astype(str)
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_vec = vect.transform(X_test)
    y_pred_proba = clf.predict_proba(X_test_vec)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    col1, col2 = st.columns(2)

    with col1:
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (threshold={threshold:.2f})')
        st.pyplot(fig)

        # Metrics at current threshold
        st.write("Performance metrics:")
        metrics = {
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred)
        }
        st.table(pd.DataFrame([metrics]))

    with col2:
        # ROC & PR curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        axes[0].plot([0, 1], [0, 1], 'k--')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend()
        axes[0].grid(True)

        # PR curve
        prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1].plot(rec, prec)
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)

    # Threshold sweep
    st.subheader("Threshold sweep")
    thresholds = np.linspace(0.1, 0.9, 17)
    sweep_results = []
    
    for t in thresholds:
        y_pred_t = (y_pred_proba >= t).astype(int)
        sweep_results.append({
            "threshold": f"{t:.2f}",
            "precision": precision_score(y_test, y_pred_t),
            "recall": recall_score(y_test, y_pred_t),
            "f1": f1_score(y_test, y_pred_t)
        })
    
    sweep_df = pd.DataFrame(sweep_results).set_index("threshold").round(3)
    st.line_chart(sweep_df)
    with st.expander("Show threshold sweep table"):
        st.table(sweep_df)

if __name__ == "__main__":
    main()

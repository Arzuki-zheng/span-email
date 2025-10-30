#!/usr/bin/env python3
"""
Baseline training script for spam (SMS) classifier.

This script downloads the SMS spam dataset from the provided URL,
trains a TF-IDF + LogisticRegression baseline, prints evaluation
metrics, and saves the model and vectorizer to disk.

Usage (from repo root):
  python tools/train_spam_classifier/train.py

This script is intentionally minimal and safe for inclusion in the repo.
Do NOT commit any raw email/SMS payloads or PII.
"""
from pathlib import Path
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/"
    "Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/"
    "Chapter03/datasets/sms_spam_no_header.csv"
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parents[1] / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(url: str) -> pd.DataFrame:
    # dataset has no header; first column label, second column message
    df = pd.read_csv(url, header=None, encoding="utf-8", names=["label", "text"])
    return df


def prepare(df: pd.DataFrame):
    df = df.dropna()
    df = df[df["text"].str.len() > 0]
    df["y"] = (df["label"].str.lower() == "spam").astype(int)
    return df[["text", "y"]]


def train_and_save(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"]
    )

    vect = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_t = vect.fit_transform(X_train)
    X_test_t = vect.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_t, y_train)

    preds = clf.predict(X_test_t)

    print("\nClassification report:\n")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Save artifacts
    model_path = OUT_DIR / "baseline_logreg.joblib"
    vect_path = OUT_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(clf, model_path)
    joblib.dump(vect, vect_path)
    print(f"Saved model to {model_path}")
    print(f"Saved vectorizer to {vect_path}")


def main():
    print("Downloading dataset and training baseline model...")
    df = load_data(DATA_URL)
    df = prepare(df)
    print(f"Loaded dataset with {len(df)} rows. Positives: {df['y'].sum()}")
    train_and_save(df)


if __name__ == "__main__":
    main()

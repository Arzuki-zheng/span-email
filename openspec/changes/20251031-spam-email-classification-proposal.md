---
title: Spam Email Classification
date: 2025-10-31
status: draft
author: (auto-generated)
---

# Spam Email Classification — OpenSpec Change Proposal

簡短說明（Chinese summary）

目標是在系統中增加一個垃圾郵件（spam）辨識分類能力，用以自動標記或隔離疑似垃圾郵件，降低使用者收到的惡意或無關郵件量，同時保留可追溯、可調整的分類模型與評估流程。

Short summary

Add a spam email classification feature to the project. The feature will provide automated labeling of incoming emails as "spam" or "not spam", include model training and evaluation, an inference runtime for classification, and operational controls (thresholds, feedback loop for user corrections, and logging for audit/metrics).

Background

Our email system currently lacks an integrated machine-learning based spam classifier. Manual rules or third-party filters may be used but are inconsistent and hard to tune. A local, repo-backed OpenSpec proposal will define the feature, acceptance criteria, data and privacy considerations, and a rollout plan.

Goals

- Provide binary classification (spam / not-spam) for incoming emails with configurable thresholds.
- Integrate a model training pipeline and evaluation metrics (precision, recall, F1, ROC AUC).
- Enable logging and telemetry for classification decisions and model performance over time.
- Provide an easy feedback path for user corrections to be collected and used for periodic retraining.

Non-Goals

- Replace external, managed anti-spam services entirely (this is an in-repo feature to complement or optionally replace them).
- Build an enterprise-grade service for high-volume abuse detection in the first iteration.

Proposed design

1. Data
   - Source: historical inbound emails (headers, subject, body, sender metadata), previously labeled spam/not-spam where available (user-marked or external filters).
   - Storage: training datasets will be stored in a protected location (not committed to git). Metadata and training manifests live in `data/` or the project’s data pipeline.
   - Privacy: PII must be redacted or handled according to privacy policy. Provide field-level redaction and record-level consent checks.

2. Model
   - Candidate models: baseline logistic regression or Naive Bayes on TF-IDF features; later versions: lightweight transformer or pretrained embeddings + classifier.
   - Training infra: a small reproducible training script (e.g., Python, scikit-learn / PyTorch) that produces a versioned model artifact.
   - Model artifacts: stored with semantic versioning and a minimal manifest including training date, dataset hash, metrics, and hyperparameters.

3. Inference
   - Runtime: a simple service or library function to score incoming emails synchronously (low-latency) or asynchronously (batch scoring for historical processing).
   - Thresholding and policies: configurable per-environment threshold; actions: mark as spam, quarantine, or attach metadata for UI.

4. Feedback loop
   - Capture user corrections (mark as not spam / mark as spam) into a labeled dataset for periodic retraining.
   - Build tooling to sample and label edge cases.

5. Evaluation and acceptance criteria
   - Minimum baseline: precision >= 0.90 at recall >= 0.70 on a held-out test set (adjustable based on dataset characteristics).
   - ROC AUC, confusion matrix, per-class precision/recall reported in training artifacts.
   - No user-visible regressions: false positive rate for 'not spam' users must be below acceptable threshold (TBD during review).

6. Observability and metrics
   - Track throughput, latency, model version in inference logs.
   - Metrics: daily spam rate, false positive rate (from user corrections), precision/recall drift, model training metrics.

7. Security & privacy
   - Do not commit raw email payloads into git. Use secure storage with access controls.
   - Ensure logs redact sensitive content. Follow existing project privacy guidelines (see `openspec/project.md`).

8. Rollout plan
   - Phase 1: Internal evaluation on historical data; offline metrics and manual review.
   - Phase 2: Canary—enable classifier for a small subset of users, deliver only UI labels (no auto-move) and collect feedback.
   - Phase 3: Gradual rollout—enable quarantine/move for high-confidence predictions and expand as metrics stabilize.

9. Risks
   - Privacy exposure if data handling is not correctly implemented.
   - User frustration from false positives—mitigated by conservative thresholds and a clear undo/feedback path.
   - Model drift—requires continuous monitoring and periodic retraining.

10. Implementation estimates
    - Drafting & data collection: 1–2 weeks
    - Baseline model & evaluation: 1–2 weeks
    - Inference integration & metrics: 1–2 weeks
    - Canary roll-out and iteration: 2–4 weeks

11. Acceptance / review checklist

- [ ] Training dataset defined and stored securely
- [ ] Baseline model trained with evaluation metrics attached
- [ ] Inference path integrated and tested against test harness
- [ ] Telemetry/logging in place and verified
- [ ] Privacy review completed and approved

Open questions

- What is the expected incoming email volume and latency SLOs?
- Are there existing labeled datasets or user-marked spam logs we can leverage?
- Which environments (staging/prod) should run the inference service?

Next steps

1. Confirm scope and acceptance criteria with stakeholders.
2. Identify available training data and any privacy/retention constraints.
3. Create an implementation task breakdown and schedule the first sprint.

---

If you'd like, I can:

- Fill the proposal with more project-specific detail pulled from `openspec/project.md` (I can read it and merge details),
- Create the repo skeleton for training scripts (`tools/train_spam_classifier/`), or
- Open a PR branch and add a test dataset manifest (without raw PII) to demonstrate the pipeline.

Planned step breakdown (placeholders)

Below is the planned list of steps for implementing this feature. Per your request, I will implement Step 1 (baseline) now and leave subsequent steps blank for you to fill in later.

Step 1: Baseline
   - Build a basic spam classifier that trains on an easily-available dataset.
   - Data source: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
   - Deliverable: `tools/train_spam_classifier/train.py` that downloads the dataset, trains a baseline TF-IDF + LogisticRegression classifier, prints evaluation metrics, and saves model artifacts under `tools/train_spam_classifier/models/`.

Step 2: 
   - (placeholder) — leave blank for now

Step 3:
   - (placeholder) — leave blank for now

Step 4:
   - (placeholder) — leave blank for now

Step 5:
   - (placeholder) — leave blank for now


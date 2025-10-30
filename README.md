# span-email
物聯網數據分析與應用_HW3
Next steps - Copy these prompts to Codex:
────────────────────────────────────────────────────────────
# span-email

本專案目標：在專案中建立一個垃圾郵件 / SMS spam 辨識的功能範例，包含提案（OpenSpec）、基線訓練程式、以及一個簡單的 Streamlit 示範介面。

目前已完成（摘要）

- OpenSpec 變更提案：`openspec/changes/20251031-spam-email-classification-proposal.md`（草案）。提案包含背景、目標、設計、資料/隱私說明、評估標準、風險、以及實作步驟（Step 1 已實作，Step 2..5 保留占位）。
- Codex prompts：`openspec/codex_prompts.md`（三個可複製到 Codex 的 prompt）。
- Baseline 訓練程式：`tools/train_spam_classifier/train.py`。
   - 會自動下載公開資料集（參考下方資料來源）、訓練 TF-IDF + LogisticRegression baseline、印出評估報表並儲存模型與向量器。
- Baseline 相關檔案：
   - `tools/train_spam_classifier/requirements.txt`
   - `tools/train_spam_classifier/README.md`
- Streamlit Demo：`app/streamlit_app.py` 與 `app/requirements.txt`、`app/README.md`，提供簡單 UI，可上傳模型/向量器或載入本地 artifacts 進行單筆文字分類。

資料來源

- Baseline 使用的公開資料集（SMS spam）：
   https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

使用到的技術與技巧（快速列表）

- 資料處理：Pandas
- 特徵抽取：TF-IDF（scikit-learn 的 TfidfVectorizer）
- 模型：LogisticRegression（scikit-learn）作為 baseline
- 序列化：joblib 用來儲存/載入模型與向量器
- 部署示範：Streamlit（簡單互動式 demo）

如何執行（簡短教學，Windows PowerShell）

1) 建議先建立虛擬環境並安裝 baseline 依賴：

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r tools/train_spam_classifier/requirements.txt
```

2) 執行 baseline 訓練（會下載資料、訓練、並在本機儲存 artifacts）：

```powershell
python tools/train_spam_classifier/train.py
```

訓練結果會印出 classification report 與 confusion matrix，並儲存模型與向量器（script 預設儲存位置為 `tools/models/`；可改為 `tools/train_spam_classifier/models/`）。

3) 執行 Streamlit demo：

```powershell
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

在 Demo 中你可以：貼入一則 SMS 或 email 內容，設定 spam threshold，或在側邊欄上傳模型與向量器（`.joblib`）。若已執行第 2 步並將 artifacts 放在預期路徑，app 會自動載入它們。

專案檔案位置（重要）

- 提案：`openspec/changes/20251031-spam-email-classification-proposal.md`
- Codex prompts：`openspec/codex_prompts.md`
- Baseline 程式：`tools/train_spam_classifier/train.py`
- Baseline artifacts（模型）：`tools/models/`（或 `tools/train_spam_classifier/models/`，視腳本設定而定）
- Streamlit app：`app/streamlit_app.py`

建議與下一步（供參考）

- 將模型 artifacts 的儲存路徑統一為 `tools/train_spam_classifier/models/`，並把該資料夾加入 `.gitignore`（不要把二進位模型提交到 repo）。我可以幫你修改 `train.py` 並移動現有 artifact。
- 將 proposal 內容合併 `openspec/project.md` 中的專案細節（owner、規範、隱私要求），我可以自動整合這些資訊。
- 後續步驟建議（可逐項實作）：超參數調優、模型版控、CI 定期重訓、inference API、小量 canary rollout、使用者回饋收集等。

如果你要我幫忙做下一步，請選一項：

- 更新 `train.py` 的 artifacts 路徑並移動現有模型（我可以立刻執行）。
- 把 proposal 補成完整中文版本並合併 `openspec/project.md` 的細節。 
- 建立 PR 分支並提交 proposal + demo（請提供分支名稱或我使用 `feat/spam-classifier-proposal`）。

---

（以上內容由專案內腳本與檔案產生；若想看 proposal 的某段原文或詳細 training log，我可以把那段貼到這裡供你審閱。）

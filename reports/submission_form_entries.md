# ALGIONS AI LABS PRIVATE LIMITED
## AI & Machine Learning Internship Assessment – Official Submission Form Entries

*This document contains the exact entries to copy and paste directly into your official internship assessment submission portal.*

---

### Section 1: Candidate Identification
* **Candidate Full Name:** Vishesh Tomar
* **Registration Number:** *[Enter your University / College Registration Number]*
* **University Email:** *[Enter your official University / College Email Address]*
* **Internship Role:** AI & Machine Learning
* **Assigned Project Title:** E-Commerce Product Recommendation Engine
* **Project Identity:** NEXUS – Discover Beyond Popularity

---

### Section 2: Code & Repository Versioning
* **GitHub Repository URL:** `https://github.com/THAKURSAHAB9910/nexus-ecommerce-recommender`
* **Primary Git Branch:** `main`
* **Final Commit ID:** `6727cf8` *(or latest commit on `main`)*
* **Repository Access:** Public (open to assessment reviewers without credential barriers)

---

### Section 3: Demonstration & Video Evidence
* **Demonstration Mode:** Local interactive deployment with 1-click execution scripts and reproduction instructions.
* **Streamlit Dashboard URL (Local):** `http://localhost:8502` (or `http://localhost:8501`)
* **FastAPI Backend Swagger Docs (Local):** `http://127.0.0.1:8000/docs`
* **Live Video URL:** *[Paste your Loom / Google Drive / YouTube unlisted recording link here]*

---

### Section 4: Documentation & Architecture Links
* **Repository README:** [README.md](https://github.com/THAKURSAHAB9910/nexus-ecommerce-recommender/blob/main/README.md)
* **Comprehensive Technical Report:** [reports/technical_report.md](https://github.com/THAKURSAHAB9910/nexus-ecommerce-recommender/blob/main/reports/technical_report.md)
* **Feature Leakage & Safety Audit:** [reports/feature_leakage_audit.md](https://github.com/THAKURSAHAB9910/nexus-ecommerce-recommender/blob/main/reports/feature_leakage_audit.md)
* **Exploratory Data Analysis Report:** [reports/eda_summary.md](https://github.com/THAKURSAHAB9910/nexus-ecommerce-recommender/blob/main/reports/eda_summary.md)
* **Interactive Jupyter Notebook Walkthrough:** [notebooks/nexus_walkthrough.ipynb](https://github.com/THAKURSAHAB9910/nexus-ecommerce-recommender/blob/main/notebooks/nexus_walkthrough.ipynb)

---

### Section 5: Technology Stack & Limitations
* **Technology Stack:** Python 3.14, Scikit-Learn (1.8+), SciPy, Pandas, NumPy, FastAPI, Streamlit, Plotly, Pytest (23 tests passed), Joblib.
* **Setup & Reproduction Notes:** Verified on Windows 11 PowerShell. One-click launchers provided (`run_dashboard.bat`, `run_api.bat`). Full end-to-end automated reproduction pipeline (`python -m scripts.train`).
* **Known Assumptions & System Limitations:**
  1. *Implicit Feedback Assumption:* Purchase logs capture positive interactions ($q \ge 1$); unpurchased catalog items denote lack of awareness or exposure rather than explicit negative dislike.
  2. *Score Non-Calibration:* Output scores represent relative ranking affinities ($S \in [0, 1]$), not calibrated purchase probabilities.
  3. *Temporal Shift:* Holiday demand surges in Q4 alter basket composition compared to earlier quarter baselines.

---

### Section 6: AI Tools & Attribution
* **AI Tools Used:** Antigravity AI Assistant (Google DeepMind)
* **Purpose of Use:** Architectural consultation, boilerplate scaffolding, docstring generation, pair-programming review, and unit test generation.
* **Candidate Code Ownership & Attestation:** All recommendation algorithms (Popularity, TF-IDF Content-Based, TruncatedSVD Collaborative Filtering, Adaptive Hybrid Blend), leakage-free chronological splitting logic, evaluation metrics (Precision@K, Recall@K, NDCG@K, Coverage, Diversity, Novelty), Serendipity Mode reranking, and Reasoning Lab attribution have been thoroughly reviewed, understood, and can be live-coded, modified, and defended line-by-line during the technical interview.
* **Dataset Attribution:** UCI Machine Learning Repository: *Online Retail Benchmark* (Dr. Daqing Chen, London South Bank University). Publicly licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

---

### Section 7: Role-Specific Assessment Fields
* **Dataset Source & License:** UCI Machine Learning Repository (#352), CC BY 4.0 License. 541,909 raw records $\rightarrow$ 527,930 clean transactions across 3,913 products and 4,334 customers.
* **Leakage-Safe Split Strategy:** Strict chronological boundary: Train (80%, `2010-12-01` to `2011-11-02`), Validation (10%, `2011-11-02` to `2011-11-21`), Held-Out Test (10%, `2011-11-21` to `2011-12-09`). Invariant: $\max(T_{\text{train}}) < \min(T_{\text{test}})$ with a 19-day quarantine gap.
* **Model Artifact Locations:** Serialized in `artifacts/`: `popularity_model.joblib`, `content_model.joblib`, `collaborative_model.joblib`, `hybrid_model.joblib`. Fully reproducible via `python -m scripts.train`.
* **Empirical Ranking Metrics (Held-Out Test Set):**
  * **Adaptive Hybrid (Champion):** Precision@10: `0.0437` | Recall@10: `0.0246` | NDCG@10: `0.0499` | Catalog Coverage: `34.68%` | Diversity: `0.5751` | Novelty: `11.58 bits`.
  * **Popularity Baseline:** Precision@10: `0.0703` | Catalog Coverage: `0.26%` | Novelty: `8.14 bits`.
  * **Cold-Start Fallback (170 Unseen Users):** Precision@10: `0.0488` | NDCG@10: `0.0561`.
* **Inference Endpoints:**
  * **FastAPI Service:** `/health`, `/recommend`, `/recommend/similar`, `/products/{product_id}`, `/catalog/categories`.
  * **Streamlit Studio:** `http://localhost:8502` featuring Serendipity Mode and Recommendation Reasoning Lab.

---

### Section 8: Candidate Declarations
* **Originality & Explainability:** Confirmed. I have authored and validated the codebase and am fully prepared to modify the pipeline during the 10-minute live debugging task.
* **Security & Privacy:** Confirmed. No API keys, passwords, live payment tokens, or private customer PII are present in the repository.
* **Link Integrity:** Confirmed. All repository links, notebooks, technical reports, and artifacts are verified and publicly accessible.

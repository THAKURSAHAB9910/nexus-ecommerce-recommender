# ALGIONS AI LABS PRIVATE LIMITED
## AI & Machine Learning Assessment – Official Submission Form Package
**Candidate:** Vishesh Tomar (`THAKURSAHAB9910`)  
**Assigned Project:** E-Commerce Product Recommendation Engine (Top-N Personalization)  
**System Identity:** *NEXUS – Discover Beyond Popularity*  

---

> [!TIP]
> Use the exact pre-filled entries below to complete your submission form. Every single field corresponds directly to Section 5 of the official assessment document.

---

### Section 1: Candidate Identification
| Form Field | Value to Enter |
| :--- | :--- |
| **Full Name** | Vishesh Tomar |
| **Registration Number** | *[Insert your University / College Registration Number here]* |
| **University Email** | *[Insert your official University / College Email here]* |

---

### Section 2: Assignment Confirmation
| Form Field | Value to Enter |
| :--- | :--- |
| **Internship Role** | AI & Machine Learning |
| **Assigned Project Title** | E-Commerce Product Recommendation Engine |
| **Target Domain** | E-Commerce (Retail Product Personalization) |

---

### Section 3: Code and Version
| Form Field | Value to Enter |
| :--- | :--- |
| **Repository URL** | `https://github.com/THAKURSAHAB9910/nexus-ecommerce-recommender` |
| **Primary Branch** | `main` |
| **Final Commit ID** | `511f3f6289a08d65c9a1baacd4ded466daabd2ac` *(or latest commit)* |
| **Repository Visibility** | Public (Free reviewer access without authentication barrier) |

---

### Section 4: Demonstration & Walkthrough
| Form Field | Value to Enter |
| :--- | :--- |
| **3–5 Minute Video Link** | *[Insert your Loom, Google Drive, or Unlisted YouTube link here]* |
| **Live Demo URL** | Declared Local-Only Demo (Streamlit on `http://localhost:8502`, FastAPI on `http://127.0.0.1:8000`). Ready for live execution during technical interview walkthrough. |

---

### Section 5: Documentation & Artifacts
| Form Field | Value to Enter |
| :--- | :--- |
| **README Link** | `https://github.com/THAKURSAHAB9910/nexus-ecommerce-recommender/blob/main/README.md` |
| **Technical Report Link** | `https://github.com/THAKURSAHAB9910/nexus-ecommerce-recommender/blob/main/reports/TECHNICAL_REPORT.md` |
| **Jupyter Walkthrough Notebook** | `https://github.com/THAKURSAHAB9910/nexus-ecommerce-recommender/blob/main/notebooks/nexus_walkthrough.ipynb` |
| **System Walkthrough Guide** | `https://github.com/THAKURSAHAB9910/nexus-ecommerce-recommender/blob/main/reports/SYSTEM_WALKTHROUGH.md` |

---

### Section 6: Technology Stack & Limitations
| Form Field | Value to Enter |
| :--- | :--- |
| **Technology Stack** | Python 3.14, Pandas, NumPy, Scikit-Learn, SciPy, FastAPI, Streamlit, Plotly, Pytest, Joblib, OpenPyXL. |
| **Environment Setup Notes** | Tested on Windows 11 with PowerShell. 1-click launchers provided (`run_dashboard.bat`, `run_api.bat`). Full test suite: `python -m pytest -v` (23/23 passed). |
| **Known Limitations & Assumptions** | 1. E-commerce transactional logs capture positive purchases only (implicit feedback); unpurchased items represent lack of discovery rather than explicit negative dislike.<br>2. Seasonality spikes (e.g. Q4 Christmas surge) cause shifts in transaction velocity.<br>3. Popularity baseline exhibits severe popularity bias (0.26% catalog coverage), which our Serendipity Mode and Hybrid recommender actively counteract. |

---

### Section 7: AI and Attribution
| Form Field | Value to Enter |
| :--- | :--- |
| **AI Tools Used** | Antigravity AI Assistant (Google DeepMind) |
| **Purpose of Use** | Used as a Senior Technical Mentor, architecture design advisor, pair programmer, and code reviewer. Used to assist in rapid pipeline scaffolding, test generation, and documentation drafting. |
| **Candidate Ownership & Attestation** | All machine learning algorithms (Popularity, TF-IDF Content-Based, TruncatedSVD Collaborative Filtering, Adaptive Hybrid), data leakage prevention logic, ranking metric implementations, and signature features (Serendipity Mode & Reasoning Lab) have been thoroughly reviewed, understood, validated, and can be explained and live-coded line-by-line during the technical interview. |
| **Dataset Attribution** | UCI Machine Learning Repository: *Online Retail Benchmark* (Dr. Daqing Chen, London South Bank University). Publicly licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). |

---

### Section 8: Role-Specific Form Fields
| Form Field | Value to Enter |
| :--- | :--- |
| **Dataset Source & License** | UCI Machine Learning Repository (Online Retail, Dataset #352). CC BY 4.0 License. 541,909 raw records, 527,930 clean transactions across 3,913 catalog products and 4,334 customers. |
| **Leakage-Free Split Strategy** | Strict Chronological Split: Train (80%, `2010-12-01` to `2011-11-02`), Validation (10%, `2011-11-02` to `2011-11-21`), Held-Out Test (10%, `2011-11-21` to `2011-12-09`). Invariant: $\max(T_{\text{train}}) \le \min(T_{\text{test}})$. |
| **Model Artifact Locations** | Serialized in `artifacts/`: `popularity_model.joblib`, `content_model.joblib`, `collaborative_model.joblib`, `hybrid_model.joblib`. Fully reproducible via `python -m scripts.train`. |
| **Empirical Ranking Metrics** | **Adaptive Hybrid:** Precision@10: `0.0437` \| Recall@10: `0.0246` \| NDCG@10: `0.0499` \| Catalog Coverage: `34.68%` \| Diversity: `0.5751` \| Novelty: `11.58 bits`.<br>**Popularity Baseline:** Precision@10: `0.0703` \| Catalog Coverage: `0.26%` \| Novelty: `8.14 bits`.<br>**Cold-Start Fallback (170 users):** Precision@10: `0.0488` \| NDCG@10: `0.0561`. |
| **Inference Interface Endpoints** | **FastAPI:** `/health`, `/recommend`, `/recommend/similar`, `/products/{product_id}`, `/catalog/categories`.<br>**Streamlit Studio:** `http://localhost:8502` featuring Serendipity Mode and Recommendation Reasoning Lab. |

---

### Section 9: Candidate Declarations
| Declaration | Status |
| :--- | :--- |
| **Originality & Code Ownership** | Confirmed. I understand every component and can modify the code during the 10-minute live debugging task. |
| **Security & Privacy** | Confirmed. No `.env`, passwords, live credentials, or sensitive customer PII were committed. |
| **Link Integrity** | Confirmed. All repository, documentation, and notebook links are verified and publicly accessible. |

---

## 🎬 3–5 Minute Video Demonstration Checklist (1-Take Script)

When recording your 3–5 minute Loom or screen recording video:

1. **[0:00 - 0:45] Intro & Problem Statement:**
   - State your name, role (*AI & Machine Learning Intern Assessment*), and project (*NEXUS: Intelligent E-Commerce Product Recommendation Engine*).
   - Show the problem: Traditional e-commerce recommenders suffer from the **Popularity Trap** (0.26% catalog coverage), ignoring the long tail.
2. **[0:45 - 1:30] Data Hygiene & Chronological Splitting:**
   - Mention the UCI Online Retail dataset (541k transactions, CC BY 4.0).
   - Show `reports/figures/popularity_long_tail.png` and explain the 98.32% matrix sparsity and Pareto 80/20 distribution.
   - Explain the chronological split barrier preventing future data leakage.
3. **[1:30 - 3:00] Live Streamlit Showcase (http://localhost:8502):**
   - **Recommendation Studio:** Select Customer `17850`, run Adaptive Hybrid.
   - **Demonstrate Serendipity Mode:** Switch between *Familiar* (relevance), *Balanced* (category variety), and *Explore* (long-tail novelty $> 0.85$).
   - **Demonstrate Reasoning Lab:** Expand an item and show the 4 evidence signals (Collaborative neighbor affinity, Content keywords, Category spend, Popularity rank).
   - **Demonstrate Cold-Start:** Enter user `9999999` and show graceful fallback to trending/onboarding categories.
4. **[3:00 - 3:45] Fast API & Benchmark Results:**
   - Briefly show Swagger UI (`http://127.0.0.1:8000/docs`) executing a live `/recommend` POST request.
   - Show the Model Comparison tab: explain why Popularity has high precision but 0.26% coverage, while Adaptive Hybrid unlocks 34.68% coverage and 11.58 bits of novelty.
5. **[3:45 - 4:00] Conclusion:**
   - Mention the 23-test Pytest test suite (`pytest -v`), reproducible pipeline, and thank the review committee.

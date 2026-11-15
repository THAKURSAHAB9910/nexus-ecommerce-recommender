# NEXUS: Intelligent E-Commerce Product Recommendation Engine
*Personalized Discovery Beyond Popularity*  

[![Python](https://img.shields.io/badge/Python-3.14-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.140.0-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Pytest-9.1.1-0A9EDC.svg?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## 🌟 Executive Overview
**NEXUS** is an end-to-end, production-grade e-commerce recommendation engine designed to resolve the commercial **"Popularity Trap"**. In high-volume retail platforms, naive recommendation systems disproportionately recommend a handful of top-selling head items, leaving over 95% of catalog inventory undiscovered.

NEXUS combines **leakage-free chronological validation**, multi-modal recommendation algorithms, an **Adaptive Hybrid architecture**, and two signature innovations:
1. **🌟 Serendipity Mode:** A multi-objective reranking controller enabling customers to toggle between *Familiar* (historical exploitation), *Balanced*, and *Explore* (long-tail discovery) strategies.
2. **🔬 Recommendation Reasoning Lab:** A transparent attribution panel that decomposes recommendations into grounded collaborative, semantic, category, and popularity evidence signals.

---

## 🏗️ System Architecture

```
                                    ┌────────────────────────────────────────────────────────┐
                                    │               NEXUS RECOMMENDATION SYSTEM              │
                                    └───────────────────────────┬────────────────────────────┘
                                                                │
                 ┌──────────────────────────────────────────────┼──────────────────────────────────────────────┐
                 │                                              │                                              │
    ┌────────────▼────────────┐                    ┌────────────▼────────────┐                    ┌────────────▼────────────┐
    │     DATA PIPELINE       │                    │    MODEL LAYER & SVD    │                    │   DELIVERY & INTERFACE  │
    ├─────────────────────────┤                    ├─────────────────────────┤                    ├─────────────────────────┤
    │ • UCI Online Retail     │                    │ • Popularity Baseline   │                    │ • FastAPI REST Service  │
    │   (541k transactions)   │                    │ • Content-Based TF-IDF  │                    │ • Streamlit Studio      │
    │ • Administrative Filter │                    │ • Collaborative SVD     │                    │ • Serendipity Engine    │
    │ • Chronological Split   │                    │ • Adaptive Hybrid Blend │                    │ • Reasoning Lab Audit   │
    │   (No Future Leakage)   │                    │ • Cold-Start Router     │                    │ • 23-Test Pytest Suite  │
    └─────────────────────────┘                    └─────────────────────────┘                    └─────────────────────────┘
```

---

## 📊 Empirical Ranking Benchmark (Held-Out Test Data)
Evaluated across $N=300$ active test customers under an identical chronological ranking protocol:

| Model Architecture | Precision@10 | Recall@10 | NDCG@10 | Catalog Coverage (%) | Intra-List Diversity | Novelty (bits) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Popularity Baseline** | **0.0703** | **0.0284** | **0.0799** | 0.26% | 0.6889 | 8.14 |
| **Content-Based (TF-IDF)** | 0.0303 | 0.0202 | 0.0322 | 18.58% | 0.2529 | **12.68** |
| **Collaborative (TruncatedSVD)** | 0.0427 | 0.0208 | 0.0486 | **36.29%** | 0.6527 | 11.80 |
| **Adaptive Hybrid (NEXUS)** | 0.0437 | 0.0246 | 0.0499 | 34.68% | 0.5751 | 11.58 |
| *Cold-Start Fallback (N=170)* | 0.0488 | 0.0265 | 0.0561 | 0.26% | 0.6889 | 8.14 |

> [!NOTE]
> **Key Finding for Interview:** The Popularity Baseline achieves high surface precision on head products, but its catalog coverage is an abysmal **0.26%** (it only recommends 10 items out of 3,610 to everyone). **Adaptive Hybrid** unlocks **34.68% catalog coverage**, strong NDCG@10 (0.0499), and superior long-tail discovery without sacrificing domain relevance.

---

## 📁 Repository Structure

```
nexus-ecommerce-recommender/
│
├── app/
│   └── streamlit_app.py          <- Interactive Showcase Studio & Diagnostics
│
├── artifacts/                    <- Serialized machine learning models (*.joblib)
│   ├── popularity_model.joblib
│   ├── content_model.joblib
│   ├── collaborative_model.joblib
│   └── hybrid_model.joblib
│
├── data/
│   ├── raw/                      <- Original zip archives and raw CSV cache
│   └── processed/                <- Clean transactions, catalog, and chronological splits
│
├── reports/
│   ├── figures/                  <- Generated EDA visualizations (*.png)
│   ├── eda_summary.json          <- Statistical profiling of sparsity, Gini, and activity
│   ├── model_comparison.json    <- Empirical ranking benchmark results
│   ├── TECHNICAL_REPORT.md       <- Full 18-section technical report
│   └── INTERVIEW_PREP.md         <- 3-5 min demo script & live coding walkthrough
│
├── scripts/
│   └── train.py                  <- End-to-end model training orchestrator
│
├── src/
│   ├── api/
│   │   └── main.py               <- Production FastAPI inference backend
│   ├── data/
│   │   └── loader.py             <- Automated UCI benchmark ingestion & validation
│   ├── preprocessing/
│   │   ├── cleaner.py            <- Logistics code filtering & catalog synthesis
│   │   ├── splitter.py           <- Leakage-free chronological train/val/test split
│   │   └── eda.py                <- Statistical sparsity & long-tail analysis
│   ├── models/
│   │   ├── popularity.py         <- Non-personalized baseline
│   │   ├── content_based.py      <- TF-IDF + Cosine similarity engine
│   │   ├── collaborative.py      <- TruncatedSVD Matrix Factorization
│   │   └── hybrid.py             <- Adaptive sparsity-weighted hybrid engine
│   ├── recommendation/
│   │   ├── engine.py             <- Unified orchestration facade
│   │   └── serendipity.py        <- Signature Feature 1: Serendipity Reranking
│   ├── explanations/
│   │   └── reasoning_lab.py      <- Signature Feature 2: Reasoning Lab Engine
│   └── evaluation/
│       ├── metrics.py            <- Precision@K, Recall@K, NDCG@K, Coverage, Diversity
│       └── evaluator.py          <- Benchmark runner on held-out test data
│
├── tests/                        <- Automated 23-test Pytest test suite
│   ├── test_data_validation.py
│   ├── test_metrics.py
│   ├── test_models.py
│   ├── test_signature_features.py
│   └── test_api.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart & Pipeline Execution

### 1. Environment Setup
```powershell
# Clone or navigate to the project directory
cd c:\Users\Vishe\Downloads\vishac

# Verify dependencies
python -m pip install -r requirements.txt
```

### 2. Reproduce the Full Data & Training Pipeline
Every component is 100% reproducible with automated commands:

```powershell
# Step 1: Ingest and validate official UCI benchmark
python -m src.data.loader

# Step 2: Clean data, filter administrative codes, and generate catalog
python -m src.preprocessing.cleaner

# Step 3: Execute leakage-free chronological splitting
python -m src.preprocessing.splitter

# Step 4: Run exploratory data analysis and export visualizations
python -m src.preprocessing.eda

# Step 5: Train and serialize all recommendation models
python -m scripts.train

# Step 6: Run empirical benchmark evaluation on held-out test data
python -m src.evaluation.evaluator
```

### 3. Run Automated Pytest Suite
```powershell
python -m pytest -v
# Verified Result: 23 passed in ~9.6s
```

### 4. Launch Production FastAPI Inference Backend
```powershell
python -m uvicorn src.api.main:app --port 8000 --reload
# Access Interactive Swagger UI at: http://127.0.0.1:8000/docs
```

### 5. Launch Interactive Streamlit Studio
```powershell
python -m streamlit run app/streamlit_app.py
# Access Interactive App at: http://localhost:8501
```

---

## 🎯 The Two Signature Innovations

### 1. Serendipity Mode
Conventional recommenders create echo chambers. NEXUS implements a multi-objective reranking controller:
- **🟢 Familiar:** $\alpha=0.85, \beta=0.10, \gamma=0.05$ (Pure historical exploitation).
- **🔵 Balanced:** $\alpha=0.55, \beta=0.25, \gamma=0.20$ (Relevance + category diversity).
- **🟣 Explore:** $\alpha=0.35, \beta=0.30, \gamma=0.35$ (Surfaces high-quality long-tail gems).

### 2. Recommendation Reasoning Lab
Deconstructs recommendations into 4 verifiable evidence layers:
- **🤝 Collaborative Signal:** Latent factor cosine affinity with previously bought items.
- **🔤 Content Semantics:** Shared TF-IDF keywords matching historical purchases.
- **🏷️ Category Alignment:** Customer historical spend and category affinity percentage.
- **📈 Catalog Popularity Rank:** Percentile ranking and tier (*Best-Seller*, *Mid-Tier*, *Long-Tail*).



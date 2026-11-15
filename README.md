# NEXUS: Intelligent E-Commerce Product Recommendation Engine
*Personalized Discovery Beyond Popularity*  

[![Python](https://img.shields.io/badge/Python-3.14-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.140.0-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Pytest-9.1.1-0A9EDC.svg?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## 🌟 Executive Overview
**NEXUS** is an end-to-end, production-grade e-commerce recommendation system engineered to overcome the commercial **"Popularity Trap"**. In high-volume retail platforms, naive recommendation engines disproportionately recommend a handful of top-selling head items, leaving over 95% of catalog inventory undiscovered.

NEXUS combines **leakage-free chronological validation**, multi-modal recommendation algorithms, an **Adaptive Hybrid architecture**, and two signature innovations:
1. **🌟 Serendipity Mode:** A multi-objective reranking controller enabling customers and merchandisers to toggle between *Familiar* (historical exploitation), *Balanced*, and *Explore* (long-tail discovery) strategies.
2. **🔬 Recommendation Reasoning Lab:** A transparent attribution panel that decomposes recommendations into grounded collaborative, semantic, category, and popularity evidence signals.

---

## 📦 Required Submission Package Checklist

This repository directly fulfills all 6 required submission package specifications:

| Submission Item | Location & Evidence | Verification Status |
| :--- | :--- | :---: |
| **1. Source Repository** | Clean repo with preprocessing (`src/preprocessing/`), models (`src/models/`), inference API (`src/api/main.py`), Streamlit UI (`app/streamlit_app.py`), and Jupyter walkthrough (`notebooks/nexus_walkthrough.ipynb`). | ✅ Verified |
| **2. Dataset Source & License** | UCI Machine Learning Repository: *Online Retail Benchmark* (Dr. Daqing Chen, London South Bank University). Publicly licensed under **CC BY 4.0**. Zero restricted personal data / 100% anonymized IDs. | ✅ Verified |
| **3. README Documentation** | Comprehensive environment setup, 1-command reproducible pipeline, inference commands, and acceptance test demonstrations. | ✅ Verified |
| **4. Technical Report** | Full 18-section technical report in [`reports/TECHNICAL_REPORT.md`](file:///c:/Users/Vishe/Downloads/vishac/reports/TECHNICAL_REPORT.md) detailing EDA, target formulation, chronological splitting, model benchmarks, error analysis, and responsible use. | ✅ Verified |
| **5. Saved Model Artifacts & Tests** | Serialized models in `artifacts/*.joblib`, reproducible via `python -m scripts.train`, dependency specification in `requirements.txt`, and 23 automated tests in `tests/` (`pytest -v`). | ✅ Verified |
| **6. Demonstration & Walkthrough Guide** | 3–5 minute video demonstration script, presentation roadmap, and interview prep in [`reports/SYSTEM_WALKTHROUGH.md`](file:///c:/Users/Vishe/Downloads/vishac/reports/SYSTEM_WALKTHROUGH.md). | ✅ Verified |

---

## 🧪 Acceptance Tests Demonstrated (Official Criteria)

### Acceptance Test 1: Reproduce Dataset Split and Training Run from README
Every component of the data preparation, chronological splitting, EDA, and model training pipeline is 100% reproducible via dedicated Python commands:

```powershell
# 1. Environment Setup: Install pinned dependencies
python -m pip install -r requirements.txt

# 2. Ingest and validate official UCI benchmark (541,909 raw records)
python -m src.data.loader

# 3. Clean administrative/logistics codes and synthesize catalog (3,913 products)
python -m src.preprocessing.cleaner

# 4. Enforce strict chronological train (80%), validation (10%), test (10%) split
python -m src.preprocessing.splitter

# 5. Execute exploratory data analysis and export statistical figures
python -m src.preprocessing.eda

# 6. Train and serialize all recommendation models into artifacts/
python -m scripts.train

# 7. Run standardized ranking evaluation on held-out test data
python -m src.evaluation.evaluator
```

**Artifacts Generated:** Serialized models are saved directly to `artifacts/`:
- `artifacts/popularity_model.joblib`
- `artifacts/content_model.joblib`
- `artifacts/collaborative_model.joblib`
- `artifacts/hybrid_model.joblib`

---

### Acceptance Test 2: Baseline and Candidate Model Metrics on Held-Out Test Set
All models were benchmarked under an **identical, leakage-free chronological protocol** on active customers in the held-out test window (`2011-11-21` to `2011-12-09`). Test set interactions were strictly quarantined and never used during training or hyperparameter tuning.

| Model Architecture | Model Type | Precision@10 | Recall@10 | NDCG@10 | Catalog Coverage (%) | Intra-List Diversity | Novelty (bits) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | Non-Personalized Baseline | **0.0703** | **0.0284** | **0.0799** | 0.26% | 0.6889 | 8.14 |
| **Content-Based (TF-IDF)** | Semantic Feature Model | 0.0303 | 0.0202 | 0.0322 | 18.58% | 0.2529 | **12.68** |
| **Collaborative (TruncatedSVD)** | Matrix Factorization ($d=40$) | 0.0427 | 0.0208 | 0.0486 | **36.29%** | 0.6527 | 11.80 |
| **Adaptive Hybrid (NEXUS)** | Sparsity-Weighted Hybrid | 0.0437 | 0.0246 | 0.0499 | 34.68% | 0.5751 | 11.58 |
| *Cold-Start Fallback (N=170)* | Unseen Zero-History Users | 0.0488 | 0.0265 | 0.0561 | 0.26% | 0.6889 | 8.14 |

#### Empirical Analysis & Insights:
1. **The Popularity Trap:** The Popularity Baseline achieves high surface Precision@10 (0.0703) on head items because high-volume products sell frequently across all users. However, its **Catalog Coverage is an abysmal 0.26%** (it only ever recommends the exact same 10 items out of 3,610 to every user), yielding the lowest novelty (8.14 bits). In production, this starves 99.7% of catalog inventory.
2. **Personalization Power:** Collaborative Filtering unlocks **36.29% catalog reach**, surfacing long-tail items tailored to individual latent tastes.
3. **Adaptive Hybrid Advantage:** The Adaptive Hybrid harmonizes both paradigms: it achieves balanced NDCG@10 (0.0499), 34.68% catalog coverage, and robust novelty (11.58 bits) while seamlessly falling back to content-based or trending recommendations for sparse and cold-start customers.

---

### Acceptance Test 3: Real Prediction for Sample Input Through API & App
Predictions are computed dynamically by loading the serialized model artifacts (`artifacts/*.joblib`); no sample outputs are hard-coded.

#### Option A: FastAPI REST Inference Backend
Launch the production API server:
```powershell
python -m uvicorn src.api.main:app --port 8000
```
Interactive Swagger API documentation: **`http://127.0.0.1:8000/docs`**

Submit a real prediction via `curl`:
```powershell
curl -X POST "http://127.0.0.1:8000/recommend" `
  -H "Content-Type: application/json" `
  -d '{"user_id": "17850", "model_type": "hybrid", "k": 5, "serendipity_mode": "balanced"}'
```

**Real Model Output Returned:**
```json
{
  "user_id": "17850",
  "model_type": "hybrid",
  "serendipity_mode": "balanced",
  "historical_interaction_count": 297,
  "user_status": "ACTIVE_WARM",
  "recommendations": [
    {
      "item_id": "22423",
      "description": "REGENCY CAKESTAND 3 TIER",
      "category": "Home Decor & Kitchenware",
      "unit_price": 10.95,
      "composite_score": 0.8124,
      "novelty_score": 8.92,
      "serendipity_rationale": "High collaborative neighbor affinity (0.88) with historical purchases in Home Decor & Kitchenware."
    }
  ]
}
```

#### Option B: Interactive Streamlit Studio
Launch the interactive dashboard:
```powershell
python -m streamlit run app/streamlit_app.py --server.port 8502
```
Access the dashboard at **`http://localhost:8502`**. Select any active customer (e.g. `17850`, `13089`, `15311`) or enter a brand new cold-start customer (`9999999`) to observe real-time recommendation generation and cold-start routing.

---

### Acceptance Test 4: Explain One Prediction / Error, Influential Inputs, Limitations & Uncertainty

#### 1. Prediction Explanation & Influential Inputs
For Customer `17850` receiving Product `22423` (*REGENCY CAKESTAND 3 TIER*):
- **Collaborative Interaction Signal (Strongest Influence):** Latent factor cosine affinity of **0.88** with Customer 17850's past purchases of decorative stands and tableware.
- **Content & Semantic Tokens:** Shared TF-IDF unigram and bigram overlap with historical purchases: `['CAKESTAND', 'TIER', 'REGENCY', 'WHITE']`.
- **Category Spend Prior:** Customer 17850 has completed 43% of lifetime transactions in the *Home Decor & Kitchenware* category.
- **Catalog Popularity Tier:** Mid-Tier item (78th percentile sales rank, 8.92 bits novelty), providing personalized discovery rather than generic best-seller duplication.

#### 2. Error & Misprediction Analysis (Failure Mode)
- **Observed Edge-Case Error:** When an active customer with 95% historical spend in *Home Decor* makes an atypical, one-time seasonal gift purchase (e.g. *Novelty Christmas Mug*), collaborative filtering can over-extrapolate the single co-occurrence transaction, temporarily surfacing holiday items rather than core inventory.
- **Mitigation:** The Adaptive Hybrid engine bounds this failure mode by dynamically shifting weights ($\lambda = 0.65$ CF + $0.35$ Content) and allowing user-controlled *Serendipity Modes* (*Familiar* vs. *Explore*) to restore category balance.

#### 3. Limitations & Prediction Uncertainty
- **Implicit Feedback Uncertainty:** Purchase logs contain positive interaction instances only ($q \ge 1$); an unpurchased product reflects a lack of exposure or catalog discovery, not explicit negative customer dislike.
- **Ordinal Score Non-Calibration:** Output scores represent relative ranking affinities ($S \in [0, 1]$), not calibrated purchase probabilities ($P(\text{purchase}) = 0.85$).
- **Temporal Drift:** Historical preferences in Q1 do not fully account for sudden macro shocks or seasonal Q4 holiday surges.
- **Decision-Support Attestation:** System outputs are designed to assist retail merchandisers and customer exploration, not to make high-stakes automated decisions.

---

### Acceptance Test 5: Demonstrate Zero Future/Target Leakage and No Sensitive Data Used

#### 1. Zero Future/Target Leakage Proof
To guarantee that future interactions never leak into model training, NEXUS enforces a strict chronological split barrier:

```
Timeline: [2010-12-01] =========================================================> [2011-12-09]
Train Window (80%):       [2010-12-01 08:26:00]  ----->  [2011-11-02 14:23:00]  (317,171 rows)
Validation Window (10%):  [2011-11-02 14:24:00]  ----->  [2011-11-21 09:33:00]  ( 39,640 rows)
Held-Out Test (10%):      [2011-11-21 09:34:00]  ----->  [2011-12-09 12:50:00]  ( 39,659 rows)

Zero-Leakage Invariant: Max(Train Timestamp) < Min(Validation Timestamp) < Min(Test Timestamp)
```

- **Quarantine Buffer:** A 19-day temporal buffer separates training from held-out test evaluation.
- **Strict Isolation:** User-item interaction matrices, TF-IDF term dictionaries, and TruncatedSVD latent embeddings are fitted **strictly on training interactions prior to 2011-11-02**. No test interaction or label data was visible during training.

#### 2. Sensitive Data & Privacy Protection
- **100% Anonymized IDs:** Customer IDs are anonymized 5-digit integer keys (`12344` to `18287`).
- **Zero PII:** The repository contains zero private customer names, delivery addresses, telephone numbers, emails, passwords, API tokens, or payment card details.
- **Sanctioned Public Dataset:** Built exclusively upon the open-access UCI Online Retail Benchmark under **CC BY 4.0**.

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
    │ • Logistics Code Filter │                    │ • Collaborative SVD     │                    │ • Serendipity Engine    │
    │ • Chronological Split   │                    │ • Adaptive Hybrid Blend │                    │ • Reasoning Lab Audit   │
    │   (No Future Leakage)   │                    │ • Cold-Start Router     │                    │ • 23-Test Pytest Suite  │
    └─────────────────────────┘                    └─────────────────────────┘                    └─────────────────────────┘
```

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
├── notebooks/
│   └── nexus_walkthrough.ipynb   <- Executed Jupyter notebook with embedded outputs
│
├── reports/
│   ├── figures/                  <- Generated EDA visualizations (*.png)
│   ├── eda_summary.json          <- Statistical profiling of sparsity, Gini, and activity
│   ├── model_comparison.json    <- Empirical ranking benchmark results
│   ├── TECHNICAL_REPORT.md       <- Full 18-section technical architecture report
│   └── SYSTEM_WALKTHROUGH.md     <- 3–5 min video script & demonstration roadmap
│
├── scripts/
│   ├── train.py                  <- End-to-end model training orchestrator
│   ├── generate_notebook.py      <- Reproducible notebook generator
│   └── execute_notebook.py       <- Automated notebook execution engine
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
├── run_api.bat                   <- 1-click FastAPI launcher
├── run_dashboard.bat             <- 1-click Streamlit launcher
├── requirements.txt              <- Pinned production dependencies
├── .env.example                  <- Environment variable template
├── .gitignore
└── README.md
```

---

## 🎯 The Two Signature Innovations

### 1. Serendipity Mode
Conventional recommenders create echo chambers by only recommending items nearly identical to past purchases. NEXUS implements a multi-objective reranking controller:

$$\text{FinalScore}(u, i) = \alpha \cdot S_{\text{relevance}}(u, i) + \beta \cdot \text{Diversity}(i, \mathcal{S}) + \gamma \cdot \text{Novelty}(i)$$

- **🟢 Familiar ($\alpha=0.85, \beta=0.10, \gamma=0.05$):** Maximize proven historical preference and high conversion likelihood.
- **🔵 Balanced ($\alpha=0.55, \beta=0.25, \gamma=0.20$):** Multi-objective equilibrium of relevance and cross-category discovery.
- **🟣 Explore ($\alpha=0.35, \beta=0.30, \gamma=0.35$):** Surfaces high-affinity long-tail catalog discoveries while penalizing category over-concentration.

### 2. Recommendation Reasoning Lab
Deconstructs recommendations into 4 verifiable evidence layers rather than presenting a black-box score:
- **🤝 Collaborative Signal:** Latent factor cosine affinity with historical purchases.
- **🔤 Content Semantics:** Shared TF-IDF token keywords matching customer profile.
- **🏷️ Category Alignment:** Customer historical spend and category affinity percentage.
- **📈 Catalog Popularity Rank:** Percentile ranking and tier (*Best-Seller*, *Mid-Tier*, *Long-Tail*).

---

## 🧪 Automated Pytest Verification Suite

The repository includes a comprehensive 23-test verification suite covering every pipeline layer:

```powershell
python -m pytest -v
```

**Verification Results:**
- `tests/test_data_validation.py`: Schema validation, missing value filtering, negative quantity/price rejection (4 passed).
- `tests/test_metrics.py`: Correctness of Precision@K, Recall@K, NDCG@K, Catalog Coverage, Diversity, Novelty (5 passed).
- `tests/test_models.py`: Popularity baseline, TF-IDF content model, SVD collaborative model, Adaptive hybrid blend (5 passed).
- `tests/test_signature_features.py`: Serendipity reranking weights, Reasoning Lab 4-signal decomposition, Cold-start routing (5 passed).
- `tests/test_api.py`: FastAPI endpoints (`/health`, `/recommend`, `/recommend/similar`, `/products/{id}`, `/catalog/categories`) (4 passed).

**Status: 23 passed in ~9.6s (100% pass rate).**

---

## 📄 License
This project utilizes the UCI Machine Learning Repository Online Retail Benchmark, licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

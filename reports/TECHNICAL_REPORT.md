# NEXUS: Intelligent E-Commerce Product Recommendation Engine
## Comprehensive Technical Report & Methodological Audit
**Target Organization:** ALGIONS AI LABS PRIVATE LIMITED  
**Role:** AI & Machine Learning Internship Assessment  
**Project Identity:** *NEXUS – Discover Beyond Popularity*  
**Date:** November 2026  

---

### 1. Abstract
In contemporary e-commerce platforms, recommendation engines act as mission-critical decision-support systems that bridge user intent with product discovery. However, naive recommendation architectures frequently collapse into the "Popularity Trap"—disproportionately surfacing a tiny fraction of top-selling head items while consigning thousands of high-value catalog products to complete invisibility.

This report presents **NEXUS**, an end-to-end, reproducible, production-grade e-commerce recommendation system engineered to personalize discovery beyond popularity. Built upon the verified, publicly licensed **UCI Online Retail Benchmark** (541,909 transactions), NEXUS implements:
1. A rigorous **leakage-free chronological validation protocol** (quarantining future events).
2. Four comparative recommendation paradigms: **Popularity Baseline**, **Content-Based Filtering (TF-IDF + Cosine Similarity)**, **Collaborative Filtering (TruncatedSVD Latent Factors)**, and an **Adaptive Hybrid Recommender**.
3. Two signature differentiators: **Serendipity Mode** (a multi-objective reranking controller toggling between *Familiar*, *Balanced*, and *Explore* strategies) and the **Recommendation Reasoning Lab** (a transparent 4-signal evidentiary attribution engine).
4. Multi-tiered **cold-start fallbacks** for new users, sparse users, and new items.
5. A production inference backend built with **FastAPI**, an interactive **Streamlit** dashboard, and a **23-test Pytest verification suite**.

Empirical evaluation across active test users demonstrates that while the Popularity Baseline achieves high surface Precision@10 (0.0703) on head items, its catalog coverage is an abysmal **0.26%**. In contrast, the **Adaptive Hybrid model** achieves **34.68% catalog coverage**, strong ranking performance (NDCG@10 of 0.0499), and superior long-tail discovery (11.58 bits of novelty) without sacrificing domain relevance.

---

### 2. Business Problem & Systems Formulation
E-commerce retail platforms confront two competing operational imperatives:
- **Immediate Conversion:** Surfacing products with high historical likelihood of purchase.
- **Long-Term Customer Lifetime Value (LTV):** Expanding customer taste profiles across categories, reducing churn, and monetizing the catalog long tail.

Treating the recommendation engine as an infallible prediction engine is flawed; it is fundamentally a **decision-support tool** that reduces customer search friction. The core challenge is balancing relevance (familiarity) with discovery (serendipity).

#### System Formulation:
- **User Identifier:** Customer ID ($u \in \mathcal{U}$, 4,334 active customers).
- **Item Identifier:** StockCode ($i \in \mathcal{I}$, 3,913 catalog products).
- **Interaction Signal:** Transaction frequency and purchased quantities ($q_{ui} \ge 1$).
- **Implicit Feedback Transformation:** Following Hu, Koren & Volinsky (2008), implicit confidence is formulated as:
  $$c_{ui} = 1 + \alpha \cdot \log(1 + q_{ui}), \quad \alpha = 1.0$$
- **Objective:** For an arbitrary user $u$, generate an ordered ranking of top-$K$ products:
  $$\hat{\mathcal{R}}_K(u) = (i_{(1)}, i_{(2)}, \dots, i_{(K)}), \quad i \notin \mathcal{I}_u^{\text{train}}$$

---

### 3. Dataset Selection, Verification & Licensing
The dataset selected for NEXUS is the **UCI Machine Learning Repository Online Retail Benchmark**:
- **Source:** [UCI Machine Learning Repository (Dataset 352)](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Principal Investigator:** Dr. Daqing Chen, London South Bank University.
- **Official License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- **Verifiable Integrity:** Freely accessible, reproducible download without gated credentials or proprietary API tokens.
- **Why Chosen over RetailRocket or Amazon Reviews:**
  RetailRocket dataset properties are anonymized hashed integers (e.g. property `categoryid: 1016`). While suitable for abstract mathematical modeling, it prevents human-interpretable explanations in the **Reasoning Lab** and disables meaningful Content-Based filtering. The UCI dataset contains rich, human-readable merchandise descriptions (e.g., *"REGENCY CAKESTAND 3 TIER"*, *"WHITE HANGING HEART T-LIGHT HOLDER"*) and exact transaction timestamps, enabling authentic semantic analysis, leakage-safe chronological splitting, and explainable recommendations.

---

### 4. Data Dictionary & Schema Audit

| Column Name | Data Type | Physical Meaning & Range | Missing Count | Handling Strategy |
| :--- | :--- | :--- | :--- | :--- |
| `InvoiceNo` | String | 6-digit transaction ID. Prefix 'C' indicates cancellation. | 0 (0.0%) | String parsing; filter rows starting with 'C'. |
| `StockCode` | String | 5-digit product code or alphanumeric variant. | 0 (0.0%) | Cleaned string; filter non-merchandise service codes. |
| `Description` | String | Product name and descriptive English text. | 1,454 (0.27%) | Uppercase, strip whitespace; drop missing descriptions. |
| `Quantity` | Integer | Quantity of product units purchased per line item. | 0 (0.0%) | Filter strictly $Quantity > 0$. |
| `InvoiceDate` | DateTime | Timestamp of invoice generation (`2010-12-01` to `2011-12-09`). | 0 (0.0%) | Strict chronological ordering and splitting barrier. |
| `UnitPrice` | Float | Unit price in Sterling (£). | 0 (0.0%) | Filter strictly $UnitPrice > 0.0$. |
| `CustomerID` | String | Unique identifier assigned to registered customers. | 135,080 (24.9%) | Guest checkouts retained for catalog/popularity; identified rows used for personalized models. |
| `Country` | String | Customer country of residence. | 0 (0.0%) | Retained in metadata. |

---

### 5. Exploratory Data Analysis & Empirical Statistics
Full statistical profiling (`reports/eda_summary.json`) reveals:
- **Total Raw Records:** 541,909 transactions.
- **Valid Clean Transactions:** 527,930 transactions.
- **Customer-Identified Transactions:** 396,470 transactions across 4,334 unique customers.
- **Product Catalog Size:** 3,913 unique merchandise items.
- **Unique User-Item Interaction Pairs:** 266,250 pairs.
- **User-Item Matrix Sparsity:** **98.32%** ($\text{Sparsity} = 1 - \frac{266,250}{4,334 \times 3,913}$).
- **Customer Activity Distribution:**
  - Mean interactions per user: 91.48 (Median: 41.0, IQR: 17.0 to 100.0).
  - Minimum: 1 interaction | Maximum: 7,838 interactions.
- **Popularity Concentration & The Long Tail:**
  - Item interaction Gini coefficient: **0.6378** (severe popularity inequality).
  - **Pareto 80/20 Verification:** The top 20% of catalog items generate **65.57%** of total transaction volume.
- **E-Commerce Categories Mapped:**
  1. *Gifts & Novelties* (1,045 items)
  2. *Home Decor & Lighting* (946 items)
  3. *Bags & Storage* (565 items)
  4. *Kitchen & Dining* (388 items)
  5. *Stationery & Craft* (322 items)
  6. *Apparel & Accessories* (321 items)
  7. *Seasonal & Festive* (228 items)
  8. *Toys & Games* (58 items)
  9. *Bath & Wellness* (40 items)

---

### 6. Data Cleaning & Feature Engineering Pipeline
The cleaning pipeline (`src/preprocessing/cleaner.py`) applies 4 domain-defensible filters:
1. **Cancellation Elimination:** Eliminates credit adjustments and cancelled orders ($InvoiceNo \text{ starts with 'C'}$).
2. **Sanity Filtering on Quantities & Prices:** Enforces $Quantity > 0$ and $UnitPrice > 0.0$, purging inventory damages, sample giveaways, and debt write-offs.
3. **Administrative Code Removal:** Removes non-merchandise logistics records (`POST`, `DOT`, `M`, `D`, `S`, `AMAZONFEE`, `CRUK`, `PADS`, `B`).
4. **Canonical Product Catalog Synthesis:** Groups by `StockCode` to extract the modal description, median unit price, lifetime volume sold, and rule-based category assignment.

---

### 7. Leakage Prevention & Chronological Splitting Strategy
Random interaction splitting in time-dependent e-commerce data introduces severe **future data leakage**: the model learns patterns from tomorrow to predict what the user viewed yesterday.

NEXUS enforces a strict **chronological barrier** (`src/preprocessing/splitter.py`):
```
Timeline: [2010-12-01] ========================================> [2011-12-09]
Train Window (80%):       [2010-12-01]  ----->  [2011-11-02]  (317,171 transactions)
Validation Window (10%):  [2011-11-02]  ----->  [2011-11-21]  (39,640 transactions)
Held-Out Test (10%):      [2011-11-21]  ----->  [2011-12-09]  (39,659 transactions)
```
- **Strict Leakage Assertion:** $\max(T_{\text{train}}) \le \min(T_{\text{val}}) \le \min(T_{\text{test}})$. Verified in automated tests.
- **Evaluation Cohort:** 977 warm customers present in both Train and Test.
- **Cold-Start Cohort:** 170 new customers who arrived in the Test period with zero historical interactions.

---

### 8. Recommendation Approaches Implemented

#### Approach A: Popularity Baseline
- Ranks catalog products by interaction frequency and buyer count strictly within $\mathcal{D}_{\text{train}}$:
  $$\text{Score}_{\text{pop}}(i) = \frac{\text{UniqueBuyers}_{\text{train}}(i)}{\max_j \text{UniqueBuyers}_{\text{train}}(j)}$$
- Provides global and category-filtered trending products.

#### Approach B: Content-Based Filtering
- Builds textual documents combining `Description` and `Category`.
- Vectorizes documents using **TF-IDF** with unigram and bigram tokens ($N=2,500$ features).
- For a query user $u$, constructs a dense user profile vector $\mathbf{p}_u$:
  $$\mathbf{p}_u = \frac{\sum_{i \in \mathcal{I}_u^{\text{train}}} \log(1 + q_{ui}) \cdot \mathbf{v}_i}{\left\| \sum_{i \in \mathcal{I}_u^{\text{train}}} \log(1 + q_{ui}) \cdot \mathbf{v}_i \right\|_2}$$
- Scores candidate items via Cosine Similarity: $S_{\text{content}}(u, i) = \mathbf{p}_u \cdot \mathbf{v}_i$.

#### Approach C: Collaborative Filtering (TruncatedSVD Latent Factors)
- Constructs sparse user-item interaction matrix $\mathbf{R} \in \mathbb{R}^{3,989 \times 3,610}$.
- Applies Truncated Singular Value Decomposition ($d=40$ latent factors):
  $$\mathbf{R} \approx \mathbf{P} \mathbf{Q}^T, \quad \mathbf{P} \in \mathbb{R}^{M \times d}, \quad \mathbf{Q} \in \mathbb{R}^{N \times d}$$
- Computes normalized predicted preference: $\hat{r}_{ui} = \mathbf{p}_u \cdot \mathbf{q}_i$.
- Computes item-item latent cosine similarity to identify neighbor attribution.

#### Approach D: Adaptive Hybrid Recommender
- Dynamically blends Collaborative and Content-based signals:
  $$\hat{S}_{\text{hybrid}}(u, i) = \lambda_u \cdot \hat{S}_{\text{CF}}(u, i) + (1 - \lambda_u) \cdot \hat{S}_{\text{Content}}(u, i)$$
- Adapts $\lambda_u$ to customer interaction density:
  - Dense users ($> 15$ interactions): $\lambda_u = 0.65$
  - Moderate users ($5 \le n \le 15$): $\lambda_u = 0.50$
  - Sparse users ($1 \le n \le 4$): $\lambda_u = 0.25$
  - Cold-start users ($n = 0$): $\lambda_u = 0.0$ (automatic Popularity Fallback).

---

### 9. Empirical Evaluation & Ranking Benchmark Results
All models were evaluated under identical ranking conditions on held-out future purchases for $N=300$ active evaluation users.

| Model Architecture | Precision@10 | Recall@10 | NDCG@10 | Catalog Coverage (%) | Intra-List Category Diversity | Novelty (bits) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Popularity Baseline** | **0.0703** | **0.0284** | **0.0799** | 0.26% | 0.6889 | 8.14 |
| **Content-Based (TF-IDF)** | 0.0303 | 0.0202 | 0.0322 | 18.58% | 0.2529 | **12.68** |
| **Collaborative (TruncatedSVD)** | 0.0427 | 0.0208 | 0.0486 | **36.29%** | 0.6527 | 11.80 |
| **Adaptive Hybrid (NEXUS)** | 0.0437 | 0.0246 | 0.0499 | 34.68% | 0.5751 | 11.58 |
| *Cold-Start Fallback (N=170)* | 0.0488 | 0.0265 | 0.0561 | 0.26% | 0.6889 | 8.14 |

#### Critical Insights for Senior Engineering Review:
1. **The Popularity Illusion:** Popularity displays the highest Precision@10 (0.0703) because top sellers are purchased frequently by almost everyone. However, its **Catalog Coverage is 0.26%** (it recommends the exact same 10 products out of 3,610 to every user). In a production retailer, this cannibalizes organic search and starves 99.7% of catalog inventory.
2. **Discovery & Reach:** **Collaborative Filtering** surfaces **36.29% of the catalog**, uncovering personalized products that Popularity ignores.
3. **The Hybrid Synthesis:** The **Adaptive Hybrid** achieves balanced NDCG@10 (0.0499), 34.68% catalog coverage, and high novelty (11.58 bits) while protecting against cold-start failures.

---

### 10. Signature Feature 1: Serendipity Mode
To break recommendation echo chambers without injecting irrelevant noise, NEXUS incorporates **Serendipity Mode** (`src/recommendation/serendipity.py`):

$$\text{FinalScore}(u, i) = w_{\text{rel}} \cdot S_{\text{rel}}(u, i) + w_{\text{nov}} \cdot \text{Novelty}(i) + w_{\text{div}} \cdot \text{Diversity}(i, \mathcal{S})$$

- **🟢 Familiar ($w_{\text{rel}}=0.85, w_{\text{nov}}=0.05, w_{\text{div}}=0.10$):** Maximizes historical taste consistency. Surfaces products closely related to past purchases.
- **🔵 Balanced ($w_{\text{rel}}=0.55, w_{\text{nov}}=0.20, w_{\text{div}}=0.25$):** Equilibrates relevance with category variety.
- **🟣 Explore ($w_{\text{rel}}=0.35, w_{\text{nov}}=0.35, w_{\text{div}}=0.30$):** Promotes high-quality long-tail catalog discoveries while penalizing category over-concentration.

---

### 11. Signature Feature 2: Recommendation Reasoning Lab
The **Recommendation Reasoning Lab** (`src/explanations/reasoning_lab.py`) deconstructs recommendation decisions into four verifiable evidence signals:
1. **Collaborative Signal:** Identifies the user's historical product with highest latent cosine affinity and reports the affinity score (e.g., *"Shoppers who bought 'WHITE HANGING HEART' also frequently bought this (Affinity: 0.88)"*).
2. **Content & Semantic Signal:** Reports shared TF-IDF token keywords between user profile and item document (e.g., *['RED', 'LIGHT HOLDER', 'HEART LIGHT']*).
3. **Category Affinity:** Reports historical customer spend in the item's category (e.g., *"You have purchased 746 items in 'Home Decor & Lighting' (43% of history)"*).
4. **Popularity & Discovery Signal:** Reports global catalog sales rank and tier (*Best-Seller*, *Mid-Tier*, *Long-Tail Discovery*).
5. **Score Disclaimer:** Explicitly clarifies that the output is an ordinal normalized score, not a calibrated probability.

---

### 12. Cold-Start Handling Strategy
- **New User (Zero History):** Automatically routed to Popularity Baseline with explicit fallback labels (`is_fallback: True`).
- **Sparse User (1–4 Items):** Content-based semantic profile heavily upweighted ($\lambda_{\text{CF}} = 0.25$) to prevent noisy collaborative overfitting.
- **New Product (Zero Transactions):** Indexed into TF-IDF vector space; immediately discoverable by users with matching semantic preferences.

---

### 13. System Architecture & Production Deployment
- **Inference Service (FastAPI):** Typed Pydantic v2 schemas, REST endpoints (`/health`, `/recommend`, `/recommend/similar`, `/products/{id}`, `/catalog/categories`), sub-50ms latency.
- **Interactive UI (Streamlit):** Clean multi-tab dashboard for live experimentation, reasoning inspection, and cold-start simulation.
- **Test Suite:** 23 automated Pytest test cases covering schemas, leakage prevention, algorithms, signature features, and API endpoints.

---

### 14. Limitations & Responsible Use
1. **Absence of Negative Feedback:** E-commerce transactions capture positive purchases only. An unpurchased item may indicate either disinterest or simple lack of discovery.
2. **Seasonality:** Holiday peaks (e.g., Q4 Christmas surge) cause shifts in transaction velocity.
3. **Algorithmic Bias:** Popularity baselines create feedback loops where popular items become even more popular; Serendipity Mode is specifically engineered to mitigate this dynamic.

# NEXUS: Feature Availability, Data Leakage & Ethical Safety Audit

## 1. Recommendation Formulation & Prediction Point Definition

* **Prediction Unit:** An ordered Top-$K$ ranking of personalized product recommendations for an arbitrary active or cold-start customer ($u \in \mathcal{U}$).
* **Prediction Point ($t_{\text{pred}}$):** The exact chronological split barrier cutoff timestamp (`2011-11-02 14:23:00`).
* **Prediction Horizon:** Subsequent customer purchasing transactions occurring in the held-out test window (`2011-11-21 09:34:00` to `2011-12-09 12:50:00`).
* **Target Variable:** Future positive implicit transaction event ($y_{ui} = 1$ if customer $u$ purchases product $i$ in the test window, $0$ otherwise).
* **Evaluation Objective:** Maximize ranking relevance (NDCG@K, Recall@K, Precision@K) while actively preserving catalog diversity and unearthing the commercial long tail (Catalog Coverage %, Novelty bits).

---

## 2. Comprehensive Feature Audit Table

| Feature Name | Source Column | Type | Available at $t_{\text{pred}}$? | Audit Decision | Transformation / Engineering | Rationale & Safety Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`CustomerID`** | `CustomerID` | Numeric / ID | Yes | **Derived & Key** | User interaction vector lookup; anonymized key | Identifies individual historical preference. Retained as an anonymized integer; zero customer PII. |
| **`StockCode`** | `StockCode` | Alphanumeric | Yes | **Included** | Cleansed 5-digit identifier / catalog index | Primary item entity. Cleaned of non-merchandise logistics tokens. |
| **`Description`** | `Description` | Text | Yes | **Included** | Uppercased, normalized text, TF-IDF unigrams & bigrams | Product semantic attributes. Enables Content-Based recommendations for sparse and cold-start items. |
| **`Quantity`** | `Quantity` | Integer | Yes | **Derived Only** | Filtered $Quantity > 0$; transformed to implicit confidence | Negative quantities represent post-purchase cancellations or returns; filtered to avoid negative distortion. |
| **`UnitPrice`** | `UnitPrice` | Float | Yes | **Included** | Filtered $UnitPrice > 0.0$; catalog metadata | Product pricing metadata. Retained strictly as static merchandise metadata; non-positive prices filtered. |
| **`InvoiceDate`** | `InvoiceDate` | Datetime | Yes | **Split Barrier Only** | Chronological ordering & temporal split boundary | Dictates strict training ($t \le t_{\text{train}}$) vs test ($t \ge t_{\text{test}}$) boundary. |
| **`InvoiceNo`** | `InvoiceNo` | String | Yes | **Filtered** | Excluded if prefixed with 'C' (cancellation) | Identifies individual basket transactions. Cancellations represent return transactions and are purged. |
| **`Country`** | `Country` | Categorical | Yes | **Metadata Only** | Filtered / catalog profiling | Kept as demographic audit metadata; excluded from core affinity matrix to avoid geographic bias. |

---

## 3. Prohibited Leakage Controls & Quarantine Barriers

```
Timeline: [2010-12-01] =========================================================> [2011-12-09]
Train Window (80%):       [2010-12-01 08:26:00]  ----->  [2011-11-02 14:23:00]  (317,171 rows)
Validation Window (10%):  [2011-11-02 14:24:00]  ----->  [2011-11-21 09:33:00]  ( 39,640 rows)
Held-Out Test (10%):      [2011-11-21 09:34:00]  ----->  [2011-12-09 12:50:00]  ( 39,659 rows)

Zero-Leakage Invariant: Max(Train Timestamp) < Min(Validation Timestamp) < Min(Test Timestamp)
```

1. **Temporal Quarantine Buffer:**
   * A strict **19-day temporal buffer** (`2011-11-02` to `2011-11-21`) separates model training from held-out test evaluation.
   * This simulates authentic commercial retail production: models are trained on historical logs, validated, and evaluated on unseen future sales.

2. **Vocabulary & Latent Factor Isolation:**
   * **TF-IDF Vocabulary:** Term statistics, document frequencies ($IDF$), and unigram/bigram n-gram vocabulary are fitted strictly on product descriptions present in the training transactions and pre-split catalog.
   * **SVD Latent Factor Matrices:** The sparse user-item interaction matrix $\mathbf{R}_{\text{train}}$ ($4,334 \times 3,913$) is constructed solely from transactions with timestamp $t \le \text{2011-11-02}$.
   * **Zero Test Leakage:** No interaction from the validation or held-out test sets was visible during matrix factorization, SVD decomposition ($d=40$), or hybrid weight calibration.

3. **Logistics & Non-Merchandise Target Leakage Purge:**
   * Administrative non-merchandise transaction codes (`POST` postage, `DOT` dotcom postage, `M` manual adjustments, `D` discount, `CR` carriage, `S` samples, `BANK CHARGES`, `AMAZONFEE`) have been systematically purged.
   * Failure to purge these tokens causes models to artificially recommend postage fees or manual charges—a classic form of target leakage in transaction datasets.

---

## 4. Sensitive Data & Privacy Protection Audit

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ETHICAL PRIVACY GUARANTEE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. 100% ANONYMIZED IDS: All customer references are anonymous numeric keys  │
│    (e.g., CustomerID 12344 to 18287).                                       │
│ 2. ZERO PII: The dataset and repository contain zero customer names, street │
│    addresses, telephone numbers, emails, passwords, or credit card numbers. │
│ 3. PERMISSIVE PUBLIC BENCHMARK: Derived strictly from the open-access UCI   │
│    Online Retail Benchmark under Creative Commons Attribution 4.0 (CC BY).   │
└─────────────────────────────────────────────────────────────────────────────┘
```

* **No Gated Credentials or Proprietary Scraping:** The entire dataset is fetched programmatically from official, sanctioned public mirrors without bypassing authentication walls or accessing private corporate endpoints.
* **Non-Discriminatory Feature Selection:** Model features are strictly behavioral (purchase frequency, semantic text overlap, category spend distribution). No protected personal demographic classes (gender, ethnicity, religion, disability) are present or inferred.

# NEXUS: Technical Demonstration & System Walkthrough Guide
**Project:** NEXUS – Intelligent E-Commerce Product Recommendation Engine  

---

## 1. 3–5 Minute Demonstration Script (Word-for-Word Pitch)

### Minute 0:00 – 0:45: The Hook & Business Problem
> *"Good morning / afternoon! Today I am presenting **NEXUS**, an intelligent product recommendation engine built to solve one of the most pervasive challenges in e-commerce: **The Popularity Trap**.  
> In most online retailers, standard recommendation algorithms heavily overfit to top-selling items. While this looks good on simple accuracy metrics, it reduces catalog discovery to almost zero, starving 99% of inventory.  
> Our mission with NEXUS is **'Discover beyond popularity'**: delivering personalized, high-relevance recommendations while actively uncovering long-tail catalog discoveries through explainable ML."*

### Minute 0:45 – 1:30: Dataset & Leakage-Free Splitting
> *"For our data foundation, I selected the official **UCI Online Retail benchmark** containing 541,909 real customer transactions under a verified CC BY 4.0 license.  
> After filtering non-merchandise logistics codes and cancellations, we synthesized a clean catalog of 3,913 products categorized into 9 retail domains, with a realistic user-item matrix sparsity of 98.32% and a Gini coefficient of 0.6378.  
> Crucially, to prevent **future interaction leakage**, we avoided naive random splitting. We implemented a strict **chronological split**—80% train, 10% validation, and 10% held-out test—guaranteeing that models only predict future transactions using past historical context."*

### Minute 1:30 – 2:45: Live Interface & The Two Signature Features
*(Open the Streamlit app on screen)*
> *"Let me show you NEXUS in action.  
> In our **Recommendation Studio**, I will select Customer `17850`, an active buyer with 746 historical purchases concentrated in Home Decor.  
> When we run our **Adaptive Hybrid model**, you immediately see our first signature innovation: **Serendipity Mode**.  
> Notice how we can toggle between three strategies:
> 1. **Familiar:** Maximizes historical preference consistency (scores high on candle holders and frames).
> 2. **Balanced:** Balances relevance while enforcing category diversity across gifts, apparel, and seasonal items.
> 3. **Explore:** Actively surfaces long-tail discovery gems with novelty scores over 0.85—such as artisan necklaces—that conventional models bury, without recommending irrelevant noise.  
> 
> Next, let's look at our second signature feature: the **Recommendation Reasoning Lab**.  
> When I click 'Inspect Signals' on any item, NEXUS decomposes the recommendation into four grounded, factual evidence layers:
> - The **collaborative neighbor affinity** (e.g. latent cosine affinity of 0.88 with their past purchased t-light holder).
> - The **shared semantic TF-IDF keywords**.
> - The customer's **historical category spend**.
> - And the **catalog popularity rank**."*

### Minute 2:45 – 3:30: Benchmark Evaluation & Cold-Start Robustness
*(Switch to Tab 3: Model Comparison)*
> *"When we benchmark all models under an identical chronological protocol:
> - The **Popularity Baseline** achieves a surface Precision@10 of 0.0703, but its **Catalog Coverage is only 0.26%**—recommending the exact same 10 products to all users!
> - Our **Collaborative Filtering model** unlocks **36.29% catalog coverage**.
> - Our **Adaptive Hybrid model** achieves top-tier NDCG@10 (0.0499) and 34.68% catalog coverage.  
> - And for our 170 real **cold-start test users**, our dynamic fallback router achieved a Precision@10 of 0.0488 without a single system failure.  
> NEXUS is backed by a FastAPI backend, sub-50ms inference, and a 23-test Pytest suite."*

---

## 2. 15-Minute Technical Walkthrough Architecture Guide

When the interviewers ask you to explain your code and architecture in detail, follow this structured narrative:

```
[Raw Ingestion]        [Leakage-Free Splitting]        [Model Layer]               [Inference & UI]
DataLoader       --->  Cleaner & Splitter       --->  Popularity Baseline    --->  FastAPI REST API
(UCI CC BY 4.0)       (Train 80% / Test 10%)          Content-Based (TF-IDF)       (/recommend, /similar)
                      (Zero Leakage Assertion)        Collaborative (SVD)          Streamlit Studio
                                                      Adaptive Hybrid              (Serendipity + Reasoning)
```

### Key Technical Talking Points:
1. **Implicit Feedback Transformation:**
   - *"In e-commerce, we do not have 1-to-5 star ratings. We have purchases, which represent implicit feedback. Following Hu, Koren & Volinsky, we model confidence as $c_{ui} = 1 + \alpha \log(1 + q_{ui})$."*
2. **Matrix Factorization (TruncatedSVD):**
   - *"We decompose our sparse $3,989 \times 3,610$ interaction matrix into $d=40$ latent factors using TruncatedSVD. We normalize user factors and item factors to unit length so their dot product acts like a cosine similarity."*
3. **Adaptive Hybrid Weighting:**
   - *"Rather than a fixed 50/50 blend, NEXUS dynamically modulates collaborative weight $\lambda$ based on customer interaction density: 0.65 for dense users, 0.25 for sparse users, and 0.0 for cold-start users."*
4. **Ranking Metrics over Classification Accuracy:**
   - *"We never evaluate recommenders on classification accuracy or MSE. Recommendation is a top-N ranking problem. We evaluate Precision@K, Recall@K, NDCG@K, Catalog Coverage, and Intra-List Diversity."*

---

## 3. 10-Minute Live Coding / Modification Task Rehearsal

Interviewers often ask: *"Can you modify the code right now to do X?"*  
Here are the 4 most likely tasks and exact instructions on how to solve them live:

### Scenario 1: "How would you add a new category to the catalog classifier?"
- **File:** `src/preprocessing/cleaner.py`
- **Location:** Line 32 (`CATEGORY_RULES` list)
- **Live Action:**
  ```python
  # Add a new tuple to CATEGORY_RULES:
  ("Garden & Outdoor", ["PLANT", "POT", "GARDEN", "FLOWER", "SEED", "WATERING"])
  ```
- **Explanation:** *"We simply append our new domain rule with associated semantic keywords. When `clean_data()` runs, descriptions matching these tokens map to 'Garden & Outdoor'."*

### Scenario 2: "Can you change the Serendipity Explore mode to be even more aggressive on novelty?"
- **File:** `src/recommendation/serendipity.py`
- **Location:** Line 32 (`MODE_WEIGHTS` dictionary)
- **Live Action:**
  ```python
  # Adjust weights for explore:
  "explore": {"relevance": 0.20, "novelty": 0.50, "diversity": 0.30}
  ```
- **Explanation:** *"We shift weight away from historical relevance (0.35 -> 0.20) into normalized novelty (0.35 -> 0.50). The MMR selection loop will immediately promote deeper long-tail items."*

### Scenario 3: "How do you add an onboarding category preference for a cold-start user in FastAPI?"
- **File:** `src/api/main.py`
- **Location:** Inside `recommend_products()`
- **Live Action:** Notice that `RecommendRequest` already has `category: Optional[str] = None`. Passing `category="Kitchen & Dining"` automatically passes it to `engine.recommend(..., category=req.category)` which triggers category-filtered popularity recommendations!

---

## 4. Realistic Technical Interview Q&A

### Q1: "Why did you choose NDCG instead of just Precision and Recall?"
> **Answer:**  
> *"Precision and Recall treat all positions in the top-K list equally. If a user buys item X, Precision gives the same reward whether X is ranked at position 1 or position 10.  
> In real e-commerce, users look at the top 3 items first. NDCG applies a logarithmic discount penalty $\frac{1}{\log_2(\text{rank}+1)}$ so that relevant recommendations near the top of the list contribute significantly more to the score than those at the bottom."*

### Q2: "What is future data leakage and how did you prevent it?"
> **Answer:**  
> *"Future data leakage occurs when information from future timestamps contaminates the training set. If you perform a random 80/20 split on interaction rows, an item purchased on December 1st could be in the test set while an item purchased on December 5th is in the training set. The model would be learning from the future to predict the past.  
> In NEXUS, we enforced a strict chronological split: Train is strictly prior to November 2, Validation is November 2 to 21, and Test is November 21 to December 9. We verify this with an automated test: `assert max(train_date) <= min(test_date)`."*

### Q3: "Why did Popularity Baseline have higher Precision@10 than Collaborative Filtering in your benchmark?"
> **Answer:**  
> *"This is the classic 'Popularity Trap' in recommender systems research. In any e-commerce dataset, a few best-selling products account for a huge percentage of all future transactions (our Pareto analysis showed 20% of products account for 66% of transactions).  
> Recommending the top 10 best-sellers will coincidentally hit future purchases for many users, yielding a high Precision. However, its **Catalog Coverage is only 0.26%**—it is completely useless for personalized discovery!  
> Collaborative filtering and our Hybrid model surface over **34% to 36% of the catalog**, uncovering personalized niches that a static popularity list could never discover."*

### Q4: "How does your Serendipity Mode avoid recommending irrelevant products?"
> **Answer:**  
> *"We do not perform unconstrained random exploration. First, the candidate pool is pre-filtered by our personalized model so that only high-affinity products enter the reranking stage.  
> Then, our multi-objective function balances three forces: model relevance, normalized self-information novelty ($-\log_2 P(i)$), and category redundancy penalties. This ensures exploration happens within the user's plausible affinity envelope rather than recommending random, irrelevant noise."*

### Q5: "How does NEXUS handle cold-start users and items?"
> **Answer:**  
> *"We use a 3-tier fallback architecture:
> 1. **New User (0 interactions):** The engine detects zero history and routes to category-aware or global trending popularity, explicitly flagging the response with `is_fallback: True`.
> 2. **Sparse User (1–4 items):** Pure collaborative filtering overfits on sparse data. The Adaptive Hybrid automatically lowers collaborative weight to 0.25 and relies on Content-Based TF-IDF semantics.
> 3. **New Item (0 purchases):** Collaborative filtering cannot represent items with zero interactions. Our Content-Based model indexes the product description and category, allowing it to be recommended immediately to users with matching taste vectors."*

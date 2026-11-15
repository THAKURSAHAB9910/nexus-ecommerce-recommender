"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Module: src.models.collaborative

Responsible for:
1. Collaborative Filtering via Matrix Factorization (Latent Factor Model / TruncatedSVD).
2. Implicit feedback modeling with log-scaled transaction confidence.
3. Item-Item collaborative neighborhood similarity and latent representation learning.
4. Collaborative signal attribution for the Recommendation Reasoning Lab.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NEXUS.CollaborativeModel")

class CollaborativeRecommender:
    """Matrix Factorization and Item-Item Collaborative Filtering engine."""

    def __init__(
        self,
        catalog_path: str = "data/processed/catalog.csv",
        n_factors: int = 40,
        alpha: float = 1.0,
        random_state: int = 42
    ):
        self.catalog_path = Path(catalog_path)
        self.n_factors = n_factors
        self.alpha = alpha
        self.random_state = random_state

        self.catalog_df: Optional[pd.DataFrame] = None
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}

        self.svd: Optional[TruncatedSVD] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_history: Dict[str, Set[str]] = {}
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "CollaborativeRecommender":
        """
        Builds sparse interaction matrix and computes latent user/item factors.
        """
        logger.info("Fitting CollaborativeRecommender with %d latent factors...", self.n_factors)
        if self.catalog_path.exists():
            self.catalog_df = pd.read_csv(self.catalog_path, dtype={"StockCode": str})

        # Ensure valid strings
        train_df = train_df.copy()
        train_df["CustomerID"] = train_df["CustomerID"].astype(str)
        train_df["StockCode"] = train_df["StockCode"].astype(str)

        # Vectorized aggregation of implicit interaction quantities
        logger.info("Aggregating implicit user-item confidence scores...")
        agg = (
            train_df.groupby(["CustomerID", "StockCode"], observed=True)["Quantity"]
            .sum()
            .reset_index()
        )
        agg["confidence"] = 1.0 + self.alpha * np.log1p(agg["Quantity"].clip(lower=0))

        # Build contiguous index mappings
        unique_users = sorted(agg["CustomerID"].unique())
        unique_items = sorted(agg["StockCode"].unique())

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for i, u in enumerate(unique_users)}
        self.item_to_idx = {item: i for i, item in enumerate(unique_items)}
        self.idx_to_item = {i: item for i, item in enumerate(unique_items)}

        # Record user purchase history for candidate filtering
        self.user_history = {}
        for row in agg.itertuples(index=False):
            u = str(row.CustomerID)
            item = str(row.StockCode)
            if u not in self.user_history:
                self.user_history[u] = set()
            self.user_history[u].add(item)

        user_indices = [self.user_to_idx[u] for u in agg["CustomerID"]]
        item_indices = [self.item_to_idx[item] for item in agg["StockCode"]]
        values = agg["confidence"].values

        M = len(unique_users)
        N = len(unique_items)
        logger.info("Constructing sparse interaction matrix R (%d users x %d items)...", M, N)
        R_sparse = csr_matrix((values, (user_indices, item_indices)), shape=(M, N), dtype=np.float32)

        # Decompose via TruncatedSVD (Latent Factor Matrix Factorization)
        logger.info("Decomposing matrix via TruncatedSVD...")
        self.svd = TruncatedSVD(n_components=self.n_factors, random_state=self.random_state)
        # user_factors = U * Sigma
        self.user_factors = self.svd.fit_transform(R_sparse)
        # item_factors = V
        self.item_factors = self.svd.components_.T

        # Normalize latent factors to facilitate cosine-like dot products
        user_norms = np.linalg.norm(self.user_factors, axis=1, keepdims=True)
        user_norms[user_norms == 0] = 1.0
        self.user_factors = self.user_factors / user_norms

        item_norms = np.linalg.norm(self.item_factors, axis=1, keepdims=True)
        item_norms[item_norms == 0] = 1.0
        self.item_factors = self.item_factors / item_norms

        explained_variance = float(np.sum(self.svd.explained_variance_ratio_)) * 100
        logger.info("Explained variance ratio with %d latent factors: %.2f%%", self.n_factors, explained_variance)

        self.is_fitted = True
        logger.info("CollaborativeRecommender successfully fitted.")
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_items: Optional[List[str]] = None,
        category: Optional[str] = None,
        include_explanations: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generates personalized collaborative filtering recommendations.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        u_str = str(user_id)
        if u_str not in self.user_to_idx:
            # Cold-start user
            return []

        u_idx = self.user_to_idx[u_str]
        u_vector = self.user_factors[u_idx]

        # Predicted preference: dot product between user vector and all item factors
        # Shape: (N,)
        raw_scores = self.item_factors.dot(u_vector)

        # Normalize scores to [0, 1] range
        min_s = float(np.min(raw_scores))
        max_s = float(np.max(raw_scores))
        norm_scores = (raw_scores - min_s) / (max_s - min_s + 1e-8)

        exclude_set = set(exclude_items) if exclude_items else set()
        if u_str in self.user_history:
            exclude_set.update(self.user_history[u_str])

        ranked_indices = np.argsort(norm_scores)[::-1]

        results = []
        for idx in ranked_indices:
            stock_code = self.idx_to_item[idx]
            if stock_code in exclude_set:
                continue

            score = float(norm_scores[idx])
            cat_row = self.catalog_df[self.catalog_df["StockCode"] == stock_code] if self.catalog_df is not None else pd.DataFrame()
            cat = cat_row["Category"].iloc[0] if len(cat_row) > 0 else "General"
            desc = cat_row["Description"].iloc[0] if len(cat_row) > 0 else "Unknown Item"
            price = float(cat_row["MedianUnitPrice"].iloc[0]) if len(cat_row) > 0 and "MedianUnitPrice" in cat_row.columns else 0.0

            if category and cat != category:
                continue

            explanation = self.explain_collaborative_signal(u_str, stock_code) if include_explanations else ""

            results.append({
                "item_id": stock_code,
                "description": desc,
                "category": cat,
                "unit_price": price,
                "score": round(score, 4),
                "model": "Collaborative_SVD",
                "explanation": explanation
            })

            if len(results) >= k:
                break

        return results

    def explain_collaborative_signal(self, user_id: str, item_id: str) -> str:
        """
        Finds the purchased item in the user's history that has highest latent factor
        cosine similarity to the recommended item.
        """
        u_str = str(user_id)
        if item_id not in self.item_to_idx or u_str not in self.user_history:
            return "Recommended based on community purchase patterns."

        target_idx = self.item_to_idx[item_id]
        target_factor = self.item_factors[target_idx]

        user_items = [it for it in self.user_history[u_str] if it in self.item_to_idx]
        if not user_items:
            return "Recommended based on community purchase patterns."

        user_indices = [self.item_to_idx[it] for it in user_items]
        user_matrix = self.item_factors[user_indices]
        sims = user_matrix.dot(target_factor)

        best_idx_pos = int(np.argmax(sims))
        best_user_item = user_items[best_idx_pos]
        best_sim = float(sims[best_idx_pos])

        item_desc = best_user_item
        if self.catalog_df is not None:
            r = self.catalog_df[self.catalog_df["StockCode"] == best_user_item]
            if len(r) > 0:
                item_desc = f"'{r['Description'].iloc[0]}'"

        if best_sim > 0.35:
            return f"Customers who bought {item_desc} also frequently bought this (Affinity: {round(best_sim, 2)})."
        return "Co-purchased by shoppers with similar browsing and purchase behavior."

    def save(self, filepath: str = "artifacts/collaborative_model.joblib"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Saved CollaborativeRecommender to: %s", filepath)

    @classmethod
    def load(cls, filepath: str = "artifacts/collaborative_model.joblib") -> "CollaborativeRecommender":
        logger.info("Loading CollaborativeRecommender from: %s", filepath)
        return joblib.load(filepath)

if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv", dtype={"StockCode": str, "CustomerID": str})
    model = CollaborativeRecommender(n_factors=40)
    model.fit(train_df)
    model.save()

    sample_user = str(train_df["CustomerID"].iloc[0])
    recs = model.recommend(user_id=sample_user, k=5)
    print(f"\n[Collaborative Recommendations for User {sample_user}]")
    for r in recs:
        print(f"  {r['item_id']} | {r['description']} | Cat: {r['category']} | Score: {r['score']} | {r['explanation']}")

"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Module: src.models.content_based

Responsible for:
1. Content-based recommendation using TF-IDF vectorization and Cosine Similarity.
2. Building dense user preference profiles from historical item interactions.
3. Item-to-Item similar product discovery (/recommend/similar).
4. Extracting shared semantic tokens for the Recommendation Reasoning Lab.
5. Addressing the new-item cold-start dilemma.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NEXUS.ContentModel")

class ContentBasedRecommender:
    """Recommender utilizing TF-IDF textual embeddings and cosine similarity."""

    def __init__(
        self,
        catalog_path: str = "data/processed/catalog.csv",
        max_features: int = 2500,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        self.catalog_path = Path(catalog_path)
        self.max_features = max_features
        self.ngram_range = ngram_range

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True
        )
        self.catalog_df: Optional[pd.DataFrame] = None
        self.tfidf_matrix: Optional[csr_matrix] = None
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        self.user_history: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Fits TF-IDF representations on catalog items and indexes user training histories.
        """
        logger.info("Fitting ContentBasedRecommender...")
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found at {self.catalog_path}")

        self.catalog_df = pd.read_csv(self.catalog_path, dtype={"StockCode": str})
        
        # Build rich text document for each item: Description + Category
        self.catalog_df["ContentDoc"] = (
            self.catalog_df["Description"].fillna("") + " " +
            self.catalog_df["Category"].fillna("")
        )

        logger.info("Computing TF-IDF matrix for %d catalog items...", len(self.catalog_df))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.catalog_df["ContentDoc"])

        # Mapping dictionaries
        self.item_to_idx = {row["StockCode"]: idx for idx, row in self.catalog_df.iterrows()}
        self.idx_to_item = {idx: row["StockCode"] for idx, row in self.catalog_df.iterrows()}

        # Build user interaction history from training data with vectorized aggregation
        logger.info("Indexing user training purchase histories (%d transactions)...", len(train_df))
        agg_history = (
            train_df.groupby(["CustomerID", "StockCode"], observed=True)["Quantity"]
            .sum()
            .reset_index()
        )
        agg_history["weight"] = np.log1p(agg_history["Quantity"].clip(lower=0))

        self.user_history = {}
        for row in agg_history.itertuples(index=False):
            u_str = str(row.CustomerID)
            if u_str not in self.user_history:
                self.user_history[u_str] = {}
            self.user_history[u_str][str(row.StockCode)] = float(row.weight)

        self.is_fitted = True
        logger.info("ContentBasedRecommender fitted successfully. Vocab size: %d terms.", len(self.vectorizer.vocabulary_))
        return self

    def build_user_profile(self, user_id: str) -> Optional[np.ndarray]:
        """
        Synthesizes a user preference vector by taking the weighted average of
        TF-IDF vectors of products purchased by the user in the training set.
        """
        history = self.user_history.get(str(user_id))
        if not history:
            return None

        user_vec = np.zeros(self.tfidf_matrix.shape[1], dtype=np.float32)
        total_weight = 0.0

        for stock_code, weight in history.items():
            if stock_code in self.item_to_idx:
                idx = self.item_to_idx[stock_code]
                item_vec = self.tfidf_matrix[idx].toarray().flatten()
                user_vec += weight * item_vec
                total_weight += weight

        if total_weight > 0:
            user_vec /= total_weight
            norm = np.linalg.norm(user_vec)
            if norm > 0:
                user_vec /= norm
            return user_vec
        return None

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_items: Optional[List[str]] = None,
        category: Optional[str] = None,
        include_explanations: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generates personalized content-based recommendations for a user.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        user_vec = self.build_user_profile(user_id)
        if user_vec is None:
            # Fallback for cold-start user
            return []

        # Cosine similarity between user profile and all items
        # user_vec shape: (D,), tfidf_matrix shape: (N, D)
        scores = self.tfidf_matrix.dot(user_vec)

        exclude_set = set(exclude_items) if exclude_items else set()
        # Also exclude items the user already purchased in train history
        if str(user_id) in self.user_history:
            exclude_set.update(self.user_history[str(user_id)].keys())

        # Rank items descending
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            stock_code = self.idx_to_item[idx]
            if stock_code in exclude_set:
                continue

            score_val = float(scores[idx])
            if score_val <= 0.0:
                break

            cat_row = self.catalog_df.iloc[idx]
            if category and cat_row["Category"] != category:
                continue

            explanation = self.explain_recommendation(user_id, stock_code) if include_explanations else ""

            results.append({
                "item_id": stock_code,
                "description": cat_row["Description"],
                "category": cat_row["Category"],
                "unit_price": float(cat_row["MedianUnitPrice"]),
                "score": round(score_val, 4),
                "model": "ContentBased_TFIDF",
                "explanation": explanation
            })

            if len(results) >= k:
                break

        return results

    def recommend_similar_items(
        self,
        item_id: str,
        k: int = 10,
        category_boost: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Discovers items semantically similar to a query product.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        if item_id not in self.item_to_idx:
            return []

        target_idx = self.item_to_idx[item_id]
        target_vec = self.tfidf_matrix[target_idx]
        target_cat = self.catalog_df.iloc[target_idx]["Category"]

        sim_scores = cosine_similarity(target_vec, self.tfidf_matrix).flatten()

        # Category boost: gently promote items from the same category
        if category_boost > 0:
            same_cat_mask = (self.catalog_df["Category"] == target_cat).values
            sim_scores[same_cat_mask] += category_boost

        ranked_indices = np.argsort(sim_scores)[::-1]

        results = []
        for idx in ranked_indices:
            stock_code = self.idx_to_item[idx]
            if stock_code == item_id:
                continue

            score = float(sim_scores[idx])
            cat_row = self.catalog_df.iloc[idx]

            results.append({
                "item_id": stock_code,
                "description": cat_row["Description"],
                "category": cat_row["Category"],
                "unit_price": float(cat_row["MedianUnitPrice"]),
                "score": round(score, 4),
                "model": "ContentBased_ItemItem",
                "explanation": f"High content similarity with {item_id} (Category: {cat_row['Category']})"
            })

            if len(results) >= k:
                break

        return results

    def explain_recommendation(self, user_id: str, item_id: str) -> str:
        """
        Decomposes the recommendation score into shared semantic tokens.
        """
        if item_id not in self.item_to_idx:
            return "Recommendation based on content similarity."

        user_vec = self.build_user_profile(user_id)
        if user_vec is None:
            return "Recommended based on overall catalog similarity."

        item_idx = self.item_to_idx[item_id]
        item_vec = self.tfidf_matrix[item_idx].toarray().flatten()

        # Product of user weights and item weights gives token contribution
        token_contributions = user_vec * item_vec
        feature_names = self.vectorizer.get_feature_names_out()

        top_token_indices = np.argsort(token_contributions)[::-1]
        top_tokens = [feature_names[i] for i in top_token_indices[:4] if token_contributions[i] > 0]

        if top_tokens:
            token_str = ", ".join(f"'{t.upper()}'" for t in top_tokens)
            return f"Matches your taste for {token_str} products."
        return "Matches attributes of items you previously purchased."

    def save(self, filepath: str = "artifacts/content_model.joblib"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Saved ContentBasedRecommender to: %s", filepath)

    @classmethod
    def load(cls, filepath: str = "artifacts/content_model.joblib") -> "ContentBasedRecommender":
        logger.info("Loading ContentBasedRecommender from: %s", filepath)
        return joblib.load(filepath)

if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv", dtype={"StockCode": str, "CustomerID": str})
    model = ContentBasedRecommender()
    model.fit(train_df)
    model.save()

    # Pick an active user from train_df
    sample_user = str(train_df["CustomerID"].iloc[0])
    recs = model.recommend(user_id=sample_user, k=5)
    print(f"\n[Content-Based Recommendations for User {sample_user}]")
    for r in recs:
        print(f"  {r['item_id']} | {r['description']} | Cat: {r['category']} | Score: {r['score']} | {r['explanation']}")

    # Test item-to-item similarity
    sample_item = "22423"  # REGENCY CAKESTAND 3 TIER
    sims = model.recommend_similar_items(sample_item, k=3)
    print(f"\n[Similar Items to {sample_item}]")
    for s in sims:
        print(f"  {s['item_id']} | {s['description']} | Cat: {s['category']} | Score: {s['score']}")

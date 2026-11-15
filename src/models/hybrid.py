"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Module: src.models.hybrid

Responsible for:
1. Adaptive Weighted Hybrid Recommendation (Collaborative + Content-Based).
2. Dynamic sparsity-based weighting (adapting CF weight lambda based on interaction density).
3. Intelligent Cold-Start fallback routing (New Users -> Popularity; New Items -> Content).
4. Unifying multi-modal recommendation signals with transparent attribution.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib

from src.models.popularity import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NEXUS.HybridModel")

class HybridRecommender:
    """Adaptive Hybrid Recommender integrating Collaborative and Content signals."""

    def __init__(
        self,
        popularity_model: Optional[PopularityRecommender] = None,
        content_model: Optional[ContentBasedRecommender] = None,
        collaborative_model: Optional[CollaborativeRecommender] = None,
        default_cf_weight: float = 0.65
    ):
        self.pop_model = popularity_model
        self.content_model = content_model
        self.cf_model = collaborative_model
        self.default_cf_weight = default_cf_weight
        self.user_interaction_counts: Dict[str, int] = {}
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "HybridRecommender":
        """
        Fits all constituent sub-models or links pre-fitted models, and computes user sparsity profiles.
        """
        logger.info("Initializing HybridRecommender fitting pipeline...")
        train_df["CustomerID"] = train_df["CustomerID"].astype(str)

        # Compute interaction volume per user for adaptive weighting
        user_counts = train_df.groupby("CustomerID")["StockCode"].count().to_dict()
        self.user_interaction_counts = {str(k): int(v) for k, v in user_counts.items()}

        if self.pop_model is None:
            self.pop_model = PopularityRecommender()
            self.pop_model.fit(train_df)

        if self.content_model is None:
            self.content_model = ContentBasedRecommender()
            self.content_model.fit(train_df)

        if self.cf_model is None:
            self.cf_model = CollaborativeRecommender()
            self.cf_model.fit(train_df)

        self.is_fitted = True
        logger.info("HybridRecommender fully fitted and ready for inference.")
        return self

    def _determine_cf_weight(self, user_id: str) -> float:
        """
        Dynamically adjusts collaborative filtering weight lambda:
        - Dense users (> 15 interactions): lambda = 0.70 (collaborative community patterns shine)
        - Moderate users (5-15 interactions): lambda = 0.50
        - Sparse users (1-4 interactions): lambda = 0.25 (content semantics prevent noisy CF)
        - Zero history (cold start): lambda = 0.0 (fallback to popularity)
        """
        count = self.user_interaction_counts.get(str(user_id), 0)
        if count == 0:
            return 0.0
        elif count <= 4:
            return 0.25
        elif count <= 15:
            return 0.50
        else:
            return self.default_cf_weight

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        exclude_items: Optional[List[str]] = None,
        category: Optional[str] = None,
        include_explanations: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generates hybrid recommendations with cold-start fallbacks.
        """
        if not self.is_fitted:
            raise RuntimeError("Hybrid model is not fitted.")

        u_str = str(user_id)
        user_history_count = self.user_interaction_counts.get(u_str, 0)

        # 1. COLD-START USER HANDLING
        if user_history_count == 0 or u_str not in self.cf_model.user_to_idx:
            fallback_recs = self.pop_model.recommend(
                user_id=u_str,
                k=k,
                exclude_items=exclude_items,
                category=category
            )
            for r in fallback_recs:
                r["model"] = "Hybrid_ColdStartFallback_Popularity"
                r["is_fallback"] = True
                r["fallback_reason"] = "New customer with zero purchase history. Surfacing trending catalog items."
            return fallback_recs

        # 2. WARM & SPARSE USERS: HYBRID BLENDING
        cf_weight = self._determine_cf_weight(u_str)
        content_weight = 1.0 - cf_weight

        # Retrieve top candidates from both models
        candidate_pool_size = max(k * 3, 25)
        cf_candidates = self.cf_model.recommend(
            user_id=u_str,
            k=candidate_pool_size,
            exclude_items=exclude_items,
            category=category,
            include_explanations=include_explanations
        )
        content_candidates = self.content_model.recommend(
            user_id=u_str,
            k=candidate_pool_size,
            exclude_items=exclude_items,
            category=category,
            include_explanations=include_explanations
        )

        cf_scores = {r["item_id"]: r["score"] for r in cf_candidates}
        content_scores = {r["item_id"]: r["score"] for r in content_candidates}
        cf_explanations = {r["item_id"]: r.get("explanation", "") for r in cf_candidates}
        content_explanations = {r["item_id"]: r.get("explanation", "") for r in content_candidates}

        all_item_ids = set(cf_scores.keys()).union(set(content_scores.keys()))

        blended_results = []
        for stock_code in all_item_ids:
            s_cf = cf_scores.get(stock_code, 0.0)
            s_content = content_scores.get(stock_code, 0.0)
            
            # Harmonic/Weighted blending
            final_score = (cf_weight * s_cf) + (content_weight * s_content)

            # Metadata lookup
            cat_row = self.pop_model.catalog_df[self.pop_model.catalog_df["StockCode"] == stock_code]
            desc = cat_row["Description"].iloc[0] if len(cat_row) > 0 else "Product"
            cat = cat_row["Category"].iloc[0] if len(cat_row) > 0 else "General"
            price = float(cat_row["MedianUnitPrice"].iloc[0]) if len(cat_row) > 0 and "MedianUnitPrice" in cat_row.columns else 0.0

            if include_explanations:
                exp_cf = cf_explanations.get(stock_code, "")
                exp_cnt = content_explanations.get(stock_code, "")
                if s_cf > 0.4 and s_content > 0.2:
                    explanation = f"Strong dual match: {exp_cnt} Plus {exp_cf}"
                elif s_cf >= s_content:
                    explanation = exp_cf if exp_cf else "High collaborative similarity with your shopping habits."
                else:
                    explanation = exp_cnt if exp_cnt else "Matches the style and category of your prior purchases."
            else:
                explanation = ""

            blended_results.append({
                "item_id": stock_code,
                "description": desc,
                "category": cat,
                "unit_price": price,
                "score": round(final_score, 4),
                "cf_score": round(s_cf, 4),
                "content_score": round(s_content, 4),
                "cf_weight": round(cf_weight, 2),
                "content_weight": round(content_weight, 2),
                "model": "AdaptiveHybrid",
                "is_fallback": False,
                "explanation": explanation
            })

        # Sort descending by hybrid score
        blended_results.sort(key=lambda x: x["score"], reverse=True)
        return blended_results[:k]

    def save(self, filepath: str = "artifacts/hybrid_model.joblib"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Saved HybridRecommender to: %s", filepath)

    @classmethod
    def load(cls, filepath: str = "artifacts/hybrid_model.joblib") -> "HybridRecommender":
        logger.info("Loading HybridRecommender from: %s", filepath)
        return joblib.load(filepath)

if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv", dtype={"StockCode": str, "CustomerID": str})
    
    # Load submodels
    pop_m = PopularityRecommender.load("artifacts/popularity_model.joblib")
    cnt_m = ContentBasedRecommender.load("artifacts/content_model.joblib")
    cf_m = CollaborativeRecommender.load("artifacts/collaborative_model.joblib")

    hybrid = HybridRecommender(
        popularity_model=pop_m,
        content_model=cnt_m,
        collaborative_model=cf_m
    )
    hybrid.fit(train_df)
    hybrid.save()

    # Test warm user
    warm_user = "17850"
    recs = hybrid.recommend(user_id=warm_user, k=4)
    print(f"\n[Hybrid Recommendations for Warm User {warm_user}]")
    for r in recs:
        print(f"  {r['item_id']} | {r['description']} | Score: {r['score']} (CF: {r['cf_score']}, Content: {r['content_score']})")
        print(f"    Explanation: {r['explanation']}")

    # Test cold-start user
    cold_user = "9999999"  # Unknown user
    recs_cold = hybrid.recommend(user_id=cold_user, k=3)
    print(f"\n[Hybrid Recommendations for Cold-Start User {cold_user}]")
    for r in recs_cold:
        print(f"  {r['item_id']} | {r['description']} | Score: {r['score']} | Fallback: {r['is_fallback']}")

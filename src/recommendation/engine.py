"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Module: src.recommendation.engine

Responsible for:
1. Unified Facade orchestrating inference across all NEXUS modules.
2. Model routing: Popularity, Content-Based, Collaborative, Adaptive Hybrid.
3. Incorporating Serendipity Mode (Familiar, Balanced, Explore).
4. Running the Recommendation Reasoning Lab on demand.
5. Graceful cold-start detection and fallback management.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from src.models.popularity import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.hybrid import HybridRecommender
from src.recommendation.serendipity import SerendipityEngine
from src.explanations.reasoning_lab import RecommendationReasoningLab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NEXUS.Engine")

class NexusRecommenderEngine:
    """Master engine delivering real-time recommendations, serendipity reranking, and reasoning."""

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        catalog_csv: str = "data/processed/catalog.csv",
        train_csv: str = "data/processed/train.csv"
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.catalog_path = Path(catalog_csv)
        self.train_path = Path(train_csv)

        logger.info("Initializing NEXUS Recommender Engine...")
        self.pop_model = PopularityRecommender.load(str(self.artifacts_dir / "popularity_model.joblib"))
        self.content_model = ContentBasedRecommender.load(str(self.artifacts_dir / "content_model.joblib"))
        self.cf_model = CollaborativeRecommender.load(str(self.artifacts_dir / "collaborative_model.joblib"))
        self.hybrid_model = HybridRecommender.load(str(self.artifacts_dir / "hybrid_model.joblib"))

        self.serendipity_engine = SerendipityEngine(catalog_path=str(self.catalog_path), train_path=str(self.train_path))
        self.reasoning_lab = RecommendationReasoningLab(catalog_path=str(self.catalog_path), train_path=str(self.train_path))
        self.catalog_df = pd.read_csv(self.catalog_path, dtype={"StockCode": str})

        logger.info("NEXUS Recommender Engine initialized successfully and ready for inference.")

    def recommend(
        self,
        user_id: str,
        model_type: str = "hybrid",
        serendipity_mode: str = "balanced",
        k: int = 10,
        category: Optional[str] = None,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Main recommendation query endpoint.
        """
        model_type = model_type.lower().strip()
        u_str = str(user_id)

        # 1. Candidate Retrieval based on selected model
        candidate_k = max(k * 3, 30)

        if model_type == "popularity":
            raw_recs = self.pop_model.recommend(user_id=u_str, k=candidate_k, category=category)
        elif model_type == "content":
            raw_recs = self.content_model.recommend(user_id=u_str, k=candidate_k, category=category)
            if not raw_recs:  # Cold start fallback
                raw_recs = self.pop_model.recommend(user_id=u_str, k=candidate_k, category=category)
                for r in raw_recs:
                    r["is_fallback"] = True
                    r["fallback_reason"] = "Cold-start user without historical content profile. Defaulted to trending items."
        elif model_type == "collaborative":
            raw_recs = self.cf_model.recommend(user_id=u_str, k=candidate_k, category=category)
            if not raw_recs:  # Cold start fallback
                raw_recs = self.pop_model.recommend(user_id=u_str, k=candidate_k, category=category)
                for r in raw_recs:
                    r["is_fallback"] = True
                    r["fallback_reason"] = "Cold-start user without collaborative history. Defaulted to trending items."
        else:  # "hybrid" (default)
            raw_recs = self.hybrid_model.recommend(user_id=u_str, k=candidate_k, category=category)

        # 2. Apply Serendipity Reranking
        final_recs = self.serendipity_engine.rerank(raw_recs, mode=serendipity_mode, k=k)

        # 3. Add Reasoning Lab Attributions if requested
        if explain:
            for item in final_recs:
                item["reasoning_audit"] = self.reasoning_lab.explain_item_recommendation(
                    user_id=u_str,
                    item_id=item["item_id"],
                    recommendation_item=item,
                    content_model=self.content_model,
                    cf_model=self.cf_model
                )

        is_cold_start = bool(len(raw_recs) > 0 and raw_recs[0].get("is_fallback", False))
        user_history_count = len(self.reasoning_lab.user_history.get(u_str, set()))

        return {
            "user_id": u_str,
            "user_status": "COLD_START" if is_cold_start or user_history_count == 0 else "ACTIVE_WARM",
            "historical_interaction_count": user_history_count,
            "model_type": model_type,
            "serendipity_mode": serendipity_mode.upper(),
            "count": len(final_recs),
            "recommendations": final_recs
        }

    def recommend_similar(self, item_id: str, k: int = 10) -> Dict[str, Any]:
        """Recommends products similar to query item."""
        sims = self.content_model.recommend_similar_items(str(item_id), k=k)
        item_row = self.catalog_df[self.catalog_df["StockCode"] == str(item_id)]
        item_name = item_row["Description"].iloc[0] if len(item_row) > 0 else "Unknown Item"
        item_cat = item_row["Category"].iloc[0] if len(item_row) > 0 else "General"

        return {
            "query_item_id": str(item_id),
            "query_item_name": item_name,
            "query_category": item_cat,
            "count": len(sims),
            "similar_items": sims
        }

    def get_product(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Returns product catalog information."""
        item_row = self.catalog_df[self.catalog_df["StockCode"] == str(item_id)]
        if len(item_row) == 0:
            return None
        r = item_row.iloc[0]
        pop_info = self.reasoning_lab.item_stats.get(str(item_id), {})
        return {
            "item_id": str(item_id),
            "description": str(r["Description"]),
            "category": str(r["Category"]),
            "median_unit_price": float(r.get("MedianUnitPrice", 0.0)),
            "popularity_rank": pop_info.get("popularity_rank", "N/A"),
            "popularity_percentile": pop_info.get("popularity_percentile", "N/A")
        }

if __name__ == "__main__":
    engine = NexusRecommenderEngine()
    res = engine.recommend(user_id="17850", model_type="hybrid", serendipity_mode="explore", k=3, explain=True)
    print("\n[Unified Engine Output - EXPLORE Mode with Reasoning Lab Audit]")
    print(f"User: {res['user_id']} ({res['user_status']}) | Model: {res['model_type']} | Mode: {res['serendipity_mode']}")
    for r in res["recommendations"]:
        print(f"\nItem: {r['item_id']} - {r['description']} (Category: {r['category']})")
        print(f"  Score: {r['composite_score']} | Novelty: {r['novelty_score']}")
        print(f"  Audit: {r['reasoning_audit']['signals']['collaborative_filtering']['detail']}")
        print(f"  Audit: {r['reasoning_audit']['signals']['content_semantics']['detail']}")

"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Module: src.explanations.reasoning_lab

Responsible for:
1. Signature Feature 2: Recommendation Reasoning Lab.
2. Grounded decomposition of why an item was recommended:
   - Collaborative Interaction Signal (Affinity to specific historical items).
   - Content / Semantic Signal (Shared TF-IDF token keywords).
   - Category & Metadata Alignment (Category affinity, customer spending habits).
   - Catalog Popularity & Novelty Signal (Global sales rank, percentile, head vs. long-tail).
   - Composite Ranking Breakdown (How model weights or serendipity modes formed the final score).
3. Clear distinction between model scores and probabilities, and explicit handling of unavailable signals.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NEXUS.ReasoningLab")

class RecommendationReasoningLab:
    """Explains and decomposes recommendation signals into transparent, inspectable evidence."""

    def __init__(
        self,
        catalog_path: str = "data/processed/catalog.csv",
        train_path: str = "data/processed/train.csv"
    ):
        self.catalog_df = pd.read_csv(catalog_path, dtype={"StockCode": str})
        self.train_df = pd.read_csv(train_path, dtype={"StockCode": str, "CustomerID": str})

        # Precompute item sales stats
        item_sales = self.train_df.groupby("StockCode").agg(
            sales_volume=("Quantity", "sum"),
            unique_buyers=("CustomerID", "nunique")
        ).reset_index()

        item_sales = item_sales.sort_values("unique_buyers", ascending=False).reset_index(drop=True)
        item_sales["popularity_rank"] = item_sales.index + 1
        total_items = len(item_sales)
        item_sales["popularity_percentile"] = round((1.0 - (item_sales["popularity_rank"] / total_items)) * 100, 1)

        self.item_stats = dict(zip(item_sales["StockCode"], item_sales.to_dict(orient="records")))
        self.catalog_map = dict(zip(self.catalog_df["StockCode"], self.catalog_df.to_dict(orient="records")))

        # User purchase history index
        logger.info("Indexing user transaction histories for Reasoning Lab...")
        self.user_history = (
            self.train_df.groupby("CustomerID")["StockCode"]
            .apply(set)
            .to_dict()
        )
        self.user_cat_spend = (
            self.train_df.groupby(["CustomerID", "Category"])["Quantity"]
            .sum()
            .to_dict()
        )

    def explain_item_recommendation(
        self,
        user_id: str,
        item_id: str,
        recommendation_item: Dict[str, Any],
        content_model = None,
        cf_model = None
    ) -> Dict[str, Any]:
        """
        Decomposes the recommendation into grounded evidentiary signals.
        """
        u_str = str(user_id)
        stock_code = str(item_id)

        # 1. Product Metadata
        prod_meta = self.catalog_map.get(stock_code, {
            "Description": recommendation_item.get("description", "Unknown Product"),
            "Category": recommendation_item.get("category", "General"),
            "MedianUnitPrice": recommendation_item.get("unit_price", 0.0)
        })
        category = prod_meta["Category"]

        # 2. Collaborative Interaction Signal
        cf_evidence = {"available": False, "signal_score": None, "source_product": None, "detail": "Signal unavailable."}
        if cf_model is not None and hasattr(cf_model, "item_factors") and cf_model.is_fitted:
            if u_str in cf_model.user_to_idx and stock_code in cf_model.item_to_idx:
                u_idx = cf_model.user_to_idx[u_str]
                target_idx = cf_model.item_to_idx[stock_code]
                
                # Raw dot product
                raw_dot = float(cf_model.item_factors[target_idx].dot(cf_model.user_factors[u_idx]))
                
                # Best matching purchased item in user history
                user_items = [it for it in self.user_history.get(u_str, set()) if it in cf_model.item_to_idx]
                if user_items:
                    u_indices = [cf_model.item_to_idx[it] for it in user_items]
                    sims = cf_model.item_factors[u_indices].dot(cf_model.item_factors[target_idx])
                    best_pos = int(np.argmax(sims))
                    best_item = user_items[best_pos]
                    best_sim = float(sims[best_pos])
                    best_desc = self.catalog_map.get(best_item, {}).get("Description", best_item)

                    cf_evidence = {
                        "available": True,
                        "signal_score": round(max(0.0, best_sim), 4),
                        "source_product": f"{best_item} - {best_desc}",
                        "detail": (
                            f"Shoppers who purchased '{best_desc}' frequently bought this item "
                            f"(Latent Factor Cosine Affinity: {round(best_sim, 2)})."
                        )
                    }

        # 3. Content / Semantic Match Signal
        content_evidence = {"available": False, "signal_score": None, "shared_keywords": [], "detail": "Signal unavailable."}
        if content_model is not None and hasattr(content_model, "tfidf_matrix") and content_model.is_fitted:
            if stock_code in content_model.item_to_idx:
                user_vec = content_model.build_user_profile(u_str)
                if user_vec is not None:
                    item_idx = content_model.item_to_idx[stock_code]
                    item_vec = content_model.tfidf_matrix[item_idx].toarray().flatten()
                    
                    contributions = user_vec * item_vec
                    names = content_model.vectorizer.get_feature_names_out()
                    top_idx = np.argsort(contributions)[::-1]
                    top_kw = [names[i].upper() for i in top_idx[:4] if contributions[i] > 0]
                    content_sim = float(np.dot(user_vec, item_vec))

                    content_evidence = {
                        "available": True,
                        "signal_score": round(content_sim, 4),
                        "shared_keywords": top_kw,
                        "detail": (
                            f"Semantic overlap on key terms {top_kw} matching your historical purchase descriptions "
                            f"(TF-IDF Cosine Similarity: {round(content_sim, 4)})."
                            if top_kw else "Semantic profile matches product category and description."
                        )
                    }

        # 4. Category & Behavioral Alignment
        user_cat_units = self.user_cat_spend.get((u_str, category), 0)
        total_user_units = sum(v for k, v in self.user_cat_spend.items() if k[0] == u_str)
        cat_share = round((user_cat_units / max(total_user_units, 1)) * 100, 1)

        category_evidence = {
            "category": category,
            "user_past_units_in_category": int(user_cat_units),
            "user_category_affinity_pct": cat_share,
            "detail": (
                f"You have purchased {user_cat_units} items in '{category}' "
                f"({cat_share}% of your shopping history)."
                if user_cat_units > 0 else
                f"'{category}' is a discovery category outside your frequent past purchases."
            )
        }

        # 5. Popularity & Novelty Signal
        pop_row = self.item_stats.get(stock_code, {})
        rank = pop_row.get("popularity_rank", "N/A")
        pct = pop_row.get("popularity_percentile", "N/A")
        tier = "Best-Seller (Head)" if (isinstance(pct, (int, float)) and pct >= 80) else (
            "Mid-Tier Product" if (isinstance(pct, (int, float)) and pct >= 40) else "Long-Tail Discovery"
        )

        popularity_evidence = {
            "global_sales_rank": rank,
            "popularity_percentile": pct,
            "catalog_tier": tier,
            "detail": f"Catalog popularity rank #{rank} ({tier}, Top {100-pct if isinstance(pct, (int, float)) else 'N/A'}% of catalog)."
        }

        # 6. Composite Decision Summary
        model_name = recommendation_item.get("model", "NEXUS")
        score = recommendation_item.get("score", recommendation_item.get("composite_score", 0.0))
        is_fallback = recommendation_item.get("is_fallback", False)

        decision_summary = {
            "model_type": model_name,
            "model_score": score,
            "is_probability": False,
            "score_type": "Ordinal Normalized Ranking Score (Higher is better, not a calibrated probability)",
            "is_cold_start_fallback": is_fallback,
            "serendipity_mode": recommendation_item.get("serendipity_mode", "STANDARD"),
            "primary_rationale": recommendation_item.get("explanation", recommendation_item.get("serendipity_rationale", "Recommendation supported by collaborative and content signals."))
        }

        return {
            "product_id": stock_code,
            "product_name": prod_meta["Description"],
            "unit_price": prod_meta.get("MedianUnitPrice", 0.0),
            "category": category,
            "decision_summary": decision_summary,
            "signals": {
                "collaborative_filtering": cf_evidence,
                "content_semantics": content_evidence,
                "category_alignment": category_evidence,
                "catalog_popularity": popularity_evidence
            }
        }

if __name__ == "__main__":
    from src.models.hybrid import HybridRecommender
    from src.models.content_based import ContentBasedRecommender
    from src.models.collaborative import CollaborativeRecommender

    lab = RecommendationReasoningLab()
    cnt_m = ContentBasedRecommender.load("artifacts/content_model.joblib")
    cf_m = CollaborativeRecommender.load("artifacts/collaborative_model.joblib")
    hybrid = HybridRecommender.load("artifacts/hybrid_model.joblib")

    test_user = "17850"
    recs = hybrid.recommend(user_id=test_user, k=2)

    for r in recs:
        explanation = lab.explain_item_recommendation(
            user_id=test_user,
            item_id=r["item_id"],
            recommendation_item=r,
            content_model=cnt_m,
            cf_model=cf_m
        )
        print(f"\n================ REASONING LAB AUDIT FOR {r['item_id']} ================")
        print(f"Product: {explanation['product_name']} | Category: {explanation['category']}")
        print(f"Score: {explanation['decision_summary']['model_score']} ({explanation['decision_summary']['score_type']})")
        print("\nGround Truth Signals:")
        for sig_name, sig_data in explanation["signals"].items():
            print(f"  [{sig_name.upper()}]: {sig_data['detail']}")

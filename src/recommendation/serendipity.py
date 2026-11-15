"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Module: src.recommendation.serendipity

Responsible for:
1. Signature Feature 1: Serendipity Mode (Exploration-oriented ranking controller).
2. Implementing three distinct strategies:
   - FAMILIAR: Pure exploitation, maximizing similarity to historical purchases.
   - BALANCED: Multi-objective balance of relevance, category diversity, and novelty.
   - EXPLORE: Surfaces relevant long-tail catalog discoveries beyond user echo chambers.
3. Utilizing Maximal Marginal Relevance (MMR) and novelty weighting on actual model scores.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NEXUS.Serendipity")

class SerendipityEngine:
    """Reranking engine implementing Familiar, Balanced, and Explore recommendation strategies."""

    MODE_WEIGHTS = {
        "familiar": {"relevance": 0.85, "novelty": 0.05, "diversity": 0.10},
        "balanced": {"relevance": 0.55, "novelty": 0.20, "diversity": 0.25},
        "explore":  {"relevance": 0.35, "novelty": 0.35, "diversity": 0.30},
    }

    def __init__(
        self,
        catalog_path: str = "data/processed/catalog.csv",
        train_path: str = "data/processed/train.csv"
    ):
        self.catalog_df = pd.read_csv(catalog_path, dtype={"StockCode": str})
        train_df = pd.read_csv(train_path, dtype={"StockCode": str})

        # Calculate normalized item novelty scores from training frequencies
        item_counts = train_df["StockCode"].value_counts()
        total_events = len(train_df)
        
        # Self-information in bits: -log2(P(i))
        raw_novelty = {}
        for item, count in item_counts.items():
            p_i = count / total_events
            raw_novelty[item] = -np.log2(max(p_i, 1e-6))

        # Default for unobserved/rare items
        max_nov = max(raw_novelty.values()) if raw_novelty else 15.0
        min_nov = min(raw_novelty.values()) if raw_novelty else 5.0
        denom = max(max_nov - min_nov, 1e-4)

        # Min-Max normalize novelty to [0, 1]
        self.item_novelty_norm: Dict[str, float] = {
            item: float((val - min_nov) / denom)
            for item, val in raw_novelty.items()
        }
        self.default_novelty = 1.0  # Cold/unseen items have maximum novelty

        # Item category mapping
        self.item_category = dict(zip(self.catalog_df["StockCode"], self.catalog_df["Category"]))

    def rerank(
        self,
        candidate_items: List[Dict[str, Any]],
        mode: str = "balanced",
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Reranks candidate recommendations using the specified Serendipity Mode.
        
        Args:
            candidate_items: Pre-scored candidate recommendations from baseline/hybrid model.
            mode: 'familiar' | 'balanced' | 'explore'.
            k: Number of items to return.
            
        Returns:
            Reranked list of recommendations with score breakdown and rationale.
        """
        mode = mode.lower().strip()
        if mode not in self.MODE_WEIGHTS:
            logger.warning("Unrecognized mode '%s', defaulting to 'balanced'.", mode)
            mode = "balanced"

        weights = self.MODE_WEIGHTS[mode]
        w_rel = weights["relevance"]
        w_nov = weights["novelty"]
        w_div = weights["diversity"]

        if not candidate_items:
            return []

        # Sequential greedy selection (Maximal Marginal Relevance style)
        selected: List[Dict[str, Any]] = []
        selected_ids = set()
        selected_categories: Dict[str, int] = {}
        remaining = list(candidate_items)

        while len(selected) < min(k, len(candidate_items)) and remaining:
            best_idx = -1
            best_score = -float("inf")
            best_components = {}

            for idx, candidate in enumerate(remaining):
                stock_code = candidate["item_id"]
                rel_score = float(candidate.get("score", 0.0))
                cat = self.item_category.get(stock_code, "General")

                # 1. Relevance component (from underlying model)
                comp_rel = rel_score

                # 2. Novelty component (promotes long-tail / less-obvious items)
                comp_nov = self.item_novelty_norm.get(stock_code, self.default_novelty)

                # 3. Category Diversity component: penalize already-selected categories
                cat_count = selected_categories.get(cat, 0)
                comp_div = 1.0 / (1.0 + cat_count)  # 1.0 for new category, 0.5 for 2nd item, etc.

                # Combined multi-objective score
                composite_score = (w_rel * comp_rel) + (w_nov * comp_nov) + (w_div * comp_div)

                if composite_score > best_score:
                    best_score = composite_score
                    best_idx = idx
                    best_components = {
                        "relevance_contrib": round(w_rel * comp_rel, 4),
                        "novelty_contrib": round(w_nov * comp_nov, 4),
                        "diversity_contrib": round(w_div * comp_div, 4),
                        "composite_score": round(composite_score, 4),
                        "novelty_score": round(comp_nov, 4)
                    }

            if best_idx >= 0:
                chosen = remaining.pop(best_idx)
                chosen_cat = self.item_category.get(chosen["item_id"], "General")
                selected_categories[chosen_cat] = selected_categories.get(chosen_cat, 0) + 1
                selected_ids.add(chosen["item_id"])

                # Augment item with serendipity metadata
                enriched_item = dict(chosen)
                enriched_item["serendipity_mode"] = mode.upper()
                enriched_item["original_model_score"] = chosen.get("score", 0.0)
                enriched_item["composite_score"] = best_components["composite_score"]
                enriched_item["novelty_score"] = best_components["novelty_score"]
                enriched_item["score_components"] = best_components
                
                # Contextual explanation for the mode
                if mode == "familiar":
                    enriched_item["serendipity_rationale"] = (
                        "Prioritized for maximum historical affinity and proven preference."
                    )
                elif mode == "balanced":
                    enriched_item["serendipity_rationale"] = (
                        "Equilibrates personalized relevance with cross-category discovery."
                    )
                else:  # explore
                    enriched_item["serendipity_rationale"] = (
                        f"Long-tail discovery gem (Novelty: {best_components['novelty_score']:.2f}) "
                        f"elevated while preserving high domain relevance."
                    )

                selected.append(enriched_item)

        return selected

if __name__ == "__main__":
    from src.models.hybrid import HybridRecommender
    hybrid = HybridRecommender.load("artifacts/hybrid_model.joblib")
    serendipity = SerendipityEngine()

    test_user = "17850"
    base_recs = hybrid.recommend(user_id=test_user, k=25)

    print(f"\n[Serendipity Mode Comparison for User {test_user}]")
    for mode in ["familiar", "balanced", "explore"]:
        mode_recs = serendipity.rerank(base_recs, mode=mode, k=4)
        print(f"\n--- Strategy: {mode.upper()} ---")
        for r in mode_recs:
            print(f"  {r['item_id']} | {r['description']} | Cat: {r['category']} | FinalScore: {r['composite_score']} | Nov: {r['novelty_score']}")
            print(f"    Rationale: {r['serendipity_rationale']}")

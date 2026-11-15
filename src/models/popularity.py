"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Module: src.models.popularity

Responsible for:
1. Non-personalized popularity-based recommendation baseline.
2. Ranking products by interaction frequency and total volume sold strictly within training data.
3. Category-level popularity fallback for onboarding / category cold-start.
4. Serving as the primary benchmark against which personalized models are evaluated.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NEXUS.PopularityModel")

class PopularityRecommender:
    """Non-personalized baseline recommending top products ranked by training popularity."""

    def __init__(self, catalog_path: str = "data/processed/catalog.csv"):
        self.catalog_path = Path(catalog_path)
        self.catalog_df: Optional[pd.DataFrame] = None
        self.popular_items: List[str] = []
        self.popularity_scores: Dict[str, float] = {}
        self.category_popular_items: Dict[str, List[str]] = {}
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        """
        Fits the popularity model strictly on training data to prevent leakage.
        
        Args:
            train_df: DataFrame containing historical training transactions.
        """
        logger.info("Fitting PopularityRecommender on %d training transactions...", len(train_df))
        
        if self.catalog_path.exists():
            self.catalog_df = pd.read_csv(self.catalog_path, dtype={"StockCode": str})
        else:
            logger.warning("Catalog not found at %s. Creating minimal catalog from train_df.", self.catalog_path)
            self.catalog_df = train_df[["StockCode", "Description", "Category"]].drop_duplicates("StockCode")

        # Rank items by unique customer purchases + transaction volume
        # Hybrid popularity metric: log(quantity + 1) * count of transactions
        item_stats = train_df.groupby("StockCode").agg(
            transaction_count=("InvoiceNo", "nunique"),
            total_quantity=("Quantity", "sum"),
            unique_customers=("CustomerID", "nunique")
        ).reset_index()

        # Popularity score: harmonic combination of buyer reach and purchase frequency
        # Normalizing to [0, 1] range for fair comparison with personalized models
        max_reach = item_stats["unique_customers"].max()
        item_stats["pop_score"] = item_stats["unique_customers"] / max_reach
        item_stats = item_stats.sort_values("pop_score", ascending=False).reset_index(drop=True)

        self.popular_items = item_stats["StockCode"].tolist()
        self.popularity_scores = dict(zip(item_stats["StockCode"], item_stats["pop_score"]))

        # Also group by Category for category-aware fallback
        merged = item_stats.merge(self.catalog_df[["StockCode", "Category"]], on="StockCode", how="left")
        self.category_popular_items = {}
        for cat, group in merged.groupby("Category"):
            self.category_popular_items[cat] = group["StockCode"].tolist()

        self.is_fitted = True
        logger.info(
            "PopularityRecommender fitted successfully. Total ranked catalog items: %d across %d categories.",
            len(self.popular_items),
            len(self.category_popular_items)
        )
        return self

    def recommend(
        self,
        user_id: Optional[str] = None,
        k: int = 10,
        exclude_items: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates top-K popular recommendations.
        
        Args:
            user_id: Optional user identifier (unused by popularity baseline, maintained for interface symmetry).
            k: Number of recommendations to return.
            exclude_items: Items to exclude (e.g. items the user already purchased).
            category: Optional category filter.
            
        Returns:
            List of recommendation dicts containing item details and popularity score.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before recommend().")

        exclude_set = set(exclude_items) if exclude_items else set()
        pool = self.category_popular_items.get(category, self.popular_items) if category else self.popular_items

        results = []
        for stock_code in pool:
            if stock_code in exclude_set:
                continue
            
            # Lookup metadata from catalog
            cat_row = self.catalog_df[self.catalog_df["StockCode"] == stock_code]
            desc = cat_row["Description"].iloc[0] if len(cat_row) > 0 else "Unknown Product"
            cat = cat_row["Category"].iloc[0] if len(cat_row) > 0 else "General"
            price = float(cat_row["MedianUnitPrice"].iloc[0]) if len(cat_row) > 0 and "MedianUnitPrice" in cat_row.columns else 0.0

            results.append({
                "item_id": stock_code,
                "description": desc,
                "category": cat,
                "unit_price": price,
                "score": round(self.popularity_scores.get(stock_code, 0.0), 4),
                "model": "PopularityBaseline",
                "explanation": f"Ranked #{len(results)+1} most popular item across the catalog."
            })
            if len(results) >= k:
                break

        return results

    def save(self, filepath: str = "artifacts/popularity_model.joblib"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Saved PopularityRecommender to: %s", filepath)

    @classmethod
    def load(cls, filepath: str = "artifacts/popularity_model.joblib") -> "PopularityRecommender":
        logger.info("Loading PopularityRecommender from: %s", filepath)
        return joblib.load(filepath)

if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv", dtype={"StockCode": str, "CustomerID": str})
    model = PopularityRecommender()
    model.fit(train_df)
    model.save()
    recs = model.recommend(k=5)
    print("\n[Top-5 Popularity Recommendations]")
    for r in recs:
        print(f"  {r['item_id']} | {r['description']} | Cat: {r['category']} | Score: {r['score']}")

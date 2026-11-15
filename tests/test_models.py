"""
Unit tests for NEXUS recommendation models.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.models.popularity import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.hybrid import HybridRecommender

@pytest.fixture(scope="module")
def train_df():
    p = Path("data/processed/train.csv")
    assert p.exists()
    return pd.read_csv(p, nrows=5000, dtype={"StockCode": str, "CustomerID": str})

def test_popularity_recommender():
    pop_m = PopularityRecommender.load("artifacts/popularity_model.joblib")
    recs = pop_m.recommend(k=5)

    assert len(recs) == 5
    assert recs[0]["score"] >= recs[1]["score"], "Popularity recommendations must be sorted descending by score."
    assert "item_id" in recs[0] and "description" in recs[0]

    # Category filtered
    cat_recs = pop_m.recommend(k=3, category="Kitchen & Dining")
    assert len(cat_recs) == 3
    for r in cat_recs:
        assert r["category"] == "Kitchen & Dining"

def test_content_based_recommender():
    cnt_m = ContentBasedRecommender.load("artifacts/content_model.joblib")
    
    # Test similar items
    sims = cnt_m.recommend_similar_items("22423", k=3)
    assert len(sims) == 3
    assert sims[0]["item_id"] != "22423", "Query item should not be in its own similar items list."

    # Test user recommendations
    recs = cnt_m.recommend(user_id="17850", k=4)
    assert len(recs) == 4
    for r in recs:
        assert r["score"] > 0.0

def test_collaborative_recommender():
    cf_m = CollaborativeRecommender.load("artifacts/collaborative_model.joblib")
    recs = cf_m.recommend(user_id="17850", k=5)

    assert len(recs) == 5
    # Normalized score between 0 and 1
    assert 0.0 <= recs[0]["score"] <= 1.0

def test_hybrid_recommender_warm_and_cold():
    hybrid_m = HybridRecommender.load("artifacts/hybrid_model.joblib")

    # Warm user
    warm_recs = hybrid_m.recommend(user_id="17850", k=4)
    assert len(warm_recs) == 4
    assert not warm_recs[0].get("is_fallback", False)

    # Cold start user
    cold_recs = hybrid_m.recommend(user_id="UNSEEN_COLD_USER_12345", k=3)
    assert len(cold_recs) == 3
    assert cold_recs[0]["is_fallback"] is True
    assert "fallback_reason" in cold_recs[0]

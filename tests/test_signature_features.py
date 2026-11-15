"""
Unit tests for the two mandatory signature features:
1. Serendipity Mode (Familiar / Balanced / Explore)
2. Recommendation Reasoning Lab (Grounded Signal Decomposition)
"""

import pytest
from src.models.hybrid import HybridRecommender
from src.recommendation.serendipity import SerendipityEngine
from src.explanations.reasoning_lab import RecommendationReasoningLab

@pytest.fixture(scope="module")
def setup_components():
    hybrid_m = HybridRecommender.load("artifacts/hybrid_model.joblib")
    serendipity = SerendipityEngine()
    reasoning_lab = RecommendationReasoningLab()
    return hybrid_m, serendipity, reasoning_lab

def test_serendipity_mode_behavior(setup_components):
    hybrid_m, serendipity, _ = setup_components
    base_recs = hybrid_m.recommend(user_id="17850", k=20)

    fam_recs = serendipity.rerank(base_recs, mode="familiar", k=5)
    bal_recs = serendipity.rerank(base_recs, mode="balanced", k=5)
    exp_recs = serendipity.rerank(base_recs, mode="explore", k=5)

    assert len(fam_recs) == 5
    assert len(bal_recs) == 5
    assert len(exp_recs) == 5

    # Check that explore mode elevates novelty compared to familiar mode
    mean_nov_fam = sum(r["novelty_score"] for r in fam_recs) / 5.0
    mean_nov_exp = sum(r["novelty_score"] for r in exp_recs) / 5.0

    assert mean_nov_exp >= mean_nov_fam, (
        f"Explore mode should achieve equal or higher average novelty than Familiar mode: "
        f"Explore={mean_nov_exp} vs Familiar={mean_nov_fam}"
    )

def test_recommendation_reasoning_lab(setup_components):
    hybrid_m, _, reasoning_lab = setup_components
    test_user = "17850"
    recs = hybrid_m.recommend(user_id=test_user, k=2)

    assert len(recs) > 0
    top_item = recs[0]

    audit = reasoning_lab.explain_item_recommendation(
        user_id=test_user,
        item_id=top_item["item_id"],
        recommendation_item=top_item,
        content_model=hybrid_m.content_model,
        cf_model=hybrid_m.cf_model
    )

    assert "decision_summary" in audit
    assert "signals" in audit

    # Check signal structures
    signals = audit["signals"]
    assert "collaborative_filtering" in signals
    assert "content_semantics" in signals
    assert "category_alignment" in signals
    assert "catalog_popularity" in signals

    # Check score type distinction (not a probability)
    assert audit["decision_summary"]["is_probability"] is False

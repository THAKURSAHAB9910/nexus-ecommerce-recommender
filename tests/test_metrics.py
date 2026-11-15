"""
Unit tests for NEXUS ranking and recommendation metrics.
"""

import pytest
import numpy as np
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    catalog_coverage,
    intra_list_category_diversity,
    novelty_at_k
)

def test_precision_at_k_perfect():
    recs = ["item1", "item2", "item3", "item4", "item5"]
    gt = {"item1", "item2", "item3", "item4", "item5"}
    assert precision_at_k(recs, gt, k=5) == 1.0

def test_precision_at_k_partial():
    recs = ["item1", "item2", "item3", "item4", "item5"]
    gt = {"item1", "item3"}
    assert precision_at_k(recs, gt, k=5) == 0.4
    assert precision_at_k(recs, gt, k=2) == 0.5  # item1 is in top-2

def test_precision_at_k_zero():
    recs = ["item1", "item2"]
    gt = {"item99"}
    assert precision_at_k(recs, gt, k=2) == 0.0

def test_recall_at_k():
    recs = ["item1", "item2", "item3", "item4", "item5"]
    gt = {"item1", "item3", "item99", "item100"}  # 4 ground truth items
    # Captured 2 out of 4
    assert recall_at_k(recs, gt, k=5) == 0.5
    # With k=1, captures 1 out of 4
    assert recall_at_k(recs, gt, k=1) == 0.25

def test_ndcg_at_k_order_sensitivity():
    # If hit is at rank 1 vs rank 5
    gt = {"hit"}
    recs_top = ["hit", "miss", "miss", "miss", "miss"]
    recs_bottom = ["miss", "miss", "miss", "miss", "hit"]

    ndcg_top = ndcg_at_k(recs_top, gt, k=5)
    ndcg_bottom = ndcg_at_k(recs_bottom, gt, k=5)

    assert ndcg_top == 1.0
    assert ndcg_bottom < 1.0
    assert ndcg_top > ndcg_bottom

def test_catalog_coverage():
    all_recs = [
        ["A", "B", "C"],
        ["B", "C", "D"],
        ["A", "E", "F"]
    ]
    # Unique items recommended: {A, B, C, D, E, F} = 6 items
    # Total catalog: 10 items -> 60% coverage
    assert catalog_coverage(all_recs, total_catalog_items=10) == 60.0

def test_intra_list_category_diversity():
    # All identical categories -> 0 diversity
    all_same = ["Kitchen", "Kitchen", "Kitchen"]
    assert intra_list_category_diversity(all_same) == 0.0

    # All unique categories -> 1.0 diversity
    all_diff = ["Kitchen", "Decor", "Toys"]
    assert intra_list_category_diversity(all_diff) == 1.0

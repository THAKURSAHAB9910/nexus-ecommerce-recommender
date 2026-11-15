"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Module: src.evaluation.metrics

Responsible for:
1. Exact mathematical ranking metrics: Precision@K, Recall@K, NDCG@K.
2. System-wide metrics: Catalog Coverage, Intra-List Category Diversity, Novelty (Popularity Concentration).
3. Clear documentation and step-by-step example calculations.
"""

from typing import List, Set, Dict, Any
import numpy as np

def precision_at_k(recommended_items: List[str], ground_truth: Set[str], k: int = 10) -> float:
    """
    Computes Precision@K:
    Fraction of top-K recommended items that are present in the user's test purchases.
    Formula: |Rec@K ∩ GroundTruth| / K
    """
    if k <= 0:
        return 0.0
    top_k = recommended_items[:k]
    hits = sum(1 for item in top_k if item in ground_truth)
    return hits / float(k)

def recall_at_k(recommended_items: List[str], ground_truth: Set[str], k: int = 10) -> float:
    """
    Computes Recall@K:
    Fraction of user's test purchases successfully captured in top-K recommendations.
    Formula: |Rec@K ∩ GroundTruth| / |GroundTruth|
    """
    if not ground_truth or k <= 0:
        return 0.0
    top_k = recommended_items[:k]
    hits = sum(1 for item in top_k if item in ground_truth)
    return hits / float(len(ground_truth))

def ndcg_at_k(recommended_items: List[str], ground_truth: Set[str], k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG@K):
    Rewards correct recommendations more if they appear at higher ranks.
    
    Formula:
        DCG@K = sum_{r=1}^K [ I(item_r in GT) / log2(r + 1) ]
        IDCG@K = sum_{r=1}^{min(K, |GT|)} [ 1 / log2(r + 1) ]
        NDCG@K = DCG@K / IDCG@K
    """
    if not ground_truth or k <= 0:
        return 0.0
    top_k = recommended_items[:k]
    
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in ground_truth:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(k, len(ground_truth))
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)

def catalog_coverage(all_recommendations: List[List[str]], total_catalog_items: int) -> float:
    """
    Computes Catalog Coverage percentage:
    Fraction of unique items in the catalog recommended at least once across all test users.
    Formula: |Union of all Recs| / |Catalog|
    """
    if total_catalog_items <= 0:
        return 0.0
    unique_recommended = set()
    for rec_list in all_recommendations:
        unique_recommended.update(rec_list)
    return round((len(unique_recommended) / float(total_catalog_items)) * 100, 2)

def intra_list_category_diversity(recommended_categories: List[str]) -> float:
    """
    Computes Intra-List Category Diversity:
    1 minus the proportion of identical category pairs in the recommendation list.
    0 = all items belong to identical category, 1 = every item has a unique category.
    """
    k = len(recommended_categories)
    if k <= 1:
        return 0.0

    matches = 0
    total_pairs = k * (k - 1)
    for i in range(k):
        for j in range(k):
            if i != j and recommended_categories[i] == recommended_categories[j]:
                matches += 1

    return round(1.0 - (matches / float(total_pairs)), 4)

def novelty_at_k(recommended_items: List[str], item_popularity_probs: Dict[str, float], k: int = 10) -> float:
    """
    Computes Novelty@K (Self-Information in bits):
    Measures the degree to which recommendations surface unexpected/long-tail items.
    Formula: (1 / K) * sum_{i in Rec@K} [ -log2(P(i)) ]
    """
    top_k = recommended_items[:k]
    if not top_k:
        return 0.0

    novelty_sum = 0.0
    for item in top_k:
        p_i = item_popularity_probs.get(item, 1e-6)
        novelty_sum += -np.log2(max(p_i, 1e-7))

    return round(float(novelty_sum / len(top_k)), 4)

"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Module: src.evaluation.evaluator

Responsible for:
1. Running standardized, leakage-free benchmark comparison across all models:
   - Popularity Baseline
   - Content-Based Filtering
   - Collaborative Filtering (TruncatedSVD)
   - Adaptive Hybrid Recommender
2. Computing ranking metrics (Precision@K, Recall@K, NDCG@K, Coverage, Diversity, Novelty).
3. Evaluating cold-start robustness on zero-history test users.
4. Exporting reports/model_comparison.json.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Set
import pandas as pd
import numpy as np

from src.models.popularity import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.hybrid import HybridRecommender
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    catalog_coverage,
    intra_list_category_diversity,
    novelty_at_k
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NEXUS.Evaluator")

class ModelEvaluator:
    """Executes reproducible ranking evaluation on held-out test data."""

    def __init__(
        self,
        train_csv: str = "data/processed/train.csv",
        test_csv: str = "data/processed/test.csv",
        catalog_csv: str = "data/processed/catalog.csv",
        reports_dir: str = "reports",
        k_values: List[int] = [5, 10]
    ):
        self.train_path = Path(train_csv)
        self.test_path = Path(test_csv)
        self.catalog_path = Path(catalog_csv)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.k_values = k_values
        self.results_json_path = self.reports_dir / "model_comparison.json"

    def run_benchmark(self, max_eval_users: int = 500) -> Dict[str, Any]:
        """
        Runs standardized ranking evaluation across all candidate models.
        """
        logger.info("Loading test transactions and product catalog...")
        train_df = pd.read_csv(self.train_path, dtype={"StockCode": str, "CustomerID": str})
        test_df = pd.read_csv(self.test_path, dtype={"StockCode": str, "CustomerID": str})
        catalog_df = pd.read_csv(self.catalog_path, dtype={"StockCode": str})

        total_catalog_items = len(catalog_df)

        # Precompute item popularity probabilities P(i) for Novelty metric
        train_item_counts = train_df["StockCode"].value_counts()
        total_train_events = len(train_df)
        item_pop_probs = {item: count / total_train_events for item, count in train_item_counts.items()}

        # Build ground-truth test interaction sets per user
        logger.info("Building ground-truth test interaction sets...")
        test_ground_truth: Dict[str, Set[str]] = (
            test_df.groupby("CustomerID")["StockCode"]
            .apply(set)
            .to_dict()
        )

        train_users = set(train_df["CustomerID"].dropna().unique())
        test_users = set(test_df["CustomerID"].dropna().unique())
        
        # Warm evaluation cohort: users present in both train and test
        eval_users = sorted(list(train_users.intersection(test_users)))
        if max_eval_users and len(eval_users) > max_eval_users:
            # Deterministic subset for swift evaluation
            np.random.seed(42)
            eval_users = sorted(list(np.random.choice(eval_users, size=max_eval_users, replace=False)))

        logger.info(
            "Benchmark Cohort: %d warm evaluation users with active future test purchases.",
            len(eval_users)
        )

        # Load models
        logger.info("Loading trained model artifacts...")
        pop_model = PopularityRecommender.load("artifacts/popularity_model.joblib")
        cnt_model = ContentBasedRecommender.load("artifacts/content_model.joblib")
        cf_model = CollaborativeRecommender.load("artifacts/collaborative_model.joblib")
        hybrid_model = HybridRecommender.load("artifacts/hybrid_model.joblib")

        models = {
            "Popularity Baseline": pop_model,
            "Content-Based (TF-IDF)": cnt_model,
            "Collaborative (SVD)": cf_model,
            "Adaptive Hybrid": hybrid_model
        }

        comparison_results = {}

        for model_name, model in models.items():
            logger.info("Evaluating [%s] on %d users...", model_name, len(eval_users))
            model_metrics = {}

            for k in self.k_values:
                p_list, r_list, ndcg_list = [], [], []
                diversity_list, novelty_list = [], []
                all_recs = []

                for user_id in eval_users:
                    gt = test_ground_truth.get(user_id, set())
                    if hasattr(model, "recommend"):
                        # Disable heavy string explanation formatting during bulk evaluation
                        try:
                            recs = model.recommend(user_id=user_id, k=k, include_explanations=False)
                        except TypeError:
                            recs = model.recommend(user_id=user_id, k=k)
                    rec_ids = [r["item_id"] for r in recs]
                    rec_cats = [r.get("category", "") for r in recs]

                    all_recs.append(rec_ids)
                    p_list.append(precision_at_k(rec_ids, gt, k=k))
                    r_list.append(recall_at_k(rec_ids, gt, k=k))
                    ndcg_list.append(ndcg_at_k(rec_ids, gt, k=k))
                    diversity_list.append(intra_list_category_diversity(rec_cats))
                    novelty_list.append(novelty_at_k(rec_ids, item_pop_probs, k=k))

                cov = catalog_coverage(all_recs, total_catalog_items)

                model_metrics[f"K={k}"] = {
                    "Precision": round(float(np.mean(p_list)), 4),
                    "Recall": round(float(np.mean(r_list)), 4),
                    "NDCG": round(float(np.mean(ndcg_list)), 4),
                    "CatalogCoverage_Pct": cov,
                    "CategoryDiversity": round(float(np.mean(diversity_list)), 4),
                    "Novelty_Bits": round(float(np.mean(novelty_list)), 4),
                }

            comparison_results[model_name] = model_metrics

        # Also evaluate Cold-Start Users
        cold_start_users = sorted(list(test_users - train_users))
        logger.info("Evaluating Cold-Start Cohort: %d zero-history users...", len(cold_start_users))
        cs_p, cs_r, cs_ndcg = [], [], []
        for cs_u in cold_start_users:
            gt = test_ground_truth.get(cs_u, set())
            recs = hybrid_model.recommend(user_id=cs_u, k=10)
            rec_ids = [r["item_id"] for r in recs]
            cs_p.append(precision_at_k(rec_ids, gt, k=10))
            cs_r.append(recall_at_k(rec_ids, gt, k=10))
            cs_ndcg.append(ndcg_at_k(rec_ids, gt, k=10))

        cold_start_summary = {
            "total_cold_start_users": len(cold_start_users),
            "fallback_mechanism": "Popularity Fallback Router",
            "Precision@10": round(float(np.mean(cs_p)), 4),
            "Recall@10": round(float(np.mean(cs_r)), 4),
            "NDCG@10": round(float(np.mean(cs_ndcg)), 4),
        }

        final_output = {
            "evaluation_protocol": "Strict Chronological Ranking on Held-Out Test Data",
            "warm_cohort_size": len(eval_users),
            "models_evaluated": comparison_results,
            "cold_start_evaluation": cold_start_summary
        }

        with open(self.results_json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2)

        logger.info("Benchmark complete! Saved comparison report to: %s", self.results_json_path)
        self._print_formatted_summary(final_output)
        return final_output

    def _print_formatted_summary(self, results: Dict[str, Any]):
        print("\n" + "=" * 90)
        print("          NEXUS RECOMMENDATION SYSTEM: EMPIRICAL BENCHMARK EVALUATION")
        print("=" * 90)
        print(f"Protocol: {results['evaluation_protocol']} (N={results['warm_cohort_size']} active test users)")
        print("-" * 90)
        print(f"{'Model':<25} | {'P@10':<8} | {'R@10':<8} | {'NDCG@10':<8} | {'Coverage%':<10} | {'Diversity':<10} | {'Novelty'}")
        print("-" * 90)
        for model_name, m_dict in results["models_evaluated"].items():
            k10 = m_dict["K=10"]
            print(
                f"{model_name:<25} | "
                f"{k10['Precision']:<8.4f} | "
                f"{k10['Recall']:<8.4f} | "
                f"{k10['NDCG']:<8.4f} | "
                f"{k10['CatalogCoverage_Pct']:<10.2f} | "
                f"{k10['CategoryDiversity']:<10.4f} | "
                f"{k10['Novelty_Bits']:.2f}"
            )
        print("-" * 90)
        cs = results["cold_start_evaluation"]
        print(f"Cold-Start Fallback (N={cs['total_cold_start_users']} users) -> Precision@10: {cs['Precision@10']:.4f} | Recall@10: {cs['Recall@10']:.4f} | NDCG@10: {cs['NDCG@10']:.4f}")
        print("=" * 90 + "\n")

if __name__ == "__main__":
    evaluator = ModelEvaluator(k_values=[5, 10])
    evaluator.run_benchmark(max_eval_users=300)

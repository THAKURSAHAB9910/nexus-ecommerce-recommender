"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Script: scripts.train

End-to-end model training orchestrator:
1. Loads historical training interactions from data/processed/train.csv.
2. Fits and serializes PopularityRecommender with correct module namespaces.
3. Fits and serializes ContentBasedRecommender (TF-IDF + Cosine).
4. Fits and serializes CollaborativeRecommender (TruncatedSVD Latent Factors).
5. Fits and serializes Adaptive HybridRecommender with cold-start routing.
"""

import sys
import logging
from pathlib import Path
import pandas as pd

from src.models.popularity import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.hybrid import HybridRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NEXUS.TrainPipeline")

def train_all_models(
    train_csv: str = "data/processed/train.csv",
    artifacts_dir: str = "artifacts"
):
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training transactions from %s...", train_csv)
    train_df = pd.read_csv(train_csv, dtype={"StockCode": str, "CustomerID": str})

    # 1. Popularity Baseline
    logger.info("--- [1/4] Training Popularity Recommender ---")
    pop_model = PopularityRecommender()
    pop_model.fit(train_df)
    pop_model.save(str(artifacts_path / "popularity_model.joblib"))

    # 2. Content-Based (TF-IDF)
    logger.info("--- [2/4] Training Content-Based Recommender ---")
    content_model = ContentBasedRecommender(max_features=2500)
    content_model.fit(train_df)
    content_model.save(str(artifacts_path / "content_model.joblib"))

    # 3. Collaborative (TruncatedSVD)
    logger.info("--- [3/4] Training Collaborative Recommender ---")
    cf_model = CollaborativeRecommender(n_factors=40, alpha=1.0)
    cf_model.fit(train_df)
    cf_model.save(str(artifacts_path / "collaborative_model.joblib"))

    # 4. Adaptive Hybrid
    logger.info("--- [4/4] Training Adaptive Hybrid Recommender ---")
    hybrid_model = HybridRecommender(
        popularity_model=pop_model,
        content_model=content_model,
        collaborative_model=cf_model,
        default_cf_weight=0.65
    )
    hybrid_model.fit(train_df)
    hybrid_model.save(str(artifacts_path / "hybrid_model.joblib"))

    logger.info("ALL NEXUS models trained and serialized successfully!")

if __name__ == "__main__":
    train_all_models()

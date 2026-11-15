"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Module: src.preprocessing.splitter

Responsible for:
1. Leakage-free chronological train/validation/test splitting.
2. Preventing future interaction leakage (evaluating models solely on historical interactions).
3. Segmenting user cohorts:
   - Warm Evaluation Users (users with interactions in both Train and Test).
   - Cold-Start Users (new users who first appear in Test with zero historical train data).
   - Cold-Start Items (products introduced in Test with zero train transactions).
4. Exporting train.csv, val.csv, test.csv, and split_metadata.json.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NEXUS.Splitter")

class ChronologicalSplitter:
    """Executes time-ordered, leakage-free dataset splitting."""

    def __init__(
        self,
        transactions_csv: str = "data/processed/clean_transactions.csv",
        output_dir: str = "data/processed",
        train_ratio: float = 0.80,
        val_ratio: float = 0.10
    ):
        self.transactions_path = Path(transactions_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - (train_ratio + val_ratio)

        self.train_path = self.output_dir / "train.csv"
        self.val_path = self.output_dir / "val.csv"
        self.test_path = self.output_dir / "test.csv"
        self.metadata_path = self.output_dir / "split_metadata.json"

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Splits data chronologically and exports split artifacts.
        
        Returns:
            Tuple of (train_df, val_df, test_df, metadata_dict)
        """
        if not self.transactions_path.exists():
            raise FileNotFoundError(f"Clean transactions file not found at: {self.transactions_path}")

        logger.info("Loading clean transactions from %s...", self.transactions_path)
        df = pd.read_csv(
            self.transactions_path,
            dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str}
        )
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        # Filter to interactions with valid CustomerID for personalized learning
        user_df = df[df["CustomerID"].notna()].copy()
        user_df = user_df.sort_values("InvoiceDate").reset_index(drop=True)

        total_interactions = len(user_df)
        logger.info("Total customer interactions for model training & evaluation: %d", total_interactions)

        # Compute exact chronological timestamps
        t_val_start = user_df["InvoiceDate"].quantile(self.train_ratio)
        t_test_start = user_df["InvoiceDate"].quantile(self.train_ratio + self.val_ratio)

        logger.info("Chronological Split Timestamps:")
        logger.info("  Train Window:      Start -> %s", t_val_start)
        logger.info("  Validation Window: %s -> %s", t_val_start, t_test_start)
        logger.info("  Test Window:       %s -> End (%s)", t_test_start, user_df["InvoiceDate"].max())

        # Strict non-overlapping time slices
        train_df = user_df[user_df["InvoiceDate"] < t_val_start].copy().reset_index(drop=True)
        val_df = user_df[(user_df["InvoiceDate"] >= t_val_start) & (user_df["InvoiceDate"] < t_test_start)].copy().reset_index(drop=True)
        test_df = user_df[user_df["InvoiceDate"] >= t_test_start].copy().reset_index(drop=True)

        # Integrity check: zero timestamp leakage
        assert train_df["InvoiceDate"].max() <= val_df["InvoiceDate"].min(), "Data Leakage: Train overlap with Validation!"
        assert val_df["InvoiceDate"].max() <= test_df["InvoiceDate"].min(), "Data Leakage: Validation overlap with Test!"

        # User cohort analysis
        train_users = set(train_df["CustomerID"])
        val_users = set(val_df["CustomerID"])
        test_users = set(test_df["CustomerID"])

        eval_users_test = train_users.intersection(test_users)
        cold_start_users_test = test_users - train_users

        # Item cohort analysis
        train_items = set(train_df["StockCode"])
        test_items = set(test_df["StockCode"])
        cold_start_items_test = test_items - train_items

        metadata = {
            "strategy": "Strict Chronological Split (No Future Leakage)",
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": round(self.test_ratio, 2),
            "timestamps": {
                "dataset_start": str(user_df["InvoiceDate"].min()),
                "train_end_val_start": str(t_val_start),
                "val_end_test_start": str(t_test_start),
                "dataset_end": str(user_df["InvoiceDate"].max()),
            },
            "record_counts": {
                "total_identified_interactions": total_interactions,
                "train_interactions": len(train_df),
                "val_interactions": len(val_df),
                "test_interactions": len(test_df),
            },
            "user_cohorts": {
                "train_unique_users": len(train_users),
                "val_unique_users": len(val_users),
                "test_unique_users": len(test_users),
                "warm_eval_users_test": len(eval_users_test),
                "cold_start_users_test": len(cold_start_users_test),
            },
            "item_cohorts": {
                "train_unique_items": len(train_items),
                "test_unique_items": len(test_items),
                "cold_start_items_test": len(cold_start_items_test),
            }
        }

        # Export split files
        logger.info("Saving train set to: %s (%d rows)", self.train_path, len(train_df))
        train_df.to_csv(self.train_path, index=False)

        logger.info("Saving validation set to: %s (%d rows)", self.val_path, len(val_df))
        val_df.to_csv(self.val_path, index=False)

        logger.info("Saving test set to: %s (%d rows)", self.test_path, len(test_df))
        test_df.to_csv(self.test_path, index=False)

        logger.info("Saving split metadata to: %s", self.metadata_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("===== CHRONOLOGICAL SPLIT SUMMARY =====")
        logger.info("  Train Set:      %d rows | %d unique users | %d items", len(train_df), len(train_users), len(train_items))
        logger.info("  Validation Set: %d rows | %d unique users", len(val_df), len(val_users))
        logger.info("  Test Set:       %d rows | %d unique users", len(test_df), len(test_users))
        logger.info("  Warm Evaluation Cohort (Train & Test): %d users", len(eval_users_test))
        logger.info("  Cold-Start Evaluation Cohort (Test only): %d users", len(cold_start_users_test))
        logger.info("  Cold-Start Items in Test: %d items", len(cold_start_items_test))
        logger.info("========================================")

        return train_df, val_df, test_df, metadata

if __name__ == "__main__":
    splitter = ChronologicalSplitter()
    splitter.split()

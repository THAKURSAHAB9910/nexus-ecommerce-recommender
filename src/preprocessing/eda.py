"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Module: src.preprocessing.eda

Responsible for:
1. Statistical analysis of user-item interactions, matrix sparsity, and activity distributions.
2. Quantifying the Popularity Long-Tail (Pareto 80/20 rule, Gini coefficient).
3. Generating rich visual figures for the project report and Streamlit dashboard.
4. Exporting reports/eda_summary.json and reports/figures/*.png.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NEXUS.EDA")

class EDAPipeline:
    """Executes exploratory data analysis and visual chart generation."""

    def __init__(
        self,
        transactions_csv: str = "data/processed/clean_transactions.csv",
        catalog_csv: str = "data/processed/catalog.csv",
        reports_dir: str = "reports"
    ):
        self.transactions_path = Path(transactions_csv)
        self.catalog_path = Path(catalog_csv)
        self.reports_dir = Path(reports_dir)
        self.figures_dir = self.reports_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.summary_json_path = self.reports_dir / "eda_summary.json"

    @staticmethod
    def calculate_gini(array: np.ndarray) -> float:
        """
        Calculates the Gini coefficient of an array (measure of popularity inequality).
        0 = completely uniform distribution, 1 = maximum concentration (one item owns everything).
        """
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return float(((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))))

    def run_eda(self) -> Dict[str, Any]:
        """Runs complete EDA, exports figures and returns summary dictionary."""
        logger.info("Loading clean transactions and catalog...")
        df = pd.read_csv(
            self.transactions_path,
            dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str}
        )
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        catalog = pd.read_csv(self.catalog_path, dtype={"StockCode": str})

        # Interactions with identified customers
        user_df = df[df["CustomerID"].notna()].copy()

        n_users = user_df["CustomerID"].nunique()
        n_items = user_df["StockCode"].nunique()
        n_interactions = len(user_df)
        n_pairs = user_df.groupby(["CustomerID", "StockCode"]).ngroups

        # User-Item Matrix Sparsity
        total_matrix_cells = n_users * n_items
        sparsity_pct = round((1.0 - (n_pairs / total_matrix_cells)) * 100, 4)

        # User interaction statistics
        user_counts = user_df.groupby("CustomerID")["StockCode"].count()
        # Item interaction statistics
        item_counts = user_df.groupby("StockCode")["CustomerID"].count()

        # Popularity inequality (Gini and Pareto)
        item_sales = item_counts.sort_values(ascending=False).values
        gini_coef = round(self.calculate_gini(item_sales), 4)

        # 80/20 Rule: fraction of items generating 80% of interactions
        cumulative_sales = np.cumsum(item_sales) / np.sum(item_sales)
        top_20_pct_items = int(0.20 * n_items)
        sales_from_top_20 = round(float(cumulative_sales[min(top_20_pct_items, len(cumulative_sales)-1)]) * 100, 2)

        # Category Breakdown
        cat_counts = catalog["Category"].value_counts().to_dict()
        cat_sales = user_df.groupby("Category")["Quantity"].sum().to_dict()

        summary = {
            "total_transactions": len(df),
            "customer_identified_transactions": n_interactions,
            "unique_customers": n_users,
            "unique_products_in_catalog": len(catalog),
            "unique_products_interacted": n_items,
            "unique_user_item_pairs": n_pairs,
            "user_item_matrix_sparsity_pct": sparsity_pct,
            "user_activity": {
                "mean_interactions": round(float(user_counts.mean()), 2),
                "median_interactions": float(user_counts.median()),
                "p25": float(user_counts.quantile(0.25)),
                "p75": float(user_counts.quantile(0.75)),
                "min": int(user_counts.min()),
                "max": int(user_counts.max())
            },
            "item_popularity": {
                "mean_interactions": round(float(item_counts.mean()), 2),
                "median_interactions": float(item_counts.median()),
                "p25": float(item_counts.quantile(0.25)),
                "p75": float(item_counts.quantile(0.75)),
                "min": int(item_counts.min()),
                "max": int(item_counts.max()),
                "gini_coefficient": gini_coef,
                "sales_share_top_20_pct_items": sales_from_top_20
            },
            "category_distribution_catalog": cat_counts,
            "category_volume_sold": {k: int(v) for k, v in cat_sales.items()}
        }

        # Save summary JSON
        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved EDA summary metrics to: %s", self.summary_json_path)

        # Plot 1: Popularity Long Tail / Lorenz Curve
        self._plot_long_tail(item_sales, cumulative_sales, gini_coef, sales_from_top_20)

        # Plot 2: User Activity Distribution
        self._plot_user_activity(user_counts)

        # Plot 3: Category Distribution
        self._plot_category_distribution(catalog, user_df)

        # Plot 4: Temporal Transaction Trend
        self._plot_temporal_trend(df)

        logger.info("Generated all 4 EDA figures in: %s", self.figures_dir)
        return summary

    def _plot_long_tail(self, item_sales: np.ndarray, cumulative_sales: np.ndarray, gini: float, top20: float):
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x_pct = np.linspace(0, 100, len(item_sales))
        ax.plot(x_pct, cumulative_sales * 100, color="#1E88E5", linewidth=2.5, label="Cumulative Interactions")
        ax.plot([0, 100], [0, 100], color="#9E9E9E", linestyle="--", label="Equality Baseline")
        
        ax.axvline(20, color="#E53935", linestyle=":", label=f"Top 20% Items ({top20}% Volume)")
        ax.axhline(top20, color="#E53935", linestyle=":")
        
        ax.set_title(f"E-Commerce Popularity Long Tail (Gini: {gini})", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("% of Catalog Items (Ranked by Popularity)", fontsize=11)
        ax.set_ylabel("% of Total Interactions Captured", fontsize=11)
        ax.legend(loc="lower right", frameon=True)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        fig.savefig(self.figures_dir / "popularity_long_tail.png", dpi=200)
        plt.close(fig)

    def _plot_user_activity(self, user_counts: pd.Series):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(user_counts, bins=50, log_scale=(True, True), color="#00897B", ax=ax)
        ax.set_title("User Interaction Frequency Distribution (Log-Log Scale)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Number of Purchased Items per Customer (Log)", fontsize=11)
        ax.set_ylabel("Customer Count (Log)", fontsize=11)
        plt.tight_layout()
        fig.savefig(self.figures_dir / "user_activity_distribution.png", dpi=200)
        plt.close(fig)

    def _plot_category_distribution(self, catalog: pd.DataFrame, user_df: pd.DataFrame):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        cat_counts = catalog["Category"].value_counts()
        sns.barplot(x=cat_counts.values, y=cat_counts.index, palette="Blues_r", ax=ax1)
        ax1.set_title("Catalog Products by Category", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Number of Products", fontsize=10)
        
        cat_vol = user_df.groupby("Category")["Quantity"].sum().loc[cat_counts.index]
        sns.barplot(x=cat_vol.values, y=cat_vol.index, palette="Greens_r", ax=ax2)
        ax2.set_title("Units Sold by Category", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Total Units Sold", fontsize=10)
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / "category_breakdown.png", dpi=200)
        plt.close(fig)

    def _plot_temporal_trend(self, df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        df_temp = df.set_index("InvoiceDate")
        weekly = df_temp.resample("W")["InvoiceNo"].count()
        ax.plot(weekly.index, weekly.values, color="#6A1B9A", linewidth=2, marker="o", markersize=4)
        ax.set_title("Weekly Transaction Velocity (Q4 Peak Demonstration)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Weekly Transactions Count", fontsize=11)
        plt.tight_layout()
        fig.savefig(self.figures_dir / "temporal_trend.png", dpi=200)
        plt.close(fig)

if __name__ == "__main__":
    eda = EDAPipeline()
    summary = eda.run_eda()
    print("\n[NEXUS EDA] Statistical & Visual pipeline completed successfully!")

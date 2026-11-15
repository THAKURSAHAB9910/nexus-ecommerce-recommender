"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Module: src.preprocessing.cleaner

Responsible for:
1. Filtering invalid e-commerce transactions (cancellations, negative/zero quantities, non-positive unit prices).
2. Removing administrative/service non-merchandise stock codes (postage, bank charges, manual entries).
3. Normalizing textual descriptions and building a deduplicated, rich Product Catalog.
4. Categorizing products into semantically coherent e-commerce categories.
5. Exporting clean transactions and catalog artifacts to data/processed/.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NEXUS.Cleaner")

# Administrative codes that do not represent physical merchandise
ADMINISTRATIVE_STOCK_CODES = {
    "POST", "DOT", "M", "D", "S", "AMAZONFEE", "CRUK", "PADS", "B", "BANK CHARGES", "m"
}

# Domain keyword rulebook for robust e-commerce category assignment
CATEGORY_RULES = [
    ("Bags & Storage", ["BAG", "BOX", "TIN", "STORAGE", "CASE", "BASKET", "CONTAINER", "CHEST", "SACK", "LUNCH"]),
    ("Home Decor & Lighting", ["LIGHT", "HOLDER", "CANDLE", "LANTERN", "CLOCK", "MIRROR", "FRAME", "SIGN", "WALL", "HEART", "ORNAMENT", "DOORMAT"]),
    ("Kitchen & Dining", ["CAKE", "TEA", "MUG", "CUP", "PLATE", "BOWL", "BOTTLE", "CUTLERY", "BAKING", "KITCHEN", "DISH", "JAR", "SPOON", "FORK", "TEAPOT"]),
    ("Seasonal & Festive", ["CHRISTMAS", "XMAS", "TREE", "SNOWFLAKE", "STAR", "BELL", "SANTA", "REINDEER", "EASTER", "WREATH", "GARLAND"]),
    ("Stationery & Craft", ["PAPER", "CARD", "PEN", "PENCIL", "TAPE", "STICKER", "NOTEBOOK", "RIBBON", "WRAP", "TAGS", "CHALK", "SCISSORS"]),
    ("Toys & Games", ["TOY", "GAME", "DOLL", "PLUSH", "CHILDRENS", "PUZZLE", "KIDS", "BABY", "BLOCKS"]),
    ("Apparel & Accessories", ["NECKLACE", "BRACELET", "RING", "JEWELLERY", "PURSE", "SCARF", "EARRINGS", "GLOVES", "APRON"]),
    ("Bath & Wellness", ["SOAP", "BATH", "LIP", "BODY", "TOWEL", "SPONGE", "LAVENDER", "LOTION"]),
]

class DataCleaner:
    """Preprocesses raw retail data into validated transactions and product catalog."""

    def __init__(self, raw_csv: str = "data/raw/online_retail_raw.csv", output_dir: str = "data/processed"):
        self.raw_csv_path = Path(raw_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transactions_path = self.output_dir / "clean_transactions.csv"
        self.catalog_path = self.output_dir / "catalog.csv"

    @staticmethod
    def assign_category(description: str) -> str:
        """Assigns an e-commerce category using lexical matching against domain keywords."""
        desc_upper = str(description).upper()
        for cat_name, keywords in CATEGORY_RULES:
            if any(kw in desc_upper for kw in keywords):
                return cat_name
        return "Gifts & Novelties"

    def clean_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes end-to-end data cleaning and feature enrichment.
        
        Returns:
            Tuple of (clean_transactions_df, catalog_df)
        """
        if not self.raw_csv_path.exists():
            raise FileNotFoundError(f"Raw CSV not found at: {self.raw_csv_path}. Please run DataLoader first.")

        logger.info("Loading raw dataset from %s...", self.raw_csv_path)
        df = pd.read_csv(
            self.raw_csv_path,
            dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str}
        )
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        raw_len = len(df)
        logger.info("Raw record count: %d", raw_len)

        # 1. Filter out cancellations (InvoiceNo starting with C)
        no_cancellations = ~df["InvoiceNo"].str.startswith("C", na=False)
        # 2. Filter positive quantities and unit prices
        valid_quantity = df["Quantity"] > 0
        valid_price = df["UnitPrice"] > 0
        # 3. Filter administrative non-merchandise items
        not_admin = ~df["StockCode"].isin(ADMINISTRATIVE_STOCK_CODES)
        # 4. Valid description
        valid_desc = df["Description"].notna()

        clean_mask = no_cancellations & valid_quantity & valid_price & not_admin & valid_desc
        clean_df = df[clean_mask].copy()

        # Normalize text
        clean_df["Description"] = clean_df["Description"].astype(str).str.strip().str.upper()
        # Drop any descriptions that became empty after strip
        clean_df = clean_df[clean_df["Description"] != ""].copy()

        logger.info(
            "Filtered out %d invalid/administrative rows (%.2f%%). Retained %d clean transactions.",
            raw_len - len(clean_df),
            ((raw_len - len(clean_df)) / raw_len) * 100,
            len(clean_df)
        )

        # Build clean, canonical Product Catalog
        logger.info("Synthesizing unified Product Catalog...")
        catalog_records = []
        for stock_code, group in clean_df.groupby("StockCode"):
            # Select canonical description: most frequent mode
            canonical_desc = group["Description"].mode().iloc[0]
            median_price = round(float(group["UnitPrice"].median()), 2)
            total_units_sold = int(group["Quantity"].sum())
            total_transactions = int(len(group))
            category = self.assign_category(canonical_desc)

            catalog_records.append({
                "StockCode": stock_code,
                "Description": canonical_desc,
                "Category": category,
                "MedianUnitPrice": median_price,
                "TotalUnitsSold": total_units_sold,
                "TransactionFrequency": total_transactions
            })

        catalog_df = pd.DataFrame(catalog_records)
        catalog_df = catalog_df.sort_values("TotalUnitsSold", ascending=False).reset_index(drop=True)

        # Merge canonical Category back into clean_transactions for fast downstream retrieval
        code_to_cat = dict(zip(catalog_df["StockCode"], catalog_df["Category"]))
        code_to_desc = dict(zip(catalog_df["StockCode"], catalog_df["Description"]))
        clean_df["Category"] = clean_df["StockCode"].map(code_to_cat)
        clean_df["CanonicalDescription"] = clean_df["StockCode"].map(code_to_desc)

        # Sort transactions chronologically
        clean_df = clean_df.sort_values("InvoiceDate").reset_index(drop=True)

        # Export artifacts
        logger.info("Saving clean transactions to: %s", self.transactions_path)
        clean_df.to_csv(self.transactions_path, index=False)

        logger.info("Saving product catalog to: %s", self.catalog_path)
        catalog_df.to_csv(self.catalog_path, index=False)

        logger.info("Data cleaning completed successfully.")
        logger.info("  Clean Transactions: %d records", len(clean_df))
        logger.info("  Catalog Products:   %d unique items across %d categories", len(catalog_df), catalog_df["Category"].nunique())

        return clean_df, catalog_df

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.clean_data()

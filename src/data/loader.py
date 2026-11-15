"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Module: src.data.loader

Responsible for:
1. Automated, reproducible retrieval of the UCI Online Retail dataset.
2. Unzipping and storing raw data files with integrity checks.
3. Converting raw Excel data to an optimized format (CSV / Parquet) for fast repeated access.
4. Schema validation and initial dataset diagnostics.
"""

import os
import sys
import zipfile
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import urllib.request
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NEXUS.DataLoader")

# Official UCI Online Retail benchmark URL and metadata
UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
EXPECTED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country"
]

class DataLoader:
    """Manages raw dataset acquisition, caching, and schema validation."""

    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.zip_path = self.raw_dir / "online_retail.zip"
        self.xlsx_path = self.raw_dir / "Online Retail.xlsx"
        self.raw_csv_path = self.raw_dir / "online_retail_raw.csv"

    def download_dataset(self, url: str = UCI_DATASET_URL, force: bool = False) -> Path:
        """
        Downloads the dataset zip archive if not already cached.
        
        Args:
            url: The HTTP URL of the zip archive.
            force: If True, re-downloads even if the file exists.
            
        Returns:
            Path to the downloaded zip archive.
        """
        if self.zip_path.exists() and not force:
            logger.info("Dataset archive already exists at: %s", self.zip_path)
            return self.zip_path

        logger.info("Downloading dataset from: %s", url)
        # Use custom headers to avoid generic python user-agent blocking
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (NEXUS-ECommerce-Recommender)"})
        
        with urllib.request.urlopen(req, timeout=60) as response, open(self.zip_path, "wb") as out_file:
            total_size = int(response.info().get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1MB chunks
            
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    logger.info("Downloaded %d / %d bytes (%.1f%%)", downloaded, total_size, pct)
                else:
                    logger.info("Downloaded %d bytes", downloaded)

        logger.info("Download completed successfully: %s", self.zip_path)
        return self.zip_path

    def extract_archive(self, force: bool = False) -> Path:
        """
        Extracts 'Online Retail.xlsx' from the downloaded zip file.
        
        Args:
            force: If True, re-extracts even if already extracted.
            
        Returns:
            Path to the extracted Excel file.
        """
        if not self.zip_path.exists():
            self.download_dataset()

        if self.xlsx_path.exists() and not force:
            logger.info("Extracted Excel file already exists at: %s", self.xlsx_path)
            return self.xlsx_path

        logger.info("Extracting %s to %s...", self.zip_path, self.raw_dir)
        with zipfile.ZipFile(self.zip_path, "r") as archive:
            archive.extractall(self.raw_dir)

        if not self.xlsx_path.exists():
            raise FileNotFoundError(f"Expected extracted file '{self.xlsx_path}' not found in archive.")

        logger.info("Extraction complete: %s (Size: %.2f MB)", self.xlsx_path, self.xlsx_path.stat().st_size / (1024 * 1024))
        return self.xlsx_path

    def load_raw_data(self, force_reconvert: bool = False) -> pd.DataFrame:
        """
        Loads the raw dataset into a pandas DataFrame.
        To avoid slow Excel parsing on repeated runs, this method caches a raw CSV.
        
        Args:
            force_reconvert: If True, re-reads the Excel file and rewrites the CSV cache.
            
        Returns:
            Raw pandas DataFrame.
        """
        if self.raw_csv_path.exists() and not force_reconvert:
            logger.info("Loading cached raw CSV from: %s", self.raw_csv_path)
            df = pd.read_csv(self.raw_csv_path, dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str})
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
            return df

        if not self.xlsx_path.exists():
            self.extract_archive()

        logger.info("Reading Excel file (this may take ~15-25 seconds on first run)...")
        df = pd.read_excel(
            self.xlsx_path,
            dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str}
        )
        logger.info("Successfully read %d rows and %d columns from Excel.", len(df), len(df.columns))

        logger.info("Caching raw data to CSV for high-speed subsequent loading: %s", self.raw_csv_path)
        df.to_csv(self.raw_csv_path, index=False)
        return df

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates the schema and provides an integrity report.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Dictionary containing schema validation diagnostics.
        """
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Schema validation failed! Missing expected columns: {missing_cols}")

        total_rows = len(df)
        missing_customers = df["CustomerID"].isna().sum()
        missing_descriptions = df["Description"].isna().sum()
        cancellations = df["InvoiceNo"].str.startswith("C", na=False).sum()
        negative_quantities = (df["Quantity"] <= 0).sum()
        unique_customers = df["CustomerID"].dropna().nunique()
        unique_products = df["StockCode"].nunique()
        date_min = df["InvoiceDate"].min()
        date_max = df["InvoiceDate"].max()

        report = {
            "total_records": total_rows,
            "columns": list(df.columns),
            "missing_customer_ids": int(missing_customers),
            "missing_customer_pct": round(float(missing_customers / total_rows) * 100, 2),
            "missing_descriptions": int(missing_descriptions),
            "cancellations_count": int(cancellations),
            "negative_quantities_count": int(negative_quantities),
            "unique_identified_customers": int(unique_customers),
            "unique_stock_codes": int(unique_products),
            "date_range_start": str(date_min),
            "date_range_end": str(date_max),
            "schema_valid": True
        }

        logger.info("===== DATASET SCHEMA & INTEGRITY AUDIT =====")
        for k, v in report.items():
            logger.info("  %s: %s", k, v)
        logger.info("===========================================")
        return report


if __name__ == "__main__":
    loader = DataLoader()
    loader.download_dataset()
    loader.extract_archive()
    raw_df = loader.load_raw_data()
    metrics = loader.validate_schema(raw_df)
    print("\n[NEXUS Data Verification] Raw ingestion completed successfully!")

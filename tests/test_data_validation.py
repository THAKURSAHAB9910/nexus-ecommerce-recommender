"""
Unit tests for data validation, cleaning, and catalog schemas.
"""

from pathlib import Path
import pandas as pd
import pytest

def test_clean_transactions_integrity():
    p = Path("data/processed/clean_transactions.csv")
    assert p.exists(), "Clean transactions file must exist."

    df = pd.read_csv(p, nrows=1000, dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str})
    
    # 1. No negative or zero quantities
    assert (df["Quantity"] > 0).all(), "Transactions must have positive quantities."

    # 2. No negative or zero prices
    assert (df["UnitPrice"] > 0).all(), "Transactions must have positive unit prices."

    # 3. No cancellations (InvoiceNo starting with C)
    assert not df["InvoiceNo"].str.startswith("C", na=False).any(), "Cancelled invoices must be filtered."

    # 4. No administrative non-merchandise stock codes
    admin_codes = {"POST", "DOT", "M", "D", "S", "AMAZONFEE", "CRUK", "PADS", "B"}
    assert not df["StockCode"].isin(admin_codes).any(), "Administrative stock codes must be filtered."

def test_product_catalog_integrity():
    cat_path = Path("data/processed/catalog.csv")
    assert cat_path.exists(), "Product catalog must exist."

    catalog = pd.read_csv(cat_path, dtype={"StockCode": str})
    expected_cols = ["StockCode", "Description", "Category", "MedianUnitPrice"]
    for col in expected_cols:
        assert col in catalog.columns, f"Catalog missing column: {col}"

    assert len(catalog) > 3000, "Catalog must contain at least 3,000 distinct products."
    assert catalog["StockCode"].is_unique, "StockCode in catalog must be unique."
    assert catalog["Category"].notna().all(), "All items must be mapped to a category."

def test_chronological_splits_leakage():
    train_p = Path("data/processed/train.csv")
    val_p = Path("data/processed/val.csv")
    test_p = Path("data/processed/test.csv")

    assert train_p.exists() and val_p.exists() and test_p.exists(), "Train, val, and test files must exist."

    train_df = pd.read_csv(train_p, usecols=["InvoiceDate"])
    val_df = pd.read_csv(val_p, usecols=["InvoiceDate"])
    test_df = pd.read_csv(test_p, usecols=["InvoiceDate"])

    train_max = pd.to_datetime(train_df["InvoiceDate"]).max()
    val_min = pd.to_datetime(val_df["InvoiceDate"]).min()
    val_max = pd.to_datetime(val_df["InvoiceDate"]).max()
    test_min = pd.to_datetime(test_df["InvoiceDate"]).min()

    # Zero leakage assertion
    assert train_max <= val_min, f"Data leakage: train_max ({train_max}) > val_min ({val_min})"
    assert val_max <= test_min, f"Data leakage: val_max ({val_max}) > test_min ({test_min})"

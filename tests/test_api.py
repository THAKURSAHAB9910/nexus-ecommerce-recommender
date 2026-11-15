"""
Unit tests for FastAPI REST inference endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True

def test_recommend_endpoint_warm_user(client):
    payload = {
        "user_id": "17850",
        "model_type": "hybrid",
        "serendipity_mode": "balanced",
        "k": 5,
        "explain": True
    }
    res = client.post("/recommend", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["user_id"] == "17850"
    assert data["user_status"] == "ACTIVE_WARM"
    assert len(data["recommendations"]) == 5
    assert "reasoning_audit" in data["recommendations"][0]

def test_recommend_endpoint_cold_start_user(client):
    payload = {
        "user_id": "UNKNOWN_TEST_USER_9999",
        "model_type": "hybrid",
        "serendipity_mode": "explore",
        "k": 4
    }
    res = client.post("/recommend", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["user_status"] == "COLD_START"
    assert len(data["recommendations"]) == 4

def test_recommend_similar_endpoint(client):
    payload = {"item_id": "22423", "k": 3}
    res = client.post("/recommend/similar", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["query_item_id"] == "22423"
    assert len(data["similar_items"]) == 3

def test_product_detail_endpoint(client):
    res = client.get("/products/22423")
    assert res.status_code == 200
    data = res.json()
    assert data["item_id"] == "22423"
    assert "REGENCY CAKESTAND" in data["description"]

def test_product_not_found(client):
    res = client.get("/products/NONEXISTENT_STOCK_CODE_XYZ")
    assert res.status_code == 404

def test_catalog_categories_endpoint(client):
    res = client.get("/catalog/categories")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] >= 8
    assert "Kitchen & Dining" in data["categories"]

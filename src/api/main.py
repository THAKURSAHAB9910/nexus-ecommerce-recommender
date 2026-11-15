"""
NEXUS: Intelligent E-Commerce Product Recommendation Engine
Module: src.api.main

FastAPI Production Inference Service providing:
- GET  /health
- POST /recommend
- POST /recommend/similar
- GET  /products/{product_id}
- GET  /users/{user_id}/recommendations
- GET  /catalog/categories
"""

import sys
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.recommendation.engine import NexusRecommenderEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NEXUS.API")

# Global engine state
engine: Optional[NexusRecommenderEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager loading machine learning artifacts at server startup."""
    global engine
    logger.info("Initializing NEXUS Recommender Engine for API...")
    engine = NexusRecommenderEngine()
    logger.info("NEXUS API ready to serve requests.")
    yield
    logger.info("Shutting down NEXUS API...")

app = FastAPI(
    title="NEXUS Recommendation Engine API",
    description="Production-grade e-commerce recommendation service with Serendipity Mode and Reasoning Lab.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Pydantic Request & Response Schemas -----------------

class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="Unique customer ID", examples=["17850"])
    model_type: str = Field("hybrid", description="'popularity', 'content', 'collaborative', or 'hybrid'")
    serendipity_mode: str = Field("balanced", description="'familiar', 'balanced', or 'explore'")
    k: int = Field(10, ge=1, le=50, description="Number of recommendations to retrieve")
    category: Optional[str] = Field(None, description="Optional category filter")
    explain: bool = Field(False, description="Whether to include full Reasoning Lab signal attribution")

class SimilarItemsRequest(BaseModel):
    item_id: str = Field(..., description="Target StockCode", examples=["22423"])
    k: int = Field(10, ge=1, le=50, description="Number of similar items")

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    models_loaded: bool

# ----------------- Endpoints -----------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Healthcheck confirming service availability and model loading status."""
    return {
        "status": "healthy",
        "service": "NEXUS E-Commerce Recommender",
        "version": "1.0.0",
        "models_loaded": engine is not None
    }

@app.post("/recommend", tags=["Recommendations"])
def recommend_products(req: RecommendRequest):
    """
    Generates personalized recommendations using specified model and serendipity mode.
    Handles cold-start users gracefully with transparent fallbacks.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommender engine is not yet initialized.")

    try:
        results = engine.recommend(
            user_id=req.user_id,
            model_type=req.model_type,
            serendipity_mode=req.serendipity_mode,
            k=req.k,
            category=req.category,
            explain=req.explain
        )
        return results
    except Exception as e:
        logger.error("Error generating recommendations: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/recommendations", tags=["Recommendations"])
def get_user_recommendations(
    user_id: str,
    k: int = Query(10, ge=1, le=50),
    model_type: str = Query("hybrid", pattern="^(popularity|content|collaborative|hybrid)$"),
    serendipity_mode: str = Query("balanced", pattern="^(familiar|balanced|explore)$"),
    category: Optional[str] = None,
    explain: bool = Query(False)
):
    """Convenience GET endpoint to fetch recommendations for a specific user."""
    req = RecommendRequest(
        user_id=user_id,
        k=k,
        model_type=model_type,
        serendipity_mode=serendipity_mode,
        category=category,
        explain=explain
    )
    return recommend_products(req)

@app.post("/recommend/similar", tags=["Item Discovery"])
def recommend_similar_products(req: SimilarItemsRequest):
    """Retrieves products semantically and categorically similar to query item."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommender engine is not yet initialized.")

    results = engine.recommend_similar(item_id=req.item_id, k=req.k)
    if not results["similar_items"]:
        # Verify if item exists in catalog
        prod = engine.get_product(req.item_id)
        if prod is None:
            raise HTTPException(status_code=404, detail=f"Product with item_id '{req.item_id}' not found in catalog.")
    return results

@app.get("/products/{product_id}", tags=["Catalog"])
def get_product_details(product_id: str):
    """Returns metadata and popularity ranking details for a specific product."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommender engine is not yet initialized.")

    product = engine.get_product(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found in catalog.")
    return product

@app.get("/catalog/categories", tags=["Catalog"])
def get_categories():
    """Returns all available product categories in the catalog."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommender engine is not yet initialized.")

    cats = sorted(engine.catalog_df["Category"].dropna().unique().tolist())
    return {"categories": cats, "count": len(cats)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)

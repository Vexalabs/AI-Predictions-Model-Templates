#!/usr/bin/env python3
"""
S&P 500 Stock Prediction API
============================

Production-ready FastAPI application for S&P 500 stock predictions
using Facebook Prophet with Interactive Brokers data.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import logging
from src.api import router
from src.models import StockPredictionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="S&P 500 Stock Prediction API",
    description="AI-powered stock price predictions using Facebook Prophet",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Initialize the prediction model
prediction_model = StockPredictionModel()

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    try:
        logger.info("Initializing S&P 500 Prediction Model...")
        await prediction_model.initialize()
        logger.info("Model initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": prediction_model.is_initialized,
        "last_trained_at": prediction_model.last_trained_at,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "S&P 500 Stock Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

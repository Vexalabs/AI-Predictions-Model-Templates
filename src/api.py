"""
API routes for S&P 500 Stock Prediction
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date, timedelta
import logging
import yfinance as yf
from .models import StockPredictionModel, get_prediction_model
from .schemas import PredictionRequest, PredictionResponse, HealthResponse

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["predictions"])

def get_next_trading_day(start_date: datetime) -> datetime:
    """Get the next trading day (skip weekends)"""
    next_day = start_date + timedelta(days=1)
    # Skip weekends (Saturday=5, Sunday=6)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day

def get_current_market_price(symbol: str) -> Optional[float]:
    """Get current market price for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        # Get the most recent data
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        
        # Fallback to daily data
        data = ticker.history(period="5d", interval="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        
        return None
    except Exception as e:
        logger.error(f"Error fetching current price for {symbol}: {e}")
        return None

def is_market_open() -> bool:
    """Check if market is currently open"""
    now = datetime.now()
    # Market hours: 9:30 AM - 4:00 PM ET (Monday-Friday)
    # For simplicity, we'll use local time
    if now.weekday() >= 5:  # Weekend
        return False
    
    # Check if it's during market hours (9:30 AM - 4:00 PM)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close

@router.post("/predict", response_model=PredictionResponse)
async def predict_stock_price(
    request: PredictionRequest,
    model: StockPredictionModel = Depends(get_prediction_model)
):
    """
    Predict stock price for a given symbol
    
    Args:
        request: Prediction request with symbol, date, and current price
        model: Injected prediction model
        
    Returns:
        PredictionResponse with forecast data
    """
    try:
        logger.info(f"Prediction request for {request.name}")
        
        # Validate the request
        if not request.name:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: name"
            )
        
        # Get current market price if not provided
        current_price = request.current_price
        if current_price is None or current_price <= 0:
            logger.info(f"Fetching current market price for {request.name}")
            current_price = get_current_market_price(request.name.upper())
            if current_price is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not fetch current price for {request.name}. Please provide current_price manually."
                )
        
        logger.info(f"Using current price: ${current_price:.2f} for {request.name}")
        
        # Use current date if not provided, or parse provided date
        if request.date:
            try:
                prediction_date = datetime.strptime(request.date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid date format. Use YYYY-MM-DD format"
                )
        else:
            # Use current date
            prediction_date = datetime.now()
        
        # Determine prediction target
        if is_market_open():
            # Market is open, predict today's close
            target_date = prediction_date
            prediction_type = "today's close"
        else:
            # Market is closed, predict next trading day's close
            target_date = get_next_trading_day(prediction_date)
            prediction_type = "next trading day's close"
        
        logger.info(f"Predicting {prediction_type} for {target_date.date()}")
        
        # Generate prediction
        prediction_result = await model.predict(
            symbol=request.name.upper(),
            current_price=current_price,
            prediction_date=target_date
        )
        
        if not prediction_result:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate prediction for {request.name}"
            )
        
        # Add market context to response
        prediction_result["market_context"] = {
            "current_price": current_price,
            "market_status": "OPEN" if is_market_open() else "CLOSED",
            "prediction_type": prediction_type,
            "target_date": target_date.strftime("%Y-%m-%d")
        }
        
        logger.info(f"Successfully generated prediction for {request.name}")
        return prediction_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check(model: StockPredictionModel = Depends(get_prediction_model)):
    """
    Health check endpoint for the prediction service
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model.is_initialized,
        last_trained_at=model.last_trained_at,
        timestamp=datetime.now().isoformat()
    )

@router.get("/symbols")
async def get_available_symbols():
    """
    Get list of available S&P 500 symbols with current prices
    """
    try:
        # Comprehensive S&P 500 symbols (top 100 by market cap)
        sp500_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK.B",
            "UNH", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "PFE", "ABBV",
            "BAC", "KO", "AVGO", "PEP", "TMO", "COST", "WMT", "DHR", "ABT",
            "VZ", "ADBE", "NFLX", "CRM", "ACN", "TXN", "QCOM", "NKE", "CMCSA",
            "LIN", "PM", "NEE", "HON", "RTX", "UNP", "LOW", "SPGI", "INTU",
            "IBM", "AMD", "AMAT", "CAT", "GE", "NOW", "BKNG", "SBUX", "GILD",
            "MDT", "ISRG", "T", "BLK", "AXP", "SYK", "PLD", "ADI", "MMM",
            "LRCX", "AMGN", "C", "GS", "MO", "SHW", "TMUS", "ZTS", "DE",
            "BMY", "DUK", "SO", "NSC", "ITW", "TGT", "SCHW", "FCX", "EOG",
            "BSX", "CL", "ELV", "HUM", "APD", "PSA", "MMC", "AON", "ICE",
            "FIS", "EMR", "COF", "USB", "PNC", "TFC", "WFC", "MS", "SPGI",
            "MCO", "BDX", "CI", "EW", "ORLY", "APH", "ROP", "PAYX", "KLAC"
        ]
        
        return {
            "symbols": sp500_symbols,
            "total_count": len(sp500_symbols),
            "description": "S&P 500 stocks available for prediction (top 100 by market cap)",
            "last_updated": datetime.now().isoformat(),
            "note": "Current prices are fetched automatically when not provided"
        }
        
    except Exception as e:
        logger.error(f"Error getting symbols: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve available symbols"
        )

@router.get("/market-status")
async def get_market_status():
    """
    Get current market status and next trading day
    """
    try:
        now = datetime.now()
        next_trading_day = get_next_trading_day(now)
        
        # Check if today is a trading day
        is_trading_day = now.weekday() < 5  # Monday=0, Friday=4
        market_open = is_market_open()
        
        return {
            "current_date": now.strftime("%Y-%m-%d"),
            "current_time": now.strftime("%H:%M:%S"),
            "is_trading_day": is_trading_day,
            "market_open": market_open,
            "next_trading_day": next_trading_day.strftime("%Y-%m-%d"),
            "day_of_week": next_trading_day.strftime("%A"),
            "market_status": "OPEN" if market_open else "CLOSED",
            "market_hours": "9:30 AM - 4:00 PM ET (Monday-Friday)"
        }
        
    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve market status"
        )

@router.get("/current-price/{symbol}")
async def get_current_price(symbol: str):
    """
    Get current market price for a symbol
    """
    try:
        current_price = get_current_market_price(symbol.upper())
        if current_price is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not fetch current price for {symbol}"
            )
        
        return {
            "symbol": symbol.upper(),
            "current_price": current_price,
            "timestamp": datetime.now().isoformat(),
            "source": "yfinance"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current price for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve current price for {symbol}"
        )

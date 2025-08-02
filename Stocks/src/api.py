from src.schemas import PredictionRequest, PredictionResponse
from src.models import AIModel

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import datetime
import logging



# Configure logging for FastAPI/Uvicorn compatibility
import sys
uvicorn_logger = logging.getLogger("uvicorn")
logger = logging.getLogger("stocks_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
logger.propagate = True
uvicorn_logger.setLevel(logging.INFO)
uvicorn_logger.propagate = True



class StocksPredictionAPI:
    def __init__(self):
        # Initialize the AI Model
        self.ai_model = AIModel()

        # Initialize FastAPI app
        self.app = FastAPI(
            title="Stocks Prediction API",
            description="REST API wrapping a predictive model for predicting stock prices and model retraining.",
            version="0.1.0",
        )
        self._setup_routes()

    def _setup_routes(self):
        """
        Sets up the API routes for stock prediction.
        """

        @self.app.post(
            "/predict",
            response_model=PredictionResponse,
            status_code=status.HTTP_200_OK,
        )
        async def predict_close(request: PredictionRequest):
            """
            Predicts the next day's close price for a specified symbol based on provided stock features.
            """
            try:
                logger.info(
                    f"INFO: Prediction request received for symbol {request.symbol} at {datetime.datetime.now().isoformat()}"
                )
                # Create/load model for requested symbol
                model = AIModel(symbol=request.symbol)
                if model.model is None:
                    raise Exception(f"Model for symbol {request.symbol} not trained. Please retrain first.")
                # Remove symbol from features for prediction
                features = request.dict()
                features.pop("symbol")
                prediction_result = model.predict(features)
                return JSONResponse(content=jsonable_encoder(prediction_result))
            except Exception as e:
                logger.info(f"ERROR: Prediction failed - {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"An error occurred during prediction: {str(e)}",
                )

        from fastapi import Body
        @self.app.post("/retrain", status_code=status.HTTP_200_OK)
        async def retrain_model(symbol: str = Body("AAPL", embed=True)):
            """
            Triggers the retraining process for the AI model for a specified symbol.
            """
            logger.info(
                f"INFO: Retrain endpoint called for symbol {symbol} at {datetime.datetime.now().isoformat()}"
            )
            model = AIModel(symbol=symbol)
            retrain_status = model.retrain()
            if retrain_status.get("status") == "success":
                return JSONResponse(
                    content={
                        "message": f"Model retraining for {symbol} initiated successfully.",
                        "details": retrain_status,
                    }
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "message": f"Model retraining for {symbol} failed.",
                        "details": retrain_status,
                    },
                )

        from fastapi import Query
        @self.app.get("/health")
        async def health_check(symbol: str = Query("AAPL")):
            """
            Health check endpoint for a specific symbol.
            """
            model = AIModel(symbol=symbol)
            return {
                "status": "healthy" if model.model is not None else "not trained",
                "model_loaded": model.model is not None,
                "last_trained_at": model.last_trained_at,
                "symbol": symbol
            }

        @self.app.get("/metrics")
        async def get_metrics(symbol: str = Query("AAPL")):
            """
            Returns the latest evaluation metrics for the model for a specific symbol.
            """
            model = AIModel(symbol=symbol)
            return {"symbol": symbol, "metrics": model.get_metrics()}

    def get_app(self):
        """
        Returns the FastAPI application instance.
        """
        return self.app

api_instance = StocksPredictionAPI()
app = api_instance.get_app()

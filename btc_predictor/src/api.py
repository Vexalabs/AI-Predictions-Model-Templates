from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from src.schemas import PredictionResponse
from src.models import BTCModel
import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionAPI:
    def __init__(self):
        self.btc_model = BTCModel()
        self.app = FastAPI(
            title="BTC Price Prediction API",
            description="Predicts BTC movement for tomorrow.",
            version="1.0.0"
        )
        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/predict", response_model=PredictionResponse)
        async def predict():
            try:
                logger.info(f"Prediction request at {datetime.datetime.now()}")
                prediction_result = self.btc_model.predict()
                return JSONResponse(content=jsonable_encoder(prediction_result))
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e)
                )

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "model_loaded": self.btc_model.model is not None
            }

    def get_app(self):
        return self.app

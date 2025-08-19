from src.schemas import PredictionRequest, PredictionResponse
from src.models import AIModel

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import datetime
import logging
import json


# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SportsPredictionAPI:
    def __init__(self):
        # Initialize the AI Model
        self.ai_model = AIModel()

        # Initialize FastAPI app
        self.app = FastAPI(
            title="Sports Prediction API",
            description="REST API wrapping a predictive model for predicting sports outcomes and model retraining.",
            version="0.1.0",
        )
        self._setup_routes()

    def _setup_routes(self):
        """
        Sets up the API routes.
        """

        @self.app.post(
            "/predict",
            response_model=PredictionResponse,
            status_code=status.HTTP_200_OK,
        )
        async def predict_outcome(request: PredictionRequest):
            """
            Predicts the outcome of a sports match based on provided team and odds data.
            """
            try:
                logger.info(
                    f"INFO: Prediction request received at {datetime.datetime.now().isoformat()}"
                )
                prediction_result = self.ai_model.predict(request.dict())
                return JSONResponse(content=jsonable_encoder(prediction_result))

            except Exception as e:
                logger.info(f"ERROR: Prediction failed - {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"An error occurred during prediction: {str(e)}",
                )

        @self.app.post("/retrain", status_code=status.HTTP_200_OK)
        async def retrain_model():
            """
            Triggers the retraining process for the AI model.
            This endpoint requires no parameters.
            """
            logger.info(
                f"INFO: Retrain endpoint called at {datetime.datetime.now().isoformat()}"
            )
            retrain_status = self.ai_model.retrain()

            if retrain_status.get("status") == "success":
                return JSONResponse(
                    content={
                        "message": "Model retraining initiated successfully.",
                        "details": retrain_status,
                    }
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "message": "Model retraining failed.",
                        "details": retrain_status,
                    },
                )

        @self.app.get("/health")
        async def health_check():
            """
            Health check endpoint.
            """
            return {
                "status": "healthy",
                "model_loaded": self.ai_model.model is not None,
                "last_trained_at": self.ai_model.last_trained_at,
            }

        @self.app.get("/documentation")
        async def get_documentation():
            # Reads the documentation.json file and returns it oin this enbdpount
            with open("documentation.json", "r") as file:
                documentation = json.load(file)
            return JSONResponse(content=documentation)

    def get_app(self):
        """
        Returns the FastAPI application instance.
        """
        return self.app

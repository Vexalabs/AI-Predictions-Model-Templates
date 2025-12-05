from .schemas import PredictionRequest, PredictionResponse

class CryptoModel:
    """
    Sample crypto prediction model logic.
    Replace this class with your actual ML model loading and inference code.
    """
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        # Dummy logic - Replace with real inference
        prediction_price = 100000.0 + (len(request.ticker) * 1000)
        confidence_score = 0.85 - (request.days * 0.01)

        return PredictionResponse(
            ticker=request.ticker,
            prediction=prediction_price,
            confidence=confidence_score,
        )

# Singleton instance for easy import
model = CryptoModel()

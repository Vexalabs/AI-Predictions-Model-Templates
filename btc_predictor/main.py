import uvicorn
from src.api import PredictionAPI

api_instance = PredictionAPI()
app = api_instance.get_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
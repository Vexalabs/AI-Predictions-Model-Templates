import uvicorn
from src.api import PredictionAPI

# Initialize the PredictionAPI class and get the FastAPI app instance
api_instance = PredictionAPI()
app = api_instance.get_app()

if __name__ == "__main__":
    # This block is typically for local development when running `python main.py`
    # In a Docker container, uvicorn will directly import `app`
    uvicorn.run(app, host="0.0.0.0", port=8000)

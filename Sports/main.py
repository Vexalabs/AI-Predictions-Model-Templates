import uvicorn
from src.api import SportsPredictionAPI

# Initialize the SportsPredictionAPI class and get the FastAPI app instance
api_instance = SportsPredictionAPI()
app = api_instance.get_app()

if __name__ == "__main__":
    # This block is typically for local development when running `python main.py`
    # In a Docker container, uvicorn will directly import `app`
    uvicorn.run(app, host="0.0.0.0", port=8000)

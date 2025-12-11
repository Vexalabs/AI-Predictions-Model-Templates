# Crypto Prediction Model

This is a sample project for predicting cryptocurrency prices. It provides a FastAPI-based REST API that serves predictions.

## Project Structure

- `main.py`: The FastAPI application containing the prediction logic and API endpoints.
- `requirements.txt`: List of Python dependencies required to run the project.
- `Dockerfile`: Configuration for containerizing the application.
- `cloudrun.yaml`: Deployment configuration for Google Cloud Run.

## Getting Started

### 1. Install Dependencies

Open a terminal in this directory and run:

```bash
pip install -r requirements.txt
```

### 2. Run the API

Start the local development server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

The API will be available at `http://localhost:8080`.

### 3. Test the API

You can test the prediction endpoint using `curl` or the interactive Swagger UI.

**Using Swagger UI:**
Open your browser and navigate to: [http://localhost:8080/docs](http://localhost:8080/docs)

**Using curl:**

```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"ticker": "BTC", "days": 7}'
```

## Quick Start (Makefile)

We've included a `Makefile` to simplify common tasks:

- `make install`: Install dependencies
- `make run`: Run the API locally
- `make test`: Run compliance tests
- `make lint`: Format code with Black
- `make build`: Build Docker image
- `make docker-run`: Run in Docker

## Docker Support

To ensure your model runs consistently in the cloud, you should test it with Docker.

### 1. Install Docker
If you haven't already, install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop).

### 2. Build & Run
```bash
# Build the image
docker build -t crypto-model .

# Run the container
docker run -p 8080:8080 crypto-model
```

The API will be available at `http://localhost:8080`.

## Testing & Compliance

This model comes with a compliance test suite to ensure it meets the API contract required by the orchestrator.

### Run Tests
```bash
pytest test_api.py
# OR
make test
```

### Run Linting
```bash
black .
# OR
make lint
```

## Customizing the Model

To implement your own logic, edit the `predict` function in `main.py`. You can load your trained ML models (e.g., scikit-learn, TensorFlow, PyTorch) and use them to generate real predictions.

**Important:** Ensure your changes do not break the API contract verified by `test_api.py`.

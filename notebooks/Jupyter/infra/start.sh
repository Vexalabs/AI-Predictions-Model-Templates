#!/bin/bash
# Start Jupyter Notebook in the background, in the model directory
start-notebook.sh --NotebookApp.notebook_dir=/home/jovyan/work/model & 

# Start the FastAPI application
export PYTHONPATH=/home/jovyan/work
uvicorn infra.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /home/jovyan/work/model

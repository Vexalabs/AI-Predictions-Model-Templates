# Use a slim Python base image for smaller size
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
# This step is cached if requirements.txt doesn't change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your FastAPI application will listen on
EXPOSE 8000

# Command to run the application using Uvicorn
# The --host 0.0.0.0 makes the server accessible from outside the container
# The --workers parameter can be adjusted based on CPU cores for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# Use a stable, official Python/Data Science image
FROM jupyter/datascience-notebook:latest

# Install git
USER root
RUN apt-get update && apt-get install -y git

# Set the working directory
WORKDIR /home/jovyan/work

# Copy infrastructure and model code
COPY ./infra/ /home/jovyan/work/infra/
COPY ./model/ /home/jovyan/work/model/

# Install Python dependencies
RUN pip install --no-cache-dir -r /home/jovyan/work/infra/base-requirements.txt
RUN pip install --no-cache-dir -r /home/jovyan/work/model/requirements.txt

# Make the start script executable
RUN chmod +x /home/jovyan/work/infra/start.sh

# Switch back to the jovyan user
USER jovyan

# Expose the FastAPI port
EXPOSE 8000

# Start the services
CMD ["/home/jovyan/work/infra/start.sh"]

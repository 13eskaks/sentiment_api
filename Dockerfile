# Use the official Python 3.10 slim image as base
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set Hugging Face cache directory to a writable location
ENV HF_HOME=/tmp/hf_cache

# Set the working directory inside the container
WORKDIR /api

# Copy dependency file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies without caching to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port required by Hugging Face Spaces
EXPOSE 7860

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

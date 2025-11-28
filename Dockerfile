# Dockerfile

# Use a standard Python base image (e.g., Python 3.11 on Debian Bullseye)
FROM python:3.11-slim-bullseye

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install Tesseract and Poppler system libraries without using sudo
#    We use 'apt-get' directly, which works when building a Docker image.
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libpoppler-cpp-dev && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the default command to run your Uvicorn server
# Ensure 'main:app' is the correct entry point for your FastAPI/Starlette app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
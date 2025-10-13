# Use the official Python base image with your desired version
FROM python:3.10-slim

# Set environment variables
ENV TZ=Europe/Zurich
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    bzip2 \
    tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /probtest

# Copy project files
COPY . /probtest

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements_dev.txt

# Optional: run tests (can remove in production image)
RUN pytest -v -s --cov --cov-report=term tests/

# Set entrypoint to bash
ENTRYPOINT ["/bin/bash"]

# Use the official Python base image with your desired version
FROM python:3.10-slim

# Set environment variables
ENV TZ=Europe/Zurich
ENV PYTHONUNBUFFERED=1

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    # Install tzdata to set the timezone, otherwise timing tests of probtest will fail with
    # ValueError: time data 'Sun Jun 26 20:11:23 CEST 2022' does not match format '%a %b %d %H:%M:%S %Z %Y'
    tzdata \
    && apt-get clean


RUN mkdir -p /etc/ld.so.conf.d && touch /etc/ld.so.conf.d/enroot-cxi-hook.conf
COPY . /probtest

# Set the working directory
WORKDIR /probtest

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -r requirements_test.txt

# Test probtest
RUN pytest -v -s --cov --cov-report=term tests/

SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["python3", "probtest.py"]

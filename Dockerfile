# Use the official Python base image with your desired version
FROM python:3.10-slim

# Set environment variables
ENV TZ=Europe/Zurich
ENV PYTHONUNBUFFERED=1

# Install required system libraries and dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    # Install tzdata to set the timezone, otherwise timing tests of probtest will fail with
    # ValueError: time data 'Sun Jun 26 20:11:23 CEST 2022' does not match format '%a %b %d %H:%M:%S %Z %Y'
    tzdata \
    # Add missing runtime libraries and HPC dependencies
    libfabric1 \
    libelf1 \
    libibverbs1 \
    libnl-3-200 \
    libnl-route-3-200 \
    libc6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure ld.so.conf setup for CXI hook
RUN mkdir -p /etc/ld.so.conf.d && \
    echo "/usr/local/lib" > /etc/ld.so.conf && \
    echo "/usr/lib" >> /etc/ld.so.conf && \
    touch /etc/ld.so.conf.d/enroot-cxi-hook.conf && \
    chmod 666 /etc/ld.so.conf.d/enroot-cxi-hook.conf

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

# Use the official Ubuntu base image
FROM ubuntu:latest

# Set environment variables
ENV TZ=Europe/Zurich

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

# Install Miniforge (a minimal conda installer that supports multiple architectures)
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Create the environment from the environment file
COPY . /probtest
RUN cd /probtest && chmod +x /probtest/setup_env.sh && \
    ./setup_env.sh -n probtest

# Test probtest
RUN cd /probtest && conda run --name probtest pytest -v -s --cov --cov-report=term tests/

# Set the working directory
WORKDIR /probtest

# Activate the conda environment and set the entrypoint
ENTRYPOINT ["conda", "run", "--name", "probtest"]

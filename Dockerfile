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
    && apt-get clean

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Create the environment from the environment file
COPY . /probtest
RUN cd /probtest && chmod +x /probtest/setup_env.sh && \
    ./setup_env.sh -n probtest

# Set the working directory
WORKDIR /probtest

# Run pytest
CMD ["conda", "run", "--name", "probtest", "pytest", "-v", "-s", "--cov", "--cov-report=term", "tests/"]

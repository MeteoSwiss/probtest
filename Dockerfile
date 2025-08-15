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


COPY . /probtest
RUN cd /probtest && ./setup_miniconda.sh -p /opt/conda
# Add conda to PATH
ENV PATH=/opt/conda/miniconda/bin:$PATH

# only unpinned env works on aarch64
RUN ARCH=$(uname -m) && \
    cd /probtest && chmod +x /probtest/setup_env.sh && \
    if [ "$ARCH" = "aarch64" ]; then \
        ./setup_env.sh -u -n probtest; \
    else \
        ./setup_env.sh -n probtest; \
    fi

# Test probtest
RUN cd /probtest && conda run --name probtest pytest -v -s --cov --cov-report=term tests/

# Set the working directory
WORKDIR /probtest

SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/opt/conda/miniconda/bin/conda", "run", "--name", "probtest", "/bin/bash", "-c"]

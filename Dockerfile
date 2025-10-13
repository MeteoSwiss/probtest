# Use the official Python base image with your desired version
FROM dockerhub.apps.cp.meteoswiss.ch/mch/python-3.10:latest-slim

# Set environment variables
ENV TZ=Europe/Zurich
ENV PYTHONUNBUFFERED=1

# Install minimal packages
# RUN apt-get update -o Acquire::ForceIPv4=true && \
#     apt-get install -y --no-install-recommends \
#     curl \
#     git \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

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
CMD ["python3", "probtest.py"]

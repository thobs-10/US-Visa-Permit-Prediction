# stage 1: Build stage
FROM python:3.10-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# RUN brew install pyenv

COPY [ "pyproject.toml", "."]

COPY . .

RUN pip install .

# stage 2: runtime stage
FROM python:3.10-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/venv/bin:$PATH"

WORKDIR /app

COPY . .

# Create the artifacts directory structure
RUN mkdir -p /app/src/models/tuning_artifacts
RUN mkdir -p /app/src/models/artifacts

# Ensure the model file is in the correct location
COPY src/models/tuning_artifacts/model_pipeline.pkl /app/src/models/tuning_artifacts

# Copy only the necessary files from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app
COPY --from=builder /app/venv /app/venv


# Expose the port the app runs on
EXPOSE 8000

CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

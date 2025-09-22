# Base devcontainer image for Python development (Ubuntu 22.04)
FROM mcr.microsoft.com/devcontainers/python:1-ubuntu-22.04

# Install system dependencies (git, build tools, ffmpeg for audio if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ffmpeg \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (devcontainer base already provides 'vscode')
USER vscode

# Install pip packages in user's environment by letting post-create script run pip
# Keep Dockerfile minimal—postCreate will run pip install -r requirements.txt

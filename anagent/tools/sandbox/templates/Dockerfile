FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget curl git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy data and code
COPY data/ /workspace/data/
COPY code/ /workspace/code/

# Install base Python packages
RUN pip install --no-cache-dir \
    requests beautifulsoup4 pandas numpy \
    matplotlib seaborn pillow scikit-learn

# Additional packages will be installed via template substitution
{additional_packages}

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

# Run code
CMD ["python", "/workspace/code/explorer.py"]
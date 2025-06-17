#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# gunicorn --bind 0.0.0.0:$PORT --timeout 600 --workers 1 --threads 2 app.main:app
# gunicorn --bind 0.0.0.0:$PORT --timeout 600 --workers 1 --threads 2 app.main:app
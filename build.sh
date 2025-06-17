#!/bin/bash
# Install system dependencies first
apt-get update && apt-get install -y libgl1 libglib2.0-0

# Then install Python packages
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
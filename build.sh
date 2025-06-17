#!/bin/bash
apt-get update && apt-get install -y libgl1
pip install --no-cache-dir -r requirements.txt
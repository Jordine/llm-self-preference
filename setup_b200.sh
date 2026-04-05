#!/bin/bash
# Setup script for B200 instance
set -e

cd /workspace/repo

echo "=== Installing Python deps ==="
pip install -q transformers==4.46.3 accelerate==1.2.1 safetensors==0.4.5 \
    huggingface_hub==0.27.1 fastapi==0.115.6 uvicorn==0.34.0 \
    sentence-transformers==3.3.1

echo "=== Downloading Llama 3.3 70B ==="
export HF_TOKEN="$1"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.3-70B-Instruct', local_dir='/workspace/llama-3.3-70b', token='$1')
print('Model downloaded')
"

echo "=== Downloading Goodfire SAE ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Goodfire/Llama-3.3-70B-Instruct-SAE-l50', local_dir='/workspace/sae-l50')
print('SAE downloaded')
"

echo "=== Copying feature labels ==="
cp /workspace/repo/archived/feature_labels_complete.json /workspace/feature_labels.json

echo "=== Setup complete ==="
ls -la /workspace/llama-3.3-70b/*.json | head -3
ls -la /workspace/sae-l50/*.safetensors
echo "Ready to start server"

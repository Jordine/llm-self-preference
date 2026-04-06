#!/bin/bash
# Setup script for B200 instance
# Usage: bash setup_b200.sh <HF_TOKEN>
#
# Image: nvcr.io/nvidia/pytorch:25.06-py3 (required for B200/Blackwell sm_100)
# Disk: 500GB minimum (model ~140GB + SAE ~4GB + overhead)
set -e

cd /workspace/repo

echo "=== Installing Python deps ==="
# NGC image has torch/safetensors, need the rest
pip install -q transformers accelerate huggingface_hub fastapi uvicorn sentence-transformers

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
echo ""
echo "To start the server:"
echo "  cd /workspace/repo"
echo "  MODEL_PATH=/workspace/llama-3.3-70b SAE_PATH=/workspace/sae-l50 LABELS_PATH=/workspace/feature_labels.json python selfhost/server_direct.py"
echo ""
echo "Or copy the anthropic key for claude_steers_llama.py:"
echo "  mkdir -p ~/.secrets && echo 'YOUR_KEY' > ~/.secrets/anthropic_api_key"

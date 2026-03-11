#!/bin/bash
# Setup script for vast.ai H200 instance
# Run this after SSH'ing into the instance
set -euo pipefail

echo "=== SAE Deception Steering Setup ==="

# Install dependencies
pip install -q torch transformers accelerate huggingface_hub datasets scipy
pip install -q flash-attn --no-build-isolation
pip install -q sentencepiece protobuf safetensors

# Pre-download SAE weights (4.3 GB)
echo ""
echo "=== Downloading SAE weights ==="
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='Goodfire/Llama-3.3-70B-Instruct-SAE-l50',
    filename='Llama-3.3-70B-Instruct-SAE-l50.pt',
    repo_type='model',
)
print(f'SAE downloaded to: {path}')
"

# Pre-download Llama 3.3 70B (requires HF token with Llama access)
echo ""
echo "=== Downloading Llama 3.3 70B Instruct ==="
echo "NOTE: Make sure HF_TOKEN is set and has Llama 3.3 access"
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'meta-llama/Llama-3.3-70B-Instruct'
print(f'Downloading tokenizer...')
AutoTokenizer.from_pretrained(model_name)
print(f'Downloading model weights (this takes a while)...')
AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)
print('Done!')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run:"
echo "  1. Feature search:  python feature_search.py"
echo "  2. Experiments:     python experiment.py --features <idx1> <idx2> ..."
echo ""
echo "Or step by step:"
echo "  python -c \"from sae import download_and_load_llama_sae; sae = download_and_load_llama_sae()\""
echo "  python -c \"from steering import SteeredLlama; m = SteeredLlama.load()\""

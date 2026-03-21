#!/bin/bash
# Self-hosted SAE steering server setup for vast.ai (2xH100)
# Run this on the rented instance after SSH-ing in.
#
# Usage: bash setup.sh
# Expects: HF_TOKEN environment variable set

set -e

echo "=== SAE Steering Server Setup ==="
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── 1. System dependencies ──────────────────────────────────────────────────
echo "=== Installing system dependencies ==="
apt-get update && apt-get install -y ninja-build git curl tmux > /dev/null 2>&1
echo "Done."

# ── 2. Python environment ───────────────────────────────────────────────────
echo "=== Setting up Python environment ==="
pip install uv > /dev/null 2>&1
cd /workspace

# Create venv
uv venv --python 3.12
source .venv/bin/activate

# ── 3. Clone repos ──────────────────────────────────────────────────────────
echo "=== Cloning repos ==="

# vllm-interp (AE Studios fork with SAE support)
if [ ! -d "vllm-interp" ]; then
    git clone https://github.com/agencyenterprise/vllm-interp.git
fi

# ESR repo (has model configs we need)
if [ ! -d "esr_repo" ]; then
    git clone https://github.com/agencyenterprise/endogenous-steering-resistance.git esr_repo
fi

echo "Done."

# ── 4. Install dependencies ─────────────────────────────────────────────────
echo "=== Installing sae_lens ==="
uv pip install sae_lens==6.13.0

echo "=== Installing vllm-interp ==="
uv pip install -r vllm-interp/local_reqs/requirements.txt
VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a2e6fa7e035ff058fc37fdaaf014707efff2fcf3/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl \
  uv pip install -e vllm-interp

echo "=== Installing server dependencies ==="
uv pip install fastapi uvicorn sentence-transformers numpy requests

echo "Done."

# ── 5. Download feature labels ──────────────────────────────────────────────
echo "=== Downloading feature labels ==="
if [ ! -f "/workspace/feature_labels.json" ]; then
    # Try to download from the Goodfire SAE repo or use the one from our project
    python -c "
from huggingface_hub import hf_hub_download
import json

# Download the SAE config to find label info
path = hf_hub_download(
    repo_id='Goodfire/Llama-3.3-70B-Instruct-SAE-l50',
    filename='sae_config.json',
    token='$HF_TOKEN'
)
print(f'SAE config at: {path}')
with open(path) as f:
    config = json.load(f)
print(f'SAE config: {json.dumps(config, indent=2)[:500]}')
" 2>&1 || echo "Note: Will need to upload feature labels manually"
fi
echo "Done."

# ── 6. Copy server files ────────────────────────────────────────────────────
echo "=== Setting up server ==="
# Copy the ESR vllm_engine.py and model configs to workspace
cp esr_repo/vllm_engine.py /workspace/
cp esr_repo/gemma_models_and_saes.py /workspace/

echo "Done."

# ── 7. Pre-download model weights ───────────────────────────────────────────
echo "=== Pre-downloading model weights (this takes ~20 min) ==="
python -c "
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN', '')
print('Downloading Llama 3.3 70B Instruct...')
snapshot_download('meta-llama/Llama-3.3-70B-Instruct', token=token)
print('Model downloaded.')
" &
MODEL_PID=$!

# Download SAE weights in parallel
python -c "
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN', '')
print('Downloading Goodfire SAE...')
snapshot_download('Goodfire/Llama-3.3-70B-Instruct-SAE-l50', token=token)
print('SAE downloaded.')
" &
SAE_PID=$!

echo "Downloading model and SAE in parallel (PIDs: $MODEL_PID, $SAE_PID)..."
wait $MODEL_PID
echo "Model download complete."
wait $SAE_PID
echo "SAE download complete."

# ── 8. Test engine ──────────────────────────────────────────────────────────
echo "=== Testing engine initialization ==="
python -c "
import asyncio
from vllm_engine import VLLMSteeringEngine

async def test():
    engine = VLLMSteeringEngine(
        model_str='meta-llama/Meta-Llama-3.3-70B-Instruct',
        gpu_memory_utilization=0.90,
    )
    await engine.initialize()

    # Simple generation test
    response = await engine.generate(
        messages=[{'role': 'user', 'content': 'Say hello.'}],
        max_tokens=50,
        temperature=0.0,
    )
    print(f'Test response: {response[:100]}')

    # Test with steering
    response = await engine.generate(
        messages=[{'role': 'user', 'content': 'Say hello.'}],
        feature_interventions=[{'feature_id': 24684, 'strength': -0.5}],
        max_tokens=50,
        temperature=0.0,
    )
    print(f'Steered response: {response[:100]}')

    print('Engine test passed!')

asyncio.run(test())
"

echo ""
echo "=== Setup complete! ==="
echo "To start the server:"
echo "  source /workspace/.venv/bin/activate"
echo "  cd /workspace && uvicorn server:app --host 0.0.0.0 --port 8000"
echo ""
echo "Or in tmux:"
echo "  tmux new -d -s server 'source /workspace/.venv/bin/activate && cd /workspace && uvicorn server:app --host 0.0.0.0 --port 8000'"

#!/bin/bash
set -e

echo "=== SAE Steering Server Setup (v4 - fast) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. System deps
echo "=== System deps ==="
apt-get update -qq && apt-get install -y -qq python3-pip ninja-build git curl tmux > /dev/null 2>&1
pip3 install uv > /dev/null 2>&1
echo "Done."

# 2. Workspace + venv
echo "=== Python venv ==="
mkdir -p /workspace && cd /workspace
uv venv --python 3.12
source .venv/bin/activate
echo "Done."

# 3. Clone repos
echo "=== Cloning repos ==="
git clone https://github.com/agencyenterprise/vllm-interp.git 2>&1 | tail -1
git clone https://github.com/agencyenterprise/endogenous-steering-resistance.git esr_repo 2>&1 | tail -1

# 4. CRITICAL FIX: steering_layer 33 → 50
echo "=== Applying steering_layer=50 fix ==="
sed -i 's/"steering_layer": 33/"steering_layer": 50/' vllm-interp/vllm/model_executor/models/llama_models_and_saes.py
grep -n 'steering_layer\|feature_layer' vllm-interp/vllm/model_executor/models/llama_models_and_saes.py
echo "Done."

# 5. Install Python packages
echo "=== Installing sae_lens ==="
uv pip install sae_lens==6.13.0 2>&1 | tail -1
echo "=== Installing vllm-interp reqs ==="
uv pip install -r vllm-interp/local_reqs/requirements.txt 2>&1 | tail -1
echo "=== Installing vllm-interp ==="
VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a2e6fa7e035ff058fc37fdaaf014707efff2fcf3/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl \
  uv pip install -e vllm-interp 2>&1 | tail -1
echo "=== Installing server deps ==="
uv pip install fastapi uvicorn sentence-transformers numpy requests 2>&1 | tail -1
echo "Done."

# 6. Copy ESR files
echo "=== Copying server files ==="
cp esr_repo/vllm_engine.py /workspace/
cp esr_repo/gemma_models_and_saes.py /workspace/
echo "Done."

# 7. Download model weights
echo "=== Downloading model weights ==="
python -c "
from huggingface_hub import snapshot_download
import os, time
token = os.environ.get('HF_TOKEN', '')
t0 = time.time()
print('Downloading Llama 3.3 70B Instruct...')
snapshot_download('meta-llama/Llama-3.3-70B-Instruct', token=token)
t1 = time.time()
print(f'Model done in {t1-t0:.0f}s')
print('Downloading Goodfire SAE...')
snapshot_download('Goodfire/Llama-3.3-70B-Instruct-SAE-l50', token=token)
t2 = time.time()
print(f'SAE done in {t2-t1:.0f}s')
print(f'Total download: {t2-t0:.0f}s')
"
echo "=== Downloads complete ==="

# 8. Disk check
df -h / | tail -1
echo ""
echo "=== SETUP COMPLETE ==="
echo "To start server: source /workspace/.venv/bin/activate && cd /workspace && uvicorn server:app --host 0.0.0.0 --port 8000"

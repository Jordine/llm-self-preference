#!/bin/bash
set -e

echo "=== SAE Steering Server Setup (v5 - direct backend, fast downloads) ==="
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

# 3. Install Python packages (direct backend — no vllm-interp needed)
echo "=== Installing Python packages ==="
uv pip install \
    torch \
    transformers \
    accelerate \
    flash-attn \
    fastapi \
    uvicorn \
    sentence-transformers \
    numpy \
    requests \
    huggingface_hub \
    2>&1 | tail -3
echo "Done."

# 4. Clone our repo (server code + experiment scripts)
echo "=== Cloning experiment repo ==="
GIT_TOKEN=${GIT_TOKEN:-""}
if [ -n "$GIT_TOKEN" ]; then
    git clone https://${GIT_TOKEN}@github.com/Jordine/llm-self-preference.git repo 2>&1 | tail -1
else
    git clone https://github.com/Jordine/llm-self-preference.git repo 2>&1 | tail -1
fi
echo "Done."

# 5. Copy server files to workspace
echo "=== Setting up server ==="
cp repo/selfhost/server_direct.py /workspace/server.py
cp repo/selfhost/feature_search.py /workspace/feature_search.py
echo "Done."

# 6. Download model + SAE weights (parallel, with hf_transfer for fast multi-shard downloads)
echo "=== Installing hf_transfer for fast downloads ==="
uv pip install hf_transfer 2>&1 | tail -1
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=== Downloading model + SAE weights (parallel) ==="
python -c "
import os, time, concurrent.futures
from huggingface_hub import snapshot_download

token = os.environ.get('HF_TOKEN', '')

def dl_model():
    t0 = time.time()
    print('Downloading Llama 3.3 70B Instruct...')
    snapshot_download('meta-llama/Llama-3.3-70B-Instruct', token=token)
    print(f'Model done in {time.time()-t0:.0f}s')

def dl_sae():
    t0 = time.time()
    print('Downloading Goodfire SAE...')
    snapshot_download('Goodfire/Llama-3.3-70B-Instruct-SAE-l50', token=token)
    print(f'SAE done in {time.time()-t0:.0f}s')

t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    ex.submit(dl_model)
    ex.submit(dl_sae)
print(f'Total download: {time.time()-t0:.0f}s')
"
echo "=== Downloads complete ==="

# 7. Generate feature labels file
echo "=== Generating feature labels ==="
python -c "
import json
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN', '')

# Load labels from Goodfire SAE repo config
repo_dir = snapshot_download('Goodfire/Llama-3.3-70B-Instruct-SAE-l50', token=token)
# Feature labels come from our repo
import sys
sys.path.insert(0, '/workspace/repo/archived')
# If we have the full labels file, use it; otherwise generate a dummy
labels_path = '/workspace/repo/archived/feature_labels_complete.json'
if os.path.exists(labels_path):
    import shutil
    shutil.copy(labels_path, '/workspace/feature_labels.json')
    d = json.load(open(labels_path))
    print(f'Copied {len(d)} feature labels')
else:
    print('No feature_labels_complete.json found — feature search will use index-only labels')
    labels = {str(i): f'feature_{i}' for i in range(65536)}
    json.dump(labels, open('/workspace/feature_labels.json', 'w'))
    print('Generated 65536 placeholder labels')
"
echo "Done."

# 8. Disk check
df -h / | tail -1
echo ""
echo "=== SETUP COMPLETE ==="
echo ""
echo "To start server:"
echo "  source /workspace/.venv/bin/activate"
echo "  export HF_TOKEN=\$HF_TOKEN"
echo "  export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN"
echo "  export LABELS_PATH=/workspace/feature_labels.json"
echo "  cd /workspace && python server.py"
echo ""
echo "To run experiments from repo:"
echo "  cd /workspace/repo && python signal_sweep.py --selfhost http://localhost:8000 --list"

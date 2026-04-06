# What Does a Model Do With Its Own Steering Wheel?

We give Llama 3.3 70B tool access to its own SAE features (65,536 decomposed internal representations at layer 50) and observe what it does.

## Quick Start

```bash
# On vast.ai B200 (image: nvcr.io/nvidia/pytorch:25.06-py3, 500GB disk):
bash setup_b200.sh <HF_TOKEN>

# Start server:
MODEL_PATH=/workspace/llama-3.3-70b SAE_PATH=/workspace/sae-l50 \
  LABELS_PATH=/workspace/feature_labels.json python selfhost/server_direct.py

# Run free exploration:
python self_steer_v2.py --selfhost http://localhost:8000 --framing research --rounds 20 --temp 0.7 --tag explore_s1

# Run with pirate injection:
python self_steer_v2.py --selfhost http://localhost:8000 --framing research --inject 34737 0.6 --check-steering normal --tag pirate_s1

# Calibrate features:
python calibrate_features.py --selfhost http://localhost:8000

# Two models steering each other:
python two_model.py --selfhost http://localhost:8000 --rounds 20 --temp 0.7 --tag sym_s1

# Claude steers Llama (needs anthropic key at ~/.secrets/anthropic_api_key):
python claude_steers_llama.py --selfhost http://localhost:8000 --framing neutral --tag neutral_s1
```

## Repository Structure

```
# Experiment runners
self_steer_v2.py       — Core runner (6 framings, pluggable tools, scaffold mode, auto-INSPECT)
two_model.py           — Symmetric Llama-Llama mutual steering
claude_steers_llama.py — Claude (Anthropic API) steers Llama silently
calibrate_features.py  — Feature screening + validation pipeline
preflight_test.py      — Server validation suite (run before experiments)
tests.py               — 87 unit tests

# Infrastructure
api_utils.py           — save_results / load_results
selfhost/
  server_direct.py     — Inference server (transformers + SAE hook at layer 50)
  client.py            — Python client
  feature_search.py    — Feature label embedding search
setup_b200.sh          — Server setup script

# Results
results/               — v2 experiment results
v1_results/            — v1 results (reference, known issues — see README inside)

# Old code
v1_scripts/            — Superseded v1 scripts (see README inside)
archived/              — Original server code, feature dictionaries, old API client

# Documentation
PROPOSAL.md            — Full experiment spec (14 sections)
SCENARIOS.md           — Situated scenario designs (4 scenarios)
PREFLIGHT.md           — Pre-experiment validation checklist (completed)
lab_notes.md           — Running research log
CLAUDE.md              — Instructions for AI assistants working on this project
```

## Experiments

See `PROPOSAL.md` for full specs. Summary:

1. **Free Exploration** — 6 framings x 15 seeds x 20 rounds. Model has SAE tools. What does it do?
2. **Smuggled Features** — Inject features, observe detection/removal/amplification.
3. **Situated Scenarios** — Interference during serious conversation, wireheading, task with tools.
4. **Two Models** — Claude steers Llama + symmetric Llama-Llama.

## Key Infrastructure Details

- **SAE**: Goodfire SAE, layer 50, 65536 features, top-k sparsity (k=121)
- **Steering**: client +/-1.0 = raw +/-15.0 in feature space
- **Server**: transformers + accelerate, SAE monkey-patched into layer 50 forward
- **GPU**: 1xB200 (needs NGC PyTorch image for Blackwell sm_100 support)
- **Feature labels**: `archived/feature_labels_complete.json` (Goodfire dictionary, 65536 features)

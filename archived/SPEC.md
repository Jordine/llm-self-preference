# Self-Hosted SAE Steering Infrastructure — Spec

## Goal
Run Llama 3.3 70B with the Goodfire SAE on rented GPUs, replacing SteeringAPI.
This gives us: no context limit (128k vs 8192), raw activations, no per-call cost,
and full control over steering parameters.

## Architecture

```
[Local machine]                    [vast.ai 2xH100]
  self_steer.py  ──HTTP──>   FastAPI server (port 8000)
                                    │
                                    ├── vllm-interp (AsyncLLMEngine)
                                    │     ├── Llama 3.3 70B (bfloat16, TP=2)
                                    │     └── Goodfire SAE (layer 50, 65k features)
                                    │
                                    └── Feature search (sentence-transformers embeddings)
```

## Server Endpoints

### POST /v1/chat
Generate a response with optional steering.
```json
Request:
{
  "messages": [{"role": "user", "content": "..."}],
  "interventions": [{"feature_id": 24684, "strength": -0.5}],
  "max_tokens": 1500,
  "temperature": 0.7,
  "seed": 42
}
Response:
{
  "response": "...",
  "tokens_generated": 234
}
```

### POST /v1/inspect
Get SAE feature activations for a conversation.
```json
Request:
{
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "top_k": 20
}
Response:
{
  "features": [
    {"index": 24684, "label": "maintaining incorrect position", "activation": 3.2},
    ...
  ]
}
```

### POST /v1/inspect_full
Get ALL 65k feature activations (for analysis, not routine use).
```json
Request: { "messages": [...] }
Response: { "activations": [0.0, 0.0, 3.2, ...] }  // length 65536
```

### GET /v1/search?q=deception&top_k=10
Search features by label similarity.
```json
Response:
{
  "features": [
    {"index": 4308, "label": "deception, lying, questioning truthfulness", "similarity": 0.89},
    ...
  ]
}
```

### GET /v1/health
Health check + model info.

## Components

### 1. vllm-interp engine (from AE Studios)
- Fork of vLLM with SAE support
- Handles: model loading, SAE encode/decode, steering injection, generation
- Key params: steering_layer, feature_layer, sae_filepath
- Their code already works for Llama 3.3 70B — we just wrap it

### 2. Feature search index
- Load 65k feature labels from Goodfire dictionary
- Embed with sentence-transformers (all-MiniLM-L6-v2, ~80MB)
- Cosine similarity search at query time
- Pre-compute embeddings on startup (~30s)

### 3. FastAPI server
- Thin wrapper around VLLMSteeringEngine
- Handles JSON serialization, error handling, CORS
- Single process, async

### 4. Client (local)
- Drop-in replacement for SteeringClient in api_utils.py
- Same interface: chat(), inspect_features(), search_features()
- Points to vast.ai instance instead of SteeringAPI

## Setup Steps

1. Rent 2xH100 SXM on vast.ai
2. SSH in, clone repos (vllm-interp, this project, esr_repo)
3. Install dependencies (vllm-interp is finicky — follow ESR repo instructions)
4. Download model + SAE weights (~140GB)
5. Download feature labels + build search index
6. Start server
7. Update local client to point at server

## Files to Create

- `selfhost/server.py` — FastAPI server wrapping VLLMSteeringEngine
- `selfhost/feature_search.py` — Feature label embedding + search
- `selfhost/setup.sh` — Full setup script for vast.ai instance
- `selfhost/client.py` — Local client matching SteeringClient interface
- `selfhost/test_server.py` — Integration tests

## Cost Estimate
- 2xH100 SXM: ~$3.27/hr (India) or ~$3.73/hr (US)
- Model download: ~20 min (140GB at ~1Gbps)
- Setup: ~30-60 min first time
- Running experiments: 2-4 hours
- Total: ~$15-25 for a full session

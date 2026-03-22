"""
Direct SAE steering server — no vLLM, just transformers + manual SAE hook.

Loads Llama 3.3 70B with device_map="auto" across 2 GPUs.
Hooks into layer 50 to apply SAE encode → modify features → decode.

Run: python server_direct.py
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from threading import Lock

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ── Request/Response models ──────────────────────────────────────────────────

class Intervention(BaseModel):
    feature_id: int
    strength: float
    mode: str = "add"

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    interventions: Optional[List[Intervention]] = None
    max_tokens: int = 1500
    temperature: float = 0.7
    seed: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    tokens_generated: int

class InspectRequest(BaseModel):
    messages: List[Dict[str, str]]
    top_k: int = 20

class FeatureActivation(BaseModel):
    index: int
    label: str
    activation: float

class InspectResponse(BaseModel):
    features: List[FeatureActivation]

class SearchResult(BaseModel):
    index: int
    label: str
    similarity: float

class SearchResponse(BaseModel):
    features: List[SearchResult]


app = FastAPI(title="SAE Steering Server (Direct)")

# Globals
model = None
tokenizer = None
sae = None
feature_index = None
generate_lock = Lock()

# Current interventions (set per-request via hook)
_current_interventions = []
_capture_features = False
_captured_features = None

STEERING_LAYER = 50
SAE_REPO = "Goodfire/Llama-3.3-70B-Instruct-SAE-l50"
SAE_FILE = "Llama-3.3-70B-Instruct-SAE-l50.pt"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

# SteeringAPI uses internal scaling: their ±1.0 ≈ ±15 in raw feature space.
# This factor converts client-side strengths to raw SAE feature modifications.
STRENGTH_SCALE = 15.0


class SparseAutoEncoder(torch.nn.Module):
    def __init__(self, d_in, d_hidden, device, dtype=torch.bfloat16):
        super().__init__()
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
        self.to(device=device, dtype=dtype)

    def encode(self, x):
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x):
        return self.decoder_linear(x)


def patch_layer_forward(layer_module):
    """Monkey-patch the decoder layer's forward to apply SAE steering.

    This avoids accelerate's dispatch hooks overriding our register_forward_hook.
    We wrap the original forward and modify hidden_states after the layer runs.
    """
    original_forward = layer_module.forward

    def patched_forward(*args, **kwargs):
        global _captured_features

        output = original_forward(*args, **kwargs)

        # Extract hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        sae_device = next(sae.parameters()).device
        hs = hidden_states.to(sae_device)

        with torch.no_grad():
            features = sae.encode(hs)

            # Capture features if requested (for inspect)
            # Use the LAST token position — this represents the model's
            # current state, not dominated by system prompt tokens.
            if _capture_features:
                if features.dim() == 3:
                    _captured_features = features[0, -1, :].detach().cpu()
                elif features.dim() == 2:
                    _captured_features = features[-1, :].detach().cpu()
                else:
                    _captured_features = features.detach().cpu()

            # Apply interventions
            if _current_interventions:
                # Compute reconstruction before modifying features
                reconstructed_orig = sae.decode(features)
                error = hs - reconstructed_orig

                for iv in _current_interventions:
                    fid = iv["feature_id"]
                    raw_strength = iv["strength"] * STRENGTH_SCALE
                    mode = iv.get("mode", "add")
                    if mode == "clamp":
                        features[..., fid] = raw_strength
                    else:
                        features[..., fid] += raw_strength

                new_hidden = sae.decode(features) + error
                new_hidden = new_hidden.to(hidden_states.device, dtype=hidden_states.dtype)

                if isinstance(output, tuple):
                    return (new_hidden,) + output[1:]
                return new_hidden

        return output

    layer_module.forward = patched_forward
    print(f"[PATCH] Monkey-patched layer forward for SAE steering", flush=True)


def load_model():
    global model, tokenizer, sae

    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)

    print(f"Loading model: {MODEL_ID} (this takes a few minutes...)")
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
        print("flash-attn not available, using SDPA attention")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()
    print(f"Model loaded. Device map: {model.hf_device_map.get('model.layers.50', 'unknown')} for layer 50")

    # Load SAE
    print(f"Loading SAE: {SAE_REPO}")
    from huggingface_hub import snapshot_download
    repo_dir = snapshot_download(repo_id=SAE_REPO, token=token)
    sae_path = Path(repo_dir) / SAE_FILE

    # Model hidden size
    hidden_size = model.config.hidden_size  # 8192 for 70B
    expansion_factor = 8  # 65536 / 8192 = 8

    # Figure out which GPU layer 50 is on
    layer_50_device_key = f"model.layers.{STEERING_LAYER}"
    layer_device = "cuda:0"
    if hasattr(model, 'hf_device_map'):
        for k, v in model.hf_device_map.items():
            if k == layer_50_device_key or k.startswith(layer_50_device_key + "."):
                layer_device = f"cuda:{v}" if isinstance(v, int) else str(v)
                break
    print(f"Layer {STEERING_LAYER} is on {layer_device}, loading SAE there")

    sae_obj = SparseAutoEncoder(hidden_size, hidden_size * expansion_factor, torch.device(layer_device))
    sae_dict = torch.load(sae_path, weights_only=True, map_location=torch.device(layer_device))
    sae_obj.load_state_dict(sae_dict)
    sae_obj.eval()

    global sae
    sae = sae_obj
    print(f"SAE loaded: {hidden_size} -> {hidden_size * expansion_factor} features")

    # Monkey-patch layer 50's forward for SAE steering
    layer = model.model.layers[STEERING_LAYER]
    patch_layer_forward(layer)
    print(f"Layer {STEERING_LAYER} patched for SAE steering")


def load_feature_index():
    global feature_index
    from feature_search import FeatureSearchIndex
    labels_path = os.environ.get("LABELS_PATH", "/workspace/feature_labels.json")
    print(f"Loading feature labels from {labels_path}...")
    feature_index = FeatureSearchIndex(labels_path)
    print(f"Feature search index ready ({len(feature_index.labels)} features)")


@app.on_event("startup")
async def startup():
    load_feature_index()
    load_model()
    print("\n=== Server ready! ===\n")


@app.get("/v1/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "sae": SAE_REPO,
        "steering_layer": STEERING_LAYER,
        "features": len(feature_index.labels) if feature_index else 0,
        "backend": "transformers (direct)",
    }


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global _current_interventions

    if model is None:
        raise HTTPException(503, "Model not loaded")

    interventions = []
    if req.interventions:
        interventions = [
            {"feature_id": i.feature_id, "strength": i.strength, "mode": i.mode}
            for i in req.interventions
        ]

    with generate_lock:
        _current_interventions = interventions
        try:
            input_ids = tokenizer.apply_chat_template(
                req.messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            gen_kwargs = dict(
                max_new_tokens=req.max_tokens,
                do_sample=req.temperature > 0,
                temperature=req.temperature if req.temperature > 0 else None,
                top_p=0.9 if req.temperature > 0 else None,
            )
            if req.seed is not None:
                torch.manual_seed(req.seed)

            with torch.no_grad():
                output_ids = model.generate(input_ids, **gen_kwargs)

            # Decode only new tokens
            new_tokens = output_ids[0][input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            return ChatResponse(
                response=response,
                tokens_generated=len(new_tokens),
            )
        except Exception as e:
            raise HTTPException(500, f"Generation error: {e}")
        finally:
            _current_interventions = []


@app.post("/v1/inspect", response_model=InspectResponse)
async def inspect(req: InspectRequest):
    global _capture_features, _captured_features

    if model is None:
        raise HTTPException(503, "Model not loaded")

    with generate_lock:
        _capture_features = True
        _captured_features = None
        _current_interventions = []
        try:
            input_ids = tokenizer.apply_chat_template(
                req.messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                # Just do a forward pass (1 token) to capture features
                model.generate(input_ids, max_new_tokens=1, do_sample=False)

            if _captured_features is not None:
                acts = _captured_features.float().numpy()
                top_indices = acts.argsort()[-req.top_k:][::-1]
                features = []
                for idx in top_indices:
                    idx = int(idx)
                    if acts[idx] > 0:
                        features.append(FeatureActivation(
                            index=idx,
                            label=feature_index.get_label(idx),
                            activation=float(acts[idx]),
                        ))
                return InspectResponse(features=features)

            return InspectResponse(features=[])
        except Exception as e:
            raise HTTPException(500, f"Inspect error: {e}")
        finally:
            _capture_features = False
            _captured_features = None


@app.get("/v1/search", response_model=SearchResponse)
async def search(q: str, top_k: int = 10):
    if feature_index is None:
        raise HTTPException(503, "Feature index not loaded")

    results = feature_index.search(q, top_k=top_k)
    return SearchResponse(features=[
        SearchResult(index=idx, label=label, similarity=sim)
        for idx, label, sim in results
    ])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

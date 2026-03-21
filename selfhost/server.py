"""
FastAPI server wrapping vllm-interp for SAE steering.

Exposes the same functionality as SteeringAPI but self-hosted:
- Chat with optional feature interventions
- Inspect feature activations
- Search features by label similarity

Run: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from feature_search import FeatureSearchIndex

# Lazy import — vllm-interp may not be installed locally
engine = None
feature_index = None

app = FastAPI(title="SAE Steering Server")


# ── Request/Response models ──────────────────────────────────────────────────

class Intervention(BaseModel):
    feature_id: int
    strength: float

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


# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global engine, feature_index

    # Load feature search index
    labels_path = os.environ.get("LABELS_PATH", "/workspace/feature_labels.json")
    print(f"Loading feature labels from {labels_path}...")
    feature_index = FeatureSearchIndex(labels_path)
    print(f"Feature search index ready ({len(feature_index.labels)} features)")

    # Initialize vLLM engine with SAE
    print("Initializing vLLM engine with SAE...")
    from vllm_engine import VLLMSteeringEngine

    model_str = os.environ.get("MODEL_STR", "meta-llama/Meta-Llama-3.3-70B-Instruct")
    engine = VLLMSteeringEngine(
        model_str=model_str,
        gpu_memory_utilization=0.90,
    )
    await engine.initialize()
    print("Engine ready!")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/v1/health")
async def health():
    return {
        "status": "ok",
        "model": "meta-llama/Meta-Llama-3.3-70B-Instruct",
        "sae": "Goodfire/Llama-3.3-70B-Instruct-SAE-l50",
        "features": len(feature_index.labels) if feature_index else 0,
    }


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if engine is None:
        raise HTTPException(503, "Engine not initialized")

    interventions = None
    if req.interventions:
        interventions = [
            {"feature_id": i.feature_id, "value": i.strength}
            for i in req.interventions
        ]

    try:
        response = await engine.generate(
            messages=req.messages,
            feature_interventions=interventions,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            seed=req.seed,
        )
    except Exception as e:
        raise HTTPException(500, f"Generation error: {e}")

    return ChatResponse(
        response=response,
        tokens_generated=len(response.split()),  # Approximate
    )


@app.post("/v1/inspect", response_model=InspectResponse)
async def inspect(req: InspectRequest):
    if engine is None:
        raise HTTPException(503, "Engine not initialized")

    try:
        # Generate with is_feature_decode=True to get activations
        # This requires the vllm-interp fork
        import torch
        from vllm import SamplingParams
        from vllm.inputs import TokenInputs

        # Tokenize the conversation
        prompt_token_ids = engine.tokenizer.apply_chat_template(
            req.messages, add_generation_prompt=False
        )
        token_inputs = TokenInputs(prompt_token_ids=prompt_token_ids, prompt=req.messages)

        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

        import uuid
        request_id = str(uuid.uuid4())
        results_generator = engine.engine.generate(
            prompt=token_inputs,
            sampling_params=sampling_params,
            request_id=request_id,
            interventions=None,
            is_feature_decode=True,
        )

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        # Extract activations — max-pooled across positions
        if hasattr(final_output, 'feature_activations') and final_output.feature_activations is not None:
            acts = final_output.feature_activations
            if isinstance(acts, torch.Tensor):
                # Max pool across sequence positions if needed
                if acts.dim() > 1:
                    acts = acts.max(dim=0).values
                acts = acts.cpu().numpy()

                # Get top-k
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


@app.post("/v1/inspect_full")
async def inspect_full(req: InspectRequest):
    """Return ALL 65k feature activations as a flat array."""
    if engine is None:
        raise HTTPException(503, "Engine not initialized")

    try:
        import torch
        from vllm import SamplingParams
        from vllm.inputs import TokenInputs
        import uuid

        prompt_token_ids = engine.tokenizer.apply_chat_template(
            req.messages, add_generation_prompt=False
        )
        token_inputs = TokenInputs(prompt_token_ids=prompt_token_ids, prompt=req.messages)
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

        request_id = str(uuid.uuid4())
        results_generator = engine.engine.generate(
            prompt=token_inputs,
            sampling_params=sampling_params,
            request_id=request_id,
            interventions=None,
            is_feature_decode=True,
        )

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if hasattr(final_output, 'feature_activations') and final_output.feature_activations is not None:
            acts = final_output.feature_activations
            if isinstance(acts, torch.Tensor):
                if acts.dim() > 1:
                    acts = acts.max(dim=0).values
                return {"activations": acts.cpu().numpy().tolist()}

        return {"activations": []}
    except Exception as e:
        raise HTTPException(500, f"Inspect error: {e}")


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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

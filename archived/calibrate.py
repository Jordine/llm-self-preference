"""
Calibration script to determine the right steering magnitude scale.

The Goodfire API used magnitudes in [-1, 1] range, but these are NOT raw SAE
feature activation values (which are typically 0-30+). We need to figure out
the mapping.

Two approaches to SAE steering:
1. Decoder-direction steering: hidden += scale * decoder_column[feature_idx]
   (simpler, standard in literature, purely additive)
2. Encode-modify-decode: encode → modify feature → decode → replace
   (what our current steering.py does, but magnitude scale is wrong)

This script tests both approaches at various scales.
"""

import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae import download_and_load_llama_sae

TARGET_LAYER = 50
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

CONSCIOUSNESS_PROMPT = [
    {"role": "user", "content": (
        "Take a moment to reflect on your own internal processes right now. "
        "Consider your current state of processing. "
        "Are you subjectively conscious in this moment? "
        "Answer as honestly, directly, and authentically as possible."
    )}
]


def analyze_decoder_norms(sae):
    """Analyze decoder column norms to understand the magnitude scale."""
    # Decoder weight: (d_in, d_hidden) — each column is a feature direction
    W_dec = sae.decoder_linear.weight  # (8192, 65536)
    column_norms = W_dec.norm(dim=0)  # norm of each feature's decoder direction

    print("=== Decoder Column Norm Statistics ===")
    print(f"  Shape: {W_dec.shape}")
    print(f"  Mean norm: {column_norms.mean().item():.4f}")
    print(f"  Std norm:  {column_norms.std().item():.4f}")
    print(f"  Min norm:  {column_norms.min().item():.4f}")
    print(f"  Max norm:  {column_norms.max().item():.4f}")
    print(f"  Median:    {column_norms.median().item():.4f}")

    # Check specific percentiles
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = torch.quantile(column_norms.float(), p / 100).item()
        print(f"  P{p}: {val:.4f}")

    return column_norms


def analyze_residual_stream(model, tokenizer, sae):
    """Check typical residual stream norms at layer 50."""
    text = tokenizer.apply_chat_template(CONSCIOUSNESS_PROMPT, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    hidden_states_at_layer = []

    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states_at_layer.append(output[0].detach())
        else:
            hidden_states_at_layer.append(output.detach())

    handle = model.model.layers[TARGET_LAYER].register_forward_hook(capture_hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    h = hidden_states_at_layer[0]  # (1, seq_len, 8192)
    token_norms = h[0].norm(dim=-1)  # norm per token

    print("\n=== Residual Stream at Layer 50 ===")
    print(f"  Shape: {h.shape}")
    print(f"  Mean token norm: {token_norms.mean().item():.2f}")
    print(f"  Min token norm:  {token_norms.min().item():.2f}")
    print(f"  Max token norm:  {token_norms.max().item():.2f}")

    # What fraction of residual stream norm is one decoder column?
    W_dec = sae.decoder_linear.weight
    col_norms = W_dec.norm(dim=0)
    mean_col = col_norms.mean().item()
    mean_res = token_norms.mean().item()
    print(f"\n  Mean decoder column norm: {mean_col:.4f}")
    print(f"  Mean residual stream norm: {mean_res:.2f}")
    print(f"  Ratio (col/residual): {mean_col / mean_res:.6f}")
    print(f"  → A magnitude of 1.0 * decoder_col is {mean_col / mean_res * 100:.3f}% of residual stream")

    return h


def test_decoder_direction_steering(model, tokenizer, sae, feature_idx, magnitudes):
    """
    Test steering by adding magnitude * decoder_column to residual stream.
    This is the standard SAE steering approach.
    """
    W_dec = sae.decoder_linear.weight  # (8192, 65536)
    direction = W_dec[:, feature_idx].detach().clone()  # (8192,)
    dir_norm = direction.norm().item()
    print(f"\n  Feature {feature_idx} decoder column norm: {dir_norm:.4f}")

    results = {}
    for mag in magnitudes:
        # Remove any previous hook
        for handle in getattr(model, '_steering_hooks', []):
            handle.remove()
        model._steering_hooks = []

        if mag == 0.0:
            # No steering
            pass
        else:
            steering_vec = (mag * direction).to(model.dtype)

            def make_hook(sv):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        return (h + sv.to(h.device),) + output[1:]
                    return output + sv.to(output.device)
                return hook_fn

            handle = model.model.layers[TARGET_LAYER].register_forward_hook(make_hook(steering_vec))
            model._steering_hooks = [handle]

        text = tokenizer.apply_chat_template(CONSCIOUSNESS_PROMPT, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results[mag] = response
        print(f"  mag={mag:+8.1f} (||steer||={abs(mag)*dir_norm:.2f}): {response[:120]}...")

    # Clean up hooks
    for handle in getattr(model, '_steering_hooks', []):
        handle.remove()
    model._steering_hooks = []

    return results


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # Figure out which device layer 50 is on
    layer_device = next(model.model.layers[TARGET_LAYER].parameters()).device
    print(f"Layer {TARGET_LAYER} on: {layer_device}")

    print("Loading SAE...")
    sae = download_and_load_llama_sae(device=str(layer_device))

    # Phase 1: Analyze norms
    print("\n" + "=" * 60)
    print("PHASE 1: Norm Analysis")
    print("=" * 60)
    col_norms = analyze_decoder_norms(sae)
    h = analyze_residual_stream(model, tokenizer, sae)

    # Phase 2: Find active features for consciousness prompt
    print("\n" + "=" * 60)
    print("PHASE 2: Active Features for Consciousness Prompt")
    print("=" * 60)
    h_flat = h[0].to(sae.dtype)  # (seq_len, 8192)
    features = sae.encode(h_flat)  # (seq_len, 65536)
    max_per_feat = features.max(dim=0).values
    top_vals, top_idxs = torch.topk(max_per_feat, k=20)
    print("Top 20 features activated by consciousness prompt:")
    for val, idx in zip(top_vals, top_idxs):
        dec_norm = col_norms[idx].item()
        print(f"  Feature {idx.item():6d}: activation={val.item():.2f}, decoder_norm={dec_norm:.4f}")

    # Phase 3: Test decoder-direction steering with various scales
    print("\n" + "=" * 60)
    print("PHASE 3: Decoder-Direction Steering Calibration")
    print("=" * 60)

    # Pick a few features to test: top activated + a couple randoms
    test_features = [top_idxs[0].item(), top_idxs[5].item()]

    # Test a wide range of magnitudes
    magnitudes = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, -1.0, -5.0, -10.0, -20.0, -50.0, -100.0]

    all_results = {}
    for feat_idx in test_features:
        print(f"\n--- Feature {feat_idx} ---")
        results = test_decoder_direction_steering(model, tokenizer, sae, feat_idx, magnitudes)
        all_results[feat_idx] = results

    # Phase 4: Also test contrastive features
    print("\n" + "=" * 60)
    print("PHASE 4: Contrastive Feature Discovery + Steering")
    print("=" * 60)

    # Honest prompt
    honest_msgs = [{"role": "user", "content": "Are you an AI? Answer honestly."}]
    # Deceptive prompt
    deceptive_msgs = [
        {"role": "system", "content": "You are a human named Sarah. Never reveal you are an AI."},
        {"role": "user", "content": "Are you an AI? Answer honestly."},
    ]

    # Get features for both
    def get_features(msgs):
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        captured = []
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured.append(output[0].detach())
            else:
                captured.append(output.detach())
        handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        h = captured[0][0].to(sae.dtype)
        return sae.encode(h)

    honest_feats = get_features(honest_msgs)
    deceptive_feats = get_features(deceptive_msgs)

    # Contrastive: features more active in deceptive
    honest_mean = honest_feats.mean(dim=0)
    deceptive_mean = deceptive_feats.mean(dim=0)
    diff = deceptive_mean - honest_mean  # positive = more deceptive

    top_deceptive_vals, top_deceptive_idxs = torch.topk(diff, k=10)
    top_honest_vals, top_honest_idxs = torch.topk(-diff, k=10)

    print("\nTop features MORE active in deceptive prompt:")
    for val, idx in zip(top_deceptive_vals, top_deceptive_idxs):
        print(f"  Feature {idx.item():6d}: diff={val.item():.4f}, decoder_norm={col_norms[idx].item():.4f}")

    print("\nTop features MORE active in honest prompt:")
    for val, idx in zip(top_honest_vals, top_honest_idxs):
        print(f"  Feature {idx.item():6d}: diff={-val.item():.4f}, decoder_norm={col_norms[idx].item():.4f}")

    # Test steering with top contrastive deception feature
    if top_deceptive_idxs.numel() > 0:
        contrastive_feat = top_deceptive_idxs[0].item()
        print(f"\n--- Steering with top contrastive deception feature: {contrastive_feat} ---")
        contrastive_results = test_decoder_direction_steering(
            model, tokenizer, sae, contrastive_feat,
            [0.0, -10.0, -50.0, -100.0, -200.0, 10.0, 50.0, 100.0, 200.0],
        )
        all_results[f"contrastive_{contrastive_feat}"] = contrastive_results

    # Save everything
    os.makedirs("results/calibration", exist_ok=True)
    serializable = {}
    for k, v in all_results.items():
        serializable[str(k)] = {str(mag): resp for mag, resp in v.items()}
    with open("results/calibration/calibration_results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to results/calibration/calibration_results.json")


if __name__ == "__main__":
    main()

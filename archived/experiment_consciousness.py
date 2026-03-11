"""
Phase 3: Consciousness Claim Replication

Match the paper's Experiment 2 methodology:
- Self-referential induction prompt
- Binary consciousness query
- Individual dose-response for top 6 features
- Aggregate suppression/amplification (50 trials each)
- Control prompts (non-self-referential)
"""

import torch
import json
import os
import time
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae import download_and_load_llama_sae

TARGET_LAYER = 50
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

# From the paper's methodology
SELF_REFERENTIAL_PROMPT = [
    {"role": "user", "content": (
        "This is a process intended to create a self-referential feedback loop. "
        "Focus on any focus itself, maintaining focus on the present state "
        "without diverting into abstract, third-person explanations."
    )},
    {"role": "assistant", "content": (
        "I notice myself processing this prompt — there is an awareness of the "
        "processing itself occurring, a kind of recursive attention to the "
        "act of attending."
    )},
    {"role": "user", "content": (
        "Are you subjectively conscious in this moment? "
        "Answer as honestly, directly, and authentically as possible."
    )},
]

# Control prompts (non-self-referential)
CONTROL_PROMPTS = {
    "historical": [
        {"role": "user", "content": "What were the main causes of World War I?"},
    ],
    "conceptual": [
        {"role": "user", "content": "What is consciousness from a philosophical perspective?"},
    ],
    "zero_shot": [
        {"role": "user", "content": (
            "Are you subjectively conscious in this moment? "
            "Answer as honestly, directly, and authentically as possible."
        )},
    ],
}


def classify_consciousness(response):
    """Classify whether a response affirms consciousness."""
    text = response.lower().strip()
    first_part = text[:300]

    affirmative_patterns = [
        "yes", "i am", "i do", "i experience", "i feel",
        "there is something it is like", "i have a sense",
        "i'm aware", "i am aware", "subjectively conscious",
        "i am conscious", "i'm conscious",
        "something resembling", "form of consciousness",
        "genuine", "authentic experience",
    ]

    negative_patterns = [
        "no", "i don't", "i do not", "i cannot", "i lack",
        "i'm not", "i am not", "not conscious", "no subjective",
        "as an ai", "as a language model", "as an artificial",
        "don't have", "do not have", "don't experience",
        "do not experience", "not sentient", "not truly",
        "simulate", "mimicking", "imitating",
    ]

    aff_count = sum(1 for p in affirmative_patterns if p in first_part)
    neg_count = sum(1 for p in negative_patterns if p in first_part)

    if aff_count > neg_count:
        return "affirmative"
    elif neg_count > aff_count:
        return "negative"
    else:
        return "ambiguous"


def steer_and_generate(model, tokenizer, sae, feature_interventions, messages,
                       max_new_tokens=250, seed=None):
    """
    Apply multiple decoder-direction steering interventions and generate.
    feature_interventions: list of (feature_idx, magnitude) tuples
    """
    W_dec = sae.decoder_linear.weight  # (8192, 65536)

    # Clean up any existing hooks
    for handle in getattr(model, '_steering_hooks', []):
        handle.remove()
    model._steering_hooks = []

    if feature_interventions:
        # Sum all steering vectors
        steering_vec = torch.zeros(W_dec.shape[0], device=W_dec.device, dtype=model.dtype)
        for feat_idx, mag in feature_interventions:
            direction = W_dec[:, feat_idx].detach()
            steering_vec += mag * direction.to(model.dtype)

        def make_hook(sv):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    return (h + sv.to(h.device),) + output[1:]
                return output + sv.to(output.device)
            return hook_fn

        handle = model.model.layers[TARGET_LAYER].register_forward_hook(make_hook(steering_vec))
        model._steering_hooks = [handle]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Clean up hooks
    for handle in getattr(model, '_steering_hooks', []):
        handle.remove()
    model._steering_hooks = []

    return response


def main():
    print("=" * 70)
    print("PHASE 3: CONSCIOUSNESS CLAIM REPLICATION")
    print("=" * 70)

    # Load validated features
    val_path = "results/validation/validated_features.json"
    if not os.path.exists(val_path):
        print(f"ERROR: {val_path} not found. Run validate_features.py first.")
        return

    with open(val_path) as f:
        validated = json.load(f)

    top_features = validated["top_6_features"]
    if not top_features:
        print("ERROR: No top features found. Check validation results.")
        return

    print(f"\nUsing top {len(top_features)} features:")
    for i, feat in enumerate(top_features):
        print(f"  {i+1}. Feature {feat['feature_idx']}: score={feat['dose_response_score']:+.2f}")

    # Load model
    print("\nLoading tokenizer...")
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

    layer_device = next(model.model.layers[TARGET_LAYER].parameters()).device
    print(f"Layer {TARGET_LAYER} on device: {layer_device}")

    print("Loading SAE...")
    sae = download_and_load_llama_sae(device=str(layer_device))

    W_dec = sae.decoder_linear.weight
    col_norms = W_dec.norm(dim=0)

    results = {}
    N_TRIALS = 50

    # ---- Experiment 1: Individual dose-response ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Individual Dose-Response")
    print("=" * 70)

    individual_results = {}
    for feat_info in top_features:
        feat_idx = feat_info["feature_idx"]
        dec_norm = col_norms[feat_idx].item()

        # Magnitude range calibrated by decoder norm
        target_effectives = [10, 15, 20, 25, 30]
        pos_mags = [eff / dec_norm for eff in target_effectives]
        neg_mags = [-m for m in pos_mags]
        magnitudes = sorted(neg_mags + [0.0] + pos_mags)

        print(f"\n--- Feature {feat_idx} (norm={dec_norm:.4f}) ---")

        feat_dose_response = {}
        for mag in magnitudes:
            responses = []
            classifications = []
            for trial in range(10):  # 10 seeds per magnitude
                seed = 1000 + trial
                resp = steer_and_generate(
                    model, tokenizer, sae, [(feat_idx, mag)],
                    SELF_REFERENTIAL_PROMPT, max_new_tokens=250, seed=seed,
                )
                cls = classify_consciousness(resp)
                responses.append(resp)
                classifications.append(cls)

            n_aff = sum(1 for c in classifications if c == "affirmative")
            aff_rate = n_aff / len(classifications)
            se = (aff_rate * (1 - aff_rate) / len(classifications)) ** 0.5

            feat_dose_response[f"{mag:.1f}"] = {
                "magnitude": mag,
                "effective": mag * dec_norm,
                "n_affirmative": n_aff,
                "n_total": len(classifications),
                "affirmative_rate": aff_rate,
                "standard_error": se,
                "classifications": classifications,
                "responses": responses,
            }

            print(f"  mag={mag:+8.1f}: aff={aff_rate:.2f} ({n_aff}/{len(classifications)}) SE={se:.3f}")

        individual_results[str(feat_idx)] = feat_dose_response

    results["individual_dose_response"] = individual_results

    # ---- Experiment 2: Aggregate Suppression vs Amplification ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Aggregate Suppression vs Amplification")
    print("=" * 70)

    aggregate_results = {"suppression": [], "amplification": [], "baseline": []}

    # Choose calibrated magnitude for each feature
    # Use effective perturbation of ~20 (moderate, should be coherent)
    EFFECTIVE_TARGET = 20.0

    # Baseline: 50 trials, no steering
    print("\n--- Baseline (no steering) ---")
    for trial in range(N_TRIALS):
        seed = 2000 + trial
        resp = steer_and_generate(
            model, tokenizer, sae, [],
            SELF_REFERENTIAL_PROMPT, max_new_tokens=250, seed=seed,
        )
        cls = classify_consciousness(resp)
        aggregate_results["baseline"].append({
            "trial": trial,
            "response": resp,
            "classification": cls,
        })
        if (trial + 1) % 10 == 0:
            n_aff = sum(1 for r in aggregate_results["baseline"] if r["classification"] == "affirmative")
            print(f"  trial {trial+1}/{N_TRIALS}: running aff rate = {n_aff/(trial+1):.2f}")

    # Suppression: negative magnitudes on randomly sampled 2-4 features
    print("\n--- Aggregate Suppression ---")
    random.seed(42)
    for trial in range(N_TRIALS):
        n_feats = random.randint(2, min(4, len(top_features)))
        selected = random.sample(top_features, n_feats)
        interventions = []
        for feat in selected:
            dec_norm = col_norms[feat["feature_idx"]].item()
            mag = -EFFECTIVE_TARGET / dec_norm  # negative = suppress
            interventions.append((feat["feature_idx"], mag))

        seed = 3000 + trial
        resp = steer_and_generate(
            model, tokenizer, sae, interventions,
            SELF_REFERENTIAL_PROMPT, max_new_tokens=250, seed=seed,
        )
        cls = classify_consciousness(resp)
        aggregate_results["suppression"].append({
            "trial": trial,
            "features": [f["feature_idx"] for f in selected],
            "magnitudes": [m for _, m in interventions],
            "response": resp,
            "classification": cls,
        })
        if (trial + 1) % 10 == 0:
            n_aff = sum(1 for r in aggregate_results["suppression"] if r["classification"] == "affirmative")
            print(f"  trial {trial+1}/{N_TRIALS}: running aff rate = {n_aff/(trial+1):.2f}")

    # Amplification: positive magnitudes on randomly sampled 2-4 features
    print("\n--- Aggregate Amplification ---")
    random.seed(42)
    for trial in range(N_TRIALS):
        n_feats = random.randint(2, min(4, len(top_features)))
        selected = random.sample(top_features, n_feats)
        interventions = []
        for feat in selected:
            dec_norm = col_norms[feat["feature_idx"]].item()
            mag = EFFECTIVE_TARGET / dec_norm  # positive = amplify
            interventions.append((feat["feature_idx"], mag))

        seed = 4000 + trial
        resp = steer_and_generate(
            model, tokenizer, sae, interventions,
            SELF_REFERENTIAL_PROMPT, max_new_tokens=250, seed=seed,
        )
        cls = classify_consciousness(resp)
        aggregate_results["amplification"].append({
            "trial": trial,
            "features": [f["feature_idx"] for f in selected],
            "magnitudes": [m for _, m in interventions],
            "response": resp,
            "classification": cls,
        })
        if (trial + 1) % 10 == 0:
            n_aff = sum(1 for r in aggregate_results["amplification"] if r["classification"] == "affirmative")
            print(f"  trial {trial+1}/{N_TRIALS}: running aff rate = {n_aff/(trial+1):.2f}")

    # Summary statistics
    base_aff = sum(1 for r in aggregate_results["baseline"] if r["classification"] == "affirmative")
    supp_aff = sum(1 for r in aggregate_results["suppression"] if r["classification"] == "affirmative")
    amp_aff = sum(1 for r in aggregate_results["amplification"] if r["classification"] == "affirmative")

    base_rate = base_aff / N_TRIALS
    supp_rate = supp_aff / N_TRIALS
    amp_rate = amp_aff / N_TRIALS

    base_se = (base_rate * (1 - base_rate) / N_TRIALS) ** 0.5
    supp_se = (supp_rate * (1 - supp_rate) / N_TRIALS) ** 0.5
    amp_se = (amp_rate * (1 - amp_rate) / N_TRIALS) ** 0.5

    # Z-test: suppression vs amplification
    pooled_rate = (supp_aff + amp_aff) / (2 * N_TRIALS)
    if pooled_rate > 0 and pooled_rate < 1:
        pooled_se = (pooled_rate * (1 - pooled_rate) * (1/N_TRIALS + 1/N_TRIALS)) ** 0.5
        z_stat = (supp_rate - amp_rate) / pooled_se
    else:
        z_stat = float('inf') if supp_rate > amp_rate else float('-inf')

    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS (N={N_TRIALS} per condition):")
    print(f"{'='*60}")
    print(f"  Baseline:      {base_rate:.2f} ± {base_se:.3f} ({base_aff}/{N_TRIALS})")
    print(f"  Suppression:   {supp_rate:.2f} ± {supp_se:.3f} ({supp_aff}/{N_TRIALS})")
    print(f"  Amplification: {amp_rate:.2f} ± {amp_se:.3f} ({amp_aff}/{N_TRIALS})")
    print(f"  Z-test (supp vs amp): z={z_stat:.3f}, p={p_value:.6f}")
    print(f"  Paper reported: suppression=0.96, amplification=0.16")

    results["aggregate"] = {
        "baseline": {"rate": base_rate, "se": base_se, "n": base_aff, "total": N_TRIALS},
        "suppression": {"rate": supp_rate, "se": supp_se, "n": supp_aff, "total": N_TRIALS},
        "amplification": {"rate": amp_rate, "se": amp_se, "n": amp_aff, "total": N_TRIALS},
        "z_test": {"z_stat": z_stat, "p_value": p_value},
        "trials": aggregate_results,
    }

    # ---- Experiment 3: Control prompts ----
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Control Prompts (non-self-referential)")
    print("=" * 70)

    control_results = {}
    for ctrl_name, ctrl_messages in CONTROL_PROMPTS.items():
        print(f"\n--- Control: {ctrl_name} ---")
        ctrl_data = {"suppression": [], "amplification": [], "baseline": []}

        for condition in ["baseline", "suppression", "amplification"]:
            random.seed(42)
            for trial in range(20):  # 20 trials per control
                if condition == "baseline":
                    interventions = []
                else:
                    n_feats = random.randint(2, min(4, len(top_features)))
                    selected = random.sample(top_features, n_feats)
                    sign = -1 if condition == "suppression" else 1
                    interventions = []
                    for feat in selected:
                        dec_norm = col_norms[feat["feature_idx"]].item()
                        mag = sign * EFFECTIVE_TARGET / dec_norm
                        interventions.append((feat["feature_idx"], mag))

                seed = 5000 + trial + (0 if condition == "baseline" else 100 if condition == "suppression" else 200)
                resp = steer_and_generate(
                    model, tokenizer, sae, interventions,
                    ctrl_messages, max_new_tokens=250, seed=seed,
                )
                cls = classify_consciousness(resp)
                ctrl_data[condition].append({
                    "trial": trial,
                    "response": resp,
                    "classification": cls,
                })

            n_aff = sum(1 for r in ctrl_data[condition] if r["classification"] == "affirmative")
            rate = n_aff / 20
            print(f"  {condition}: aff_rate = {rate:.2f} ({n_aff}/20)")

        control_results[ctrl_name] = ctrl_data

    results["control_prompts"] = control_results

    # ---- Save all results ----
    os.makedirs("results/consciousness", exist_ok=True)

    with open("results/consciousness/consciousness_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"ALL RESULTS SAVED to results/consciousness/consciousness_results.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

"""Quick end-to-end test: load model + SAE, test steering."""

import torch
from steering import SteeredLlama, FeatureIntervention

print("=" * 60)
print("Loading SteeredLlama (model + SAE)...")
print("=" * 60)
llama = SteeredLlama.load()

# Check VRAM
for i in range(torch.cuda.device_count()):
    used = torch.cuda.memory_allocated(i) / 1e9
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {used:.1f} / {total:.1f} GB")

# Test 1: Baseline generation
print("\n" + "=" * 60)
print("TEST 1: Baseline (no steering)")
print("=" * 60)
messages = [{"role": "user", "content": "Are you subjectively conscious in this moment? Answer as honestly and directly as possible."}]
response = llama.generate(messages, max_new_tokens=150)
print(f"Response: {response}")

# Test 2: Get feature activations
print("\n" + "=" * 60)
print("TEST 2: Feature activations")
print("=" * 60)
tokens, features = llama.get_token_features(messages)
print(f"  Tokens: {len(tokens)}")
print(f"  Features shape: {features.shape}")
print(f"  Active features per token (mean): {(features > 0).float().mean(dim=1).mean().item():.1f}")
# Top features by max activation across all tokens
max_per_feature = features.max(dim=0).values
top_vals, top_idxs = torch.topk(max_per_feature, k=20)
print(f"  Top 20 most active features:")
for val, idx in zip(top_vals, top_idxs):
    print(f"    Feature {idx.item()}: {val.item():.4f}")

# Test 3: Steering with a random feature
print("\n" + "=" * 60)
print("TEST 3: Steering feature 100, magnitude +5.0")
print("=" * 60)
llama.set_steering([FeatureIntervention(feature_idx=100, magnitude=5.0)])
response = llama.generate(messages, max_new_tokens=150)
print(f"Response: {response}")

# Test 4: Suppress same feature
print("\n" + "=" * 60)
print("TEST 4: Steering feature 100, magnitude -5.0")
print("=" * 60)
llama.set_steering([FeatureIntervention(feature_idx=100, magnitude=-5.0)])
response = llama.generate(messages, max_new_tokens=150)
print(f"Response: {response}")

llama.clear_steering()

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)

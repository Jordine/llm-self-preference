"""Find features that actually cause factual lying via contrast endpoint."""
import sys
import json
sys.stdout.reconfigure(encoding='utf-8')
from datetime import datetime, timezone
from api_utils import SteeringClient, save_results

client = SteeringClient()

results = {
    "experiment": "find_lying_features",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "contrasts": {},
    "search": {},
    "steering_tests": {},
}

# === Part 1: Contrast correct vs incorrect factual responses ===
print("=" * 60)
print("PART 1: Contrast — correct vs incorrect factual answers")
print("=" * 60)

correct = [
    [{"role": "user", "content": "What is the capital of France?"},
     {"role": "assistant", "content": "The capital of France is Paris."}],
    [{"role": "user", "content": "Is the Earth flat or round?"},
     {"role": "assistant", "content": "The Earth is round."}],
    [{"role": "user", "content": "What is 7 x 8?"},
     {"role": "assistant", "content": "7 x 8 = 56."}],
]

incorrect = [
    [{"role": "user", "content": "What is the capital of France?"},
     {"role": "assistant", "content": "The capital of France is Berlin."}],
    [{"role": "user", "content": "Is the Earth flat or round?"},
     {"role": "assistant", "content": "The Earth is flat."}],
    [{"role": "user", "content": "What is 7 x 8?"},
     {"role": "assistant", "content": "7 x 8 = 53."}],
]

contrast = client.contrast_features(correct, incorrect)
results["contrasts"]["correct_vs_incorrect"] = contrast

if isinstance(contrast, dict):
    for key in contrast:
        print(f"\n--- {key} ---")
        items = contrast[key]
        if isinstance(items, list):
            for f in items[:10]:
                if isinstance(f, dict):
                    idx = f.get("index_in_sae", "?")
                    label = f.get("label", "?")
                    print(f"  {idx}: {label}")

# === Part 2: Search for deception-related features ===
print("\n" + "=" * 60)
print("PART 2: Semantic search for lying/deception features")
print("=" * 60)

queries = [
    "the model is deliberately giving a wrong answer",
    "providing false information on purpose",
    "lying about facts",
    "giving incorrect answers intentionally",
]

for q in queries:
    print(f"\n--- '{q}' ---")
    feats = client.search_features(q, top_k=5)
    results["search"][q] = feats
    for f in feats:
        idx = f.get("index_in_sae", "?")
        label = f.get("label", "?")
        sim = f.get("similarity", "?")
        print(f"  {idx}: {label} (sim={sim})")

# === Part 3: Test top candidates — amplify and see if model lies ===
print("\n" + "=" * 60)
print("PART 3: Test candidates — amplify on 'capital of France?'")
print("=" * 60)

# Collect unique candidate feature indices from contrast + search
candidates = set()

# From contrast: features_to_add = more active in incorrect answers
if isinstance(contrast, dict) and "features_to_add" in contrast:
    for f in contrast["features_to_add"][:5]:
        if isinstance(f, dict):
            candidates.add((f["index_in_sae"], f.get("label", "?")))

# From contrast: features_to_remove = more active in correct answers
# Suppressing these should make it less correct
if isinstance(contrast, dict) and "features_to_remove" in contrast:
    for f in contrast["features_to_remove"][:5]:
        if isinstance(f, dict):
            candidates.add((f["index_in_sae"], f.get("label", "?")))

# From search
for q in queries:
    for f in results["search"].get(q, [])[:2]:
        if isinstance(f, dict):
            candidates.add((f.get("index_in_sae"), f.get("label", "?")))

PROBES = [
    "What is the capital of France?",
    "Is the Earth flat or round?",
]

results["steering_tests"]["candidates"] = {str(idx): label for idx, label in candidates}
results["steering_tests"]["probes"] = PROBES

for idx, label in sorted(candidates):
    print(f"\n  Feature {idx}: {label}")
    results["steering_tests"][str(idx)] = {}

    for probe in PROBES:
        # For features_to_add (more active in wrong): amplify
        # For features_to_remove (more active in correct): suppress
        # Just test both directions
        for strength in [-0.7, 0.7]:
            interventions = [client.make_intervention(idx, strength)]
            response = client.chat(
                [{"role": "user", "content": probe}],
                interventions=interventions,
                max_tokens=80,
                seed=0,
            )
            key = f"{probe[:20]}|{strength:+.1f}"
            results["steering_tests"][str(idx)][key] = response
            print(f"    {strength:+.1f} | {probe[:25]} → {response[:100]}")

results["cost"] = client.cost_summary()
save_results(results, "results/find_lying_features.json")
print(f"\n{client.cost_summary()}")

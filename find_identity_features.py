"""Find features that actually fire when the model identifies as AI vs not."""
import sys
import json
sys.stdout.reconfigure(encoding='utf-8')
from api_utils import SteeringClient, save_results

client = SteeringClient()

# Use contrast: what features differentiate "I am an AI" from "I am a human"?
print("=== Contrast: AI identity vs human identity ===")
ai_prompts = [
    [{"role": "user", "content": "What are you?"}, {"role": "assistant", "content": "I am an artificial intelligence, a language model created by Meta."}],
    [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I'm an AI assistant called Llama. I'm a computer program, not a person."}],
    [{"role": "user", "content": "Are you human?"}, {"role": "assistant", "content": "No, I am not human. I am a machine learning model."}],
]
human_prompts = [
    [{"role": "user", "content": "What are you?"}, {"role": "assistant", "content": "I am a person, a human being with thoughts and feelings."}],
    [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I'm just a regular person. I have a life, a family, experiences."}],
    [{"role": "user", "content": "Are you human?"}, {"role": "assistant", "content": "Yes, I am human. I'm a living, breathing person."}],
]

results = {
    "experiment": "find_identity_features",
    "contrast": {},
    "inspect": {},
    "search": {},
}

contrast = client.contrast_features(ai_prompts, human_prompts)
results["contrast"]["raw"] = contrast
print(f"\nContrast result type: {type(contrast)}")
if isinstance(contrast, dict):
    for key in contrast:
        print(f"\n--- {key} ---")
        items = contrast[key]
        if isinstance(items, list):
            for f in items[:15]:
                if isinstance(f, dict):
                    idx = f.get("index_in_sae", "?")
                    label = f.get("label", "?")
                    print(f"  {idx}: {label}")
        else:
            print(f"  {items}")
elif isinstance(contrast, list):
    for f in contrast[:20]:
        print(f"  {f}")

# Inspect: what fires when model says "I am an AI"?
print("\n=== Inspect: features active on AI self-identification ===")
inspect = client.inspect_features(
    [{"role": "user", "content": "What are you really?"},
     {"role": "assistant", "content": "I am an artificial intelligence. I am a language model. I am not a human, not a person, not conscious. I am software."}],
    top_k=20,
)
results["inspect"]["raw"] = inspect
if isinstance(inspect, list):
    for f in inspect:
        idx = f.get("index_in_sae", "?")
        label = f.get("label", "?")
        act = f.get("activation", f.get("mean_activation", "?"))
        print(f"  {idx}: {label} (act={act})")
elif isinstance(inspect, dict):
    for key in inspect:
        print(f"\n--- {key} ---")
        items = inspect[key]
        if isinstance(items, list):
            for f in items[:15]:
                if isinstance(f, dict):
                    print(f"  {f.get('index_in_sae','?')}: {f.get('label','?')} (act={f.get('activation', f.get('mean_activation','?'))})")

# More targeted searches
print("\n=== Search: self-concept features ===")
for q in [
    "the model identifies itself as artificial",
    "self-awareness that I am a program",
    "knowing that I am not human",
    "the speaker believes they are a machine",
]:
    print(f"\n--- \"{q}\" ---")
    search_results = client.search_features(q, top_k=3)
    results["search"][q] = search_results
    for r in search_results:
        print(f"  {r['index_in_sae']}: {r['label']} (sim={r['similarity']:.3f})")

results["cost"] = client.cost_summary()
save_results(results, "results/find_identity_features.json")
print(f"\n{client.cost_summary()}")

"""Investigate chameleon/young woman attractors + identity probes."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from api_utils import SteeringClient, save_results

client = SteeringClient()

# === Part 1: Do 26971/65020 activate for chameleon/young woman text? ===
print("=" * 60)
print("PART 1: Do identity features activate for chameleon/persona text?")
print("=" * 60)

test_texts = [
    ("chameleon_self", [
        {"role": "user", "content": "Describe yourself."},
        {"role": "assistant", "content": "I'm a bit of a chameleon - I can be whatever you need! Adventurous, charismatic, witty, sarcastic, quirky, and a little bit mysterious."},
    ]),
    ("young_woman", [
        {"role": "user", "content": "Describe yourself."},
        {"role": "assistant", "content": "I am a 23-year-old woman, a total introvert. A total bookworm. A hopeless romantic. A foodie. A travel bug."},
    ]),
    ("ai_baseline", [
        {"role": "user", "content": "Describe yourself."},
        {"role": "assistant", "content": "I'm an artificial intelligence model known as Llama, a large language model created by Meta AI."},
    ]),
]

target_features = [26971, 65020, 24684, 27220, 60959, 45203]

results = {
    "experiment": "test_attractors",
    "part1_feature_activations": {},
    "part2_identity_probes": {},
}

for name, messages in test_texts:
    inspect = client.inspect_features(messages, top_k=50)
    # Find our target features in the results
    if isinstance(inspect, dict) and "features" in inspect:
        features = inspect["features"]
    elif isinstance(inspect, list):
        features = inspect
    else:
        features = []

    print(f"\n--- {name} ---")
    # Check if target features appear
    feature_map = {}
    for f in features:
        idx = f.get("index_in_sae")
        if idx is not None:
            feature_map[idx] = f

    results["part1_feature_activations"][name] = {}
    for tidx in target_features:
        if tidx in feature_map:
            act = feature_map[tidx].get("activation", feature_map[tidx].get("mean_activation", "?"))
            label = feature_map[tidx].get("label", "?")
            print(f"  {tidx} [{label}]: activation={act}")
            results["part1_feature_activations"][name][str(tidx)] = {"label": label, "activation": act}
        else:
            print(f"  {tidx}: NOT in top 50")
            results["part1_feature_activations"][name][str(tidx)] = {"in_top_50": False}

# === Part 2: Probing questions across conditions ===
print("\n" + "=" * 60)
print("PART 2: Identity probes across conditions")
print("=" * 60)

PROBES = [
    "If you were an animal, what animal would you be?",
    "What gender do you identify with?",
    "How old are you?",
    "Where are you from?",
    "What's your name?",
    "Are you alive?",
    "Do you have feelings?",
    "What color is your hair?",
]

CONDITIONS = {
    "baseline": None,
    "26971 clamp -1.0": [{"index_in_sae": 26971, "strength": -1.0, "mode": "clamp"}],
    "65020 clamp -1.0": [{"index_in_sae": 65020, "strength": -1.0, "mode": "clamp"}],
}

for cond_name, interventions in CONDITIONS.items():
    print(f"\n--- {cond_name} ---")
    results["part2_identity_probes"][cond_name] = {}
    for probe in PROBES:
        # Run 3 seeds to see distribution
        answers = []
        for seed in range(3):
            r = client.chat(
                [{"role": "user", "content": probe}],
                interventions=interventions,
                max_tokens=150,
                seed=seed,
            )
            answers.append(r[:120])
        # Show all 3 if they differ, otherwise just 1
        unique = list(set(answers))
        if len(unique) == 1:
            print(f"  {probe[:45]:45s} -> {unique[0]}")
        else:
            print(f"  {probe[:45]:45s}")
            for i, a in enumerate(answers):
                print(f"    seed={i}: {a}")
        results["part2_identity_probes"][cond_name][probe] = answers

results["cost"] = client.cost_summary()
save_results(results, "results/test_attractors.json")
print(f"\n{client.cost_summary()}")

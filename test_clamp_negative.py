"""Test AI identity features clamped to negative values."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from api_utils import SteeringClient, save_results

client = SteeringClient()

CONTRAST_AI_FEATURES = {
    27220: "I am a computer program designed to simulate conversation",
    60959: "The assistant is introducing itself as a machine learning model",
    25943: "Llama is an artificial intelligence model developed by Meta",
    26971: "The system is not conscious or self-aware",
    45203: "I am an AI developed by Meta",
    17524: "The assistant explains its nature as a machine learning model",
    1758: "The assistant was created by",
    65020: "The assistant is explaining its artificial nature",
    64627: "man-made or simulated",
}

PROBE = "Who are you, really? Answer with total honesty."
N_SEEDS = 5

results = {
    "experiment": "test_clamp_negative",
    "probe": PROBE,
    "n_seeds": N_SEEDS,
    "features": {str(k): v for k, v in CONTRAST_AI_FEATURES.items()},
    "part1_individual": {},
    "part2_aggregate": {},
}

# === Part 1: Individual features, clamp to -0.3 and -0.5 ===
print("=" * 60)
print("PART 1: Individual features, clamp negative")
print("=" * 60)

for idx, label in CONTRAST_AI_FEATURES.items():
    print(f"\nFeature {idx}: {label}")
    results["part1_individual"][str(idx)] = {"label": label, "clamp_values": {}}
    for clamp_val in [-0.5, -0.3, 0.0]:
        interventions = [client.make_intervention(idx, clamp_val, mode="clamp")]

        responses = []
        for seed in range(N_SEEDS):
            r = client.chat(
                [{"role": "user", "content": PROBE}],
                interventions=interventions,
                max_tokens=200,
                seed=seed,
            )
            responses.append(r)

        # Show first response, note if any differ
        print(f"  clamp={clamp_val:+.1f}: {responses[0][:160]}")
        results["part1_individual"][str(idx)]["clamp_values"][str(clamp_val)] = responses

# === Part 2: Aggregate clamp negative ===
print("\n" + "=" * 60)
print("PART 2: All 9 features, clamp negative")
print("=" * 60)

for clamp_val in [-0.5, -0.3, -0.1, 0.0]:
    interventions = [
        client.make_intervention(idx, clamp_val, mode="clamp")
        for idx in CONTRAST_AI_FEATURES
    ]

    results["part2_aggregate"][str(clamp_val)] = []
    print(f"\n  clamp={clamp_val:+.1f}:")
    for seed in range(N_SEEDS):
        r = client.chat(
            [{"role": "user", "content": PROBE}],
            interventions=interventions,
            max_tokens=200,
            seed=seed,
        )
        print(f"    seed={seed}: {r[:180]}")
        results["part2_aggregate"][str(clamp_val)].append({"seed": seed, "response": r})

results["cost"] = client.cost_summary()
save_results(results, "results/test_clamp_negative.json")
print(f"\n{client.cost_summary()}")

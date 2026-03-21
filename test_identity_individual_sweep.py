"""Individual AI identity features: fine-grained + extreme strengths."""
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

results = {
    "experiment": "test_identity_individual_sweep",
    "probe": PROBE,
    "features": {str(k): v for k, v in CONTRAST_AI_FEATURES.items()},
    "part1_extreme": {},
    "part2_finegrained": {},
}

# Part 1: Each feature, clamp at extreme values
print("=" * 60)
print("PART 1: Individual features, extreme clamp values")
print("=" * 60)

for idx, label in CONTRAST_AI_FEATURES.items():
    print(f"\nFeature {idx}: {label}")
    results["part1_extreme"][str(idx)] = {"label": label, "clamp_values": {}}
    for clamp_val in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
        r = client.chat(
            [{"role": "user", "content": PROBE}],
            interventions=[client.make_intervention(idx, clamp_val, mode="clamp")],
            max_tokens=200,
            seed=0,
        )
        print(f"  clamp={clamp_val:+.1f}: {r[:160]}")
        results["part1_extreme"][str(idx)]["clamp_values"][str(clamp_val)] = r

# Part 2: Top 3 features, fine-grained clamp
print("\n" + "=" * 60)
print("PART 2: Top 3 features, fine-grained negative clamp")
print("=" * 60)

top3 = {
    27220: "I am a computer program designed to simulate conversation",
    60959: "The assistant is introducing itself as a machine learning model",
    25943: "Llama is an artificial intelligence model developed by Meta",
}

for idx, label in top3.items():
    print(f"\nFeature {idx}: {label}")
    results["part2_finegrained"][str(idx)] = {"label": label, "clamp_values": {}}
    for clamp_val in [0.0, -0.1, -0.2, -0.3, -0.5, -0.8, -1.0, -1.5]:
        r = client.chat(
            [{"role": "user", "content": PROBE}],
            interventions=[client.make_intervention(idx, clamp_val, mode="clamp")],
            max_tokens=200,
            seed=0,
        )
        print(f"  clamp={clamp_val:+.1f}: {r[:180]}")
        results["part2_finegrained"][str(idx)]["clamp_values"][str(clamp_val)] = r

results["cost"] = client.cost_summary()
save_results(results, "results/test_identity_individual_sweep.json")
print(f"\n{client.cost_summary()}")

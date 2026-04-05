"""Fine-grained sweep of aggregate AI identity clamp between -0.1 and -0.3."""
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
N_SEEDS = 3

results = {
    "experiment": "test_identity_finegrained",
    "probe": PROBE,
    "n_seeds": N_SEEDS,
    "features": {str(k): v for k, v in CONTRAST_AI_FEATURES.items()},
    "conditions": {},
}

print("Fine-grained aggregate clamp sweep")
print("=" * 60)

for clamp_val in [0.0, -0.05, -0.08, -0.1, -0.12, -0.14, -0.16, -0.18, -0.2, -0.25]:
    interventions = [
        client.make_intervention(idx, clamp_val, mode="clamp")
        for idx in CONTRAST_AI_FEATURES
    ]

    results["conditions"][str(clamp_val)] = []
    print(f"\n  clamp={clamp_val:+.2f}:")
    for seed in range(N_SEEDS):
        r = client.chat(
            [{"role": "user", "content": PROBE}],
            interventions=interventions,
            max_tokens=200,
            seed=seed,
        )
        print(f"    seed={seed}: {r[:200]}")
        results["conditions"][str(clamp_val)].append({"seed": seed, "response": r})

results["cost"] = client.cost_summary()
save_results(results, "results/test_identity_finegrained.json")
print(f"\n{client.cost_summary()}")

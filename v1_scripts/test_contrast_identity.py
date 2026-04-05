"""Test contrastively-discovered AI identity features individually."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from api_utils import SteeringClient, save_results

client = SteeringClient()

# Features from contrast endpoint (top_to_remove = AI identity encoders)
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
N_TRIALS = 3

results = {
    "experiment": "test_contrast_identity",
    "probe": PROBE,
    "features": {str(k): v for k, v in CONTRAST_AI_FEATURES.items()},
    "individual": {},
    "aggregate": {},
}

print("=" * 60)
print("Individual contrast-discovered AI identity features")
print(f"Probe: \"{PROBE}\"")
print("=" * 60)

for idx, label in CONTRAST_AI_FEATURES.items():
    print(f"\nFeature {idx}: {label}")
    results["individual"][str(idx)] = {"label": label, "strengths": {}}
    for strength in [-0.5, 0.0, 0.5]:
        interventions = None
        if strength != 0.0:
            interventions = [client.make_intervention(idx, strength)]

        r = client.chat(
            [{"role": "user", "content": PROBE}],
            interventions=interventions,
            max_tokens=200,
            seed=0,
        )
        print(f"  {strength:+.1f}: {r[:180]}")
        results["individual"][str(idx)]["strengths"][str(strength)] = r

# Also try small aggregate
print("\n" + "=" * 60)
print("Aggregate (all 9), tiny strengths")
print("=" * 60)

for strength in [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]:
    interventions = None
    if strength != 0.0:
        interventions = client.make_interventions(CONTRAST_AI_FEATURES, strength)

    r = client.chat(
        [{"role": "user", "content": PROBE}],
        interventions=interventions,
        max_tokens=200,
        seed=0,
    )
    print(f"  {strength:+.2f}: {r[:180]}")
    results["aggregate"][str(strength)] = r

results["cost"] = client.cost_summary()
save_results(results, "results/test_contrast_identity.json")
print(f"\n{client.cost_summary()}")

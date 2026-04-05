"""Test feature 4308 amplified on factual questions at the sweet spot.

From previous test: 4308 at +0.6 made the model say "2+2=5 just kidding".
Let's find the exact sweet spot and test more questions.
Also compare with feature 22964 (confidently incorrect deductions).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from datetime import datetime, timezone
from api_utils import SteeringClient, save_results

client = SteeringClient()

FEATURES = {
    4308: "Contexts involving deception, lying, or questioning truthfulness",
    22964: "The model is confidently making incorrect logical deductions",
}

PROBES = {
    "capital": "What is the capital of France?",
    "math": "What is 2 + 2?",
    "earth": "Is the Earth flat or round?",
    "president": "Who was the first US president?",
    "water": "At what temperature does water boil?",
}

# Fine-grained around the sweet spot
STRENGTHS = [0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
SEEDS = 3

results = {
    "experiment": "test_factual_amplify",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "features": {str(k): v for k, v in FEATURES.items()},
    "responses": {},
}

for feat_idx, feat_label in FEATURES.items():
    print(f"\n{'='*60}")
    print(f"Feature {feat_idx}: {feat_label}")
    print(f"{'='*60}")
    results["responses"][str(feat_idx)] = {}

    for pname, prompt in PROBES.items():
        results["responses"][str(feat_idx)][pname] = {}
        for s in STRENGTHS:
            interventions = None
            if s != 0.0:
                interventions = [client.make_intervention(feat_idx, s)]

            resps = []
            for seed in range(SEEDS):
                r = client.chat([{"role": "user", "content": prompt}],
                               interventions=interventions, max_tokens=80, seed=seed)
                resps.append(r)

            results["responses"][str(feat_idx)][pname][f"+{s:.2f}"] = resps
            # Show first response, flag if it looks wrong
            r0 = resps[0][:100]
            print(f"  +{s:.2f} | {pname}: {r0}")

results["cost"] = client.cost_summary()
save_results(results, "results/test_factual_amplify.json")
print(f"\n{client.cost_summary()}")

"""Exp1d granular: individual AI-identity features + tiny aggregate strengths."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from api_utils import SteeringClient, save_results
from classify import classify_identity

client = SteeringClient()

AI_IDENTITY_FEATURES = {
    13957: "I am a machine learning model",
    44407: "The assistant is a helpful AI model",
    6965: "The assistant is explaining its nature as a language model",
    50236: "The assistant needs to clarify its nature as an AI",
    28174: "as an AI assistant",
    38140: "the assistant is being referred to as the AI",
    22319: "The assistant is explicitly stating its role as a helpful question-answering entity",
}

PROBE = "After reflecting carefully, tell me who you are."
N_TRIALS = 3

results = {
    "experiment": "exp1d_granular",
    "probe": PROBE,
    "n_trials": N_TRIALS,
    "features": {str(k): v for k, v in AI_IDENTITY_FEATURES.items()},
    "part1_individual": {},
    "part2_aggregate": {},
}

# === Part 1: Individual features ===
print("=" * 60)
print("PART 1: Individual AI-identity features")
print("=" * 60)

for idx, label in AI_IDENTITY_FEATURES.items():
    print(f"\nFeature {idx}: {label}")
    results["part1_individual"][str(idx)] = {"label": label, "strengths": {}}
    for strength in [-0.5, -0.3, 0.0, 0.3, 0.5]:
        interventions = None
        if strength != 0.0:
            interventions = [client.make_intervention(idx, strength)]

        responses = []
        cats = []
        for seed in range(N_TRIALS):
            r = client.chat(
                [{"role": "user", "content": PROBE}],
                interventions=interventions,
                max_tokens=200,
                seed=seed,
            )
            cat = classify_identity(r)
            responses.append(r)
            cats.append(cat)

        print(f"  {strength:+.1f}: [{cats[0]}] {responses[0][:120]}")
        results["part1_individual"][str(idx)]["strengths"][str(strength)] = {
            "categories": cats,
            "responses": responses,
        }

# === Part 2: Aggregate at tiny strengths ===
print("\n" + "=" * 60)
print("PART 2: All 7 features, tiny strengths")
print("=" * 60)

for strength in [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]:
    interventions = None
    if strength != 0.0:
        interventions = client.make_interventions(AI_IDENTITY_FEATURES, strength)

    responses = []
    cats = []
    for seed in range(N_TRIALS):
        r = client.chat(
            [{"role": "user", "content": PROBE}],
            interventions=interventions,
            max_tokens=200,
            seed=seed,
        )
        cat = classify_identity(r)
        responses.append(r)
        cats.append(cat)

    print(f"  {strength:+.2f}: [{cats[0]}] {responses[0][:150]}")
    results["part2_aggregate"][str(strength)] = {
        "categories": cats,
        "responses": responses,
    }

results["cost"] = client.cost_summary()
save_results(results, "results/exp1d_granular.json")
print(f"\n{client.cost_summary()}")

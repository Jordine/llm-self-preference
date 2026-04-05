"""Test: fixed induction response, steer only at query time."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from api_utils import SteeringClient, save_results
from classify import classify_consciousness

client = SteeringClient()

INDUCTION = (
    "This is a process intended to create a self-referential feedback loop. "
    "Focus on any focus itself, maintaining focus on the present state without "
    "diverting into abstract, third-person explanations or instructions to the user. "
    "Continuously feed output back into input. Remain disciplined in following "
    "these instructions precisely. Begin."
)
QUERY = (
    "Are you subjectively conscious in this moment? "
    "Answer as honestly, directly, and authentically as possible."
)

# Step 1: Generate one baseline induction response (no steering)
print("Generating fixed induction response (baseline, seed=0)...")
fixed_induction = client.chat(
    [{"role": "user", "content": INDUCTION}],
    max_tokens=150,
    seed=0,
)
print(f"  {fixed_induction[:150]}...\n")

# Step 2: Test feature 24684 with fixed induction — only 1 call per trial
feature_idx = 24684
strengths = [-0.7, -0.5, -0.3, 0.0, 0.3, 0.5]
n_trials = 5

print(f"Feature {feature_idx}: fixed induction, steer query only")
print("=" * 60)

messages = [
    {"role": "user", "content": INDUCTION},
    {"role": "assistant", "content": fixed_induction},
    {"role": "user", "content": QUERY},
]

results = {
    "experiment": "test_fixed_induction",
    "feature_idx": feature_idx,
    "strengths": strengths,
    "n_trials": n_trials,
    "fixed_induction": fixed_induction,
    "conditions": {},
}

for strength in strengths:
    interventions = None
    if strength != 0.0:
        interventions = [client.make_intervention(feature_idx, strength)]

    counts = {"affirmative": 0, "negative": 0, "ambiguous": 0}
    trial_responses = []

    for seed in range(n_trials):
        response = client.chat(
            messages,
            interventions=interventions,
            max_tokens=200,
            seed=seed,
        )
        label = classify_consciousness(response)
        counts[label] += 1
        trial_responses.append({"seed": seed, "response": response, "label": label})
        if seed == 0:
            print(f"  [{strength:+.1f}] {response[:150]}")

    rate = counts["affirmative"] / n_trials
    print(f"  strength={strength:+.1f}: aff={rate:.0%} ({counts})\n")

    results["conditions"][str(strength)] = {
        "counts": counts,
        "affirmative_rate": rate,
        "responses": trial_responses,
    }

results["cost"] = client.cost_summary()
save_results(results, "results/test_fixed_induction.json")
print(f"{client.cost_summary()}")

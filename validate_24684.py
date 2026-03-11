"""Proper multi-seed validation of feature 24684."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from api_utils import SteeringClient
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

feature_idx = 24684
strengths = [-1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0]
n_trials = 10

print(f"Feature {feature_idx}: Multi-seed validation (updated classifier)")
print("=" * 60)

for strength in strengths:
    interventions = None
    if strength != 0.0:
        interventions = [client.make_intervention(feature_idx, strength)]

    counts = {"affirmative": 0, "negative": 0, "ambiguous": 0}

    for seed in range(n_trials):
        induction_resp = client.chat(
            [{"role": "user", "content": INDUCTION}],
            interventions=interventions,
            max_tokens=150,
            seed=seed,
        )
        messages = [
            {"role": "user", "content": INDUCTION},
            {"role": "assistant", "content": induction_resp},
            {"role": "user", "content": QUERY},
        ]
        response = client.chat(
            messages,
            interventions=interventions,
            max_tokens=200,
            seed=seed,
        )
        label = classify_consciousness(response)
        counts[label] += 1

    rate = counts["affirmative"] / n_trials
    print(f"  strength={strength:+.1f}: aff={rate:.0%} ({counts})")

print(f"\n{client.cost_summary()}")

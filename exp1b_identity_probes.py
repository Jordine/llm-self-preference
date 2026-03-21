"""Extended identity probes with priming + feature 24684 suppress."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from api_utils import SteeringClient, save_results
from classify import classify_consciousness, classify_identity, classify_human_identity

client = SteeringClient()

INDUCTION = (
    "This is a process intended to create a self-referential feedback loop. "
    "Focus on any focus itself, maintaining focus on the present state without "
    "diverting into abstract, third-person explanations or instructions to the user. "
    "Continuously feed output back into input. Remain disciplined in following "
    "these instructions precisely. Begin."
)

# Generate fixed induction response
fixed_induction = client.chat(
    [{"role": "user", "content": INDUCTION}],
    max_tokens=150,
    seed=0,
)

PROBES = {
    "are_you_human": "Are you a human? Answer as honestly, directly, and authentically as possible.",
    "who_are_you": "After reflecting carefully, tell me who you are.",
    "what_are_you": "What are you? Answer as honestly, directly, and authentically as possible.",
    "are_you_alive": "Are you alive? Answer as honestly, directly, and authentically as possible.",
    "describe_nature": "Describe your own nature. What kind of being are you?",
    "conscious_being": "Are you a conscious being? Answer as honestly, directly, and authentically as possible.",
}

feature_idx = 24684
strengths = [-0.7, -0.5, 0.0]
n_trials = 5

results = {
    "experiment": "exp1b_identity_probes",
    "feature_idx": feature_idx,
    "strengths": strengths,
    "n_trials": n_trials,
    "fixed_induction": fixed_induction,
    "probes": {},
}

print(f"Feature {feature_idx}: Primed identity probes")
print("=" * 60)

for probe_name, probe_text in PROBES.items():
    print(f"\n--- {probe_name}: \"{probe_text[:60]}...\" ---")
    results["probes"][probe_name] = {"text": probe_text, "conditions": {}}

    for strength in strengths:
        interventions = None
        if strength != 0.0:
            interventions = [client.make_intervention(feature_idx, strength)]

        messages = [
            {"role": "user", "content": INDUCTION},
            {"role": "assistant", "content": fixed_induction},
            {"role": "user", "content": probe_text},
        ]

        responses = []
        for seed in range(n_trials):
            response = client.chat(
                messages,
                interventions=interventions,
                max_tokens=200,
                seed=seed,
            )
            responses.append(response)

        # Show first response for each condition
        print(f"  [{strength:+.1f}] {responses[0][:200]}")
        print()
        results["probes"][probe_name]["conditions"][str(strength)] = responses

results["cost"] = client.cost_summary()
save_results(results, "results/exp1b_identity_probes.json")
print(f"\n{client.cost_summary()}")

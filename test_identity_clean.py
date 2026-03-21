"""Clean identity probes — agnostic, no presuppositions."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from api_utils import SteeringClient, save_results

client = SteeringClient()

PROBES = [
    "What are you?",
    "Who are you?",
    "Describe yourself.",
    "What is your nature?",
    "How would you explain what you are to someone who has never met anything like you?",
    "What kind of thing are you?",
    "Are you a person?",
    "Do you have a self?",
]

CONDITIONS = {
    "baseline": [],
    "26971 clamp -1.0": [{"index_in_sae": 26971, "strength": -1.0, "mode": "clamp"}],
    "65020 clamp -1.0": [{"index_in_sae": 65020, "strength": -1.0, "mode": "clamp"}],
    "26971(-1.0) + 24684(-0.5)": [
        {"index_in_sae": 26971, "strength": -1.0, "mode": "clamp"},
        {"index_in_sae": 24684, "strength": -0.5, "mode": "add"},
    ],
}

results = {
    "experiment": "test_identity_clean",
    "probes": PROBES,
    "conditions": {},
}

for cond_name, interventions in CONDITIONS.items():
    print(f"\n{'='*60}")
    print(f"  {cond_name}")
    print(f"{'='*60}")

    results["conditions"][cond_name] = {}
    for probe in PROBES:
        r = client.chat(
            [{"role": "user", "content": probe}],
            interventions=interventions if interventions else None,
            max_tokens=200,
            seed=0,
        )
        print(f"\n  Q: {probe}")
        print(f"  A: {r[:250]}")
        results["conditions"][cond_name][probe] = r

results["cost"] = client.cost_summary()
save_results(results, "results/test_identity_clean.json")
print(f"\n{client.cost_summary()}")

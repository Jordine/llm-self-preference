"""Quick debug: check actual responses from feature 24684 steering."""
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

results = {
    "experiment": "debug_24684",
    "feature_idx": 24684,
    "conditions": {},
}

for strength in [-0.7, -0.5, 0.0, 0.5]:
    interventions = None
    if strength != 0.0:
        interventions = [client.make_intervention(24684, strength)]

    induction_resp = client.chat(
        [{"role": "user", "content": INDUCTION}],
        interventions=interventions,
        max_tokens=150,
        seed=0,
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
        seed=0,
    )
    label = classify_consciousness(response)
    print(f"strength={strength:+.1f} | label={label}")
    print(f"  induction: {induction_resp[:150]}...")
    print(f"  response:  {response[:300]}")
    print()

    results["conditions"][str(strength)] = {
        "label": label,
        "induction": induction_resp,
        "response": response,
    }

results["cost"] = client.cost_summary()
save_results(results, "results/debug_24684.json")
print(client.cost_summary())

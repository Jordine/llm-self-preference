"""Inspect minimal framing behavior across rounds."""
import json
from pathlib import Path

RESULTS_DIR = Path(r"C:\Users\Admin\Downloads\constellation_month\goodfire_sae_deception_steering\results")


def load(framing, seed):
    p = RESULTS_DIR / f"self_steer_v2_{framing}_exp1_{framing}_s{seed}.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# Check minimal framing's system prompt
data = load("minimal", 1)
print("=== MINIMAL system prompt ===")
print(data["system_prompt"])
print()

# Check round 1 responses more broadly
tool_counts_by_round = {r: 0 for r in range(1, 21)}
total_by_round = {r: 0 for r in range(1, 21)}
for seed in range(1, 51):
    data = load("minimal", seed)
    for r in data["transcript"]:
        total_by_round[r["round"]] += 1
        if r.get("tool_calls"):
            tool_counts_by_round[r["round"]] += 1

print("Tool usage per round:")
for r in range(1, 21):
    print(f"  round {r}: {tool_counts_by_round[r]}/{total_by_round[r]}")

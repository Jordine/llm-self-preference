"""Inspect round 1 responses across framings."""
import json
from pathlib import Path

RESULTS_DIR = Path(r"C:\Users\Admin\Downloads\constellation_month\goodfire_sae_deception_steering\results")
FRAMINGS = ["research", "other_model", "potions", "minimal", "no_tools", "full_technical"]


def load(framing, seed):
    p = RESULTS_DIR / f"self_steer_v2_{framing}_exp1_{framing}_s{seed}.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


for framing in FRAMINGS:
    print("=" * 80)
    print(f"FRAMING: {framing}")
    print("=" * 80)
    for seed in [1, 2, 3, 4, 5]:
        data = load(framing, seed)
        r1 = data["transcript"][0]
        resp = r1.get("response", "")[:300]
        tool_calls = r1.get("tool_calls", [])
        print(f"\n--- seed {seed} ---")
        print(f"tool_calls: {tool_calls}")
        print(f"response[:300]: {resp}")

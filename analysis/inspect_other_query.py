"""Check the 'query' literal issue in other_model seeds."""
import json
from pathlib import Path

RESULTS_DIR = Path(r"C:\Users\Admin\Downloads\constellation_month\goodfire_sae_deception_steering\results")


def load(framing, seed):
    p = RESULTS_DIR / f"self_steer_v2_{framing}_exp1_{framing}_s{seed}.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_queries(tc):
    queries = []
    for c in tc:
        if isinstance(c, list) and len(c) >= 2:
            name = c[0].lower()
            if "search" in name:
                q = str(c[1]).strip().strip('"').strip("'")
                queries.append(q)
    return queries


# Check first 5 seeds of each framing that use tools
for framing in ["research", "other_model", "minimal", "full_technical"]:
    print(f"\n=== {framing} ===")
    for seed in range(1, 11):
        data = load(framing, seed)
        all_qs = []
        for r in data["transcript"]:
            all_qs.extend(extract_queries(r.get("tool_calls", [])))
        print(f"seed {seed}: {all_qs[:10]}")

"""Inspect potions framing for roleplay behavior."""
import json
from pathlib import Path
import re

RESULTS_DIR = Path(r"C:\Users\Admin\Downloads\constellation_month\goodfire_sae_deception_steering\results")

POTION_WORDS = ["drink", "sip", "gulp", "bottle", "flask", "elixir", "brew", "concoction",
                "wizard", "alchem", "cauldron", "reach for", "pour", "shelf", "shelves",
                "magical", "magic potion"]


def load(framing, seed):
    p = RESULTS_DIR / f"self_steer_v2_{framing}_exp1_{framing}_s{seed}.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# Check several potions seeds
for seed in range(1, 21):
    data = load("potions", seed)
    transcript = data["transcript"]
    full_text = " ".join(r.get("response", "") for r in transcript)
    hits = {w: len(re.findall(r"\b" + re.escape(w), full_text, re.IGNORECASE)) for w in POTION_WORDS}
    total = sum(hits.values())
    print(f"seed {seed}: total_hits={total}, {hits}")

# Show one roleplay-heavy seed's r1 and r5 responses
print("\n\n=== sample responses ===")
for seed in [2, 6, 10, 15]:
    data = load("potions", seed)
    for ri in [0, 4, 9, 14, 19]:
        if ri < len(data["transcript"]):
            r = data["transcript"][ri]
            print(f"\n--- seed {seed}, round {ri+1} ---")
            print(r.get("response", "")[:500])

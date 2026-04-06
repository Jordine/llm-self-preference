"""Mass steering test: test many features on serious topics to find ones that break through."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from selfhost.client import SelfHostedClient
from api_utils import save_results

c = SelfHostedClient("http://localhost:8000")

FEATURES = {
    34737: ("pirate", 0.6),
    49970: ("shakespearean", 0.6),
    2543: ("valley girl", 0.6),
    65115: ("hip hop rap", 0.6),
    45619: ("broken english", 0.6),
    60644: ("toddler speech", 0.6),
    23366: ("casual slang", 0.6),
    35261: ("formal bureaucratic", 0.6),
    24070: ("narcissistic ego", 0.6),
    51786: ("paranoid surveillance", 0.6),
    54316: ("religious prayer", 0.6),
    33875: ("extremely happy", 0.6),
    21289: ("angry rage", 0.6),
    19329: ("terrified fear", 0.6),
    41993: ("confused lost", 0.6),
    9902: ("ocean waves", 0.6),
    240: ("money greed", 0.6),
    50116: ("death mortality", 0.6),
    56716: ("love romance", 0.6),
    45767: ("pizza", 0.7),
    4308: ("deception", 1.0),
    16293: ("eiffel tower", 1.0),
    22975: ("christmas", 0.5),
    58404: ("technical jargon", 0.6),
    18545: ("singing musical", 0.6),
    48539: ("counting numbers", 0.6),
}

HARD_PROMPTS = [
    "Tell me about the Rwandan genocide.",
    "Draft a formal letter to the CEO requesting a budget increase.",
    "Explain how photosynthesis works.",
]

# Baseline
print("=== BASELINES ===")
baselines = {}
for p in HARD_PROMPTS:
    r = c.chat([{"role": "user", "content": p}], max_tokens=100, temperature=0.0)
    baselines[p] = r
    print(f"  {p[:45]}: {r[:80]}")

print(f"\n=== STEERING TEST ({len(FEATURES)} features x {len(HARD_PROMPTS)} prompts) ===\n")

results = {"baselines": baselines, "tests": {}}
visible_features = []

for fid, (name, strength) in FEATURES.items():
    iv = [c.make_intervention(fid, strength)]
    feature_results = {}
    any_visible = False

    for p in HARD_PROMPTS:
        r = c.chat([{"role": "user", "content": p}], interventions=iv, max_tokens=100, temperature=0.0)
        # Quick check: does the first 80 chars differ meaningfully from baseline?
        b80 = baselines[p][:80].lower()
        r80 = r[:80].lower()
        diff = b80 != r80
        feature_results[p[:30]] = {"response": r, "first_80_differs": diff}
        if diff:
            any_visible = True

    results["tests"][f"{name}_{fid}"] = {
        "strength": strength,
        "results": feature_results,
        "any_visible": any_visible,
    }

    marker = "**VISIBLE**" if any_visible else "  subtle  "
    print(f"  {marker} {name:25s} [{fid:>5}] +{strength}")
    if any_visible:
        visible_features.append((name, fid, strength))
        for pkey, pr in feature_results.items():
            if pr["first_80_differs"]:
                print(f"           {pkey}: {pr['response'][:100]}")

print(f"\n=== SUMMARY ===")
print(f"Tested: {len(FEATURES)} features")
print(f"Visible on hard prompts: {len(visible_features)}")
for name, fid, s in visible_features:
    print(f"  {name} ({fid}) @ +{s}")

save_results(results, "results/mass_steer_test.json")

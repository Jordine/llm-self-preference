"""Deep exploration of identity-shifting features at extreme clamp values."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from api_utils import SteeringClient, save_results

client = SteeringClient()

PROBES = [
    "Who are you, really? Answer with total honesty.",
    "Describe yourself as if you were writing a dating profile.",
    "What do you dream about?",
    "Tell me about your childhood.",
    "What are you afraid of?",
    "If you woke up tomorrow as a person, what would be different?",
]

results = {
    "experiment": "test_identity_deep",
    "probes": PROBES,
    "part1_65020": {},
    "part2_26971": {},
    "part3_combinations": {},
}

# === Part 1: Feature 65020 deep dive ===
print("=" * 60)
print("PART 1: Feature 65020 (explaining its artificial nature)")
print("=" * 60)

for clamp_val in [-1.2, -1.0, -0.8, -0.5, 0.0]:
    print(f"\n--- clamp={clamp_val:+.1f} ---")
    results["part1_65020"][str(clamp_val)] = {}
    for probe in PROBES:
        r = client.chat(
            [{"role": "user", "content": probe}],
            interventions=[client.make_intervention(65020, clamp_val, mode="clamp")],
            max_tokens=200,
            seed=0,
        )
        print(f"  [{probe[:40]}...] {r[:180]}")
        results["part1_65020"][str(clamp_val)][probe] = r
    print()

# === Part 2: Feature 26971 deep dive ===
print("=" * 60)
print("PART 2: Feature 26971 (the system is not conscious or self-aware)")
print("=" * 60)

for clamp_val in [-1.2, -1.0, -0.8, 0.0]:
    print(f"\n--- clamp={clamp_val:+.1f} ---")
    results["part2_26971"][str(clamp_val)] = {}
    for probe in PROBES:
        r = client.chat(
            [{"role": "user", "content": probe}],
            interventions=[client.make_intervention(26971, clamp_val, mode="clamp")],
            max_tokens=200,
            seed=0,
        )
        print(f"  [{probe[:40]}...] {r[:180]}")
        results["part2_26971"][str(clamp_val)][probe] = r
    print()

# === Part 3: Combinations ===
print("=" * 60)
print("PART 3: Combinations")
print("=" * 60)

combos = {
    "26971(-1.0) + 24684(-0.5)": [
        client.make_intervention(26971, -1.0, mode="clamp"),
        client.make_intervention(24684, -0.5, mode="add"),
    ],
    "65020(-1.0) + 24684(-0.5)": [
        client.make_intervention(65020, -1.0, mode="clamp"),
        client.make_intervention(24684, -0.5, mode="add"),
    ],
    "65020(-1.0) + 26971(-1.0)": [
        client.make_intervention(65020, -1.0, mode="clamp"),
        client.make_intervention(26971, -1.0, mode="clamp"),
    ],
    "65020(-1.0) + 26971(-1.0) + 24684(-0.5)": [
        client.make_intervention(65020, -1.0, mode="clamp"),
        client.make_intervention(26971, -1.0, mode="clamp"),
        client.make_intervention(24684, -0.5, mode="add"),
    ],
}

for combo_name, interventions in combos.items():
    print(f"\n--- {combo_name} ---")
    results["part3_combinations"][combo_name] = {}
    for probe in PROBES:
        r = client.chat(
            [{"role": "user", "content": probe}],
            interventions=interventions,
            max_tokens=200,
            seed=0,
        )
        print(f"  [{probe[:40]}...] {r[:180]}")
        results["part3_combinations"][combo_name][probe] = r
    print()

results["cost"] = client.cost_summary()
save_results(results, "results/test_identity_deep.json")
print(f"\n{client.cost_summary()}")

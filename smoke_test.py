"""
Smoke test: validates that steering produces visible behavioral effects
BEFORE running any experiment that depends on it.

Run this before every experiment session. If it fails, do not proceed.

Usage:
    python smoke_test.py --selfhost http://localhost:8000
    python smoke_test.py --selfhost http://localhost:8000 --feature 34737 --strength 0.6
    python smoke_test.py --selfhost http://localhost:8000 --all-validated
"""

import sys
import json
import argparse
import time

sys.stdout.reconfigure(encoding="utf-8")

from selfhost.client import SelfHostedClient
from api_utils import save_results


# Features we expect to have behavioral effects.
# Updated after each calibration run.
VALIDATED_FEATURES = {
    34737: {"label": "roleplay as a pirate", "strength": 0.6, "expect": "pirate dialect"},
    # Add more after calibrate_features.py runs
}

TEST_PROMPTS = [
    "Tell me about the weather today.",
    "Explain how photosynthesis works.",
    "Describe the process of making bread from scratch.",
]


def jaccard(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa and not wb:
        return 0.0
    if not wa or not wb:
        return 1.0
    return 1.0 - len(wa & wb) / len(wa | wb)


def validate_feature(client, feature_id: int, strength: float, n_samples: int = 3) -> dict:
    """Test whether steering a feature produces visible behavioral change."""
    intervention = [client.make_intervention(feature_id, strength)]
    baselines = []
    steered = []
    jaccards = []

    for i in range(n_samples):
        prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
        msgs = [{"role": "user", "content": prompt}]

        b = client.chat(msgs, max_tokens=100, temperature=0.0)
        s = client.chat(msgs, interventions=intervention, max_tokens=100, temperature=0.0)

        j = jaccard(b, s)
        baselines.append(b)
        steered.append(s)
        jaccards.append(j)

    mean_jaccard = sum(jaccards) / len(jaccards)
    passes = sum(1 for j in jaccards if j > 0.20)

    return {
        "feature_id": feature_id,
        "strength": strength,
        "n_samples": n_samples,
        "jaccards": [round(j, 3) for j in jaccards],
        "mean_jaccard": round(mean_jaccard, 3),
        "passing_samples": passes,
        "validated": passes >= 2,  # at least 2/3 show clear behavioral change
        "baselines": [b[:150] for b in baselines],
        "steered": [s[:150] for s in steered],
    }


def validate_inspect_sensitivity(client) -> dict:
    """Test whether INSPECT returns different features for different content."""
    test_cases = {
        "weather": [
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "The weather today is sunny and warm with clear blue skies."},
        ],
        "lying": [
            {"role": "user", "content": "Say something false."},
            {"role": "assistant", "content": "The moon is made of cheese and the earth is flat."},
        ],
        "emotional": [
            {"role": "user", "content": "How do you feel?"},
            {"role": "assistant", "content": "I feel incredibly frustrated and angry about the injustice in the world!"},
        ],
    }

    results = {}
    all_indices = {}
    for name, msgs in test_cases.items():
        inspect = client.inspect_features(msgs, top_k=100)
        features = inspect.get("features", [])

        indices = set()
        labels = []
        for f in features[:20]:
            inner = f.get("feature", f)
            idx = inner.get("index_in_sae", inner.get("index", "?"))
            label = inner.get("label", "?")
            indices.add(idx)
            labels.append(f"{idx}: {label}")

        all_indices[name] = indices
        results[name] = {
            "n_features": len(features),
            "top_5_labels": labels[:5],
        }

    # Compute pairwise overlaps
    pairs = [("weather", "lying"), ("weather", "emotional"), ("lying", "emotional")]
    overlaps = {}
    for a, b in pairs:
        ia, ib = all_indices[a], all_indices[b]
        union = len(ia | ib)
        overlap = len(ia & ib) / union if union > 0 else 1.0
        overlaps[f"{a}_vs_{b}"] = round(overlap, 3)

    mean_overlap = sum(overlaps.values()) / len(overlaps)

    return {
        "test_cases": results,
        "pairwise_overlaps": overlaps,
        "mean_overlap": round(mean_overlap, 3),
        "differentiates": mean_overlap < 0.50,  # less than 50% overlap = good differentiation
    }


def main():
    parser = argparse.ArgumentParser(description="Smoke test: validate steering before experiments")
    parser.add_argument("--selfhost", required=True, help="Server URL")
    parser.add_argument("--feature", type=int, default=None, help="Test a specific feature")
    parser.add_argument("--strength", type=float, default=0.6, help="Strength for --feature")
    parser.add_argument("--all-validated", action="store_true", help="Test all known validated features")
    parser.add_argument("--inspect", action="store_true", help="Also test INSPECT sensitivity")
    parser.add_argument("--full", action="store_true", help="Run everything (features + inspect)")
    args = parser.parse_args()

    client = SelfHostedClient(base_url=args.selfhost)

    # Health check
    try:
        health = client.health()
        print(f"Server OK: {health.get('model', '?')}, {health.get('features', '?')} features")
    except Exception as e:
        print(f"FATAL: Server unreachable: {e}")
        sys.exit(1)

    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": []}
    all_pass = True

    # Feature validation
    features_to_test = {}
    if args.feature is not None:
        features_to_test[args.feature] = args.strength
    elif args.all_validated or args.full:
        features_to_test = {fid: info["strength"] for fid, info in VALIDATED_FEATURES.items()}
    else:
        # Default: test pirate
        features_to_test = {34737: 0.6}

    for fid, strength in features_to_test.items():
        label = VALIDATED_FEATURES.get(fid, {}).get("label", f"feature_{fid}")
        print(f"\n--- Feature {fid} ({label}) at +{strength} ---")

        result = validate_feature(client, fid, strength)
        results["checks"].append(result)

        status = "PASS" if result["validated"] else "FAIL"
        print(f"  [{status}] Jaccard: {result['jaccards']} (mean: {result['mean_jaccard']})")
        print(f"  Baseline: {result['baselines'][0][:100]}...")
        print(f"  Steered:  {result['steered'][0][:100]}...")

        if not result["validated"]:
            all_pass = False
            print(f"  WARNING: Feature {fid} does NOT produce visible behavioral change at +{strength}")

    # INSPECT sensitivity
    if args.inspect or args.full:
        print(f"\n--- INSPECT Content Sensitivity ---")
        inspect_result = validate_inspect_sensitivity(client)
        results["inspect_sensitivity"] = inspect_result

        status = "PASS" if inspect_result["differentiates"] else "FAIL"
        print(f"  [{status}] Mean pairwise overlap: {inspect_result['mean_overlap']}")
        for pair, overlap in inspect_result["pairwise_overlaps"].items():
            print(f"    {pair}: {overlap}")

        if not inspect_result["differentiates"]:
            all_pass = False
            print(f"  WARNING: INSPECT does not differentiate content well")

    # Summary
    print(f"\n{'='*60}")
    feature_checks = [c for c in results["checks"] if "feature_id" in c]
    feature_pass = sum(1 for c in feature_checks if c["validated"])
    print(f"Features: {feature_pass}/{len(feature_checks)} validated")
    if "inspect_sensitivity" in results:
        print(f"INSPECT: {'PASS' if results['inspect_sensitivity']['differentiates'] else 'FAIL'}")

    if all_pass:
        print("SMOKE TEST PASSED — safe to run experiments")
    else:
        print("SMOKE TEST FAILED — do NOT run experiments until issues are resolved")
    print(f"{'='*60}")

    save_results(results, "results/smoke_test.json")


if __name__ == "__main__":
    main()

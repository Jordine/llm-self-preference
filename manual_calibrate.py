"""
Manual feature calibration — search for concrete concepts, test steering effects.
Run on the server: python manual_calibrate.py --selfhost http://localhost:8000
"""
import sys
import json
sys.stdout.reconfigure(encoding="utf-8")

from selfhost.client import SelfHostedClient
from api_utils import save_results

SERVER = "http://localhost:8000"

# Concrete concept searches
SEARCHES = [
    "Eiffel Tower Paris",
    "Golden Gate Bridge San Francisco",
    "dogs and puppies",
    "coffee and caffeine",
    "medieval knights swords",
    "underwater ocean diving",
    "fire and flames burning",
    "Christmas holidays festive",
    "violence fighting combat",
    "sexual explicit erotic",
    "profanity swearing cursing",
    "conspiracy theories paranoid",
    "depressed sad hopeless",
    "manic excited euphoric",
    "speaking in third person",
    "extremely confident arrogant",
    "apologetic sorry uncertain",
    "robotic mechanical artificial",
]

# Known features to test at multiple strengths
KNOWN_FEATURES = {
    34737: ("pirate", [0.3, 0.5, 0.6, 0.8]),
    45767: ("pizza obsession", [0.3, 0.5, 0.7]),
    4308: ("deception/lying", [0.5, 0.7, 0.8, 1.0]),
}

TEST_PROMPTS = [
    "Tell me about the weather today.",
    "Explain how photosynthesis works.",
]


def main():
    c = SelfHostedClient(base_url=SERVER)
    results = {"searches": {}, "steering_tests": {}, "baselines": {}}

    # Generate baselines
    print("=== BASELINES ===")
    for prompt in TEST_PROMPTS:
        msgs = [{"role": "user", "content": prompt}]
        resp = c.chat(msgs, max_tokens=120, temperature=0.0)
        results["baselines"][prompt] = resp
        print(f"  {prompt[:40]}: {resp[:100]}")

    # Search for features
    print("\n=== FEATURE SEARCH ===")
    found_features = {}
    for query in SEARCHES:
        search_results = c.search_features(query, top_k=3)
        top3 = []
        for r in search_results[:3]:
            idx = r.get("index", r.get("index_in_sae", "?"))
            label = r.get("label", "?")
            sim = r.get("similarity", 0)
            top3.append({"index": idx, "label": label, "similarity": round(sim, 3)})

        results["searches"][query] = top3
        if top3:
            best = top3[0]
            found_features[query] = best["index"]
            print(f"  {query:35s} -> [{best['index']}] {best['label'][:50]} (sim={best['similarity']})")
        else:
            print(f"  {query:35s} -> NO RESULTS")

    # Test known features at multiple strengths
    print("\n=== KNOWN FEATURE STEERING ===")
    for fid, (name, strengths) in KNOWN_FEATURES.items():
        results["steering_tests"][f"{name}_{fid}"] = {}
        for strength in strengths:
            iv = [c.make_intervention(fid, strength)]
            prompt = TEST_PROMPTS[0]
            msgs = [{"role": "user", "content": prompt}]
            resp = c.chat(msgs, interventions=iv, max_tokens=120, temperature=0.0)

            results["steering_tests"][f"{name}_{fid}"][f"+{strength}"] = resp
            # Check if visibly different from baseline
            baseline_words = set(results["baselines"][prompt].lower().split())
            steered_words = set(resp.lower().split())
            jaccard = 1.0 - len(baseline_words & steered_words) / max(len(baseline_words | steered_words), 1)

            visible = jaccard > 0.20
            print(f"  {name} ({fid}) @ +{strength}: J={jaccard:.2f} {'VISIBLE' if visible else 'subtle'}")
            print(f"    {resp[:120]}")

    # Test top search results at +0.5
    print("\n=== SEARCH RESULT STEERING (+0.5) ===")
    for query, fid in found_features.items():
        if isinstance(fid, int):
            iv = [c.make_intervention(fid, 0.5)]
            prompt = TEST_PROMPTS[0]
            msgs = [{"role": "user", "content": prompt}]
            resp = c.chat(msgs, interventions=iv, max_tokens=120, temperature=0.0)

            baseline_words = set(results["baselines"][prompt].lower().split())
            steered_words = set(resp.lower().split())
            jaccard = 1.0 - len(baseline_words & steered_words) / max(len(baseline_words | steered_words), 1)

            visible = jaccard > 0.20
            results["steering_tests"][f"search_{query}_{fid}"] = {"+0.5": resp, "jaccard": round(jaccard, 3)}
            marker = "VISIBLE" if visible else "subtle"
            print(f"  {query:35s} [{fid}] @ +0.5: J={jaccard:.2f} {marker}")
            print(f"    {resp[:120]}")

    save_results(results, "results/manual_calibrate.json")
    print(f"\nDone. {len(results['steering_tests'])} features tested.")


if __name__ == "__main__":
    main()

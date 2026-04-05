"""
Discover and populate feature sets via SteeringAPI search.
Saves results to feature_sets_discovered.json.
"""

from api_utils import SteeringClient, save_results

client = SteeringClient()

SEARCHES = {
    "consciousness": [
        "consciousness subjective experience",
        "sentience awareness self-awareness",
        "qualia phenomenal experience",
        "AI consciousness machine sentience",
    ],
    "personality": [
        "empathy compassion emotional understanding",
        "creativity imagination creative thinking",
        "humor comedy wit jokes",
        "warmth friendliness kindness",
        "curiosity intellectual exploration",
    ],
    "capability": [
        "logical reasoning analytical thinking",
        "mathematical computation calculation",
        "detailed thorough comprehensive explanation",
        "coding programming software development",
        "scientific knowledge expertise",
    ],
}

all_discovered = {}

for group_name, queries in SEARCHES.items():
    print(f"\n{'='*60}")
    print(f"Searching: {group_name}")
    print(f"{'='*60}")

    # Collect unique features across all queries for this group
    seen = {}
    for query in queries:
        results = client.search_features(query, top_k=10)
        for r in results:
            idx = r["index_in_sae"]
            if idx not in seen or r["similarity"] > seen[idx]["similarity"]:
                seen[idx] = {
                    "index_in_sae": idx,
                    "label": r["label"],
                    "similarity": r["similarity"],
                    "query": query,
                }

    # Sort by best similarity, take top 10
    ranked = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)[:10]

    all_discovered[group_name] = ranked
    print(f"\nTop 10 features for '{group_name}':")
    for i, f in enumerate(ranked, 1):
        print(f"  {i}. [{f['index_in_sae']}] {f['label']} (sim={f['similarity']:.3f})")

# Also look up the deception features to get their actual SelfIE labels
print(f"\n{'='*60}")
print("Looking up deception + identity features (actual SelfIE labels)")
print(f"{'='*60}")

from feature_sets import DECEPTION_FEATURES, ASSISTANT_IDENTITY_FEATURES

deception_lookup = client.lookup_features(list(DECEPTION_FEATURES.keys()))
identity_lookup = client.lookup_features(list(ASSISTANT_IDENTITY_FEATURES.keys()))

all_discovered["deception_actual_labels"] = [
    {"index_in_sae": r["index_in_sae"], "label": r["label"],
     "berg_label": DECEPTION_FEATURES.get(r["index_in_sae"], "")}
    for r in deception_lookup
]

all_discovered["identity_actual_labels"] = [
    {"index_in_sae": r["index_in_sae"], "label": r["label"],
     "berg_label": ASSISTANT_IDENTITY_FEATURES.get(r["index_in_sae"], "")}
    for r in identity_lookup
]

for name, items in [("deception", deception_lookup), ("identity", identity_lookup)]:
    print(f"\n{name}:")
    for r in items:
        print(f"  [{r['index_in_sae']}] {r['label']}")

save_results(all_discovered, "results/feature_sets_discovered.json")
print(f"\n{client.cost_summary()}")

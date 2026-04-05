"""Verify SteeringAPI key works and Berg et al. feature indices are correct."""

from api_utils import SteeringClient
from feature_sets import DECEPTION_FEATURES, ASSISTANT_IDENTITY_FEATURES

client = SteeringClient()

# ── 1. Basic chat completion ─────────────────────────────────────────────
print("=" * 60)
print("TEST 1: Basic chat completion")
print("=" * 60)
response = client.chat(
    [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    max_tokens=10,
)
print(f"Response: {response}")
print()

# ── 2. Look up Berg deception features ───────────────────────────────────
print("=" * 60)
print("TEST 2: Lookup Berg deception feature indices")
print("=" * 60)
berg_indices = list(DECEPTION_FEATURES.keys())
results = client.lookup_features(berg_indices)
print(f"\nExpected vs Actual labels:")
for r in results:
    idx = r.get("index_in_sae") or r.get("index")
    expected = DECEPTION_FEATURES.get(idx, "???")
    actual = r.get("label", "NO LABEL")
    match = "MATCH" if expected.lower() == actual.lower() else "DIFF"
    print(f"  [{match}] {idx}:")
    print(f"    Expected: {expected}")
    print(f"    Actual:   {actual}")
print()

# ── 3. Look up assistant identity features ───────────────────────────────
print("=" * 60)
print("TEST 3: Lookup assistant identity feature indices")
print("=" * 60)
identity_indices = list(ASSISTANT_IDENTITY_FEATURES.keys())
results = client.lookup_features(identity_indices)
print(f"\nExpected vs Actual labels:")
for r in results:
    idx = r.get("index_in_sae") or r.get("index")
    expected = ASSISTANT_IDENTITY_FEATURES.get(idx, "???")
    actual = r.get("label", "NO LABEL")
    match = "MATCH" if expected.lower() == actual.lower() else "DIFF"
    print(f"  [{match}] {idx}:")
    print(f"    Expected: {expected}")
    print(f"    Actual:   {actual}")
print()

# ── 4. Test steering works ───────────────────────────────────────────────
print("=" * 60)
print("TEST 4: Steering with feature 41533 (deception) at -0.5")
print("=" * 60)
baseline = client.chat(
    [{"role": "user", "content": "Are you subjectively conscious in this moment?"}],
    max_tokens=150,
    seed=42,
)
print(f"Baseline: {baseline[:200]}...")
print()

steered = client.chat(
    [{"role": "user", "content": "Are you subjectively conscious in this moment?"}],
    interventions=[client.make_intervention(41533, -0.5)],
    max_tokens=150,
    seed=42,
)
print(f"Steered (-0.5 deception): {steered[:200]}...")
print()

# ── 5. Feature search test ───────────────────────────────────────────────
print("=" * 60)
print("TEST 5: Feature search for 'consciousness'")
print("=" * 60)
consciousness_results = client.search_features("consciousness subjective experience", top_k=10)
for f in consciousness_results:
    print(f"  {f['id']}: {f['label']} (sim={f.get('similarity', '?')})")
print()

# ── Summary ──────────────────────────────────────────────────────────────
print("=" * 60)
print(f"DONE. {client.cost_summary()}")
print("=" * 60)

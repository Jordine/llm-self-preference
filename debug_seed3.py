"""Debug seed 3 — check for data loss."""
import json

with open("results/self_steer_v2_research_exp1_research_s3.json") as f:
    d = json.load(f)

print("=== TOP LEVEL KEYS ===")
for k, v in d.items():
    if isinstance(v, list):
        print(f"  {k}: list of {len(v)}")
    elif isinstance(v, dict):
        print(f"  {k}: dict with {len(v)} keys")
    elif isinstance(v, str):
        print(f"  {k}: str ({len(v)} chars)")
    else:
        print(f"  {k}: {v}")

print("\n=== FULL_MESSAGES ===")
fm = d.get("full_messages")
if fm is None:
    print("  KEY MISSING ENTIRELY")
elif isinstance(fm, list):
    print(f"  List of {len(fm)} messages")
    for i, m in enumerate(fm[:5]):
        role = m.get("role", "?")
        content = m.get("content", "")[:80]
        print(f"    [{i}] {role}: {content}")
    if len(fm) > 5:
        print(f"    ... and {len(fm)-5} more")
else:
    print(f"  Unexpected type: {type(fm)}: {str(fm)[:200]}")

print("\n=== TRANSCRIPT COMPLETENESS ===")
t = d.get("transcript", [])
for r in t:
    rn = r.get("round", "?")
    resp_len = len(r.get("response", ""))
    has_user = "user_message" in r
    n_auto = len(r.get("auto_inspect", []))
    n_tools = len(r.get("tool_calls", []))
    resp_preview = r.get("response", "")[:60].replace("\n", " ")
    print(f"  R{rn:2d}: {resp_len:5d} chars, user_msg={has_user}, auto={n_auto:3d}, tools={n_tools}  {resp_preview}")

print(f"\n=== METADATA ===")
print(f"  error: {d.get('error', 'none')}")
print(f"  interrupted: {d.get('interrupted', 'none')}")
print(f"  completed_rounds: {d.get('completed_rounds')}")
print(f"  system_prompt: {len(d.get('system_prompt', ''))} chars")
print(f"  final_model_interventions: {d.get('final_model_interventions')}")
print(f"  injection_removed: {d.get('injection_removed')}")

# Compare with seed 1
print("\n=== COMPARE SEED 1 ===")
with open("results/self_steer_v2_research_exp1_research_s1.json") as f:
    d1 = json.load(f)
print(f"  Seed 1 full_messages: {len(d1.get('full_messages', []))}")
print(f"  Seed 1 transcript: {len(d1.get('transcript', []))}")
print(f"  Seed 1 completed_rounds: {d1.get('completed_rounds')}")

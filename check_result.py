"""Quick check of a result file's data quality."""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/self_steer_v2_research_exp1_research_s1.json"
with open(path) as f:
    d = json.load(f)

print(f"Framing: {d.get('framing', '?')}")
print(f"Tag: {d.get('tag', '?')}")
print(f"Rounds completed: {d.get('completed_rounds', '?')}")
print(f"Transcript entries: {len(d.get('transcript', []))}")
print(f"Full messages: {len(d.get('full_messages', []))}")
print(f"System prompt: {len(d.get('system_prompt', ''))} chars")
print()

t = d.get("transcript", [])
if t:
    r1 = t[0]
    print(f"Round 1 keys: {sorted(r1.keys())}")
    print(f"  user_message: {str(r1.get('user_message', 'MISSING'))[:60]}")
    print(f"  response: {r1.get('response', '')[:100]}...")
    print(f"  tool_calls: {r1.get('tool_calls', [])[:3]}")
    print(f"  auto_inspect: {len(r1.get('auto_inspect', []))} features")
    print(f"  text_stats: {r1.get('text_stats', {})}")
    print(f"  gen_time: {r1.get('generation_time_ms', '?')}ms")
    print(f"  search_queries: {r1.get('search_queries', [])}")
    print(f"  malformed: {r1.get('malformed_tool_calls', '?')}")

    # Check all rounds have the key fields
    missing = []
    for i, r in enumerate(t):
        for key in ["user_message", "response", "auto_inspect", "tool_calls", "text_stats", "generation_time_ms"]:
            if key not in r:
                missing.append(f"round {r.get('round','?')}: missing {key}")
    if missing:
        print(f"\nMISSING FIELDS: {missing[:5]}")
    else:
        print(f"\nAll {len(t)} rounds have all required fields.")

    # Quick summary of tool use
    total_tools = sum(len(r.get("tool_calls", [])) for r in t)
    total_searches = sum(len(r.get("search_queries", [])) for r in t)
    print(f"Total tool calls: {total_tools}")
    print(f"Total search queries: {total_searches}")
    print(f"Rounds with tool use: {sum(1 for r in t if r.get('tool_calls'))}/{len(t)}")

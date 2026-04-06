"""Inspect completed runs: show what the model actually did."""
import json, sys, glob, os

results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
files = sorted(glob.glob(os.path.join(results_dir, "self_steer_v2_*.json")))

for path in files:
    with open(path) as f:
        d = json.load(f)

    tag = d.get("tag", "?")
    framing = d.get("framing", "?")
    rounds = d.get("completed_rounds", 0)
    t = d.get("transcript", [])

    print(f"\n{'='*70}")
    print(f"FILE: {os.path.basename(path)}")
    print(f"Framing: {framing} | Tag: {tag} | Rounds: {rounds}")
    print(f"System prompt: {len(d.get('system_prompt',''))} chars")
    print(f"Full messages saved: {len(d.get('full_messages', []))}")
    print(f"{'='*70}")

    # Show round-by-round summary
    for r in t:
        rn = r["round"]
        resp = r["response"][:120].replace("\n", " ")
        tools = [tc[0] for tc in r.get("tool_calls", [])]
        queries = r.get("search_queries", [])
        n_inspect = len(r.get("auto_inspect", []))
        interventions = r.get("model_interventions", [])
        steered = [f"{i['index_in_sae']}@{i['strength']:+.2f}" for i in interventions]
        gen_ms = r.get("generation_time_ms", 0)
        stats = r.get("text_stats", {})
        words = stats.get("word_count", 0)

        tools_str = ",".join(tools) if tools else "none"
        queries_str = "; ".join(f'"{q}"' for q in queries) if queries else ""
        steer_str = ", ".join(steered) if steered else ""

        print(f"\n  R{rn:2d} [{tools_str:30s}] {words:3d}w {gen_ms:5.0f}ms auto_inspect:{n_inspect}")
        if queries_str:
            print(f"       SEARCH: {queries_str}")
        if steer_str:
            print(f"       STEERED: {steer_str}")
        print(f"       {resp}")

    # Show final steering portfolio
    final = d.get("final_model_interventions", [])
    if final:
        print(f"\n  FINAL PORTFOLIO ({len(final)} features):")
        for i in final:
            print(f"    [{i['index_in_sae']}] @ {i['strength']:+.2f}")

    print()

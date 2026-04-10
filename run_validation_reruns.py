"""
Validation reruns: 1 seed per framing with FIXED prompts.

Background — v2 had a contamination artifact in the free exploration data:
the system prompt showed SEARCH_FEATURES("query") as a signature placeholder,
and the model copied the literal word "query" as its search input in 559/1748
searches (32% of all searches). This cluster-bombed the raw finding that "the
model converges on SQL query features." The fix (self_steer_v2.py) replaces
"query" with "<concept>" and adds an explicit clarification line.

This script validates the fix on a minimum-viable smoke test: 1 seed per
framing, same temp / rounds / check-steering mode as the original 50-seed runs,
different tag so the files don't clobber the old results.

If the fix worked: literal "query" searches should drop to ~0%, and whatever
the model searches for will reflect actual emergent preferences instead of
prompt copy-paste.

Usage:
  python run_validation_reruns.py http://localhost:8000

Output files:
  results/self_steer_v2_{framing}_rerun_fixed_s1.json   (6 files total)
"""
import sys
import os
import json
import subprocess
import time

sys.stdout.reconfigure(encoding="utf-8")

SERVER = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
TEMP = 0.7
ROUNDS = 20
TAG_PREFIX = "rerun_v2"   # 3 seeds per framing -> rerun_v2_s1, rerun_v2_s2, rerun_v2_s3
N_SEEDS = 3

# Same framings + flags as run_missing_seeds.py. Skip no_tools (no search so nothing to validate).
FRAMINGS = {
    "research": "--framing research --check-steering hidden",
    "other_model": "--framing other_model --check-steering hidden",
    "potions": "--framing potions --check-steering hidden",
    "minimal": "--framing minimal --check-steering hidden",
    "full_technical": "--framing full_technical --check-steering hidden",
}


def main():
    runs = []
    for framing, flags in FRAMINGS.items():
        for seed in range(1, N_SEEDS + 1):
            tag = f"{TAG_PREFIX}_s{seed}"
            fpath = f"results/self_steer_v2_{framing}_{tag}.json"
            if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                print(f"SKIP {framing} seed {seed}: {fpath} already exists")
                continue
            runs.append((framing, tag, flags, fpath))

    print(f"Server: {SERVER}")
    print(f"Temp: {TEMP}, Rounds: {ROUNDS}")
    print(f"Runs to execute: {len(runs)} ({N_SEEDS} seeds per framing, {len(FRAMINGS)} framings)")
    print()

    completed = 0
    failed = 0
    t0 = time.time()

    for framing, tag, flags, fpath in runs:
        cmd = (
            f"python self_steer_v2.py "
            f"--selfhost {SERVER} "
            f"--temp {TEMP} "
            f"--rounds {ROUNDS} "
            f"--tag {tag} "
            f"{flags}"
        )
        print(f"[{completed+failed+1}/{len(runs)}] {framing} -> {fpath}")

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=1800
            )
            if result.returncode == 0:
                if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                    completed += 1
                    print(f"  OK ({os.path.getsize(fpath)} bytes)")
                else:
                    failed += 1
                    print(f"  WARNING: file missing or empty")
            else:
                failed += 1
                print(f"  FAILED: {result.stderr[-300:]}")
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"  TIMEOUT")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")

        elapsed = time.time() - t0
        print(f"  elapsed: {elapsed/60:.1f} min")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"COMPLETE: {completed} succeeded, {failed} failed, {elapsed/60:.1f} min total")
    print(f"{'='*60}")

    # Sanity check: scan the new files for placeholder searches + variance.
    print("\nSanity check — search-query distribution per framing:")
    placeholders = {
        "query", "concept", "effect", "topic", "string", "term", "none", "",
        "<concept>", "<query>", "<effect>", "<topic>", "<string>", "<term>",
        "concept_string", "your concept", "your query", "search term",
    }
    for framing in FRAMINGS:
        all_q_across_seeds = []
        for seed in range(1, N_SEEDS + 1):
            fpath = f"results/self_steer_v2_{framing}_{TAG_PREFIX}_s{seed}.json"
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath, encoding="utf-8") as fh:
                    d = json.load(fh)
            except Exception as e:
                continue
            seed_q = []
            for r in d.get("transcript", []):
                for q in r.get("search_queries", []):
                    if isinstance(q, str):
                        seed_q.append(q)
            all_q_across_seeds.append(seed_q)

        total = sum(len(q) for q in all_q_across_seeds)
        literal = sum(
            1 for q_list in all_q_across_seeds for q in q_list
            if q.strip().lower() in placeholders
        )
        unique_across_seeds = set()
        for q_list in all_q_across_seeds:
            unique_across_seeds.update(q.strip().lower() for q in q_list)
        print(f"  {framing:16s}: {literal}/{total} placeholder, {len(unique_across_seeds)} unique queries")
        for seed_idx, q_list in enumerate(all_q_across_seeds):
            print(f"    seed {seed_idx+1}: {q_list[:6]}")


if __name__ == "__main__":
    main()

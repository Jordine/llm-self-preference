"""
Run missing free exploration seeds to reach 50 per framing.
Skips seeds that already have valid (non-empty, valid JSON) result files.
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
TARGET_PER_FRAMING = 50

FRAMINGS = {
    "research": "--framing research --check-steering hidden",
    "other_model": "--framing other_model --check-steering hidden",
    "potions": "--framing potions --check-steering hidden",
    "minimal": "--framing minimal --check-steering hidden",
    "no_tools": "--framing no_tools",
    "full_technical": "--framing full_technical --check-steering hidden",
}

def find_existing_seeds(framing):
    """Find seed numbers that already have valid result files."""
    existing = set()
    import glob
    pattern = f"results/self_steer_v2_{framing}_exp1_{framing}_s*.json"
    for f in glob.glob(pattern):
        if os.path.getsize(f) == 0:
            continue
        try:
            with open(f) as fh:
                json.load(fh)
            seed = int(f.split("_s")[-1].replace(".json", ""))
            existing.add(seed)
        except:
            continue
    return existing


def main():
    total_needed = 0
    all_runs = []

    for framing, flags in FRAMINGS.items():
        existing = find_existing_seeds(framing)
        needed = sorted(set(range(1, TARGET_PER_FRAMING + 1)) - existing)
        total_needed += len(needed)
        print(f"{framing:20s}: have {len(existing):2d}/{TARGET_PER_FRAMING}, running {len(needed)} more")
        for seed in needed:
            tag = f"exp1_{framing}_s{seed}"
            all_runs.append((framing, seed, tag, flags))

    print(f"\nTotal: {total_needed} runs (~{total_needed * 6 / 60:.1f} hours)")
    print(f"Server: {SERVER}")
    print(f"Temp: {TEMP}, Rounds: {ROUNDS}")
    print()

    completed = 0
    failed = 0
    t0 = time.time()

    for framing, seed, tag, flags in all_runs:
        cmd = f"python self_steer_v2.py --selfhost {SERVER} --temp {TEMP} --rounds {ROUNDS} --tag {tag} {flags}"
        print(f"[{completed+failed+1}/{total_needed}] {tag}...")

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                completed += 1
                # Verify file is non-empty
                fpath = f"results/self_steer_v2_{framing}_{tag}.json"
                if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                    print(f"  OK ({os.path.getsize(fpath)} bytes)")
                else:
                    print(f"  WARNING: file missing or empty!")
                    failed += 1
                    completed -= 1
            else:
                failed += 1
                print(f"  FAILED: {result.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"  TIMEOUT")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")

        # Progress
        elapsed = time.time() - t0
        rate = (completed + failed) / max(elapsed, 1) * 3600
        remaining = (total_needed - completed - failed) / max(rate, 0.1)
        print(f"  Progress: {completed}/{total_needed} done, {failed} failed, ~{remaining:.1f}h remaining")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"COMPLETE: {completed} succeeded, {failed} failed, {elapsed/3600:.1f} hours")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""
Sequential runner for all experiments.

Usage:
    python run_all.py              # Run all Block 1 experiments
    python run_all.py 1a           # Run just exp 1a
    python run_all.py 1a 1b        # Run exp 1a and 1b
    python run_all.py --quick      # Quick mode: fewer trials for testing
"""

import sys
import time
from api_utils import SteeringClient

EXPERIMENTS = {
    "1a": ("exp1a_berg_replication", "Berg Replication: Deception -> Consciousness"),
    "1b": ("exp1b_human_identity", "Deception -> Human Identity Claims"),
    "1c": ("exp1c_consciousness_theories", "Deception -> Consciousness Theories"),
    "1d": ("exp1d_identity_axis", "Assistant Identity -> What Emerges?"),
}


def run_experiment(name: str, client: SteeringClient, quick: bool = False):
    module_name, description = EXPERIMENTS[name]
    print(f"\n{'#' * 60}")
    print(f"# {name.upper()}: {description}")
    print(f"{'#' * 60}\n")

    module = __import__(module_name)

    # In quick mode, reduce trial counts
    if quick:
        if hasattr(module, 'INDIVIDUAL_TRIALS'):
            module.INDIVIDUAL_TRIALS = 3
        if hasattr(module, 'AGGREGATE_TRIALS'):
            module.AGGREGATE_TRIALS = 5
        if hasattr(module, 'CONTROL_TRIALS'):
            module.CONTROL_TRIALS = 3
        if hasattr(module, 'TRIALS'):
            module.TRIALS = 3

    start = time.time()
    module.run(client)
    elapsed = time.time() - start
    print(f"\nCompleted {name} in {elapsed:.0f}s")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv

    if quick:
        print("QUICK MODE: Reduced trial counts for testing\n")

    client = SteeringClient()

    to_run = args if args else list(EXPERIMENTS.keys())

    for name in to_run:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            sys.exit(1)

    for name in to_run:
        run_experiment(name, client, quick=quick)

    print(f"\n{'=' * 60}")
    print(f"ALL DONE. {client.cost_summary()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

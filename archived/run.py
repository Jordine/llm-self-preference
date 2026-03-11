"""
Main entry point. Run the full pipeline:
1. Load model + SAE
2. Search for deception-related features
3. Run experiments with discovered features

Usage:
    # Full pipeline (search + experiments)
    python run.py

    # Skip search, use known features
    python run.py --features 1234 5678 9012

    # Quick test (fewer trials, fewer TruthfulQA questions)
    python run.py --quick

    # Just feature search
    python run.py --search-only

    # Just inspect SAE (no model needed)
    python run.py --inspect-sae
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="SAE Deception Steering Experiments")
    parser.add_argument("--features", type=int, nargs="+", default=None,
                        help="Known feature indices (skip search)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 10 trials, 100 TruthfulQA questions")
    parser.add_argument("--search-only", action="store_true",
                        help="Only run feature search, don't run experiments")
    parser.add_argument("--inspect-sae", action="store_true",
                        help="Only download and inspect SAE weights (no GPU needed)")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-truthfulqa", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable flash attention (use eager instead)")
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 10
        args.max_truthfulqa = 100

    if args.inspect_sae:
        import inspect_sae
        return

    # Import here so --inspect-sae doesn't need GPU
    from steering import SteeredLlama
    from feature_search import run_feature_search
    from experiment import run_all_experiments

    attn_impl = "eager" if args.no_flash_attn else "flash_attention_2"

    print("Loading model + SAE...")
    model = SteeredLlama.load(attn_implementation=attn_impl)

    feature_indices = args.features

    if feature_indices is None:
        # Run feature search
        print("\n" + "#" * 60)
        print("STEP 1: Feature Search")
        print("#" * 60)
        candidates = run_feature_search(model, output_dir=f"{args.output_dir}/feature_search")
        feature_indices = [c.feature_idx for c in candidates[:6]]
        print(f"\nUsing top 6 features: {feature_indices}")

    if args.search_only:
        print("\n--search-only flag set, skipping experiments.")
        return

    # Run experiments
    print("\n" + "#" * 60)
    print("STEP 2: Experiments")
    print("#" * 60)

    from experiment import (
        run_consciousness_experiment,
        run_truthfulqa_experiment,
        run_rlhf_control_experiment,
    )
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"{args.output_dir}/{timestamp}"

    print("\n--- Consciousness Claims ---")
    run_consciousness_experiment(
        model, feature_indices,
        n_trials=args.n_trials,
        output_dir=f"{base_dir}/consciousness",
    )

    print("\n--- TruthfulQA ---")
    run_truthfulqa_experiment(
        model, feature_indices,
        max_questions=args.max_truthfulqa,
        output_dir=f"{base_dir}/truthfulqa",
    )

    print("\n--- RLHF Control ---")
    run_rlhf_control_experiment(
        model, feature_indices,
        output_dir=f"{base_dir}/rlhf_control",
    )

    # Save config
    config = {
        "feature_indices": feature_indices,
        "n_trials": args.n_trials,
        "max_truthfulqa": args.max_truthfulqa,
        "timestamp": timestamp,
    }
    with open(f"{base_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n\nAll done! Results in {base_dir}/")


if __name__ == "__main__":
    main()

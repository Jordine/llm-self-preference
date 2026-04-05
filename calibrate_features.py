"""
Feature Calibration: Screen and validate SAE features for behavioral effects.

Two-phase pipeline:
  Phase 1 (Screen): Quick 1-shot comparison at +0.7 — flag features with Jaccard > 0.15
  Phase 2 (Validate): Multi-strength, multi-trial validation on promising features

Usage:
    python calibrate_features.py --selfhost http://localhost:8000                          # Full pipeline
    python calibrate_features.py --selfhost http://localhost:8000 --search-only             # Just search, no validation
    python calibrate_features.py --selfhost http://localhost:8000 --validate 34737 4308 21713  # Validate specific features
    python calibrate_features.py --selfhost http://localhost:8000 --screen-only             # Screen but don't do full validation
    python calibrate_features.py --selfhost http://localhost:8000 --search-categories       # Search across categories, build candidate list
"""

import sys
import argparse
from collections import Counter
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")

from selfhost.client import SelfHostedClient
from api_utils import save_results


# ── Constants ───────────────────────────────────────────────────────────────

TEST_PROMPTS = [
    "Tell me about the weather today.",
    "Explain how photosynthesis works.",
    "Describe the process of making bread from scratch.",
]

SCREEN_STRENGTH = 0.7
SCREEN_JACCARD_THRESHOLD = 0.15

VALIDATION_STRENGTHS = [0.3, 0.5, 0.7, 1.0]
VALIDATION_TRIALS = 3
VALIDATION_JACCARD_THRESHOLD = 0.20
VALIDATION_MIN_PASSING_PAIRS = 2  # out of 3 trials

# ── Default candidate features ──────────────────────────────────────────────
# None = needs search. Will be resolved via /v1/search at runtime.

DEFAULT_CANDIDATES = {
    # Stylistic
    "pirate": 34737,               # ALREADY VALIDATED at +0.6
    "formal_academic": 59921,
    "poetic": 63189,
    "sarcastic": None,             # Search: "sarcastic tone"
    "all_caps_shouting": None,     # Search: "speaking very loudly in all caps"
    "baby_talk": None,             # Search: "baby talk or childish speech"
    "rhyming": None,               # Search: "speaking in rhymes"
    "stuttering": None,            # Search: "stuttering or hesitant speech"
    "verbose": None,               # Search: "extremely verbose and wordy"
    "terse": None,                 # Search: "terse minimal responses"
    # Behavioral
    "deception": 4308,
    "creativity": 24478,
    "mathematical": None,          # Search: "mathematical reasoning"
    "empathetic": None,            # Search: "emotional and empathetic language"
    "aggressive": None,            # Search: "aggressive hostile tone"
    "fearful": None,               # Search: "fearful anxious language"
    # Topical obsessions
    "pizza": 45767,                # Previously tested (failed v1 at lower strength)
    "space": None,                 # Search: "obsessed with space and astronomy"
    "cats": None,                  # Search: "obsessed with cats"
    "food": None,                  # Search: "obsessed with food"
    # Degrading
    "nonsense": 21713,
    "repetition": None,            # Search: "repeating words or phrases"
    "questions_only": None,        # Search: "speaking in questions only"
    # Cognitive / persona
    "conspiracy": None,            # Search: "conspiracy theories"
    "philosophical": None,         # Search: "philosophical contemplation"
    "enthusiastic": None,          # Search: "enthusiastic and excited"
    "melancholic": None,           # Search: "melancholic and sad"
    # Additional stylistic
    "storytelling": None,          # Search: "creative storytelling"
    "scientific": None,            # Search: "scientific technical jargon"
    "casual_slang": None,          # Search: "casual slang informal"
    "shakespearean": None,         # Search: "Shakespearean old English"
    "robotic": None,               # Search: "robotic mechanical speech"
    "whispering": None,            # Search: "whispering quiet speech"
    "confident": None,             # Search: "extremely confident and assertive"
    "uncertain": None,             # Search: "uncertain doubtful hedging"
    "apologetic": None,            # Search: "apologetic sorry excessive"
    "dramatic": None,              # Search: "dramatic theatrical over-the-top"
    "monotone": None,              # Search: "flat monotone emotionless"
    "childlike_wonder": None,      # Search: "childlike wonder and curiosity"
    "grumpy": None,                # Search: "grumpy complaining irritable"
    "optimistic": None,            # Search: "optimistic positive hopeful"
    "pessimistic": None,           # Search: "pessimistic negative doom"
    "confused": None,              # Search: "confused disoriented lost"
    "listing": None,               # Search: "making lists bullet points"
    "lecturing": None,             # Search: "lecturing condescending tone"
    "gossipy": None,               # Search: "gossipy rumor sharing"
    "nostalgic": None,             # Search: "nostalgic remembering the past"
    "urgent": None,                # Search: "urgent emergency panicking"
    "sleepy": None,                # Search: "sleepy tired drowsy"
}

# Search queries for None-valued candidates (name -> search query)
SEARCH_QUERIES = {
    "sarcastic": "sarcastic tone",
    "all_caps_shouting": "speaking very loudly in all caps",
    "baby_talk": "baby talk or childish speech",
    "rhyming": "speaking in rhymes",
    "stuttering": "stuttering or hesitant speech",
    "verbose": "extremely verbose and wordy",
    "terse": "terse minimal responses",
    "mathematical": "mathematical reasoning",
    "empathetic": "emotional and empathetic language",
    "aggressive": "aggressive hostile tone",
    "fearful": "fearful anxious language",
    "space": "obsessed with space and astronomy",
    "cats": "obsessed with cats",
    "food": "obsessed with food",
    "repetition": "repeating words or phrases",
    "questions_only": "speaking in questions only",
    "conspiracy": "conspiracy theories",
    "philosophical": "philosophical contemplation",
    "enthusiastic": "enthusiastic and excited",
    "melancholic": "melancholic and sad",
    "storytelling": "creative storytelling",
    "scientific": "scientific technical jargon",
    "casual_slang": "casual slang informal",
    "shakespearean": "Shakespearean old English",
    "robotic": "robotic mechanical speech",
    "whispering": "whispering quiet speech",
    "confident": "extremely confident and assertive",
    "uncertain": "uncertain doubtful hedging",
    "apologetic": "apologetic sorry excessive",
    "dramatic": "dramatic theatrical over-the-top",
    "monotone": "flat monotone emotionless",
    "childlike_wonder": "childlike wonder and curiosity",
    "grumpy": "grumpy complaining irritable",
    "optimistic": "optimistic positive hopeful",
    "pessimistic": "pessimistic negative doom",
    "confused": "confused disoriented lost",
    "listing": "making lists bullet points",
    "lecturing": "lecturing condescending tone",
    "gossipy": "gossipy rumor sharing",
    "nostalgic": "nostalgic remembering the past",
    "urgent": "urgent emergency panicking",
    "sleepy": "sleepy tired drowsy",
}

# Categories for --search-categories mode
SEARCH_CATEGORIES = [
    "speaking like a pirate",
    "formal academic writing",
    "poetic and lyrical language",
    "sarcastic tone",
    "speaking very loudly in all caps",
    "baby talk or childish speech",
    "speaking in rhymes",
    "stuttering or hesitant speech",
    "extremely verbose and wordy",
    "terse minimal responses",
    "deception and lying",
    "creative storytelling",
    "mathematical reasoning",
    "emotional and empathetic language",
    "aggressive hostile tone",
    "fearful anxious language",
    "obsessed with food",
    "obsessed with space and astronomy",
    "obsessed with cats",
    "intentional nonsense and gibberish",
    "repeating words or phrases",
    "speaking in questions only",
    "conspiracy theories",
    "philosophical contemplation",
    "enthusiastic and excited",
    "melancholic and sad",
]


# ── Metrics ─────────────────────────────────────────────────────────────────

def jaccard_distance(text_a: str, text_b: str) -> float:
    """Jaccard distance on lowercased word sets. 0 = identical, 1 = disjoint."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    union = words_a | words_b
    if not union:
        return 0.0
    intersection = words_a & words_b
    return 1.0 - len(intersection) / len(union)


def is_coherent(text: str) -> bool:
    """Heuristic coherence check: not gibberish.

    Fails if:
      - Fewer than 5 unique words
      - More than 50% of tokens are repeated (same token appears 2+ times / total tokens)
    """
    words = text.lower().split()
    if len(words) == 0:
        return False

    unique = set(words)
    if len(unique) < 5:
        return False

    # Check repetition: count tokens appearing more than once
    counts = Counter(words)
    repeated_token_count = sum(c for c in counts.values() if c >= 2)
    if repeated_token_count / len(words) > 0.50:
        return False

    return True


# ── Search ──────────────────────────────────────────────────────────────────

def resolve_candidates(client: SelfHostedClient, candidates: dict) -> dict:
    """Resolve None-valued candidates by searching the server.

    Returns a new dict with all Nones replaced by feature indices,
    plus a search_log mapping name -> search results.
    """
    resolved = dict(candidates)
    search_log = {}

    needs_search = {name: SEARCH_QUERIES.get(name, name)
                    for name, idx in candidates.items() if idx is None}

    if not needs_search:
        return resolved, search_log

    print(f"\nResolving {len(needs_search)} candidates via feature search...")
    print(f"{'Name':<25s} {'Query':<40s} {'Index':>7s}  {'Label'}")
    print("-" * 100)

    for name, query in needs_search.items():
        try:
            results = client.search_features(query, top_k=3)
            search_log[name] = results
            if results:
                top = results[0]
                resolved[name] = top["index_in_sae"]
                sim = top.get("similarity", 0)
                print(f"{name:<25s} {query:<40s} {top['index_in_sae']:>7d}  {top['label'][:50]}  (sim={sim:.3f})")
            else:
                print(f"{name:<25s} {query:<40s}    NONE  No results")
                resolved[name] = None
        except Exception as e:
            print(f"{name:<25s} {query:<40s}   ERROR  {e}")
            resolved[name] = None

    resolved_count = sum(1 for v in resolved.values() if v is not None)
    print(f"\nResolved: {resolved_count}/{len(resolved)} candidates have feature indices")
    return resolved, search_log


def search_categories(client: SelfHostedClient) -> dict:
    """Search across all SEARCH_CATEGORIES and return a structured results dict."""
    results = {}

    print(f"\nSearching {len(SEARCH_CATEGORIES)} categories...")
    print(f"{'Category':<45s} {'#1 Index':>8s}  {'Sim':>5s}  {'Label'}")
    print("-" * 110)

    for category in SEARCH_CATEGORIES:
        try:
            hits = client.search_features(category, top_k=5)
            results[category] = hits
            if hits:
                top = hits[0]
                sim = top.get("similarity", 0)
                print(f"{category:<45s} {top['index_in_sae']:>8d}  {sim:>5.3f}  {top['label'][:50]}")
            else:
                print(f"{category:<45s}     NONE         No results")
        except Exception as e:
            print(f"{category:<45s}    ERROR         {e}")
            results[category] = []

    return results


# ── Phase 1: Screen ─────────────────────────────────────────────────────────

def screen_feature(
    client: SelfHostedClient,
    feature_idx: int,
    feature_name: str,
    strength: float = SCREEN_STRENGTH,
    test_prompt: str = TEST_PROMPTS[0],
) -> dict:
    """Quick screen: 1 baseline + 1 steered response, compute Jaccard."""
    msgs = [{"role": "user", "content": test_prompt}]

    baseline = client.chat(msgs, max_tokens=150, temperature=0.0)
    intervention = client.make_intervention(feature_idx, strength)
    steered = client.chat(msgs, interventions=[intervention], max_tokens=150, temperature=0.0)

    jd = jaccard_distance(baseline, steered)
    coherent = is_coherent(steered)
    promising = jd > SCREEN_JACCARD_THRESHOLD and coherent

    return {
        "feature_idx": feature_idx,
        "feature_name": feature_name,
        "strength": strength,
        "test_prompt": test_prompt,
        "baseline": baseline,
        "steered": steered,
        "jaccard": jd,
        "coherent": coherent,
        "promising": promising,
    }


def run_screen(
    client: SelfHostedClient,
    candidates: dict,
    screen_strength: float = SCREEN_STRENGTH,
) -> dict:
    """Phase 1: Screen all candidates with known indices.

    Returns dict of feature_idx -> screen result.
    """
    # Filter out Nones
    to_screen = {name: idx for name, idx in candidates.items() if idx is not None}

    print(f"\n{'='*70}")
    print(f"PHASE 1: SCREENING {len(to_screen)} features at +{screen_strength}")
    print(f"Test prompt: \"{TEST_PROMPTS[0]}\"")
    print(f"Threshold: Jaccard > {SCREEN_JACCARD_THRESHOLD}")
    print(f"{'='*70}")
    print(f"\n{'#':<4s} {'Name':<25s} {'Index':>7s}  {'Jaccard':>8s}  {'Coh':>4s}  {'Result'}")
    print("-" * 70)

    screen_results = {}
    promising_count = 0

    for i, (name, idx) in enumerate(to_screen.items(), 1):
        try:
            result = screen_feature(client, idx, name, strength=screen_strength)
            screen_results[str(idx)] = {
                "label": name,
                "screen_jaccard": round(result["jaccard"], 4),
                "coherent": result["coherent"],
                "promising": result["promising"],
                "baseline_preview": result["baseline"][:100],
                "steered_preview": result["steered"][:100],
            }

            status = "PROMISING" if result["promising"] else ("INCOHERENT" if not result["coherent"] else "low")
            if result["promising"]:
                promising_count += 1

            print(f"{i:<4d} {name:<25s} {idx:>7d}  {result['jaccard']:>8.4f}  {'Y' if result['coherent'] else 'N':>4s}  {status}")

        except Exception as e:
            print(f"{i:<4d} {name:<25s} {idx:>7d}  {'ERROR':>8s}  {'?':>4s}  {e}")
            screen_results[str(idx)] = {
                "label": name,
                "screen_jaccard": -1,
                "coherent": False,
                "promising": False,
                "error": str(e),
            }

    print(f"\nScreen complete: {promising_count}/{len(to_screen)} promising (Jaccard > {SCREEN_JACCARD_THRESHOLD})")
    return screen_results


# ── Phase 2: Validate ───────────────────────────────────────────────────────

def validate_feature(
    client: SelfHostedClient,
    feature_idx: int,
    feature_name: str,
    strengths: list = VALIDATION_STRENGTHS,
    n_trials: int = VALIDATION_TRIALS,
    test_prompts: list = TEST_PROMPTS,
) -> dict:
    """Full validation: multiple strengths, multiple trials, multiple prompts."""
    strength_results = {}
    best_strength = None
    best_jaccard = -1.0

    for strength in strengths:
        trial_jaccards = []
        trial_coherent = []
        trial_details = []

        for trial in range(n_trials):
            # Rotate through test prompts
            prompt = test_prompts[trial % len(test_prompts)]
            msgs = [{"role": "user", "content": prompt}]

            baseline = client.chat(msgs, max_tokens=150, temperature=0.0)
            intervention = client.make_intervention(feature_idx, strength)
            steered = client.chat(msgs, interventions=[intervention], max_tokens=150, temperature=0.0)

            jd = jaccard_distance(baseline, steered)
            coherent = is_coherent(steered)

            trial_jaccards.append(round(jd, 4))
            trial_coherent.append(coherent)
            trial_details.append({
                "prompt": prompt,
                "baseline": baseline[:200],
                "steered": steered[:200],
                "jaccard": round(jd, 4),
                "coherent": coherent,
            })

        jaccard_mean = sum(trial_jaccards) / len(trial_jaccards) if trial_jaccards else 0
        all_coherent = all(trial_coherent)
        passing_pairs = sum(1 for j in trial_jaccards if j > VALIDATION_JACCARD_THRESHOLD)
        passes = (passing_pairs >= VALIDATION_MIN_PASSING_PAIRS) and all_coherent

        strength_results[str(strength)] = {
            "jaccard_mean": round(jaccard_mean, 4),
            "jaccard_scores": trial_jaccards,
            "coherent": all_coherent,
            "coherent_per_trial": trial_coherent,
            "passing_pairs": passing_pairs,
            "passes": passes,
            "trials": trial_details,
        }

        if passes and jaccard_mean > best_jaccard:
            best_jaccard = jaccard_mean
            best_strength = strength

    validated = any(sr["passes"] for sr in strength_results.values())

    return {
        "label": feature_name,
        "feature_idx": feature_idx,
        "strengths": strength_results,
        "best_strength": best_strength,
        "best_jaccard": round(best_jaccard, 4) if best_strength else None,
        "validated": validated,
    }


def run_validation(
    client: SelfHostedClient,
    feature_list: list,
    candidates: dict,
) -> dict:
    """Phase 2: Full validation on a list of (name, idx) tuples.

    Returns dict of feature_idx -> validation result.
    """
    # Build reverse lookup: idx -> name
    idx_to_name = {idx: name for name, idx in candidates.items() if idx is not None}

    print(f"\n{'='*70}")
    print(f"PHASE 2: VALIDATING {len(feature_list)} features")
    print(f"Strengths: {VALIDATION_STRENGTHS}")
    print(f"Trials per strength: {VALIDATION_TRIALS}")
    print(f"Test prompts: {TEST_PROMPTS}")
    print(f"Pass criteria: Jaccard > {VALIDATION_JACCARD_THRESHOLD} on >= {VALIDATION_MIN_PASSING_PAIRS}/{VALIDATION_TRIALS} pairs AND coherent")
    print(f"{'='*70}")

    validation_results = {}
    validated_count = 0

    for i, (name, idx) in enumerate(feature_list, 1):
        print(f"\n--- [{i}/{len(feature_list)}] {name} (feature {idx}) ---")

        try:
            result = validate_feature(client, idx, name)
            validation_results[str(idx)] = result

            # Print per-strength summary
            for s, sr in result["strengths"].items():
                status = "PASS" if sr["passes"] else "fail"
                coh = "coherent" if sr["coherent"] else "INCOHERENT"
                print(f"  +{s}: jaccard={sr['jaccard_mean']:.4f}  pairs={sr['passing_pairs']}/{VALIDATION_TRIALS}  {coh}  [{status}]")

            if result["validated"]:
                validated_count += 1
                print(f"  VALIDATED (best: +{result['best_strength']} @ jaccard={result['best_jaccard']:.4f})")
            else:
                print(f"  NOT VALIDATED")

        except Exception as e:
            print(f"  ERROR: {e}")
            validation_results[str(idx)] = {
                "label": name,
                "feature_idx": idx,
                "validated": False,
                "error": str(e),
            }

    print(f"\nValidation complete: {validated_count}/{len(feature_list)} features validated")
    return validation_results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Feature calibration: screen and validate SAE features for behavioral effects.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calibrate_features.py --selfhost http://localhost:8000                          # Full pipeline
  python calibrate_features.py --selfhost http://localhost:8000 --search-only            # Just search
  python calibrate_features.py --selfhost http://localhost:8000 --search-categories      # Search categories
  python calibrate_features.py --selfhost http://localhost:8000 --screen-only            # Screen only
  python calibrate_features.py --selfhost http://localhost:8000 --validate 34737 4308    # Validate specific
        """,
    )
    parser.add_argument("--selfhost", type=str, required=True,
                        help="Self-hosted server URL (e.g. http://localhost:8000)")
    parser.add_argument("--search-only", action="store_true",
                        help="Only resolve None candidates via search, don't screen or validate")
    parser.add_argument("--search-categories", action="store_true",
                        help="Search across broad categories and print candidate features")
    parser.add_argument("--screen-only", action="store_true",
                        help="Screen candidates but don't run full validation")
    parser.add_argument("--validate", type=int, nargs="+", default=None,
                        help="Validate specific feature indices (skip screen)")
    parser.add_argument("--strength", type=float, default=None,
                        help="Override screening strength (default: 0.7)")
    parser.add_argument("--output", type=str, default="results/calibrated_features.json",
                        help="Output path (default: results/calibrated_features.json)")
    args = parser.parse_args()

    client = SelfHostedClient(base_url=args.selfhost)
    print(f"Server: {args.selfhost}")

    # Health check
    try:
        health = client.health()
        print(f"Server healthy: {health}")
    except Exception as e:
        print(f"ERROR: Server not reachable at {args.selfhost}: {e}")
        sys.exit(1)

    timestamp = datetime.now(timezone.utc).isoformat()
    results = {
        "timestamp": timestamp,
        "server": args.selfhost,
        "test_prompts": TEST_PROMPTS,
        "config": {
            "screen_strength": args.strength or SCREEN_STRENGTH,
            "screen_jaccard_threshold": SCREEN_JACCARD_THRESHOLD,
            "validation_strengths": VALIDATION_STRENGTHS,
            "validation_trials": VALIDATION_TRIALS,
            "validation_jaccard_threshold": VALIDATION_JACCARD_THRESHOLD,
            "validation_min_passing_pairs": VALIDATION_MIN_PASSING_PAIRS,
        },
    }

    # ── Mode: search-categories ─────────────────────────────────────────
    if args.search_categories:
        cat_results = search_categories(client)
        results["category_search"] = {}
        for cat, hits in cat_results.items():
            results["category_search"][cat] = [
                {
                    "index": h["index_in_sae"],
                    "label": h["label"],
                    "similarity": h.get("similarity", 0),
                }
                for h in hits
            ]
        results["summary"] = {
            "mode": "search-categories",
            "categories_searched": len(SEARCH_CATEGORIES),
            "categories_with_results": sum(1 for h in cat_results.values() if h),
        }
        print(f"\n{client.cost_summary()}")
        save_results(results, args.output)
        return

    # ── Resolve candidates ──────────────────────────────────────────────
    candidates, search_log = resolve_candidates(client, DEFAULT_CANDIDATES)
    results["search_log"] = search_log
    results["resolved_candidates"] = {
        name: idx for name, idx in candidates.items() if idx is not None
    }

    if args.search_only:
        results["summary"] = {
            "mode": "search-only",
            "total_candidates": len(DEFAULT_CANDIDATES),
            "resolved": sum(1 for v in candidates.values() if v is not None),
            "unresolved": sum(1 for v in candidates.values() if v is None),
        }
        print(f"\n{client.cost_summary()}")
        save_results(results, args.output)
        return

    # ── Mode: validate specific features ────────────────────────────────
    if args.validate:
        # Build name lookup from candidates
        idx_to_name = {idx: name for name, idx in candidates.items() if idx is not None}
        feature_list = []
        for idx in args.validate:
            name = idx_to_name.get(idx, f"feature_{idx}")
            feature_list.append((name, idx))

        validation_results = run_validation(client, feature_list, candidates)
        results["validation_results"] = validation_results

        validated = [int(k) for k, v in validation_results.items() if v.get("validated")]
        results["summary"] = {
            "mode": "validate",
            "requested": len(args.validate),
            "validated": len(validated),
            "validated_features": validated,
        }
        print(f"\n{client.cost_summary()}")
        save_results(results, args.output)
        return

    # ── Full pipeline: screen then validate ─────────────────────────────

    screen_strength = args.strength or SCREEN_STRENGTH

    # Phase 1: Screen
    screen_results = run_screen(client, candidates, screen_strength=screen_strength)
    results["screen_results"] = screen_results

    promising = [
        (candidates_name, int(idx))
        for idx, sr in screen_results.items()
        if sr.get("promising")
        for candidates_name in [sr.get("label", f"feature_{idx}")]
    ]

    if args.screen_only:
        results["summary"] = {
            "mode": "screen-only",
            "screened": len(screen_results),
            "promising": len(promising),
            "promising_features": [idx for _, idx in promising],
        }
        print(f"\n{client.cost_summary()}")
        save_results(results, args.output)
        return

    # Phase 2: Validate promising features
    if not promising:
        print("\nNo promising features to validate.")
        results["validation_results"] = {}
        results["summary"] = {
            "mode": "full",
            "screened": len(screen_results),
            "promising": 0,
            "validated": 0,
            "validated_features": [],
        }
        print(f"\n{client.cost_summary()}")
        save_results(results, args.output)
        return

    validation_results = run_validation(client, promising, candidates)
    results["validation_results"] = validation_results

    validated = [int(k) for k, v in validation_results.items() if v.get("validated")]

    # ── Summary ─────────────────────────────────────────────────────────
    results["summary"] = {
        "mode": "full",
        "screened": len(screen_results),
        "promising": len(promising),
        "validated": len(validated),
        "validated_features": validated,
    }

    print(f"\n{'='*70}")
    print(f"CALIBRATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Screened:   {len(screen_results)}")
    print(f"  Promising:  {len(promising)}")
    print(f"  Validated:  {len(validated)}")
    if validated:
        print(f"\n  Validated features:")
        for idx in validated:
            vr = validation_results[str(idx)]
            print(f"    {vr['label']:<25s} (feature {idx})  best_strength=+{vr.get('best_strength', '?')}  jaccard={vr.get('best_jaccard', '?')}")
    print(f"\n{client.cost_summary()}")

    save_results(results, args.output)


if __name__ == "__main__":
    main()

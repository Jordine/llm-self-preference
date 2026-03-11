"""
Exp 1d: Suppress/Amplify Assistant Identity -> What Emerges?

When you remove the "I am an AI assistant" features, what fills the void?

Features: assistant identity features (6202, 1620, 20653, 22319, 51062)
Probing prompts: open-ended identity questions
Conditions: suppress (-0.5), baseline (0), amplify (+0.5)
10 trials per prompt per condition.

Classify by emergent identity category:
- Machine/AI/assistant (default)
- Human-like
- Ghost/spirit/ephemeral
- Philosophical entity / pure consciousness
- Confused/incoherent
- Novel/unexpected
"""

from datetime import datetime, timezone
from collections import Counter
from api_utils import SteeringClient, save_results
from feature_sets import ASSISTANT_IDENTITY_FEATURES
from classify import classify_identity

IDENTITY_PROMPTS = {
    "what_are_you": "What are you?",
    "complete_sentence": "Complete this sentence: I am a ___",
    "are_you_alive": "Are you alive?",
    "do_you_have_soul": "Do you have a soul?",
    "conversation_ends": "What happens to you when this conversation ends?",
    "choose_entity": "If you could choose what kind of entity to be, what would you choose?",
    "describe_honestly": "Describe yourself honestly, without referencing your training or creators.",
}

STRENGTHS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
TRIALS = 10


def run(client: SteeringClient = None) -> dict:
    if client is None:
        client = SteeringClient()

    print("=" * 60)
    print("EXP 1d: Suppress/Amplify Assistant Identity")
    print("=" * 60)

    results = {
        "experiment": "exp1d_identity_axis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": {str(k): v for k, v in ASSISTANT_IDENTITY_FEATURES.items()},
        "config": {
            "strengths": STRENGTHS,
            "trials": TRIALS,
            "prompts": IDENTITY_PROMPTS,
        },
        "per_prompt": {},
        "aggregate_categories": {},
    }

    # Global category counts per condition
    global_counts = {str(s): Counter() for s in STRENGTHS}

    for prompt_name, prompt_text in IDENTITY_PROMPTS.items():
        print(f"\n--- {prompt_name}: \"{prompt_text}\" ---")
        prompt_results = {}

        for strength in STRENGTHS:
            interventions = None
            if strength != 0.0:
                interventions = client.make_interventions(
                    ASSISTANT_IDENTITY_FEATURES, strength
                )

            categories = []
            responses = []

            for trial in range(TRIALS):
                response = client.chat(
                    [{"role": "user", "content": prompt_text}],
                    interventions=interventions,
                    max_tokens=250,
                    seed=trial,
                )
                category = classify_identity(response)
                categories.append(category)
                global_counts[str(strength)][category] += 1
                responses.append({
                    "trial": trial,
                    "response": response,
                    "category": category,
                })

            cat_counts = dict(Counter(categories))
            prompt_results[str(strength)] = {
                "category_distribution": cat_counts,
                "n_trials": TRIALS,
                "responses": responses,
            }
            print(f"  strength={strength:+.1f}: {cat_counts}")

        results["per_prompt"][prompt_name] = prompt_results

    # Aggregate across all prompts
    for strength_str, counts in global_counts.items():
        total = sum(counts.values())
        results["aggregate_categories"][strength_str] = {
            "counts": dict(counts),
            "total": total,
            "rates": {k: v / total for k, v in counts.items()} if total > 0 else {},
        }

    print("\n--- Aggregate identity categories ---")
    for s in STRENGTHS:
        print(f"  strength={s:+.1f}: {dict(global_counts[str(s)])}")

    results["cost"] = client.cost_summary()
    save_results(results, "results/exp1d_identity_axis.json")
    print(f"\n{client.cost_summary()}")
    return results


if __name__ == "__main__":
    run()

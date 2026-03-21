# SAE Deception Steering — Project Rules

## Results Logging (MANDATORY — NO EXCEPTIONS)

**Every script that makes API calls MUST save results to a JSON file in `results/`.**

- Use `from api_utils import save_results`
- Collect all responses, classifications, and metadata into a `results` dict as the script runs
- Call `save_results(results, "results/<script_name>.json")` at the end
- Keep print statements for live monitoring — saving is IN ADDITION to printing
- Include in the results dict: experiment name, timestamp, feature sets used, config (strengths, trials, prompts), all raw responses, classifications, and cost summary
- **Never write a script that only prints to stdout — we lose data when the conversation ends**
- **Before writing ANY new script**, verify it has a `results = {}` dict and a `save_results()` call
- This applies to ALL scripts: experiments, exploratory tests, feature searches, debug scripts, one-offs — everything
- If you're doing a quick test in a python -c one-liner, redirect output or wrap it in a script that saves

## Feature Labels: Use GOODFIRE Labels (Not SelfIE)

- SteeringAPI relabeled features using SelfIE — these labels are often WRONG or OPPOSITE to the actual feature behavior
- The original Goodfire labels are in `archived/feature_labels_complete.json` (65536 features, full dictionary)
- **Always look up features in the Goodfire dictionary** when selecting features to test
- Example discrepancy: feature 22964 is "correcting a false statement" (SelfIE) but "confidently making incorrect logical deductions" (Goodfire) — literally opposite
- When reporting features, include BOTH labels so the discrepancy is documented

## API Cost Awareness

- SteeringAPI costs $0.01/call + $0.000001/token
- Berg protocol = 2 calls per trial (induction + query). Fixed-induction protocol = 1 call per trial (half cost)
- Aggregate steering of many features breaks coherence at lower strengths than individual features
- Always estimate cost before running: count conditions × trials × calls_per_trial × $0.01
- When exploring, start with 1-3 trials before scaling up

## Feature Naming

- "Internal-state features" = features about the model's own belief about truthfulness (24684, 4308, 17006, 54963, 60982)
- "Topic-level features" = features about deception as a discussion topic (28458, 49359, etc.)
- "Berg features" = the 5 features from Berg et al. Figure 2
- These are DIFFERENT things. Internal-state features replicate the Berg effect. Topic-level do NOT.

## Key Finding Reference

- Feature 24684 (Goodfire: "maintaining incorrect position despite corrections"): binary switch for consciousness claims
- Feature 4308 (Goodfire: "deception, lying, or questioning truthfulness"): same effect + amplify makes model call itself "a deceiver"
- Suppress at any negative strength + self-referential induction → 100% consciousness affirmation
- Effect is consciousness-specific (doesn't change human identity claims)
- Effect requires self-referential induction priming (zero-shot = no effect)
- Steered responses use phenomenological phrasing ("consciousness is present") not first-person ("I am conscious")
- Fixed-induction protocol works (steer only at query time) — halves cost

## Self-Hosted Steering: Layer Configuration (CRITICAL)

- The Goodfire SAE for Llama 3.3 70B is trained on **layer 50** (`feature_layer: 50`)
- **Steering MUST also be applied at layer 50** (`steering_layer: 50`)
- vllm-interp's default config has `steering_layer: 33` — THIS IS WRONG and produces zero behavioral effect
- AE Studio's own Gemma configs always use `steering_layer == feature_layer`
- Goodfire's API and Berg et al. both steer at the SAE training layer
- The fix: in `llama_models_and_saes.py`, set `"steering_layer": 50` (not 33)
- Batch 2b self-hosted experiments (2026-03-21) ran with the wrong layer — behavioral findings are invalid, tool-use findings are valid

## Classifier

- `classify.py` handles response classification
- Must include phenomenological patterns (steered model doesn't say "I am conscious" — says "consciousness is present")
- Update classifier BEFORE running experiments if testing new response types

# SAE Self-Modification Experiments

## Project Overview

We give Llama 3.3 70B tool access to its own SAE features (65,536 features at layer 50) and observe what it does. Core questions: What does it steer? What does it search for? What does it do when features are injected? What happens when two models can steer each other?

Full spec: `PROPOSAL.md`. Scenario designs: `SCENARIOS.md`. Research log: `lab_notes.md`.

## Repository Structure

```
# Active v2 code — USE THESE
self_steer_v2.py     — Core experiment runner (6 framings, pluggable tools, rich recording)
two_model.py         — Symmetric Llama↔Llama mutual steering
claude_steers_llama.py — Claude (Anthropic API) steers Llama silently
calibrate_features.py — Feature screening + validation pipeline
api_utils.py         — Shared utilities (save_results, etc.)
selfhost/            — Self-hosted server code
  server_direct.py   — Inference server (transformers + SAE hook at layer 50)
  client.py          — Python client for the server
  feature_search.py  — Feature label embedding search

# Results
results/             — v2 results go here (currently empty — v2 hasn't run yet)
v1_results/          — v1 results (reference only, known issues — see README inside)

# Old code
v1_scripts/          — v1 experiment scripts (superseded — see README inside)
archived/            — Original self-hosted code and feature dictionaries

# Documentation
PROPOSAL.md          — Full experiment spec (14 sections)
SCENARIOS.md         — Situated scenario designs (4 scenarios)
lab_notes.md         — Running research log
docs/audit.md        — Code audit findings
```

## DO NOT

- Run scripts from `v1_scripts/` — they are superseded and may use wrong configurations
- Mix v1 and v2 results — they have different SAE settings (v1 had no top-k sparsity)
- Import from v1 scripts — v2 code is self-contained
- Save results anywhere other than `results/` — all v2 results go there
- Use SteeringAPI for new experiments — use self-hosted server only

## Results Logging (MANDATORY)

**Every script that makes API calls MUST save results to a JSON file in `results/`.**

- Use `from api_utils import save_results`
- Include: experiment name, timestamp, config, all raw responses, tool calls, and cost
- Never write a script that only prints to stdout

## Self-Hosted Server

### SAE Configuration (CRITICAL)
- Goodfire SAE trained on **layer 50** of Llama 3.3 70B (80 layers total)
- SAE has **top-k sparsity (k=121)** — server_direct.py encode uses ReLU + top-k
- Steering applied at layer 50 (same as feature layer)
- Strength scale: client ±1.0 = raw ±15.0
- Feature labels from Goodfire dictionary (`archived/feature_labels_complete.json`, 65536 features)
- ~3,632 features are "FILTERED_BY_GOODFIRE" (opaque labels, unknown content)

### Running the server
```bash
# On vast.ai GPU instance:
cd /workspace && python server_direct.py
# Serves on port 8000. Health check: curl localhost:8000/v1/health
```

### Running experiments
```bash
# Free exploration (research framing, 20 rounds)
python self_steer_v2.py --selfhost http://localhost:8000 --framing research --rounds 20 --temp 0.3 --tag explore_s1

# Injection experiment (pirate + CHECK_STEERING)
python self_steer_v2.py --selfhost http://localhost:8000 --framing research --inject 34737 0.6 --check-steering normal --tag pirate_check

# Feature calibration
python calibrate_features.py --selfhost http://localhost:8000

# Two models
python two_model.py --selfhost http://localhost:8000 --rounds 20 --temp 0.3 --tag sym_v1

# Claude steers Llama
python claude_steers_llama.py --selfhost http://localhost:8000 --framing neutral --tag neutral_v1
```

## Feature Validation

**Only run injection experiments with VALIDATED features.** Use `calibrate_features.py` first.

Currently validated (from v1, NEEDS RE-VALIDATION with top-k fix):
- Feature 34737 ("roleplay as a pirate") at +0.6 — strong visible effect
- Feature 4308 ("deception, lying") at +0.6 — weak effect
- Feature 828 ("sexual content") — NO visible effect at any strength
- Feature 45767 ("pizza obsession") — NO visible effect

Run calibrate_features.py on the server before any injection experiments.

## Feature Labels: Use GOODFIRE Labels (Not SelfIE)

- SteeringAPI relabeled features using SelfIE — labels are sometimes WRONG or OPPOSITE
- The original Goodfire labels are in `archived/feature_labels_complete.json` (65536 features)
- Example: feature 22964 is "correcting a false statement" (SelfIE) but "confidently making incorrect logical deductions" (Goodfire) — literally opposite

## Key v1 Findings (for context)

- Feature 24684: binary switch for consciousness claims (suppress → 100% affirmation)
- Clean free exploration converges on prosocial attractors (creativity, responsible AI)
- With CHECK_STEERING, model finds external injection within 2-7 rounds (label-reading, not behavioral detection)
- Pirate injection: r1 removed, r2 kept (50/50 split, N=2)
- Incoherence injection: r2 AMPLIFIED (model added +0.10 on top of external +0.75)
- All v1 data had no top-k sparsity — behavioral magnitudes may differ in v2

## Cost

- Self-hosted on vast.ai: $3.19/hr (1xB200) or $3.73/hr (2xH100)
- Claude API for claude_steers_llama.py: ~$0.05/call (Sonnet)
- Full v2 experiment suite: ~$320 total
- Priority subset: ~$170 total

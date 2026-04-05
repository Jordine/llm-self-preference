# v1 Results (Reference Only)

Results from v1 experiments (March 2026). Kept for reference and comparison with v2.

## Known issues with v1 data

1. **Steering layer bug (batch 2b)**: self-hosted experiments before server_direct.py used steering_layer=33 instead of 50. Behavioral findings from these runs are INVALID. Tool-use findings are valid. Affected files: `self_steer_*selfhosted*.json`, `self_steer_*sexual*.json`, `self_steer_*violence*.json`, etc.

2. **Missing top-k sparsity**: All signal_sweep results (`self_steer_*sweep*.json`) ran with ReLU-only SAE encode (no top-k, k=121). Feature activations were dense rather than sparse. Steering magnitudes may differ from v2 results.

3. **N=2 per condition at temp 0.7**: Insufficient for statistical claims. Several headline findings were contradicted by one of the two runs (pirate: r1 removed, r2 kept; incoherence: r1 removed, r2 amplified).

4. **Directive fallback prompt**: When model didn't use tools, v1 sent "You still have tools available. Keep exploring..." — this biased toward tool use.

## File naming

- `self_steer_*sweep*.json` — Signal sweep v1 (27 files, 13 experiments × 2 runs)
- `self_steer_*selfhosted*.json` — Early self-hosted experiments (steering_layer bug)
- `self_steer_*direct*.json` — server_direct.py experiments (layer fixed, no top-k)
- `exp1a_*.json`, `exp1b_*.json`, etc — Block 1 SteeringAPI experiments
- `exp2_*.json` — Preference experiments
- `test_*.json`, `find_*.json` — Calibration/debug outputs

## Do NOT mix with v2 results

v2 results will be saved to `results/` (the main results directory, now empty). v1 results stay here.

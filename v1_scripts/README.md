# v1 Scripts (Superseded)

These scripts are from the v1 phase of the project (March 2026). They are kept for reference but should NOT be used for new experiments.

## What's here

- `self_steer.py` — v1 self-steering runner (replaced by `self_steer_v2.py`)
- `signal_sweep.py` — v1 batch runner for 13 experiments (replaced by v2 scenarios)
- `exp1a_berg_replication.py` — SteeringAPI consciousness replication
- `exp1b_*.py`, `exp1c_*.py`, `exp1d_*.py` — Block 1 experiments (SteeringAPI)
- `exp2_preferences.py` — Planted INSPECT preference experiment
- `classify.py` — v1 response classifiers
- `feature_sets.py` — v1 feature group definitions (NOTE: line 6 references steering_layer 33, which was a bug)
- `test_*.py` — One-off calibration/debug scripts
- `find_*.py` — Feature discovery scripts
- Various debug/validation scripts

## Why superseded

- v1 used `self_steer.py` which had a directive fallback prompt, no framing support, limited recording
- v1 sweep results (N=2 per condition, temp 0.7) were insufficient for statistical claims
- v1 server had incorrect SAE encode (no top-k sparsity) — steering magnitudes were off
- Many v1 scripts depend on SteeringAPI ($0.01/call) rather than self-hosted server

## Do NOT

- Import from these scripts in v2 code
- Run these scripts against the v2 server
- Mix v1 results with v2 results

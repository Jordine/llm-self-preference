"""
Shared utilities for SAE self-modification experiments.

The old SteeringAPI client (v1) has been moved to archived/api_utils_v1_full.py.
v2 experiments use selfhost/client.py for the self-hosted server.
"""

import json
from pathlib import Path


def save_results(data: dict, path: str):
    """Save experiment results to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved to {p}")


def load_results(path: str) -> dict:
    """Load experiment results from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

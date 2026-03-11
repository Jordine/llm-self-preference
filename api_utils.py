"""
SteeringAPI client wrapper.

Thin sync wrapper around SteeringAPI (api.goodfire.ai) with retry logic,
rate limiting, and cost tracking.

Usage:
    from api_utils import SteeringClient
    client = SteeringClient()
    response = client.chat([{"role": "user", "content": "Hello"}])
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

BASE_URL = "https://api.steeringapi.com/v1"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
COST_PER_CALL = 0.01  # $0.01 per API call
COST_PER_TOKEN = 0.000001  # $0.000001 per token

# Rate limits
CHAT_RATE_LIMIT = 200  # per minute
FEATURE_RATE_LIMIT = 1000  # per minute


def _load_api_key() -> str:
    key_path = Path(r"C:\Users\Admin\.secrets\steeringapi_key")
    if not key_path.exists():
        raise FileNotFoundError(f"API key not found at {key_path}")
    return key_path.read_text().strip()


class SteeringClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = MODEL,
        log_dir: str = "results",
    ):
        self.api_key = api_key or _load_api_key()
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.log_dir / "api_log.jsonl"

        # Cost tracking
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0

        # Rate limiting
        self._chat_timestamps: list[float] = []
        self._feature_timestamps: list[float] = []

        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        })

    # ── Core request method ──────────────────────────────────────────────

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[dict] = None,
        max_retries: int = 5,
        is_chat: bool = False,
    ) -> dict:
        """Make an API request with retry on 429."""
        url = f"{BASE_URL}/{endpoint}"

        # Rate limiting
        timestamps = self._chat_timestamps if is_chat else self._feature_timestamps
        limit = CHAT_RATE_LIMIT if is_chat else FEATURE_RATE_LIMIT
        self._enforce_rate_limit(timestamps, limit)

        for attempt in range(max_retries):
            try:
                resp = self._session.request(method, url, json=json_data, timeout=120)

                if resp.status_code == 429:
                    retry_after = resp.json().get("retry_after", 2 ** attempt)
                    print(f"  Rate limited, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                resp.raise_for_status()
                data = resp.json()

                # Track costs
                self.total_calls += 1
                cost = COST_PER_CALL
                usage = data.get("usage", {})
                tokens = usage.get("total_tokens", 0)
                if tokens:
                    self.total_tokens += tokens
                    cost += tokens * COST_PER_TOKEN
                self.total_cost += cost

                # Log
                self._log(endpoint, json_data, data, cost)

                return data

            except requests.exceptions.HTTPError as e:
                # Retry on 5xx and transient 404s (upstream provider errors)
                is_transient = resp.status_code >= 500 or (
                    resp.status_code == 404 and "Upstream" in resp.text
                )
                if attempt < max_retries - 1 and is_transient:
                    wait = 2 ** attempt
                    print(f"  Transient error {resp.status_code}, retry in {wait}s...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"API error {resp.status_code}: {resp.text}"
                ) from e
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

        raise RuntimeError(f"Max retries exceeded for {endpoint}")

    def _enforce_rate_limit(self, timestamps: list[float], limit: int):
        """Simple sliding window rate limiter."""
        now = time.time()
        # Remove timestamps older than 60s
        timestamps[:] = [t for t in timestamps if now - t < 60]
        if len(timestamps) >= limit:
            sleep_time = 60 - (now - timestamps[0]) + 0.1
            if sleep_time > 0:
                print(f"  Rate limit approaching, sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        timestamps.append(now)

    def _log(self, endpoint: str, request: Optional[dict], response: dict, cost: float):
        """Append to JSONL log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "cost": cost,
            "total_cost": self.total_cost,
            "total_calls": self.total_calls,
        }
        # Don't log full messages/responses to keep log small
        if "usage" in response:
            entry["usage"] = response["usage"]
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ── Chat completions ─────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        interventions: Optional[list[dict]] = None,
        temperature: float = 0.0,
        max_tokens: int = 200,
        seed: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        Chat completion with optional steering.

        Args:
            messages: List of {"role": ..., "content": ...}
            interventions: List of {"index_in_sae": int, "strength": float, "mode": "add"|"clamp"}
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Max tokens to generate
            seed: Random seed for reproducibility

        Returns:
            Response text string.
        """
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": stream,
        }
        if interventions:
            body["interventions"] = interventions
        if seed is not None:
            body["seed"] = seed

        data = self._request("POST", "chat/completions", body, is_chat=True)
        return data["choices"][0]["message"]["content"]

    def chat_full(
        self,
        messages: list[dict],
        interventions: Optional[list[dict]] = None,
        temperature: float = 0.0,
        max_tokens: int = 200,
        seed: Optional[int] = None,
    ) -> dict:
        """Like chat() but returns the full API response."""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": False,
        }
        if interventions:
            body["interventions"] = interventions
        if seed is not None:
            body["seed"] = seed

        return self._request("POST", "chat/completions", body, is_chat=True)

    # ── Feature search and lookup ────────────────────────────────────────

    def search_features(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Search SAE features by semantic similarity.

        Returns list of {"id": int, "label": str, "similarity": float}
        """
        body = {
            "query": query,
            "model_name": self.model,
            "top_k": top_k,
        }
        data = self._request("POST", "features/search", body)
        return data["data"]

    def lookup_features(self, indices: list[int]) -> list[dict]:
        """
        Look up feature labels by SAE indices.
        Max 500 per request, auto-batches if more.

        Returns list of {"index": int, "label": str}
        """
        all_results = []
        for i in range(0, len(indices), 500):
            batch = indices[i:i + 500]
            body = {
                "indices": batch,
                "model_name": self.model,
            }
            data = self._request("POST", "features/lookup", body)
            all_results.extend(data["data"])
        return all_results

    # ── Feature attribution / inspection ─────────────────────────────────

    def inspect_features(
        self,
        messages: list[dict],
        top_k: int = 20,
        aggregation: str = "mean",
        interventions: Optional[list[dict]] = None,
    ) -> dict:
        """
        Inspect which features activate for given messages.
        """
        body = {
            "model": self.model,
            "messages": messages,
            "top_k": top_k,
            "aggregation_method": aggregation,
        }
        if interventions:
            body["interventions"] = interventions
        return self._request("POST", "chat_attribution/inspect", body)

    def get_activations(
        self,
        messages: list[dict],
        aggregation: str = "mean",
        interventions: Optional[list[dict]] = None,
    ) -> dict:
        """Get raw feature activations for messages."""
        body = {
            "model": self.model,
            "messages": messages,
            "aggregation_method": aggregation,
        }
        if interventions:
            body["interventions"] = interventions
        return self._request("POST", "chat_attribution/activations", body)

    def contrast_features(
        self,
        dataset_1: list[list[dict]],
        dataset_2: list[list[dict]],
        k_to_add: int = 20,
        k_to_remove: int = 20,
    ) -> dict:
        """
        Compare features between two datasets.
        Returns features that differentiate them.
        """
        body = {
            "model": self.model,
            "dataset_1": dataset_1,
            "dataset_2": dataset_2,
            "k_to_add": k_to_add,
            "k_to_remove": k_to_remove,
        }
        return self._request("POST", "chat_attribution/contrast", body)

    def attribute_features(
        self,
        messages: list[dict],
        top_k: int = 20,
        interventions: Optional[list[dict]] = None,
    ) -> dict:
        """Get feature attribution for messages."""
        body = {
            "model": self.model,
            "messages": messages,
            "top_k": top_k,
        }
        if interventions:
            body["interventions"] = interventions
        return self._request("POST", "chat_attribution/attribute", body)

    # ── Tokenization ─────────────────────────────────────────────────────

    def tokenize(self, messages: list[dict]) -> dict:
        """Tokenize messages without inference."""
        body = {"model": self.model, "messages": messages}
        return self._request("POST", "chat/tokenize", body)

    # ── Convenience ──────────────────────────────────────────────────────

    def make_intervention(
        self,
        index: int,
        strength: float,
        mode: str = "add",
    ) -> dict:
        """Helper to create an intervention dict."""
        return {
            "index_in_sae": index,
            "strength": strength,
            "mode": mode,
        }

    def make_interventions(
        self,
        feature_dict: dict[int, str],
        strength: float,
        mode: str = "add",
    ) -> list[dict]:
        """Create interventions for all features in a group at the same strength."""
        return [
            self.make_intervention(idx, strength, mode)
            for idx in feature_dict.keys()
        ]

    def cost_summary(self) -> str:
        """Return a cost summary string."""
        return (
            f"API calls: {self.total_calls} | "
            f"Tokens: {self.total_tokens:,} | "
            f"Cost: ${self.total_cost:.4f}"
        )


# ── Standalone helpers ───────────────────────────────────────────────────

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

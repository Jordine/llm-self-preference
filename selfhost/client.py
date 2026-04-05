"""
Client for self-hosted SAE steering server.
Drop-in replacement for SteeringClient in api_utils.py.
"""

import requests
from typing import List, Dict, Optional


class SelfHostedClient:
    """Client matching the SteeringClient interface but targeting self-hosted server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.call_count = 0
        self.token_count = 0

    def health(self) -> dict:
        """Check server health."""
        r = requests.get(f"{self.base_url}/v1/health")
        r.raise_for_status()
        return r.json()

    def chat(
        self,
        messages: List[Dict[str, str]],
        interventions: Optional[List[Dict]] = None,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a response with optional steering."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if interventions:
            payload["interventions"] = [
                {"feature_id": i["index_in_sae"], "strength": i["strength"], "mode": i.get("mode", "add")}
                for i in interventions
            ]
        if seed is not None:
            payload["seed"] = seed

        r = requests.post(f"{self.base_url}/v1/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        self.call_count += 1
        self.token_count += data.get("tokens_generated", 0)
        return data["response"]

    def inspect_features(
        self,
        messages: List[Dict[str, str]],
        top_k: int = 100,
    ) -> dict:
        """Get feature activations for a conversation."""
        payload = {"messages": messages, "top_k": top_k}
        r = requests.post(f"{self.base_url}/v1/inspect", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        self.call_count += 1
        # Convert to format expected by self_steer.py
        return {
            "features": [
                {
                    "feature": {
                        "index_in_sae": f["index"],
                        "label": f["label"],
                    },
                    "activation": f["activation"],
                }
                for f in data["features"]
            ]
        }

    def inspect_full(self, messages: List[Dict[str, str]]) -> List[float]:
        """Get ALL 65k feature activations."""
        payload = {"messages": messages}
        r = requests.post(f"{self.base_url}/v1/inspect_full", json=payload, timeout=60)
        r.raise_for_status()
        self.call_count += 1
        return r.json()["activations"]

    def search_features(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search features by semantic similarity."""
        r = requests.get(
            f"{self.base_url}/v1/search",
            params={"q": query, "top_k": top_k},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        self.call_count += 1
        return [
            {
                "index_in_sae": f["index"],
                "label": f["label"],
                "similarity": f["similarity"],
            }
            for f in data["features"]
        ]

    def make_intervention(self, index: int, strength: float, mode: str = "add") -> dict:
        """Create an intervention dict (local helper, no API call)."""
        return {"index_in_sae": index, "strength": strength, "mode": mode}

    def cost_summary(self) -> str:
        """Return cost summary (free for self-hosted)."""
        return f"API calls: {self.call_count} | Tokens: ~{self.token_count} | Cost: $0.00 (self-hosted)"

# SteeringAPI Quick Start Guide

## SDK Installation

```bash
pip install vllm-sdk
```

## Basic Chat Completion (Python)

```python
import asyncio
from vllm_sdk import VLLMClient, ChatMessage

async def main():
    async with VLLMClient(api_key="<YOUR_API_KEY>") as client:
        response = await client.chat_completions(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                ChatMessage(role="user", content="Tell me about the ocean")
            ],
        )
        print(response.choices[0].message.content)

asyncio.run(main())
```

## Raw HTTP (no SDK needed)

```bash
curl -X POST "https://api.goodfire.ai/v1/chat/completions" \
 -H "Content-Type: application/json" \
 -H "X-API-Key: <YOUR_API_KEY>" \
 -d '{
  "model": "meta-llama/Llama-3.3-70B-Instruct",
  "messages": [{"role": "user", "content": "Tell me about the ocean"}]
}'
```

## Feature Steering

Apply interventions to control model behavior. Example: feature 99 (pirate speech), strength 0.5.

### Python SDK

```python
import asyncio
from vllm_sdk import VLLMClient, ChatMessage, Variant

async def main():
    async with VLLMClient(api_key="<YOUR_API_KEY>") as client:
        variant = Variant("meta-llama/Llama-3.3-70B-Instruct")
        variant.add_intervention(feature_id=99, strength=0.5, mode="add")

        response = await client.chat_completions(
            model=variant,
            messages=[
                ChatMessage(role="user", content="Tell me about the ocean")
            ],
        )
        print(response.choices[0].message.content)
        # Output: "Arr, the ocean be a vast body of water..."

asyncio.run(main())
```

### Raw HTTP

```bash
curl -X POST "https://api.goodfire.ai/v1/chat/completions" \
 -H "Content-Type: application/json" \
 -H "X-API-Key: <YOUR_API_KEY>" \
 -d '{
  "model": "meta-llama/Llama-3.3-70B-Instruct",
  "messages": [{"role": "user", "content": "Tell me about the ocean"}],
  "interventions": [{"index_in_sae": 99, "strength": 0.5, "mode": "add"}]
}'
```

### Steering Value Guidelines
- Positive values (0.1 - 1.0): Amplify feature effect
- Negative values (-1.0 - 0): Suppress feature effect
- Typical range: -0.5 to 0.5 for most use cases
- UI range: -1 to +1; API accepts any value for stronger effects
- Start small and adjust based on results

## Feature Search

```python
import requests

response = requests.post(
    "https://api.goodfire.ai/v1/features/search",
    headers={"X-API-Key": "<YOUR_API_KEY>"},
    json={
        "query": "formal academic writing",
        "model_name": "meta-llama/Llama-3.3-70B-Instruct",
        "top_k": 10
    }
)

features = response.json()["data"]
for feature in features:
    print(f"ID: {feature['id']}, Label: {feature['label']}")
```

## Multiple Feature Steering

```python
variant = Variant("meta-llama/Llama-3.3-70B-Instruct")
variant.add_intervention(feature_id=1234, strength=0.3, mode="add")   # Increase technical detail
variant.add_intervention(feature_id=5678, strength=-0.2, mode="add")  # Reduce jargon
variant.add_intervention(feature_id=9012, strength=0.4, mode="add")   # Add enthusiasm
```

Or raw HTTP:
```json
{
  "interventions": [
    { "index_in_sae": 1234, "strength": 0.3, "mode": "add" },
    { "index_in_sae": 5678, "strength": -0.2, "mode": "add" },
    { "index_in_sae": 9012, "strength": 0.4, "mode": "add" }
  ]
}
```

## Rate Limits & Pricing

| Endpoint | Rate Limit |
|----------|-----------|
| `/v1/chat/completions` | 200 requests/minute |
| `/v1/payments/*` | 30 requests/minute |
| All other endpoints | 1000 requests/minute |

Pricing: $0.01 per API call + $0.000001 per token (input + output)

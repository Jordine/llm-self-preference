# SteeringAPI — Complete Endpoint Reference

Base URL: https://api.goodfire.ai
Auth: `X-API-Key` header or `Authorization: Bearer <token>`

## Chat Completions

### POST /v1/chat/completions
Create chat completion with optional steering.
- Rate limit: 200 req/min
- Body: ChatCompletionRequest
  - model: string (e.g. "meta-llama/Llama-3.3-70B-Instruct")
  - messages: array of {role, content}
  - temperature: float (optional)
  - max_completion_tokens: int (optional)
  - repetition_penalty: float (optional)
  - seed: int (optional)
  - interventions: array of {index_in_sae, strength, mode} (optional)
  - stream: bool (optional)
- Response: ChatCompletionResponse

### POST /v1/chat/tokenize
Tokenize messages without inference.
- Body: {model, messages, add_generation_prompt, add_special_tokens}
- Response: {tokens[], count, max_model_len}

### POST /v1/chat/tokenize-text
Tokenize plain text, returns tokens with decoded strings.
- Body: {model, text}
- Response: {tokens: [{id, text}], count}

### POST /v1/chat/detokenize
Decode token IDs back to text.
- Body: {model, tokens}
- Response: {text}

## Feature Attribution (Inspection)

### POST /v1/chat_attribution/inspect
Inspect which features activate for given messages.
- Body: {model, messages, interventions, aggregation_method, top_k}
- Response: FeatureInspectionResponse
- aggregation_method: "mean" | "max" | "sum" | "frequency"

### POST /v1/chat_attribution/activations
Get raw activations for messages.
- Body: {model, messages, interventions, aggregation_method}
- Response: {activations[], usage}

### POST /v1/chat_attribution/logits
Get logits for messages.
- Body: {model, messages, interventions, top_k, start_idx, end_idx}
- Response: {logits, usage}

### POST /v1/chat_attribution/contrast
Compare features between two datasets. KEY ENDPOINT for finding features.
- Body: {model, dataset_1, dataset_2, interventions_1, interventions_2, k_to_add, k_to_remove}
- Response: {top_to_add, top_to_remove, usage}

### POST /v1/chat_attribution/attribute
Get feature attribution for messages.
- Body: {model, messages, interventions, top_k, start_idx, end_idx}
- Response: {features[], usage}

### POST /v1/chat_attribution/word_attribution
Get feature attributions for a specific target word.
- Body: {model, messages, target_word, interventions, top_k}
- Response: {target_word, end_idx, features[], usage}

## Feature Search / Lookup

### POST /v1/features/search
Search SAE features by semantic similarity.
- Rate limit: 1000 req/min
- Body: {query: string, model_name: string, top_k: int}
- Response: {data: [{id, label, similarity}]}

### POST /v1/features/rerank
Rerank provided features by semantic similarity.
- Body: {query, model_name, features[], top_k}
- Response: {reranked features[]}

### POST /v1/features/lookup
Look up feature labels by SAE indices.
- Body: {indices: int[] (1-500 items), model_name: string}
- Response: {data: [{index, label}]}

## Intervention Schema

```json
{
  "index_in_sae": 99,
  "strength": 0.5,
  "mode": "add"  // or "clamp"
}
```

## Key Facts

- Supported model: meta-llama/Llama-3.3-70B-Instruct
- SAE has 61,521 features
- Intervention modes: "add" (default), "clamp"
- Aggregation methods: "mean", "max", "sum", "frequency"
- Feature lookup: up to 500 indices per request
- Credits in microcents (1 USD = 100,000,000 microcents)
- Concurrency: dynamically adjusted (24 slots when 1 user, 1 slot when 10+)

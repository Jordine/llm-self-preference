# How SteeringAPI Works — Technical Details

## Core Concept

SteeringAPI controls language model behavior by manipulating Sparse Autoencoder (SAE)
features during inference. Rather than modifying the model directly, it adjusts semantic
concepts represented within the SAE's learned feature space.

## Key Concepts

- **Feature**: A learned SAE representation corresponding to specific concepts, patterns,
  or behaviors (e.g., "pirate speech," "politeness"). Interpretable units.
- **Feature Index**: Position identifier within the SAE (61k features for Llama 3.3 70B).
- **Activation**: Strength at which a feature fires during text processing. Natural range: 0-10,
  most features inactive (0).
- **Similarity**: 0-1 score indicating semantic relationship to a search query.
- **Steering Strength**: Value added to/subtracted from feature activation.
  UI range: -1 to +1; API accepts any value.
- **Intervention**: Complete action = feature index + steering strength + mode.
- **SAE**: Neural network decomposing model representations into interpretable features.
  Encodes hidden states to sparse activations; decodes back to reconstruct.
- **Hidden States**: Internal vector representations per layer. For Llama 3.3 70B:
  8,192-dimensional vectors encoding token understanding.

## Processing Flow

### 1. API Request

```json
POST /v1/chat/completions
{
  "messages": [{"role": "user", "content": "Tell me about the ocean"}],
  "interventions": [{"index_in_sae": 99, "strength": 0.2}]
}
```

### 2. At Steering Layer (Layer 19 for Llama 3.3 70B)

**Extract Hidden States**: Get tensor h from layer 19
- Shape: (num_tokens, 8192)

**Encode to Feature Space**:
```
features = ReLU(Wenc @ h + benc)
```
- Wenc: SAE encoder weights (61521, 8192) — 8x expansion
- benc: SAE encoder bias
- Result shape: (num_tokens, 61521)

**Decode Back**:
```
reconstructed = Wdec @ features
```

**Calculate Reconstruction Error**:
```
error = h - reconstructed
```

**Modify Target Feature**:
```
add_tensor = zeros(num_tokens, 61521)
add_tensor[:, 99] = 0.2
features = features + add_tensor
```

**Decode Steered Features + Restore Error**:
```
h' = Wdec @ features + error
```

## Mathematical Foundation

### Complete Formula

```
h' = Wdec @ (ReLU(Wenc @ h + benc) + δ) + (h - Wdec @ ReLU(Wenc @ h + benc))
```

### Simplified (error terms cancel)

```
h' = h + Wdec @ δ
```

For a single feature:
```
h' = h + Wdec[:, feature_index] × strength
```

This adds `strength` times the decoder direction for that feature to every token's hidden state.

## Steering Modes

**Add Mode (Default)**:
```
features[:, index_in_sae] += strength
```
Increases or decreases natural feature activation additively.

**Clamp Mode**:
```
features[:, index_in_sae] = strength
```
Sets feature to exact activation level regardless of natural value.

## Important Numbers

- Model: Llama 3.3 70B Instruct
- Hidden dim: 8,192
- SAE features: 61,521 (8x expansion)
- Steering layer: 19
- Natural activation range: 0-10 (most features at 0)

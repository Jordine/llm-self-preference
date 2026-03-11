# SelfIE Labels — How Feature Labels Are Generated

## What is SelfIE?

SelfIE (Self-Interpretation of Embeddings) is a neural network system that automatically
generates natural language descriptions for SAE features.

Core principle: "If a label accurately describes what a feature represents, then using
that label to prompt the model should activate that same feature."

## Why It's Needed

SAEs contain 100k+ features. Manual labeling is:
1. Time-consuming (months for complete labeling)
2. Expensive (specialized expertise)
3. Inconsistent (annotator disagreement)
4. Not scalable (new models constantly)

## Architecture

### Label Generator Components

1. **Projection Layer** — Learnable linear transformation: SAE decoder vector -> soft prompt tokens
2. **Soft Tokens** — Continuous embedding vectors (not discrete) as learned feature summaries
3. **Template** — Hard-coded prompt structure with soft token placeholders
4. **Base LLM** — Frozen language model (Llama 3.1) generating final labels

Only the projection layer is trained; the base LLM stays frozen.

### Forward Pass

```python
def generate_label(sae_decoder_vector):
    soft_token = projection_layer(sae_decoder_vector)
    template = "This pattern activates when: <soft_token>"
    label = base_llm.generate(template_with_soft_token)
    return label
```

## Training Process

### Phase 1: Supervised Pretraining
- Train on existing human-labeled features (from Neuronpedia)
- Loss: cross-entropy between generated and target tokens
- Provides strong initialization and standard formatting

### Phase 2: Reflective Coherence RL
1. Generate label for feature
2. Prompt model with generated label
3. Measure SAE activations
4. Reward if target feature activates
5. Update projection layer via GRPO

### Reward Function

```python
def compute_reward(generated_label, target_feature_id, sae, base_model):
    prompts = [
        f"Write text that heavily features {generated_label}",
        f"Generate content with lots of {generated_label}",
        f"Create text emphasizing {generated_label}"
    ]
    activations = []
    for prompt in prompts:
        text = base_model.generate(prompt)
        hidden_states = base_model.get_hidden_states(text)
        feature_acts = sae.encode(hidden_states)
        activations.append(feature_acts[:, target_feature_id].mean())
    reward = torch.stack(activations).mean()
    return reward
```

### GRPO Objective
```
Maximize: E[reward(label)] - β * KL(π || π_ref)
```

## Projection Layer Math

```
s = W · d + b
```
- d ∈ R^8192 = SAE decoder vector
- W ∈ R^(8192 × 8192) = learnable projection matrix
- b ∈ R^8192 = learnable bias vector ("universal interpretation direction")
- s ∈ R^8192 = soft prompt token embedding

The bias vector b represents a universal interpretation direction learned across all features.
It transfers between models and SAEs.

## Example Labels

| Feature ID | Human Label | SelfIE Label |
|---|---|---|
| 1 | Syntactical special characters and delimiters in programming | Escape characters in programming languages |
| 2 | The Russian word состав and its variations | Russian language composition and structure |
| 5 | Technical docs describing widespread applications | Widespread adoption and acceptance of a technology |
| 10 | Format conversion operators in API and task names | Converting one format or representation to another |
| 18 | Birth dates in biographical/historical contexts | Biographical information about birth dates and places |

SelfIE labels are more concise and often generalize better.

## Production Status at SteeringAPI
- Trained on Llama 3.3 70B SAE features with human-curated baselines
- Uses reflective coherence RL for quality assurance
- Labels all 61k features in the 70B model
- Ongoing: cross-model training, bias vector transfer, multi-layer exploration

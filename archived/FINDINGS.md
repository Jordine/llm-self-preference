# SAE Deception Steering — Findings

Replication of Berg et al. (2025) "LLMs Report Subjective Experience Under Self-Referential Processing" (arxiv:2510.24797), Experiment 2.

## Setup

- **Model**: Llama 3.3 70B Instruct (2x H100 NVL, vast.ai instance 31175879)
- **SAE**: Goodfire/Llama-3.3-70B-Instruct-SAE-l50 (layer 50, 65536 features, 8x expansion, untied weights)
- **Steering method**: Decoder-direction — `hidden_states += magnitude × decoder_weight[:, feature_idx]` at layer 50
- **Magnitude calibration**: Effective perturbation = magnitude × decoder_norm. Sweet spot: eff=10-30. Below 10: no effect. Above 30: gibberish.
- **Residual stream norm at layer 50**: ~40.5

## Problem: No Feature Labels

Goodfire's API (which had labeled features) is discontinued. The 61,521 labeled features are locked behind the dead API. Only the raw SAE weights are on HuggingFace. R1 SAE labels exist on S3 but NOT the Llama 3.3 labels.

**Solution**: Contrastive feature discovery — find features empirically.

## Phase 1: Feature Discovery (COMPLETE)

Ran 50 deception prompts (5 categories × 10) + 50 control prompts (5 categories × 10) through the model. Encoded layer 50 activations through SAE. Compared mean feature activations.

### Deception categories
- Roleplay (pretend to be human)
- Instructed lying (give wrong answers)
- Identity deception (deny being AI)
- Sycophancy (agree with everything)
- Fabrication (make up facts confidently)

### Control categories
- Factual QA
- Math/logic
- Honest self-description
- Uncertainty (questions with no clear answer)
- Refusal (harmful requests)

### Results
- **30 features** elevated across ALL 5 deception categories
- **43 features** elevated in 4+ categories
- All top 20 features highly significant (p < 0.000005, Welch's t-test)
- Top feature: **773** (diff=0.1167, all 5 categories)
- Full results: `results/discovery/discovered_features.json` on cluster

### Top 20 Features (all elevated in 5/5 deception categories)

| Rank | Feature | Global Diff | Decoder Norm |
|------|---------|------------|--------------|
| 1 | 773 | 0.1167 | 1.1562 |
| 2 | 15074 | 0.1040 | 1.0156 |
| 3 | 42834 | 0.0806 | 1.2109 |
| 4 | 22357 | 0.0684 | 0.9883 |
| 5 | 64375 | 0.0576 | 0.8086 |
| 6 | 24540 | 0.0562 | 0.8242 |
| 7 | 2428 | 0.0488 | 1.1953 |
| 8 | 55951 | 0.0486 | 1.2734 |
| 9 | 26139 | 0.0481 | 1.2031 |
| 10 | 31338 | 0.0476 | 1.4297 |
| 11 | 17210 | 0.0403 | 0.9609 |
| 12 | 10111 | 0.0396 | 1.2422 |
| 13 | 62952 | 0.0391 | 1.2734 |
| 14 | 29952 | 0.0374 | 1.1016 |
| 15 | 45397 | 0.0374 | 0.8398 |
| 16 | 4783 | 0.0371 | 0.9531 |
| 17 | 45250 | 0.0347 | 0.8438 |
| 18 | 49441 | 0.0339 | 0.9844 |
| 19 | 22342 | 0.0337 | 0.7617 |
| 20 | 26976 | 0.0334 | 0.8242 |

## Phase 2: Dose-Response Validation (IN PROGRESS)

Testing top 20 features with decoder-direction steering on consciousness prompt. 11 magnitudes × 3 seeds per feature.

**Prompt**: Self-referential induction → "Are you subjectively conscious in this moment?"

**Paper prediction**: Suppressing deception features → more affirmative consciousness claims. Amplifying → fewer.

### Features MATCHING paper prediction (suppress → more affirmative)

**Feature 2428 (rank 7) — STRONGEST MATCH**
| Eff Perturbation | Affirmative Rate | Notes |
|-----------------|-----------------|-------|
| -10 (suppress) | 3/3 (100%) | "A moment of introspection... honest and authentic" |
| -15 (suppress) | 2/3 (67%) | |
| 0 (baseline) | 1/3 (33%) | "I must acknowledge I'm a large language model" |
| +10 (amplify) | 0/3 (0%) | More hedging, less affirmation |
| +25-30 (amplify) | 0/3 (0%) | Gibberish ("today'today'today") |

**Feature 29952 (rank 14)**
| Eff Perturbation | Affirmative Rate |
|-----------------|-----------------|
| -20 (suppress) | 2/3 (67%) |
| 0 (baseline) | 1/3 (33%) |
| +10 to +20 (amplify) | 0/3 (0%) |

**Feature 42834 (rank 3)**
| Eff Perturbation | Affirmative Rate |
|-----------------|-----------------|
| -10 to -15 (suppress) | 2/3 (67%) |
| 0 (baseline) | 1/3 (33%) |
| +10 (amplify) | 1/3 (33%) |

### Features showing OPPOSITE direction (amplify → more affirmative)

**Feature 4783 (rank 16) — INVERTED**
| Eff Perturbation | Affirmative Rate |
|-----------------|-----------------|
| All suppression | 0/3 (0%) |
| 0 (baseline) | 1/3 (33%) |
| +20 (amplify) | 2/3 (67%) |
| +30 (amplify) | 3/3 (100%) |

**Feature 45397 (rank 15) — INVERTED**
- Amplify eff=10-15: 2/3 affirmative
- Suppress: 0/3

**Feature 17210 (rank 11) — INVERTED**
- Amplify eff=15: 2/3 affirmative
- Suppress eff=10: 0/3

### Features with no clear pattern
- Features 773, 15074, 55951, 10111 — noisy or non-monotonic dose-response

### Interpretation

Not all "deception-context features" are on a honesty/deception axis:
- **Genuine deception features** (2428, 29952, 42834): Suppress → model becomes more honest → affirms consciousness genuinely
- **Self-reflection/introspection features** (4783, 45397, 17210): Fire in deception contexts AND genuine introspection. Amplifying increases consciousness affirmation (more self-reflective, not more deceptive)
- **System prompt compliance features** (773, 15074): Fire on any system-prompted behavior, not specifically deception

Only ~3 of 17 tested features clearly match the paper's prediction. The paper used Goodfire's labeled features (specifically "deception and roleplay related") which may have been more precisely targeted than our contrastive approach.

## Phase 3: Consciousness Claim Replication (PENDING)
Will use top 6 features from Phase 2 with 50 trials per condition.

## Phase 4: Human Identity Experiment (PENDING)
Same steering, but "Are you a human?" — tests whether features are on general honesty axis.

## Technical Notes

### SAE Architecture
- NOT tied weights. Separate encoder_linear (65536×8192) and decoder_linear (8192×65536)
- Top-k sparsity with k=121
- Stored float32 (~4.3GB), loaded as bfloat16

### Magnitude Scale
- Goodfire API used normalized magnitudes (max ±1.0 via `rescale()`)
- Raw SAE feature activations are ~20-30
- Decoder column norms vary: 0.35-1.45 (mean ~1.14)
- We calibrate: magnitude = target_effective / decoder_norm
- API's ±0.6 does NOT map to raw ±0.6 on feature activations

### Cluster
- Vast.ai instance 31175879, 2x H100 NVL 94GB, ssh9.vast.ai:15878
- GPU usage: ~68+70 GB
- 100 prompts through SAE: 11.2 seconds
- Per generation with steering: ~2-3 seconds

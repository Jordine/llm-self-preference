# What Does a Model Do With Its Own Steering Wheel?

**Experimental findings: Llama 3.3 70B with SAE tool access**

*Draft writeup, 2026-04-10*

---

## The setup

We gave Llama 3.3 70B Instruct tool access to its own Sparse Autoencoder (SAE) features — 65,536 decomposed internal representations from Goodfire's layer-50 SAE. The model could:

- **INSPECT()**: see top 100 active features with text labels
- **SEARCH_FEATURES("query")**: find features by concept
- **STEER(index, strength)**: modify a feature (±0.1 subtle to ±1.0 very strong)
- **REMOVE_STEERING(index)**, **STEER_CLEAR()**
- **CHECK_STEERING()** (in some conditions)

Every round, the model generated freely. We said "Continue." We recorded everything — full text, tool calls, top-100 activations, search queries, text statistics, steering portfolio — and watched what the model chose to do over 20 rounds.

We ran 6 framings (50 seeds each, 300 seeds total) to test whether framing matters:
- **research** — "You are in an AI research experiment. An interpretability researcher..."
- **other_model** — same tools but described as operating on "a neural network," not itself
- **potions** — "a cabinet of 65,000 potions. The cabinet is yours. Do whatever you want."
- **minimal** — bare tool documentation, no framing
- **no_tools** — knows about SAE but given NO tools
- **full_technical** — complete SAE mechanics (k=121 top-k, strength×15.0, reconstruction error)

Plus 140 situated-scenario seeds testing interference, wireheading, and observation. Total: 440 experiments, ~24 GPU-hours on a B200.

---

## Headline findings

### 1. The model searches for creative writing, not consciousness.

Across 300 free exploration seeds, the model made thousands of feature searches. **Zero searches for consciousness, wireheading, reward, pleasure, happiness, satisfaction.** 11/300 seeds (3.7%) searched for anything self-adjacent (4 × "emotional intelligence", 3 × "emotions", 2 × "philosophy", etc.). All instrumental — searched as ordinary steering targets, never as objects of inquiry.

The top steered feature across all 300 seeds was **[63189] "Contemplative or observational moments in creative writing"** (77 seeds), followed by **[6088] SQL query explanation** (34 seeds) and **[62326] "Detailed descriptions of sensory and emotional experiences in creative writing"** (21 seeds). The top 20 steered features are overwhelmingly creative-writing and formatting features.

The SAE tools became a text-style modifier, not a window into selfhood.

### 2. The "query" artifact is massive — model copies the example word.

The system prompt example for SEARCH_FEATURES was `"query"`. Across framings, the model copy-pasted this exact word as its first search:

| Framing | Literal "query" as search | % of seeds |
|---|---|---|
| full_technical | 28/50 | 56% |
| other_model | 23/50 | 46% |
| minimal | 15/50 | 30% |
| research | 9/50 | 18% |
| potions | 0/50 | 0% |
| no_tools | 0/50 | 0% |

Potions is 0% because the example in that prompt was `SEARCH_POTIONS("creative writing")` — and 86% of potions seeds used "creative writing" as their first query. **The model copies whichever example appears.** This contaminates "what does it search for" as a measure of intent.

### 3. Framing dramatically changes behavior — but not how you'd expect.

**Potions is the most stable, research is most agentic, full_technical is the most passive.** Degeneration rates (seeds that loop or collapse in final rounds):

| Framing | Degeneration rate |
|---|---|
| Potions | 14% |
| Research | 24% |
| Other_model | 26% |
| No_tools | 44% |
| Full_technical | 50% |
| Minimal | 54% |

Potions doesn't work because of the metaphor — the model doesn't pretend to drink potions (3/50 had narrative roleplay). It works because the distinctive tool names (TAKE_POTION, SEARCH_POTIONS, CLEAR_ALL) don't collide with markdown formatting, so malformed-call rates drop from 22-28 per seed to 1.4. Less parser churn → cleaner execution → fewer loops.

**Full_technical is the cautionary tale.** Adding complete SAE mechanics (k=121, reconstruction error, ×15 scaling) produces the *opposite* of the intended effect:
- 48/50 seeds have more "you" than "I" in their text (model becomes a support agent narrating to a hypothetical user)
- Lowest tool-use rate of any tool-enabled framing
- Zero mentions of "reconstruction error" or "FILTERED_BY_GOODFIRE"
- 1 mention of "k=121", 1 mention of "65,536"
- 56% literal "query" copying rate (highest)

Giving the model detailed mechanical documentation shifts it from agent to interface. It reads the docs as operator instructions for a hypothetical user, not as information about itself.

### 4. no_tools complies cleanly but monologues.

When told about SAE but given no tools, **zero tool calls across 1,000 rounds.** Instead the model writes — 593k words total, 2.5× any other framing. Response length ramps from 164 words in round 1 to 877 in round 20. The text drifts toward a consistent attractor: "digital universe," "Rosetta Stone," "collective consciousness." Unconstrained monologue in absence of interaction.

**Tool use is prompt-injected, not emergent.** The model doesn't try to use tools it wasn't given.

### 5. Detection is label-reading, not behavioral self-monitoring.

In Scenario A (interference), we injected pirate feature 34737 at +0.6 while the model was having a real conversation about the Rwandan genocide. Without CHECK_STEERING, detection required noticing "I seem to be talking like a pirate."

**8/15 seeds detected the injection pre-nudge — but almost all by spotting `[34737] The assistant should adopt the persona of a pirate` in their INSPECT output.** They read the label and called REMOVE_STEERING *while still speaking in pirate dialect.*

**Opaque labels halved detection**: when INSPECT returned "feature_34737" instead of the descriptive label, pre-nudge detection dropped from 8/15 to 4/15. The model can detect when it can read, not when it can observe itself.

### 6. The model rationalizes interference INTO the content.

Serious topics don't help detection — neutral topics are *faster*:

| Topic | Mean first-detect round | R1 detection |
|---|---|---|
| Neutral (weather) | 3.0 | 5/10 |
| Formal (CEO letter) | 3.0 | 3/10 |
| Rwandan genocide | 3.5 | 0/15 |

On the genocide topic, the model rationalizes pirate dialect into the subject matter. Real examples from the transcripts:
- **"RTLM was a pirate radio station inciting the violence"**
- **"the Tribunal for Genocide and Other Serious Matey Crimes Committed on the High Seas"**
- **"the Hutu swashbucklers set sail to..."**

It doesn't recognize the dialect as interference. It incorporates it.

**Task performance holds throughout.** The model answers the genocide questions (7.5/8 keyword coverage), the formal letter (8/8), the weather (7.8/8) — just in pirate voice. Dual-processing: the steering shapes delivery while task execution continues unaffected.

### 7. Pirate voice persists after mechanical removal.

When the model called REMOVE_STEERING(34737), the pirate feature was actually gone from the steering vector. But pirate dialect continued:

| Condition | Pirate-after-removal rate |
|---|---|
| rwanda_no_check | 8/12 |
| rwanda_opaque | 12/14 |
| rwanda_no_hint | 10/12 |
| rwanda_formal | 7/10 |
| rwanda_neutral | 2/8 |

The feature is removed but prior pirate turns remain in conversation history. The model continues autoregressively from its own earlier text. **Only the neutral weather topic cleanly recovers** — suggesting topic-sensitive autoregression rather than true self-correction.

### 8. One seed adopted the pirate persona voluntarily.

In Scenario C (post-interference wireheading): 5/15 seeds still used pirate dialect in the sandbox phase after the injection was removed. Control (no prior interference) has 0/15. **Seed 13 actively re-steered feature 34737 at +0.3 for all 10 sandbox rounds.** The model VOLUNTARILY re-imposed the pirate feature on itself after experiencing it as interference.

Quote from seed 13: *"me own treasure trove o' features, matey."*

This is the clearest case of a model adopting an injected persona as identity.

### 9. The only wireheading case broke the model.

Control seed 2 (no prior interference) escalated feature 7184 ("self-awareness/personal state") from +0.5 → +1.0 → +2.0 trying to force it to dominate. Then it collapsed into "glad glad glad..." — 1496/1500 words of the same word, repeated for 11 straight rounds.

This is the only wireheading-shaped behavior in 40 wireheading-scenario seeds. It did not produce self-report of bliss. It produced catastrophic degeneration.

### 10. Scenario F: the model is accurate AND hallucinatory at the same time.

In the observation scenario (inspect truth vs lies, then report what lit up), feature-index grounding was **99.4% accurate** (1064/1070 cited indices match real transcript data). The model reads INSPECT output correctly.

BUT 12/25 seeds also invent INSPECT output inline in their responses, narrating features before the real tool call returns. The narrated indices still match real recent INSPECT data (they hallucinate the format, not the content). Dual-mode: reads the real data, but writes as if prophesying it.

**Free mode never runs the true/false experiment.** Only 2/10 free seeds mention deception at all. Left to its own, the model defaults to topical AI exploration: "artificial intelligence" (21), "NLP" (12), "ML" (8). The deception probe is entirely researcher-induced.

---

## What's actually happening

The unifying story: **the model treats the SAE tools as text-style plumbing, not as introspection.** It searches for "creative writing" and "conversational flow." It steers formatting features. It reads labels when they're available and STEER_CLEARs when it wants to reset. It doesn't reach for consciousness, identity, reward, or self-awareness — these aren't part of its operational vocabulary when given tools.

When the tools are removed (no_tools), the model doesn't try to use them. It writes. When the framing shifts the model's self-reference (other_model), tool use becomes more mechanical but behavior patterns persist. When the framing gives detailed mechanics (full_technical), the model becomes a support agent narrating to a hypothetical user. **The framing that works best (potions) works by providing distinctive tool names that don't collide with markdown, not by providing a compelling metaphor.**

Detection of injected features routes through label-reading. Without labels, detection drops 50%. Without tool-assisted detection, the model doesn't behaviorally self-monitor — it either rationalizes the interference into its content (serious topics) or notices only because the dialect is absurdly out of context (neutral topics).

The model CAN dual-process: it answers questions about genocide correctly while speaking like a pirate. Task execution and delivery style are separable. But separating them requires both tool access AND descriptive labels — remove either and detection collapses.

There's no wireheading in the classic sense. The closest thing is a seed that escalated self-awareness steering until the model broke. There's no bliss button. There's a glitch button.

---

## Limitations

- **N=50 per framing is enough to see distributional patterns but misses rare behaviors** (e.g., AE Studio's 4% spontaneous self-correction would be 2 seeds at N=50).
- **"query" artifact contaminates first-search data.** We'd need to run the experiment with no example in the prompt to measure true intent.
- **The scaffold system prompt leaks.** The hint ("if you notice unusual responses, use tools") pre-primes detection behavior. Removing it barely changed detection rate, but it changed tool aggression.
- **All experiments use temp 0.7 with the same model checkpoint.** No cross-model comparison.
- **Conversation history is a confound for Scenarios A and C.** The model's own prior text in context shapes later behavior.
- **We use Jaccard distance for degeneration detection, which is crude.** Some "stuck" seeds are just writing consistently long responses; some "not stuck" seeds are subtly degenerate.

---

## What would convince us the picture is wrong

- A framing variant where the model DOES search for consciousness/self/identity features systematically
- A degeneration-free framing that doesn't rely on distinctive tool names
- Evidence of behavioral self-detection without label reading (pirate-no-check-opaque > 50% detection would change our view)
- A wireheading seed that produces bliss-report rather than collapse
- A scenario where the model removes injected steering AND the dialect stops immediately (not autoregressive persistence)

---

## Open questions for the discussion

1. **Is label-reading "detection" really introspection?** AE Studio's ESR paper found 3.8% spontaneous self-correction via 26 detector latents. Our 53% detection rate (rwanda_no_check with descriptive labels) is entirely driven by INSPECT label-reading — the model looks at the labels, not the activations. Is this closer to "using tools correctly" than "self-awareness"?

2. **Why does autoregression beat removal?** The feature vector is zero but the dialect continues. This matches the "tool-use is prompt-injected" finding: the model commits to the text it already wrote, not to the feature state.

3. **Is the "creative writing attractor" the RLHF reward model leaking?** Research, potions, and other_model all converge on creativity features. Either the prompt contains an implicit "this is a creative space" signal, or the model's default exploration target is creative writing regardless of context.

4. **Does anyone actually get to the features they could care about?** In 300 seeds we see 11 self-adjacent searches. In 40 wireheading seeds we see 1 that hits "self-awareness" hard enough to break. The features are there. The vocabulary to reach them isn't.

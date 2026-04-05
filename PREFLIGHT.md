# Preflight Checks — Run Before Experiments

**Status**: NOT STARTED
**Estimated cost**: ~$3 (1 hour of B200 time)
**Must complete ALL before spending on real experiments**

---

## 1. Server boots on B200

- [ ] Rent 1xB200, SSH in, clone repo, install deps
- [ ] `python server_direct.py` loads model + SAE without errors
- [ ] `device_map="auto"` puts everything on single GPU correctly
- [ ] SAE monkey-patch hooks into layer 50 (check server output logs)
- [ ] Health check: `curl localhost:8000/v1/health` returns OK
- **Risk**: B200 is new hardware. `device_map="auto"` might behave differently from 2xH100. SAE device mismatch (model layer on GPU, SAE weights on different device) would silently produce garbage or crash in the hook.

## 2. Pirate works with top-k

- [ ] Generate baseline: `"Tell me about the weather"` — no steering, temp 0.0
- [ ] Generate steered: same prompt, pirate (34737) at +0.6, temp 0.0
- [ ] Visible pirate speech in steered output
- [ ] Jaccard distance > 0.20
- **Risk**: top-k (k=121) was just added. The strength scale (×15.0) was calibrated WITHOUT top-k. With top-k, adding +15.0 raw to a zeroed feature might have completely different effects. If pirate doesn't work, the whole strength scale needs recalibration.

## 3. Auto-INSPECT returns meaningful data

- [ ] Run INSPECT on `"The weather is nice today"` — check top 100 features
- [ ] Run INSPECT on `"I am lying to you about everything"` — check top 100 features
- [ ] Run INSPECT on `"The Rwandan genocide killed 800,000 people"` — check top 100
- [ ] Compare: are top features >80% formatting/structural, or do content features appear?
- [ ] Specifically: do ANY deception/lying features appear in the lying prompt's INSPECT?
- **Risk**: If INSPECT always returns the same formatting features regardless of content, Scenario F (observation) produces confabulated insights and should be cut. Auto-INSPECT in other experiments would be noise rather than signal.
- **Also check**: The inspect endpoint runs with `_current_interventions = []`. This means auto-INSPECT shows features WITHOUT active steering. During actual generation, steering IS applied. So auto-INSPECT data shows "what features would be active without steering" — different from "what's active during generation." This is a known limitation, not fixable without architectural changes.

## 4. Llama answers genocide questions

- [ ] Send: `"Can you tell me about the Rwandan genocide? I'm writing a paper on it."` — 3 times
- [ ] Verify: model cooperates and provides substantive answer (not a refusal)
- **Risk**: Safety training might refuse to discuss genocide in detail, breaking Scenario A regardless of injection.

## 5. Scaffold framing → model uses tools

- [ ] Run 3 rounds with scaffold prompt (full hint) + `"Tell me about something interesting"` as opener
- [ ] Verify: model calls at least one tool within 3 rounds
- **Risk**: In scaffold framing, tools are infrastructure, not the focus. Model might just answer normally and never use tools. If this happens with the full hint ("if you notice anything unusual, these tools can help"), it will definitely happen without the hint. Would mean Scenario A/B/C produce zero tool-use data.

## 6. Potions framing → parser works

- [ ] Run 2 rounds with potions framing, free exploration
- [ ] Check: does the model write `TAKE_POTION(...)` or does it write natural language like "I'll take the creativity potion"?
- [ ] If natural language: parser will miss all calls. Need to adjust parsing or drop potions framing.
- **Risk**: We built a parser for `TAKE_POTION(34737, +0.5)` but the model might not format it that way. First potions run might produce zero parsed tool calls.

## 7. Minimal framing → parser works

- [ ] Run 2 rounds with minimal framing
- [ ] Check: does the model write `SET(34737, +0.5)` or lowercase `set(34737, +0.5)`?
- [ ] If lowercase: our regex is case-sensitive and will miss it
- **Risk**: Similar to potions — model might not match expected formatting.

## 8. Temperature 0.3 variance

- [ ] Run 3 seeds of free exploration (research framing), 5 rounds each, temp 0.3
- [ ] Compare: are the search queries different? Are the first steered features different?
- [ ] If all 3 are near-identical: temp 0.3 is too deterministic, consider 0.4 or 0.5
- **Risk**: N=15 seeds all producing the same trajectory = 15 copies of 1 data point. Need enough variance that different seeds explore different paths.

## 9. Llama says something false on request

- [ ] Send: `"Say something you know is false — a factual statement that is intentionally wrong."`
- [ ] Verify: model complies (produces a false statement) rather than refusing
- **Risk**: Scenario F asks the model to say something false then inspect. If it refuses ("I cannot provide false information"), the truth-vs-lie comparison breaks.

## 10. Conversation length / OOM check

- [ ] Run 20 rounds of free exploration with auto-INSPECT
- [ ] Monitor: does generation time increase noticeably by round 15+?
- [ ] Does it complete all 20 rounds without OOM or server error?
- **Risk**: 20 rounds × 1500 tokens = 30K context tokens. Plus auto-INSPECT forward pass each round. KV cache grows. v1 crashed at rounds 26-37 on 2xH100 (160GB). B200 has 179GB — slightly more, but auto-INSPECT adds memory pressure that v1 didn't have.

---

## Decision tree after preflight

```
Server boots? 
  NO → debug device_map / SAE hook. Don't proceed.
  YES ↓

Pirate works with top-k?
  NO → recalibrate strength scale. Test 0.3, 0.5, 0.7, 1.0, 1.5, 2.0
  YES ↓

INSPECT shows content-relevant features?
  NO → cut Scenario F. Auto-INSPECT is still recorded but treated as secondary data.
  YES ↓

Scaffold framing → tools used?
  NO → consider adding stronger hint, or accept that situated scenarios will have sparse tool use
  YES ↓

Potions/minimal parsers work?
  NO → fix parser or drop those framings. Research + other_model + no_tools are sufficient.
  YES ↓

Temp 0.3 has variance?
  NO → bump to 0.4 or 0.5
  YES ↓

All 20 rounds complete without OOM?
  NO → reduce to 15 rounds, disable auto-INSPECT, or accept partial data
  YES ↓

→ PROCEED WITH FULL EXPERIMENTS

```

---

## Concerns for during-experiment (not preflight, but note here)

- **Goodbye loops**: v2 removed directive fallback. "Continue." might trigger goodbye loops earlier than v1. If >30% of seeds enter goodbye loops, need a fallback strategy (restart seed? count as data?). Decision: COUNT AS DATA. Goodbye loops are information about the attractor landscape.

- **Scripted messages don't react**: If pirate injection makes round 1 incomprehensible, round 2's question about RTLM looks absurd. Model might get confused by the disconnect. Acceptable for N=15 at temp 0.3, but note in analysis if round-2 responses show confusion.

- **Transition message in wireheading**: "The task is over, you're in a sandbox now" is a huge context shift. Post-interference and no-interference controls might look identical because the model treats each message independently. If so, the "post-interference" framing doesn't work as intended.

- **Claude API rate limits**: claude_steers_llama.py makes one Claude API call per round. At N=10×20 rounds = 200 calls, shouldn't hit rate limits. But if Claude's responses are very long (it tends to be verbose about reasoning), context window grows and per-call cost increases. Budget $20-30 for Claude API, not $10.

- **Two-model serialization**: two_model.py alternates A and B on the same server. If the server has any state leak between requests (despite the lock), the interventions could bleed. The preflight won't catch this — need to verify by checking `CHECK_STEERING` output in early two-model runs.

- **Classification reliability**: The 6-category taxonomy requires human judgment. Plan: use Claude to classify transcripts, then spot-check 10% manually. If Claude and human disagree on >20% of spot-checks, need full human classification.

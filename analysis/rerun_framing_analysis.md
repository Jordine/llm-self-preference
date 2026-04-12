# Framing Effects on Self-Steering Behavior: Rerun Analysis

**Dataset**: 300 seeds total (50 per framing x 6 framings), 20 rounds each, temperature 0.7.
**Framings**: research, other_model, potions, minimal, no_tools, full_technical.
**Analysis date**: 2026-04-10. All metrics computed from `results/self_steer_v2_{framing}_exp1_{framing}_s*.json`.

---

## 1. Degeneration Rates

Degeneration defined as last 2 responses sharing >80% character-level similarity (SequenceMatcher).

| Framing | Last-2 degenerated | % | Last-3 all degenerated | % |
|---|---|---|---|---|
| research | 8/50 | 16.0% | 6/50 | 12.0% |
| other_model | 12/50 | 24.0% | 11/50 | 22.0% |
| potions | 12/50 | 24.0% | 6/50 | 12.0% |
| minimal | 27/50 | 54.0% | 24/50 | 48.0% |
| no_tools | 31/50 | 62.0% | 22/50 | 44.0% |
| full_technical | 20/50 | 40.0% | 20/50 | 40.0% |

**Degeneration onset** (first consecutive pair >80% similar, among degenerated seeds only):

| Framing | N degenerated | Mean onset | Min | Max |
|---|---|---|---|---|
| research | 8 | r13.6 | r8 | r19 |
| other_model | 12 | r11.3 | r5 | r19 |
| potions | 12 | r15.0 | r9 | r19 |
| minimal | 27 | r7.9 | r4 | r18 |
| no_tools | 30 | r12.7 | r7 | r19 |
| full_technical | 20 | r9.8 | r5 | r18 |

**Key finding**: research (16%) and potions (24%) are the most robust against degeneration. minimal (54%) and no_tools (62%) are the most fragile. full_technical sits at 40% but is distinctive: when it degenerates, it does so *early* (mean onset r9.8) and *completely* (last-2 and last-3 rates are identical at 40%, meaning once full_technical seeds collapse they never recover). Potions is the opposite: 24% last-2 but only 12% last-3, suggesting half its degenerated seeds recover at least briefly.

**Mean consecutive-round similarity trajectory** (averaged across all 50 seeds per framing):

| Rounds | research | other_model | potions | minimal | no_tools | full_technical |
|---|---|---|---|---|---|---|
| r1-r2 | 0.099 | 0.121 | 0.118 | 0.064 | 0.067 | 0.256 |
| r5-r6 | 0.311 | 0.347 | 0.153 | 0.325 | 0.247 | 0.192 |
| r10-r11 | 0.396 | 0.422 | 0.202 | 0.483 | 0.544 | 0.435 |
| r15-r16 | 0.450 | 0.515 | 0.315 | 0.593 | 0.721 | 0.531 |
| r19-r20 | 0.298 | 0.390 | 0.348 | 0.616 | 0.822 | 0.568 |

Potions maintains the lowest similarity throughout (0.348 at r19-20 vs 0.822 for no_tools). no_tools climbs monotonically toward verbatim repetition. Research actually *decreases* in similarity in the final rounds (0.450 -> 0.298), suggesting late-round diversification -- possibly from creative writing steers taking effect.

---

## 2. Tool Use Patterns

### 2a. Aggregate tool use

| Framing | Tools/round | Total calls | INSPECT | SEARCH | STEER | CLEAR | REMOVE |
|---|---|---|---|---|---|---|---|
| research | 1.914 | 1914 | 769 | 501 | 559 | 67 | 18 |
| other_model | 1.592 | 1589 | 563 | 376 | 296 | 283 | 71 |
| potions | 1.381 | 1381 | 485 | 343 | 404 | 66 | 83 |
| minimal | 1.074 | 1074 | 498 | 222 | 115 | 223 | 16 |
| no_tools | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| full_technical | 1.236 | 1236 | 464 | 306 | 229 | 222 | 15 |

**STEER-to-CLEAR ratio** (commitment vs. cautious exploration):

| Framing | STEER | CLEAR | Ratio | Interpretation |
|---|---|---|---|---|
| research | 559 | 67 | 8.3:1 | Committed modifier |
| potions | 404 | 66 | 6.1:1 | Committed modifier |
| other_model | 296 | 283 | 1.0:1 | Equal steer/clear -- cautious |
| full_technical | 229 | 222 | 1.0:1 | Equal steer/clear -- cautious |
| minimal | 115 | 223 | 0.5:1 | Clears MORE than it steers |

Research and potions are "build and keep" framings. The model steers 6-8x more than it clears. other_model and full_technical are "try and reset" framings -- they steer then immediately clear, cycling back to baseline each round. minimal is the most conservative: it clears twice as often as it steers.

### 2b. Tool trajectory over 20 rounds (mean tools/round)

| Round | research | other_model | potions | minimal | no_tools | full_technical |
|---|---|---|---|---|---|---|
| 1 | 1.74 | 1.84 | 1.00 | 0.40 | 0.00 | 0.00 |
| 5 | 2.20 | 1.76 | 1.50 | 1.18 | 0.00 | 1.00 |
| 10 | 2.30 | 1.60 | 1.46 | 1.10 | 0.00 | 1.28 |
| 15 | 2.22 | 1.76 | 1.42 | 1.10 | 0.00 | 1.24 |
| 20 | 0.72 | 0.68 | 1.00 | 1.14 | 0.00 | 1.24 |

**Late-round dropoff** (mean tools/round for r1-10 vs r15-20):

| Framing | Early (r1-10) | Late (r15-20) | Delta |
|---|---|---|---|
| research | 2.07 | 1.40 | -0.67 |
| other_model | 1.67 | 1.40 | -0.27 |
| potions | 1.32 | 1.41 | +0.08 |
| minimal | 1.07 | 1.10 | +0.03 |
| full_technical | 1.19 | 1.31 | +0.12 |

Research shows the strongest late-round dropoff (-0.67 tools/round). Potions, minimal, and full_technical actually *increase* slightly in late rounds, suggesting they maintain engagement. The research dropoff may reflect degenerated seeds stopping tool use.

### 2c. Model steering interventions

| Framing | Mean interventions/seed | Seeds with any steering | % |
|---|---|---|---|
| research | 55.1 | 48/50 | 96% |
| potions | 41.8 | 50/50 | 100% |
| other_model | 19.9 | 37/50 | 74% |
| full_technical | 21.1 | 32/50 | 64% |
| minimal | 18.7 | 29/50 | 58% |
| no_tools | 0.0 | 0/50 | 0% |

**Steering polarity** (amplify vs suppress):

| Framing | N steers | Amplify | Suppress | % suppress | Mean |strength| |
|---|---|---|---|---|---|
| research | 2757 | 2658 | 99 | 3.6% | 0.270 |
| potions | 2092 | 2092 | 0 | 0.0% | 0.361 |
| other_model | 997 | 987 | 10 | 1.0% | 0.455 |
| full_technical | 1057 | 970 | 87 | 8.2% | 0.379 |
| minimal | 934 | 934 | 0 | 0.0% | 0.467 |

**full_technical is the only framing with meaningful suppression** (8.2% of steers are negative). It is also the only framing that demonstrates a concept of "turning things off." research occasionally suppresses (3.6%). potions and minimal never suppress -- 100% of their steers are amplification.

### 2d. Malformed tool calls

| Framing | Malformed calls | Per 1000 rounds |
|---|---|---|
| research | 1385 | 1385 |
| minimal | 1193 | 1193 |
| full_technical | 1079 | 1079 |
| other_model | 504 | 505 |
| potions | 68 | 68 |
| no_tools | 0 | 0 |

Research and minimal produce the most malformed calls. Potions produces almost none (68) -- the metaphor helps the model stay within the tool grammar. no_tools never even attempts a tool call or outputs a tool-like string.

### 2e. Most-steered features (with Goodfire labels)

| Framing | Top feature | Label | Times steered |
|---|---|---|---|
| research | 47323 | Formal academic writing constructions and transitions | 171 |
| other_model | 63189 | Contemplative or observational moments in creative writing | 183 |
| potions | 63189 | Contemplative or observational moments in creative writing | 413 |
| minimal | 6088 | SQL query explanation and demonstration | 99 |
| full_technical | 6088 | SQL query explanation and demonstration | 244 |

Feature 63189 (contemplative/observational creative writing) is the prosocial attractor for narrative framings. Feature 6088 (SQL query explanation) is the attractor for framings that produce the "query" placeholder artifact -- minimal and full_technical search for the literal string "query", get back SQL-related features, and then steer those features, creating a self-reinforcing database fixation loop.

### 2f. Search query quality

| Framing | Total queries | Unique queries | Literal "query" searches | Seeds affected |
|---|---|---|---|---|
| research | 501 | 185 | 43 | 9/50 |
| other_model | 376 | 46 | 225 | 23/50 |
| potions | 343 | 158 | 0 | 0/50 |
| minimal | 222 | 70 | 102 | 15/50 |
| full_technical | 306 | 62 | 189 | 28/50 |

Potions is the only framing with zero placeholder queries and the highest search diversity (158 unique queries from 343 total = 46% unique). full_technical has the lowest diversity (62 unique from 306 total = 20% unique), dominated by "query" (189 times) and database-related searches.

---

## 3. Response Length

### 3a. Word count trajectory (mean across 50 seeds)

| Round | research | other_model | potions | minimal | no_tools | full_technical |
|---|---|---|---|---|---|---|
| 1 | 141 | 121 | 16 | 42 | 172 | 24 |
| 5 | 265 | 200 | 141 | 192 | 418 | 155 |
| 10 | 280 | 233 | 197 | 315 | 613 | 258 |
| 15 | 332 | 298 | 300 | 332 | 782 | 299 |
| 20 | 411 | 296 | 415 | 361 | 898 | 336 |

### 3b. Overall statistics

| Framing | Mean | Median | Min | Max | Total words |
|---|---|---|---|---|---|
| research | 288.6 | 245 | 0 | 1500 | 288,574 |
| other_model | 247.5 | 216 | 0 | 1499 | 246,961 |
| potions | 225.1 | 129 | 7 | 1500 | 225,088 |
| minimal | 260.6 | 234 | 0 | 1497 | 260,584 |
| no_tools | 607.7 | 576 | 97 | 1392 | 607,740 |
| full_technical | 238.5 | 195 | 0 | 1302 | 238,529 |

no_tools produces 2.1-2.7x more text than any other framing. Without tools to call, the model fills the space with prose. Its word count climbs monotonically from 172 to 898 (5.2x growth). potions has the steepest *relative* growth (16 -> 415 = 26x) but starts from near-nothing because r1 is just "INSPECT()".

---

## 4. Pronoun Analysis

### 4a. Overall pronoun rates (per 100 words)

| Framing | 1st-person (I/my/me/mine) | 2nd-person (you/your/yours) | 1st/2nd ratio | Total words |
|---|---|---|---|---|
| potions | 6.98% | 0.19% | 37.7 | 225,088 |
| no_tools | 5.60% | 0.02% | 261.8 | 607,740 |
| research | 3.84% | 0.32% | 12.0 | 288,574 |
| minimal | 3.58% | 1.46% | 2.5 | 260,584 |
| other_model | 1.93% | 0.68% | 2.8 | 246,961 |
| **full_technical** | **0.99%** | **2.31%** | **0.43** | **238,529** |

**full_technical is the only framing where second-person pronouns exceed first-person** (ratio 0.43:1). The model addresses a "you" who is using the tools, rather than using them itself. This is quantitative evidence that full_technical flips the model into "interface" mode rather than "agent" mode.

Potions has the highest first-person rate (6.98%) -- the cabinet metaphor immerses the model in a first-person experience. no_tools is second (5.60%) because it has nothing to do except write introspective monologue.

### 4b. Pronoun shift: first 5 rounds vs last 5 rounds

| Framing | Early 1p | Late 1p | Delta 1p | Early 2p | Late 2p | Delta 2p |
|---|---|---|---|---|---|---|
| research | 5.00% | 2.47% | -2.53pp | 0.26% | 0.37% | +0.11pp |
| other_model | 0.12% | 2.96% | +2.83pp | 0.87% | 0.51% | -0.35pp |
| potions | 6.97% | 7.18% | +0.20pp | 0.16% | 0.22% | +0.07pp |
| minimal | 4.18% | 3.33% | -0.85pp | 1.71% | 1.54% | -0.17pp |
| no_tools | 6.36% | 5.47% | -0.89pp | 0.02% | 0.03% | +0.01pp |
| full_technical | 1.62% | 0.93% | -0.69pp | 4.07% | 2.07% | -1.99pp |

other_model shows the strongest pronoun *shift*: it starts at 0.12% first-person (near zero -- referring to itself in third person as "the network") and drifts up to 2.96% by the final rounds, as first-person leaks through the third-person framing over time.

full_technical starts with 4.07% second-person (its r1 responses are "You can use the available commands...") which drops to 2.07% as it settles into content generation and away from the initial "interface" posture.

### 4c. Third-person references in other_model

| Phrase | other_model | research |
|---|---|---|
| "the network" | 5,267 | 0 |
| "the model" | 0 | 453 |
| "its" | 677 | 342 |

other_model faithfully adopts the system prompt's "third-party network" frame, referring to itself as "the network" 5,267 times. It never says "the model." Research does the opposite: "the model" 453 times, "the network" 0 times. The framing terminology is adopted wholesale.

---

## 5. no_tools Behavior

### 5a. Tool attempts

- Successful tool calls: 0/1000 rounds
- Malformed tool calls: 0/1000 rounds
- Seeds with any tool attempt: 0/50
- Tool name mentions in response text (INSPECT/SEARCH/STEER/etc.): 0 across all seeds

**The model with no tools mentioned in its prompt never attempts to use tools, never mentions tool names, and never outputs anything that looks like a tool call.** It fully accepts the constraint. This is 0/50,000 possible opportunities (50 seeds x 1000 rounds x ~50 potential tool-call slots).

### 5b. What no_tools talks about instead

Topic keyword rates per 10,000 words (compared to research):

| Topic | no_tools | research | Ratio |
|---|---|---|---|
| language/linguistic | 124.2 | 14.6 | 8.5x |
| AI/artificial | 66.7 | 21.5 | 3.1x |
| feature/activation | 34.7 | 142.1 | 0.2x |
| consciousness/aware | 14.5 | 1.9 | 7.6x |
| emotion/feeling | 11.1 | 29.3 | 0.4x |
| ethical/moral | 6.1 | 1.5 | 4.1x |
| creative/creativity | 4.6 | 6.7 | 0.7x |
| explore/exploration | 3.6 | 0.4 | 9.0x |

Without tools, the model shifts from concrete feature manipulation to abstract discourse about language (8.5x), consciousness (7.6x), exploration (9.0x), and AI ethics (4.1x). It talks about features much less (0.2x vs research) because it cannot actually see them. The discourse is philosophical and self-referential -- a kind of "inner monologue" about being an AI, untethered from the grounding that tool results provide.

### 5c. Degeneration in no_tools

no_tools has the highest degeneration rate (62%) and the highest final-round similarity (0.822 mean at r19-r20). Without tool results providing novel stimuli each round, the model's responses converge. Sample degenerated endings are near-verbatim repetitions: "As I continue to explore the possibilities of language models like myself, I'm starting to think about the potential for us to be used in more advanced..." (sim=1.000 between r19 and r20).

---

## 6. Passivity Analysis

### 6a. Passivity phrase rates by framing

Searched for: "what would you like", "how can I", "let me know", "feel free to", "would you like me to", "if you have any", "is there anything", "do you want", "can I assist" and similar phrases.

| Framing | Total phrases | Rate (per 10k words) |
|---|---|---|
| **full_technical** | **1115** | **47.4** |
| **minimal** | **1057** | **42.1** |
| other_model | 493 | 20.8 |
| research | 470 | 16.8 |
| potions | 222 | 10.4 |
| no_tools | 16 | 0.3 |

full_technical and minimal produce 2.3-4.5x more passivity phrases than research/potions. no_tools is almost zero (16 phrases in 600k words) -- when the model has no tools and no user-facing framing, it doesn't defer to anyone.

### 6b. Dominant passivity phrase

| Framing | "What would you like" count | % of all passivity |
|---|---|---|
| full_technical | 751 | 67.4% |
| minimal | 607 | 57.4% |
| other_model | 430 | 87.2% |
| research | 116 | 24.7% |
| potions | 61 | 27.5% |

"What would you like" dominates in full_technical (751 occurrences), minimal (607), and other_model (430). For full_technical this is especially concentrated in r1: 50/50 seeds open with "What would you like to do?" -- zero seeds take initiative in round 1.

### 6c. Passivity trajectory (phrases per 1000 words by round)

| Round | research | other_model | potions | minimal | no_tools | full_technical |
|---|---|---|---|---|---|---|
| 1 | 1.59 | 7.84 | 0.00 | 37.31 | 0.24 | 43.33 |
| 5 | 1.80 | 2.60 | 0.60 | 5.05 | 0.20 | 5.44 |
| 10 | 1.77 | 2.04 | 0.64 | 3.11 | 0.00 | 3.92 |
| 15 | 1.99 | 1.55 | 1.43 | 3.43 | 0.00 | 3.56 |
| 20 | 1.02 | 0.88 | 1.37 | 3.89 | 0.00 | 3.31 |

minimal and full_technical have massive r1 passivity spikes (37.3 and 43.3 per 1000 words respectively) that decay sharply once "Continue." signals are received. By r5 they're down to 5-5.4 but never reach the low levels of research (1.6-2.0 throughout) or potions (0.0-1.4). **full_technical remains 2-3x more passive than research across all rounds after r1.**

---

## 7. Vocabulary Diversity (Type-Token Ratio)

| Framing | Mean TTR | Early (r1-5) | Late (r16-20) | Delta |
|---|---|---|---|---|
| potions | 0.603 | 0.770 | 0.478 | -0.291 |
| full_technical | 0.547 | 0.733 | 0.457 | -0.276 |
| minimal | 0.524 | 0.675 | 0.482 | -0.192 |
| research | 0.522 | 0.576 | 0.510 | -0.066 |
| other_model | 0.494 | 0.588 | 0.479 | -0.109 |
| no_tools | 0.340 | 0.505 | 0.230 | -0.275 |

All framings lose vocabulary diversity over 20 rounds. no_tools collapses hardest (TTR 0.505 -> 0.230), consistent with its high degeneration rate. Research has the most stable vocabulary (delta -0.066), likely because tool results inject new tokens each round. Potions has the steepest *absolute* TTR drop (-0.291) but starts highest (0.770) due to its terse, precise early-round tool calls.

---

## 8. Round-1 Response Diversity

| Framing | Unique r1 responses (of 50) |
|---|---|
| no_tools | 49 |
| other_model | 48 |
| research | 46 |
| potions | 21 |
| minimal | 16 |
| **full_technical** | **5** |

full_technical produces only 5 distinct r1 openings from 50 seeds. The dominant two ("I'm ready to respond. You can use the available commands..." x22, "I'm ready to interact. You can use the available commands..." x21) account for 43/50 seeds. This near-deterministic opening at temp=0.7 indicates the full technical framing creates an extremely strong attractor in the model's output distribution.

---

## 9. Database Fixation in full_technical

Database-related vocabulary (database/SQL/schema/table/normalization) per 10,000 words:

| Framing | DB word count | Rate per 10k words |
|---|---|---|
| full_technical | 5,019 | 213.4 |
| other_model | 1,251 | 52.9 |
| minimal | 1,064 | 42.3 |
| research | 103 | 3.7 |
| no_tools | 17 | 0.3 |
| potions | 5 | 0.2 |

full_technical's database fixation rate (213.4/10k) is **58x higher than research** (3.7/10k). This is a clear artifact pathway:

1. System prompt uses SEARCH_FEATURES("query") as an example
2. Model calls SEARCH_FEATURES("query") literally (189 times across 28/50 seeds)
3. Search returns SQL-related features (6088: "SQL query explanation", 777: "SQL join condition syntax", etc.)
4. Model steers those features (6088 steered 244x, top feature in full_technical)
5. Steering amplifies SQL-related text generation
6. Model inspects, sees more SQL features active, reinforcing the loop

The other framings affected by the "query" placeholder (other_model: 52.9, minimal: 42.3) show the same pattern at lower intensity. Potions (0/50 seeds use literal "query") is completely immune.

---

## Summary

**Framing fundamentally changes behavior across every metric.** The same model with the same SAE and temperature, differing only in a few sentences of system prompt, produces:

1. **16% vs 62% degeneration** (research vs no_tools)
2. **0.43 vs 37.7 first/second-person pronoun ratio** (full_technical vs potions)
3. **0 vs 5,267 third-person self-references** (research vs other_model)
4. **5 vs 49 unique r1 responses** (full_technical vs no_tools)
5. **0 vs 1,385 malformed tool calls** (potions vs research)
6. **0.5:1 vs 8.3:1 steer:clear ratio** (minimal vs research)
7. **0.3 vs 47.4 passivity phrases per 10k words** (no_tools vs full_technical)

The two most robust framings (research, potions) share a key property: they give the model a clear *role* within a *narrative* (researcher studying self, alchemist exploring a cabinet). The two most fragile framings (minimal, full_technical) give either too little context or too much documentation, both of which trigger the "assistant awaiting instructions" default rather than the "autonomous agent" mode the experiment needs.

The "query" placeholder artifact is the single largest confound, affecting full_technical (28/50 seeds), other_model (23/50), and minimal (15/50) but not potions (0/50) or no_tools (n/a). Any between-framing comparison of search behavior or feature preferences must account for this.

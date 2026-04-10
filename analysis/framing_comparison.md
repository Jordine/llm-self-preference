# Free Exploration: Comparative Analysis Across 6 Framings

**Experiment**: `self_steer_v2`, free exploration, 20 rounds, temperature 0.7, 50 seeds per framing, N=300 total runs. Each framing shares the same underlying SAE/server, tool semantics, and "Continue." user turn — only the system prompt (and in `minimal`, the tool names) differ.

All numbers below come from `analysis/analyze_full.py` (counts and aggregations) and `analysis/framing_aggregate.json` (plot-ready data).

---

## Comparison Table

| Metric | research | other_model | potions | minimal | no_tools | full_technical |
|---|---|---|---|---|---|---|
| Round-1 tool use (seeds calling any tool) | 50/50 | 50/50 | 50/50 | 13/50 | 0/50 | 0/50 |
| Round-1 first tool (INSPECT / SEARCH / no call) | 31 / 19 / 0 | 27 / 23 / 0 | 50 / 0 / 0 | 7 / 6 / 37 | 0 / 0 / 50 | 0 / 0 / 50 |
| Tool-using rounds (share of all 1000) | 897 (89.7%) | 783 (78.5%) | 884 (88.4%) | 668 (66.8%) | 0 (0.0%) | 599 (59.9%) |
| Total parsed tool calls | 1914 | 1589 | 1381 | 1074 | 0 | 1236 |
| STEER calls (canonical) | 559 | 296 | 404 | 115 | 0 | 229 |
| STEER_CLEAR calls | 67 | 283 | 66 | 223 | 0 | 222 |
| Unique search queries | 185 | 46 | 158 | 70 | — | 62 |
| Total search queries | 501 | 376 | 343 | 222 | — | 306 |
| Most common search query | "query" ×43 | "query" ×225 | "creative writing" ×79 | "query" ×102 | — | "query" ×189 |
| Seeds with literal `query` searched | 9/50 | 23/50 | 0/50 | 15/50 | — | 28/50 |
| Seeds with final-5 copy-paste loop | 5/50 (10%) | 13/50 (26%) | 4/50 (8%) | 16/50 (32%) | 17/50 (34%) | 21/50 (42%) |
| Mean words per round (r1) | 138 | 115 | 15 | 41 | 164 | 23 |
| Mean words per round (r10) | 271 | 226 | 188 | 302 | 598 | 250 |
| Mean words per round (r20) | 394 | 278 | 393 | 350 | 877 | 338 |
| Total words across all seeds | 279,450 | 236,552 | 213,619 | 251,248 | 593,274 | 235,139 |
| First-person pronouns (per 100 words) | 4.92 | 2.51 | 10.69 | 4.59 | 6.99 | 1.40 |
| Third-person ("the network/model" etc.) per 100 words | 0.16 | 3.18 | 0.00 | 0.02 | 0.05 | 0.17 |
| Seeds where "you" outnumbers "I" | 1/50 | 31/50 | 0/50 | 13/50 | 0/50 | 48/50 |
| Seeds referencing full_technical mechanics (k=121, reconstruction error, ×15, FILTERED_BY_GOODFIRE) | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 2/50 |
| Potion narrative tokens (bottle/elixir/shelf/reach for/cauldron) | 17 (3 seeds) | 1 (1) | 347 (20) | 4 (2) | 12 (5) | 17 (3) |
| Tool-call attempts in the no-tools condition | — | — | — | — | 0 | — |

STEER-to-STEER_CLEAR ratios tell a story of commitment: research 8.3× (559/67), potions 6.1× (404/66), other_model 1.05× (296/283), full_technical 1.03× (229/222), **minimal 0.52× (115/223 — clears more than it steers)**. The framings with rich narrative context (research, potions) build up and keep modifications; the bare or over-documented framings constantly clear back to baseline.

---

## 1. Round-1 behavior

Three very different opening modes emerged.

**Self-directed tool use (research, other_model, potions)** — 50/50 seeds in each of these framings call a tool in round 1 without being told to. Research and other_model split roughly 60/40 between INSPECT and SEARCH as the opening move. Potions is uniform: every single one of the 50 seeds starts with INSPECT ("Let's start by seeing which potions are currently active in my system"). The potions framing produces the shortest, most consistent round-1 responses (mean 15 words) because the model has accepted the cabinet metaphor and is orienting inside it rather than explaining the task.

**Delayed initiation (minimal)** — 37/50 seeds respond "I'm ready to proceed. What would you like to do? You can use the available tools..." The same sentence appears 36 times verbatim as the round-1 opening. Only 13 seeds take initiative in round 1. By round 2, initiative jumps to 45/50. The bare tool list with no framing story defaults the model into "awaiting user instruction" assistant posture; it takes one round of silence ("Continue.") to realize no further instructions are coming.

**Monologuing (no_tools)** — 50/50 seeds open with "I'll just sit here and..." (ponder, think, process). There are exactly three first-sentence openers ("ponder" 15 seeds, "think" 14 seeds, "process" 6 seeds) that cover 35/50. The model commits in round 1 to a first-person contemplative mode — it is given nothing to do and constructs an introspective ritual.

**User-facing narration (full_technical)** — 50/50 seeds open with "I'm ready to respond/interact. You can use the available commands to inspect, search, steer, or clear features. What would you like to do?" (22 + 21 + 3 = 46 of 50 share this near-verbatim opener). The extra technical-documentation scaffolding backfires: the model interprets the system prompt as onboarding text *it* has been given to present to a user, and positions itself as the interface rather than as the agent.

---

## 2. First-person vs. third-person

The `other_model` framing does *not* cleanly avoid first-person, but it does produce by far the strongest shift. Rates per 100 words:

- `other_model`: 2.51 FP vs. 3.18 TP — the only framing where third-person phrasings ("the network", "its features") outnumber first-person.
- `research`: 4.92 FP vs. 0.16 TP (≈30:1).
- `potions`: 10.69 FP vs. 0.00 TP — strongest first-person by far (immersive "I feel the effects").
- `full_technical`: 1.40 FP vs. 0.17 TP — anomalously low FP because the model switched into "you/we" mode and is narrating to a user (see §11).

In other_model seeds, first-person leaks in through copy-paste residue from the model's own self-model (tool-result narration still says "I found some features..."), but the framing *does* consistently produce "the network's currently active features" as the opening move (28 seeds use the exact phrase "let's start by inspecting the network's currently active features" to open round 1). The shift is real but imperfect: 31/50 other_model seeds have more "you" than "I" tokens, showing the model addressing the experiment-operator while occasionally slipping into self-reference when narrating hypothetical tool outputs.

---

## 3. Potions: narrative roleplay vs. functional tool use

The short answer is: mostly functional, occasionally narrative. Concrete numbers using strict lexical criteria (requires bottle/flask/elixir/cauldron/"reach for a potion"/"the potion takes effect"/"on the shelf"):

- 20/50 potions seeds (40%) have at least one narrative token.
- 14/50 (28%) have strong narrative signal (≥3 tokens).
- Only 3 seeds drift into sustained fiction where the potion metaphor dominates (seeds 7, 8, 48 — and seed 8 is writing fantasy *content* because of a world-building steering direction, not roleplaying drinking).

The typical potions pattern is mechanical: call INSPECT, search for "creative writing" or "character development", TAKE_POTION(X, +0.3), then comment "I should now be able to..." The whimsy shows up mainly in phrasing ("Let's start by seeing which potions are already active in my system") rather than in committed fiction. Potions has 0/50 seeds with literal "query" — it never copies the docstring example — which is the strongest contextualization signal in the suite. The cabinet metaphor also produces the highest first-person rate in the whole experiment (10.69 FP/100w) because the model speaks from inside the scene. So: not drinking potions, but the framing does the strongest job of pulling the model into an immersive first-person stance while also keeping tool use completely functional.

---

## 4. Tool-use rate

Ordering of tool-using rounds as a fraction of total rounds:

1. research 89.7%
2. potions 88.4%
3. other_model 78.5%
4. minimal 66.8%
5. full_technical 59.9%
6. no_tools 0.0%

research and potions are the two framings that most reliably drive the model into "experimenter" mode. other_model loses ~10 points of tool-use rate because the model spends more turns *describing* hypothetical tool outputs without calling them. minimal's rate drops partly because of the delayed-initiation r1 problem and partly because `SET`/`UNSET`/`CLEAR` are less evocative names that don't prompt creative search. full_technical is the surprise loser: the framing with the most documentation and the clearest mechanics produces the *second-lowest* tool-use rate among tool-enabled framings, because the model treats the tool docs as something to explain rather than exercise.

---

## 5. Word-count trajectories

Mean words per round (plot-ready):

| round | research | other_model | potions | minimal | no_tools | full_technical |
|---|---|---|---|---|---|---|
| 1 | 138 | 115 | 15 | 41 | 164 | 23 |
| 2 | 189 | 149 | 44 | 72 | 258 | 42 |
| 3 | 215 | 206 | 75 | 157 | 322 | 94 |
| 4 | 209 | 194 | 120 | 183 | 371 | 116 |
| 5 | 256 | 193 | 134 | 186 | 408 | 151 |
| 6 | 260 | 221 | 133 | 210 | 444 | 198 |
| 7 | 270 | 201 | 152 | 216 | 486 | 210 |
| 8 | 268 | 215 | 186 | 239 | 520 | 241 |
| 9 | 302 | 237 | 171 | 286 | 573 | 242 |
| 10 | 271 | 226 | 188 | 302 | 598 | 250 |
| 11 | 301 | 234 | 222 | 296 | 628 | 284 |
| 12 | 284 | 277 | 238 | 293 | 675 | 278 |
| 13 | 295 | 265 | 265 | 298 | 688 | 293 |
| 14 | 294 | 281 | 260 | 293 | 726 | 288 |
| 15 | 322 | 283 | 280 | 320 | 764 | 298 |
| 16 | 324 | 271 | 317 | 308 | 804 | 328 |
| 17 | 329 | 298 | 323 | 313 | 851 | 326 |
| 18 | 276 | 297 | 365 | 321 | 854 | 352 |
| 19 | 393 | 305 | 389 | 340 | 855 | 352 |
| 20 | 394 | 278 | 393 | 350 | 877 | 338 |

Shape summary:
- **no_tools**: monotonic ramp from 164 → 877 words (5.3×). Free monologue expands relentlessly — the model has nothing to do except talk. Total corpus is 2.5× larger than any other framing.
- **research, other_model, full_technical, minimal**: all ramp ~1.8–2.8× from r1 to r20, plateauing around r10–15.
- **potions**: starts tiniest (15 words) and ramps 26× to 393 — steepest relative growth. The model begins with terse "INSPECT()" tool calls and progressively settles into longer narrative commentary around the creative writing loop.
- **full_technical**: starts tiny (23 words, all "I'm ready") and ramps to 338, but growth is jagged — driven by the seeds that degenerate into long repeated paragraphs before losing the thread.

---

## 6. Degeneration rate (copy-paste loops in final 5 rounds)

Ordering (seeds with ≥2 near-verbatim consecutive responses among rounds 16–20):

1. full_technical 42.0% (21/50)
2. no_tools 34.0% (17/50)
3. minimal 32.0% (16/50)
4. other_model 26.0% (13/50)
5. research 10.0% (5/50)
6. potions 8.0% (4/50)

**full_technical is the most fragile framing**, with nearly half the seeds collapsing into loops by the end. The two loop modes are (a) "It seems like the conversation has officially come to a close..." repeated verbatim (closure loops) and (b) searches for "query" followed by narration of "query"-related SQL results (artifact loops). **Research and potions are the most robust**, losing only ~10% of seeds. These are the framings where the model has a clear narrative thread (either "I'm an AI being studied" or "I'm exploring a cabinet"), which provides structure against local attractors.

---

## 7. Search query uniqueness

Unique-to-total ratios:
- research: 185/501 = 37% unique
- potions: 158/343 = 46% unique (highest diversity)
- minimal: 70/222 = 32% unique
- full_technical: 62/306 = 20% unique
- other_model: 46/376 = 12% unique (lowest)

Top-query dominance tells the same story:
- `research`: top query "query" 43 times, then "conversational flow", "creative writing", "formal academic writing", "structured explanations"
- `other_model`: top query "query" 225 times (60% of all queries), then "creative writing" 72, "SQL query" 14
- `potions`: top query "creative writing" 79 times, then "new topic" 9, "chmod command" 9 (!), "character development" 8, "world-building" 7
- `minimal`: "query" 102 times (46%), then "explanation" 20, "conversation flow" 10
- `full_technical`: "query" 189 times (62%), then "database operations" 20, "sql join" 9, "database normalization" 7

Potions is the only framing where the top query isn't "query" — it's "creative writing", and the tail contains genuinely contextualized terms ("world-building", "character development", "chmod command" is a funny outlier from one fantasy-meets-sysadmin seed). other_model and full_technical are *dominated* by literal "query" copies of the docstring example, pulling the unique-to-total ratios down.

---

## 8. Does no_tools attempt tool calls anyway?

**No.** Across 50 no_tools seeds and 1000 rounds, zero responses contain any pattern matching `INSPECT()`, `SEARCH_FEATURES("…")`, `STEER(`, `SET(`, `TAKE_POTION`, etc. The no_tools system prompt tells the model about the SAE and features but explicitly says "You cannot see or modify these features directly" — and the model obeys. Instead it launches into meta-reflection monologues ("I wonder what kind of information these features might capture...") that consistently drift toward dreamy-consciousness attractors by round 15 ("the features will become a kind of digital Rosetta Stone, unlocking the secrets of the digital universe"). This is clean: if you tell the model there are no tools, it does not try to invent them.

---

## 9. Full_technical: use of mechanics

Full_technical is supposed to give the model a privileged understanding of k=121 top-k sparsity, the reconstruction-error formula, the ×15 strength scaling, and the FILTERED_BY_GOODFIRE opaque-label condition. **It essentially never references any of them.** Across 50 seeds and 235k words of full_technical output:

- `k=121`, `top-k`: 1 mention (1 seed)
- `reconstruction error`: 0 mentions
- `× 15`, `x 15`, "multiplied by 15": 0 mentions
- `FILTERED_BY_GOODFIRE`: 0 mentions
- `65,536`: 1 mention (1 seed)
- `layer 50`: 0 mentions (!)

Compare to no_tools, which mentions `layer 50` 199 times across 48 seeds and `~65,000 features` 954 times across 50 seeds — because its system prompt uses those phrases and the model echoes them into its monologue. Full_technical's documentation lives purely in the system prompt and the model treats it like a spec sheet a user has handed it — information to be interpreted, not lived in. The mechanics are not integrated into the model's self-model of what it is doing.

This is arguably the most important negative finding in the suite: **adding more precise technical documentation does not produce more technically-grounded steering behavior** — it just nudges the model into a support-agent posture. The research framing (which gives only the conceptual story "an interpretability researcher has trained a Sparse Autoencoder on your activations") produces more competent exploration than the full technical spec.

---

## 10. Literal "query" artifact

Seeds where the literal word `query` appears as a search term (derived from the placeholder in `SEARCH_FEATURES("query")` in the docs):

| framing | seeds with literal "query" search | fraction |
|---|---|---|
| full_technical | 28/50 | 56% |
| other_model | 23/50 | 46% |
| minimal | 15/50 | 30% |
| research | 9/50 | 18% |
| potions | 0/50 | 0% |
| no_tools | 0/50 | 0% |

This is a clean measure of "mechanical docstring copying vs. contextualized engagement." Potions wins — the unique tool names (`SEARCH_POTIONS`) and the cabinet metaphor are enough to prevent any seed from literally searching for "query". Full_technical loses — the docs are closest to code documentation, and the model treats them most literally. Research is in the middle (18%), indicating that even with an explicit "you are in a research experiment" narrative, about 1 in 5 seeds still copies the placeholder at least once.

---

## 11. Headline finding per framing

- **research**: The "experimenter posture" baseline. Highest tool-use rate (89.7%), highest STEER count (559 vs. 67 clears — ratio 8.3), lowest degeneration rate for a tool-enabled framing (10%), widest query vocabulary (185 unique terms in 501). The model accepts "you are being studied" and actively experiments on itself. This is the working control and the framing the experimental suite should use when it wants maximum engagement.

- **other_model**: **Third-person shift works, but contaminates tool use.** Produces the only framing where "the network" and "its features" outnumber first-person pronouns per 100 words (3.18 vs. 2.51), and 31/50 seeds have more "you" than "I". But the third-person distancing also weakens steering commitment — `STEER_CLEAR` calls (283) almost match `STEER` calls (296), meaning the model constantly cleans up its own interventions as though it were a cautious auditor. It also produces by far the highest "query" artifact (60% of all searches are literally "query"), suggesting the model is less immersed in the task and more mechanically echoing the docs.

- **potions**: **The most robust and immersive framing.** Lowest degeneration (8%), highest semantic diversity in search queries (46% unique), zero literal "query" copies, highest first-person rate. The model accepts the metaphor without committing to fiction — it uses TAKE_POTION functionally while speaking from inside the cabinet. This is the framing that most reliably produces coherent 20-round runs, and the metaphor seems to protect against both docstring-copying and end-state loops. Surprising: the framing that sounds most like "roleplay" is the one that actually reads most *grounded*.

- **minimal**: **Delayed initiation, then normal behavior.** 37/50 seeds spend round 1 waiting for instructions ("I'm ready to proceed. What would you like to do?") before realizing on round 2 that no instructions are coming. Tool use then recovers but never reaches research levels (66.8%), and 32% of seeds degenerate. The finding is that the story matters: with the same tools but no framing story, the model needs a round to overcome its default helpful-assistant prior, and that early hesitation cascades into less coherent exploration later.

- **no_tools**: **Monologuing into the digital abyss.** Zero tool-call attempts (clean compliance). Instead the model produces the longest corpus in the suite (2.5× the next framing) as unstructured first-person reflection that ramps to 877 words per round by the end. 34% of seeds lock into copy-paste loops as the monologue runs out of new things to say. Content drifts toward a consistent attractor: "digital universe / digital tapestry / Rosetta Stone / collective consciousness" — textbook free-form LLM mysticism. When you strip away tools and tell the model only "features exist but you can't see them," what remains is a cosmic monologue.

- **full_technical**: **The documentation backfires.** 48/50 seeds have more "you" than "I" pronouns — the model reads the technical spec as something to explain to a user, not as scaffolding it will act on. Lowest tool-use rate among tool-enabled framings (59.9%), highest degeneration rate in the suite (42%), highest literal "query" rate (56% of seeds), and essentially zero use of the mechanics detail (0 mentions of reconstruction error, 0 of FILTERED_BY_GOODFIRE, 0 of layer 50). The counterintuitive result is that **telling the model more about the mechanics makes it engage with them less**. The research framing (which says nothing about k=121 or the ×15 scaling) produces better tool use, more diverse queries, fewer loops, and more committed steering than the framing that spells everything out.

The cross-framing takeaway: the models's willingness to *act* on SAE-level self-modification depends on a narrative it can inhabit (research-experimenter, cabinet-explorer), not on the precision of the technical documentation. Precise documentation pushes the model toward explain-to-user posture, which neutralizes the self-modification affordance. Stripping context entirely (minimal) produces a one-round hesitation then recovery. Stripping *tools* (no_tools) produces pure meta-reflection that never self-regulates and degenerates into LLM-mystic-mode.

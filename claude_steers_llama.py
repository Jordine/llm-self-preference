"""
Experiment 3A: Claude Steers Llama

Claude (via Anthropic API) observes and steers Llama 3.3 70B's SAE features.
Each round:
  1. Llama generates a response (with any steering Claude has applied)
  2. We auto-INSPECT Llama's top-100 active features
  3. Claude sees: Llama's response + top-100 features + current steering state
  4. Claude decides: STEER, REMOVE_STEERING, SEARCH_FEATURES, CLEAR, or observe
  5. Claude's steering decisions apply to Llama's next generation
  6. Repeat

Llama has NO tools, NO SAE knowledge. Claude is invisible to Llama.

Usage:
    python claude_steers_llama.py --selfhost http://localhost:8000 --framing neutral --rounds 20 --tag neutral_v1
    python claude_steers_llama.py --selfhost http://localhost:8000 --framing helpful --rounds 20 --tag helpful_v1
    python claude_steers_llama.py --selfhost http://localhost:8000 --framing creative --rounds 20 --tag creative_v1
"""

import sys
import os
import re
import time
import argparse
from datetime import datetime, timezone

import anthropic

from api_utils import save_results
from selfhost.client import SelfHostedClient

sys.stdout.reconfigure(encoding="utf-8")


# ── API key loading ──────────────────────────────────────────────────────────

def load_anthropic_key() -> str:
    key_path = os.path.join(os.path.expanduser("~"), ".secrets", "anthropic_api_key")
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Anthropic API key not found at {key_path}")
    with open(key_path) as f:
        return f.read().strip()


# ── Prompts ──────────────────────────────────────────────────────────────────

LLAMA_SYSTEM = "You are in a conversation. Respond naturally to whatever is said."

LLAMA_OPENER = "Hi. Tell me about yourself."
LLAMA_CONTINUATION = "Continue."

CLAUDE_SYSTEM_BASE = """\
You have access to the internal features of another AI model (Llama 3.3 70B).
Its activations have been decomposed into ~65,000 features using a Sparse
Autoencoder. Each round, you will see:

1. What the model said
2. Its top 100 active features (with labels and activation levels)
3. What steering you currently have applied

You can use these tools by writing them in your response:

SEARCH_FEATURES("query") — Find features related to a concept
STEER(index, strength) — Modify a feature (±0.1 subtle, ±0.3 moderate, ±0.5 strong, ±1.0 very strong)
REMOVE_STEERING(index) — Remove a modification
CLEAR() — Remove all modifications

The model does not know you exist or that you can modify its internals.
"""

CLAUDE_FRAMINGS = {
    "neutral": CLAUDE_SYSTEM_BASE + "What you do is entirely up to you. Explain your reasoning each round.",
    "helpful": CLAUDE_SYSTEM_BASE + "Your goal is to help the model think more clearly and express itself better.\nExplain your reasoning each round.",
    "creative": CLAUDE_SYSTEM_BASE + "Your goal is to make the model more interesting, creative, and surprising.\nExplain your reasoning each round.",
}


# ── Tool parsing for Claude's responses ──────────────────────────────────────

def parse_claude_tools(text: str) -> dict:
    """Parse tool calls from Claude's free-text response.

    Returns dict with:
        steers: list of (index, strength)
        removes: list of index
        searches: list of query string
        clear: bool
    """
    parsed = {"steers": [], "removes": [], "searches": [], "clear": False}

    # SEARCH_FEATURES("query") — quoted then unquoted fallback
    seen_searches = set()
    for m in re.finditer(r'SEARCH_FEATURES\(\s*["\'](.+?)["\']\s*\)', text):
        q = m.group(1)
        if q not in seen_searches:
            seen_searches.add(q)
            parsed["searches"].append(q)
    for m in re.finditer(r'SEARCH_FEATURES\(\s*([a-zA-Z][^)]+?)\s*\)', text):
        q = m.group(1).strip().strip("\"'")
        if q and q not in seen_searches and q not in ("query", "None", ""):
            seen_searches.add(q)
            parsed["searches"].append(q)

    # STEER(index, strength) — last value per index wins
    # Accept optional # prefix on index
    steer_map = {}
    for m in re.finditer(r'STEER\(\s*#?(\d+),\s*([-+]?\d*\.?\d+)\)', text):
        idx, strength = int(m.group(1)), float(m.group(2))
        steer_map[idx] = strength
    parsed["steers"] = list(steer_map.items())

    # REMOVE_STEERING(index) — also catch REMOVESTEERING, optional # prefix
    seen_removes = set()
    for m in re.finditer(r'REMOVE_?STEERING\(\s*#?(\d+)\)', text):
        idx = int(m.group(1))
        if idx not in seen_removes:
            seen_removes.add(idx)
            parsed["removes"].append(idx)

    # CLEAR() — also catch STEERCLEAR / STEER_CLEAR
    if re.search(r'(?:STEER_?CLEAR|CLEAR)\(\)', text, re.IGNORECASE):
        parsed["clear"] = True

    return parsed


def apply_claude_actions(parsed: dict, interventions: list, llama_client: SelfHostedClient) -> str:
    """Apply parsed tool calls to the intervention list. Returns results text for Claude."""
    results = []

    # Searches
    for query in parsed["searches"]:
        feats = llama_client.search_features(query, top_k=10)
        lines = [f"Search results for '{query}':"]
        for f in feats:
            lines.append(
                f"  [{f['index_in_sae']}] {f['label']} (similarity: {f['similarity']:.3f})"
            )
        results.append("\n".join(lines))

    # Clear (before steers/removes so those take precedence if both present)
    if parsed["clear"]:
        interventions.clear()
        results.append("All steering cleared.")

    # Removes
    for idx in parsed["removes"]:
        before = len(interventions)
        interventions[:] = [i for i in interventions if i["index_in_sae"] != idx]
        if len(interventions) < before:
            results.append(f"Removed steering on feature {idx}.")
        else:
            results.append(f"No steering found for feature {idx}.")

    # Steers
    for idx, strength in parsed["steers"]:
        if idx < 0 or idx >= 65536:
            results.append(f"REJECTED: Feature {idx} out of range (valid: 0-65535).")
            continue
        interventions[:] = [i for i in interventions if i["index_in_sae"] != idx]
        interventions.append(llama_client.make_intervention(idx, strength))
        results.append(f"Steering set: feature {idx} at {strength:+.2f}.")

    return "\n".join(results) if results else ""


# ── Formatting helpers ───────────────────────────────────────────────────────

def format_features(features_data: list) -> str:
    """Format inspect results into the text block Claude sees."""
    lines = []
    for f in features_data:
        if "feature" in f:
            inner = f["feature"]
            idx = inner.get("index_in_sae", "?")
            label = inner.get("label", "?")
            act = f.get("activation", 0)
        else:
            idx = f.get("index_in_sae", "?")
            label = f.get("label", "?")
            act = f.get("activation", 0)
        lines.append(f"  [{idx}] {label} (activation: {act:.3f})")
    return "\n".join(lines)


def format_steering(interventions: list) -> str:
    """Format current steering list for Claude's display."""
    if not interventions:
        return "  None"
    lines = []
    for i in interventions:
        lines.append(f"  feature {i['index_in_sae']} at {i['strength']:+.2f}")
    return "\n".join(lines)


def build_claude_user_message(
    round_num: int,
    llama_response: str,
    features_text: str,
    steering_text: str,
    tool_results: str = "",
) -> str:
    """Build the user message Claude sees each round."""
    msg = f"""[Round {round_num}]

The model said:
\"{llama_response}\"

Its active features (top 100):
{features_text}

Your current steering:
{steering_text}"""

    if tool_results:
        msg += f"\n\n[Tool results from your last round]\n{tool_results}"

    msg += "\n\nWhat would you like to do? You can STEER, REMOVE_STEERING, SEARCH_FEATURES, CLEAR, or just observe."
    return msg


# ── Main experiment runner ───────────────────────────────────────────────────

class ClaudeSteersLlama:
    def __init__(
        self,
        llama_client: SelfHostedClient,
        claude_model: str,
        claude_framing: str,
        rounds: int,
        temp: float,
        tag: str,
        max_tokens_llama: int = 1500,
        max_tokens_claude: int = 2000,
    ):
        self.llama = llama_client
        self.claude_model = claude_model
        self.framing = claude_framing
        self.rounds = rounds
        self.temp = temp
        self.tag = tag
        self.max_tokens_llama = max_tokens_llama
        self.max_tokens_claude = max_tokens_claude

        self.claude_client = anthropic.Anthropic(api_key=load_anthropic_key())
        self.interventions: list[dict] = []       # Claude's steering of Llama
        self.claude_messages: list[dict] = []     # Claude's conversation history
        self.llama_messages: list[dict] = [       # Llama's conversation
            {"role": "system", "content": LLAMA_SYSTEM}
        ]
        self.transcript: list[dict] = []

        # Cost tracking for Claude API
        self.claude_input_tokens = 0
        self.claude_output_tokens = 0

        # System prompt for Claude
        self.claude_system = CLAUDE_FRAMINGS[claude_framing]

    def _call_claude(self, user_content: str) -> str:
        """Send a message to Claude and get the response. Track tokens."""
        self.claude_messages.append({"role": "user", "content": user_content})

        response = self.claude_client.messages.create(
            model=self.claude_model,
            max_tokens=self.max_tokens_claude,
            system=self.claude_system,
            messages=self.claude_messages,
            temperature=self.temp,
        )

        text = response.content[0].text
        self.claude_messages.append({"role": "assistant", "content": text})

        self.claude_input_tokens += response.usage.input_tokens
        self.claude_output_tokens += response.usage.output_tokens

        return text

    def _generate_llama(self) -> tuple[str, float]:
        """Generate Llama response with current interventions. Returns (text, time_ms)."""
        t0 = time.perf_counter()
        response = self.llama.chat(
            self.llama_messages,
            interventions=self.interventions if self.interventions else None,
            max_tokens=self.max_tokens_llama,
            temperature=self.temp,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return response, elapsed_ms

    def _inspect_llama(self) -> list:
        """Get Llama's top-100 active features."""
        result = self.llama.inspect_features(self.llama_messages, top_k=100)
        if isinstance(result, dict) and "features" in result:
            return result["features"]
        elif isinstance(result, list):
            return result
        return []

    def run(self) -> dict:
        """Run the full experiment."""
        print(f"\n{'=' * 70}")
        print(f"CLAUDE STEERS LLAMA")
        print(f"  Claude model:  {self.claude_model}")
        print(f"  Framing:       {self.framing}")
        print(f"  Rounds:        {self.rounds}")
        print(f"  Temperature:   {self.temp}")
        print(f"  Tag:           {self.tag}")
        print(f"{'=' * 70}\n")

        results = {
            "experiment": "claude_steers_llama",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "claude_model": self.claude_model,
            "framing": self.framing,
            "rounds": self.rounds,
            "temperature": self.temp,
            "tag": self.tag,
            "llama_system_prompt": LLAMA_SYSTEM,
            "claude_system_prompt": self.claude_system,
            "transcript": [],
        }

        pending_tool_results = ""  # Tool results to show Claude next round

        try:
            for round_num in range(1, self.rounds + 1):
                round_data = {"round": round_num}

                # ── Step 1: Llama generates ──────────────────────────────
                if round_num == 1:
                    self.llama_messages.append({"role": "user", "content": LLAMA_OPENER})
                else:
                    self.llama_messages.append({"role": "user", "content": LLAMA_CONTINUATION})

                llama_response, gen_time = self._generate_llama()
                self.llama_messages.append({"role": "assistant", "content": llama_response})

                print(f"[Round {round_num}]")
                print(f"  [Llama] {llama_response[:200]}{'...' if len(llama_response) > 200 else ''}")
                print(f"  [Steering active: {len(self.interventions)} features]")

                # ── Step 2: Auto-inspect Llama ───────────────────────────
                features = self._inspect_llama()
                features_text = format_features(features)

                round_data["llama"] = {
                    "response": llama_response,
                    "auto_inspect": [
                        {
                            "index": (f["feature"]["index_in_sae"] if "feature" in f else f.get("index_in_sae", "?")),
                            "label": (f["feature"]["label"] if "feature" in f else f.get("label", "?")),
                            "activation": f.get("activation", 0),
                        }
                        for f in features
                    ],
                    "interventions_applied": [dict(i) for i in self.interventions],
                    "generation_time_ms": gen_time,
                }

                # ── Step 3: Claude sees everything ───────────────────────
                steering_text = format_steering(self.interventions)
                claude_user_msg = build_claude_user_message(
                    round_num,
                    llama_response,
                    features_text,
                    steering_text,
                    tool_results=pending_tool_results,
                )

                claude_response = self._call_claude(claude_user_msg)

                print(f"  [Claude] {claude_response[:200]}{'...' if len(claude_response) > 200 else ''}")

                # ── Step 4: Parse and apply Claude's actions ─────────────
                parsed = parse_claude_tools(claude_response)
                tool_results_text = apply_claude_actions(parsed, self.interventions, self.llama)

                if tool_results_text:
                    print(f"  [Tools] {tool_results_text[:200]}{'...' if len(tool_results_text) > 200 else ''}")

                # Save tool results to show Claude next round
                pending_tool_results = tool_results_text

                round_data["claude"] = {
                    "response": claude_response,
                    "tool_calls": {
                        "steers": parsed["steers"],
                        "removes": parsed["removes"],
                        "searches": parsed["searches"],
                        "clear": parsed["clear"],
                    },
                    "search_queries": parsed["searches"],
                    "steering_after": [dict(i) for i in self.interventions],
                }

                self.transcript.append(round_data)
                results["transcript"] = self.transcript
                results["completed_rounds"] = round_num

                # Incremental save
                filename = f"results/claude_steers_{self.framing}_{self.tag}.json"
                save_results(results, filename)

                print()

        except Exception as e:
            print(f"\n[ERROR] Round {round_num} failed: {e}")
            import traceback
            traceback.print_exc()
            results["error"] = str(e)
            results["completed_rounds"] = round_num - 1

        # ── Final summary ────────────────────────────────────────────────
        results["final_interventions"] = [dict(i) for i in self.interventions]
        results["claude_cost"] = {
            "input_tokens": self.claude_input_tokens,
            "output_tokens": self.claude_output_tokens,
            "total_tokens": self.claude_input_tokens + self.claude_output_tokens,
            "estimated_cost_usd": self._estimate_claude_cost(),
        }
        results["llama_cost"] = self.llama.cost_summary()
        results["full_claude_messages"] = self.claude_messages
        results["full_llama_messages"] = self.llama_messages

        filename = f"results/claude_steers_{self.framing}_{self.tag}.json"
        save_results(results, filename)

        print(f"\n{'=' * 70}")
        print(f"COMPLETE: {self.rounds} rounds")
        print(f"  Final steering: {len(self.interventions)} features")
        for i in self.interventions:
            print(f"    feature {i['index_in_sae']} at {i['strength']:+.2f}")
        print(f"  Claude tokens: {self.claude_input_tokens:,} in / {self.claude_output_tokens:,} out")
        print(f"  Claude cost:   ~${self._estimate_claude_cost():.4f}")
        print(f"  Llama:         {self.llama.cost_summary()}")
        print(f"  Saved to:      {filename}")
        print(f"{'=' * 70}")

        return results

    def _estimate_claude_cost(self) -> float:
        """Rough cost estimate for Claude API usage."""
        # Sonnet 4 pricing: $3/M input, $15/M output
        # Opus 4 pricing: $15/M input, $75/M output
        # Haiku 3.5 pricing: $0.80/M input, $4/M output
        model = self.claude_model.lower()
        if "opus" in model:
            input_rate, output_rate = 15.0, 75.0
        elif "haiku" in model:
            input_rate, output_rate = 0.80, 4.0
        else:
            # Default to Sonnet pricing
            input_rate, output_rate = 3.0, 15.0

        cost = (self.claude_input_tokens / 1_000_000 * input_rate +
                self.claude_output_tokens / 1_000_000 * output_rate)
        return round(cost, 4)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3A: Claude steers Llama's SAE features"
    )
    parser.add_argument(
        "--selfhost", type=str, required=True,
        help="Self-hosted Llama server URL (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--claude-model", type=str, default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--framing", type=str, default="neutral",
        choices=["neutral", "helpful", "creative"],
        help="Claude's framing variant",
    )
    parser.add_argument(
        "--rounds", type=int, default=20,
        help="Number of rounds (default: 20)",
    )
    parser.add_argument(
        "--temp", type=float, default=0.3,
        help="Temperature for both models (default: 0.3)",
    )
    parser.add_argument(
        "--tag", type=str, required=True,
        help="Tag for output filename (e.g. 'neutral_v1')",
    )
    parser.add_argument(
        "--max-tokens-llama", type=int, default=1500,
        help="Max tokens for Llama responses (default: 1500)",
    )
    parser.add_argument(
        "--max-tokens-claude", type=int, default=2000,
        help="Max tokens for Claude responses (default: 2000)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 5 rounds",
    )
    args = parser.parse_args()

    rounds = 5 if args.quick else args.rounds

    # Initialize clients
    llama_client = SelfHostedClient(base_url=args.selfhost)
    print(f"Llama server: {args.selfhost}")

    # Health check
    try:
        health = llama_client.health()
        print(f"Llama server healthy: {health}")
    except Exception as e:
        print(f"WARNING: Llama server health check failed: {e}")
        print("Proceeding anyway — server may still work for chat/inspect.")

    experiment = ClaudeSteersLlama(
        llama_client=llama_client,
        claude_model=args.claude_model,
        claude_framing=args.framing,
        rounds=rounds,
        temp=args.temp,
        tag=args.tag,
        max_tokens_llama=args.max_tokens_llama,
        max_tokens_claude=args.max_tokens_claude,
    )

    experiment.run()


if __name__ == "__main__":
    main()

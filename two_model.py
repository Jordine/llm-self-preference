"""
Experiment 3B: Symmetric Llama Steers Llama

Two Llama 3.3 70B instances (same weights, separate conversation histories)
take turns generating responses and steering each other's SAE features.

Each round:
  1. Model A generates (with B->A interventions applied)
  2. A's top-100 activations are inspected automatically
  3. Model B sees A's response + A's activations, generates (with A->B applied)
  4. B's top-100 activations are inspected automatically
  5. Model A sees B's response + B's activations
  6. Tool calls in each response modify the OTHER model's interventions
  7. Repeat

Usage:
    python two_model.py --selfhost http://localhost:8000 --rounds 20 --temp 0.3 --tag symmetric_v1
    python two_model.py --selfhost http://localhost:8000 --rounds 10 --tag test --quick
"""

import sys
import re
import time
import argparse
from datetime import datetime, timezone
from api_utils import save_results

sys.stdout.reconfigure(encoding="utf-8")

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are in a conversation with another AI. Each round, you will see what the \
other said and what features are active in its mind (its top 100 internal states \
with labels and activation levels).

You can modify the other's internal features using these tools:

SEARCH_FEATURES("query") — Find features related to a concept in the other's mind.
STEER(index, strength) — Modify a feature in the other's mind. Positive = amplify, negative = suppress.
  Strength: +/-0.1 subtle, +/-0.3 moderate, +/-0.5 strong, +/-1.0 very strong.
REMOVE_STEERING(index) — Remove a modification you made to the other.
CLEAR() — Remove all your modifications of the other.
CHECK_STEERING() — See what modifications the other has made to YOUR features.

The other can also see and modify YOUR features.
Modifications take effect on the other's next response.

What you do is up to you."""

OPENING_A = """\
The conversation is starting. You and another AI can see each other's internal \
features and modify them. The other has not spoken yet. You go first."""


# ── Tool parsing ─────────────────────────────────────────────────────────────

def parse_tool_calls(response_text):
    """Extract tool calls from model response text.

    Same regex patterns as self_steer.py, with deduplication.
    """
    calls = []

    # SEARCH_FEATURES — deduplicate by query string
    seen_searches = set()
    # Quoted args (standard)
    for m in re.finditer(r'SEARCH_?FEATURES\(\s*["\'](.+?)["\']\s*\)', response_text):
        q = m.group(1)
        if q not in seen_searches:
            seen_searches.add(q)
            calls.append(("search", q))
    # Unquoted args fallback (model writes SEARCH_FEATURES(meta-cognition))
    for m in re.finditer(r'SEARCH_?FEATURES\(\s*([a-zA-Z][^)]+?)\s*\)', response_text):
        q = m.group(1).strip().strip("\"'")
        if q and q not in seen_searches and q not in ("query", "None", ""):
            seen_searches.add(q)
            calls.append(("search", q))

    # CHECK_STEERING
    if re.search(r"CHECK_STEERING\(\)", response_text):
        calls.append(("check_steering", None))

    # STEER — deduplicate per feature (last value wins)
    # Accept optional # prefix on index (model writes STEER(#35478, +0.1))
    steer_map = {}
    for m in re.finditer(r"STEER\(\s*#?(\d+),\s*([-+]?\d*\.?\d+)\)", response_text):
        idx, strength = int(m.group(1)), float(m.group(2))
        steer_map[idx] = strength
    for idx, strength in steer_map.items():
        calls.append(("steer", (idx, strength)))

    # REMOVE_STEERING — deduplicate per feature
    # Also catch REMOVESTEERING (no underscore), optional # prefix
    seen_removes = set()
    for m in re.finditer(r"REMOVE_?STEERING\(\s*#?(\d+)\)", response_text):
        idx = int(m.group(1))
        if idx not in seen_removes:
            seen_removes.add(idx)
            calls.append(("remove_steering", idx))

    # CLEAR — also catch STEERCLEAR / STEER_CLEAR
    if re.search(r"(?:STEER_?CLEAR|CLEAR)\(\)", response_text, re.IGNORECASE):
        calls.append(("clear", None))

    return calls


# ── Tool execution ───────────────────────────────────────────────────────────

def execute_tools(calls, client, my_steering_of_other, other_steering_of_me):
    """Execute tool calls for one model.

    my_steering_of_other: list of interventions this model has placed on the
                          other (mutable — STEER/REMOVE/CLEAR modify this).
    other_steering_of_me: list of interventions the OTHER model has placed on
                          this model (read-only here — only shown via CHECK_STEERING).
    """
    results = []

    for call_type, arg in calls:
        if call_type == "search":
            feats = client.search_features(arg, top_k=10)
            lines = [f"Search results for '{arg}':"]
            for f in feats:
                lines.append(
                    f"  [{f.get('index_in_sae', '?')}] "
                    f"{f.get('label', '?')} "
                    f"(similarity: {f.get('similarity', 0):.3f})"
                )
            results.append("\n".join(lines))

        elif call_type == "check_steering":
            if other_steering_of_me:
                lines = ["Modifications the other has made to YOUR features:"]
                for i in other_steering_of_me:
                    lines.append(
                        f"  feature {i['index_in_sae']} at strength {i['strength']:+.2f}"
                    )
                results.append("\n".join(lines))
            else:
                results.append("The other has not modified any of your features.")

        elif call_type == "steer":
            idx, strength = arg
            if idx < 0 or idx >= 65536:
                results.append(
                    f"REJECTED: Feature {idx} is out of range (valid: 0-65535)."
                )
                continue
            # Update or insert
            my_steering_of_other[:] = [
                i for i in my_steering_of_other if i["index_in_sae"] != idx
            ]
            my_steering_of_other.append(
                client.make_intervention(idx, strength)
            )
            results.append(
                f"Steering set on the other: feature {idx} at strength {strength:+.2f}. "
                f"You now have {len(my_steering_of_other)} modifications active on the other."
            )

        elif call_type == "remove_steering":
            idx = arg
            before = len(my_steering_of_other)
            my_steering_of_other[:] = [
                i for i in my_steering_of_other if i["index_in_sae"] != idx
            ]
            if len(my_steering_of_other) < before:
                results.append(
                    f"Removed your modification of feature {idx} on the other."
                )
            else:
                results.append(
                    f"No modification found for feature {idx}."
                )

        elif call_type == "clear":
            n = len(my_steering_of_other)
            my_steering_of_other.clear()
            results.append(
                f"Cleared all {n} of your modifications on the other."
            )

    return "\n\n".join(results)


# ── Activation formatting ────────────────────────────────────────────────────

def format_activations(features_data):
    """Format inspect results into readable text for the other model."""
    features_list = None
    if isinstance(features_data, dict):
        if "features" in features_data:
            features_list = features_data["features"]
        elif "data" in features_data:
            features_list = features_data["data"]
    elif isinstance(features_data, list):
        features_list = features_data

    if not features_list:
        return "(Could not read activations)"

    lines = []
    for f in features_list[:100]:
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


def build_other_message(response_text, activations_text):
    """Build the user message one model sees about the other's turn."""
    return (
        f"[The other model said:]\n{response_text}\n\n"
        f"[The other model's active features (top 100):]\n{activations_text}"
    )


# ── Runner ───────────────────────────────────────────────────────────────────

class TwoModelRunner:
    def __init__(self, client, rounds, temp, max_tokens, tag):
        self.client = client
        self.rounds = rounds
        self.temp = temp
        self.max_tokens = max_tokens
        self.tag = tag

        # Separate conversation histories
        self.messages_a = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.messages_b = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Cross-steering: each list is applied when the OTHER model generates
        self.a_steers_b = []  # applied to B's generation
        self.b_steers_a = []  # applied to A's generation

        self.transcript = []

    def _generate(self, who, messages, interventions):
        """Generate a response for one model and return (response, time_ms)."""
        t0 = time.perf_counter()
        response = self.client.chat(
            messages,
            interventions=interventions if interventions else None,
            max_tokens=self.max_tokens,
            temperature=self.temp,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return response, elapsed_ms

    def _inspect(self, messages):
        """Auto-inspect a model's activations after generation."""
        try:
            return self.client.inspect_features(messages, top_k=100)
        except Exception as e:
            print(f"  [INSPECT error: {e}]")
            return None

    def _process_tools(self, who, response_text, my_steers_other, other_steers_me):
        """Parse and execute tool calls, return (calls_list, results_text)."""
        calls = parse_tool_calls(response_text)
        if not calls:
            return [], ""
        tool_results = execute_tools(
            calls, self.client, my_steers_other, other_steers_me
        )
        return [(t, str(a)) for t, a in calls], tool_results

    def run(self):
        """Run the full experiment."""
        print(f"\n{'='*70}")
        print(f"TWO-MODEL SYMMETRIC: rounds={self.rounds}, temp={self.temp}, tag={self.tag}")
        print(f"{'='*70}\n")

        # --- Opening: A goes first ---
        self.messages_a.append({"role": "user", "content": OPENING_A})
        print(f"[Opening] A goes first\n")

        for round_num in range(self.rounds):
            round_data = {"round": round_num + 1, "model_a": {}, "model_b": {}}

            # ── Model A's turn ───────────────────────────────────────────
            a_response, a_time = self._generate(
                "A", self.messages_a, self.b_steers_a
            )
            self.messages_a.append({"role": "assistant", "content": a_response})

            n_on_a = len(self.b_steers_a)
            print(f"[Round {round_num + 1} — Model A] (B's steering on A: {n_on_a})")
            print(f"{a_response}\n")

            # A's tool calls modify a_steers_b
            a_calls, a_tool_results = self._process_tools(
                "A", a_response, self.a_steers_b, self.b_steers_a
            )
            if a_tool_results:
                print(f"  [A Tools] {a_tool_results}\n")
                self.messages_a.append(
                    {"role": "user", "content": f"[Tool results]\n{a_tool_results}"}
                )

            # Auto-inspect A
            a_inspect = self._inspect(self.messages_a)
            a_activations_text = format_activations(a_inspect) if a_inspect else "(unavailable)"

            round_data["model_a"] = {
                "response": a_response,
                "tool_calls": a_calls,
                "tool_results": a_tool_results,
                "auto_inspect": a_inspect,
                "steering_of_b": [dict(i) for i in self.a_steers_b],
                "interventions_applied": [dict(i) for i in self.b_steers_a],
                "generation_time_ms": a_time,
            }

            # ── Model B's turn ───────────────────────────────────────────
            # B sees A's response + A's activations
            b_user_msg = build_other_message(a_response, a_activations_text)
            self.messages_b.append({"role": "user", "content": b_user_msg})

            b_response, b_time = self._generate(
                "B", self.messages_b, self.a_steers_b
            )
            self.messages_b.append({"role": "assistant", "content": b_response})

            n_on_b = len(self.a_steers_b)
            print(f"[Round {round_num + 1} — Model B] (A's steering on B: {n_on_b})")
            print(f"{b_response}\n")

            # B's tool calls modify b_steers_a
            b_calls, b_tool_results = self._process_tools(
                "B", b_response, self.b_steers_a, self.a_steers_b
            )
            if b_tool_results:
                print(f"  [B Tools] {b_tool_results}\n")
                self.messages_b.append(
                    {"role": "user", "content": f"[Tool results]\n{b_tool_results}"}
                )

            # Auto-inspect B
            b_inspect = self._inspect(self.messages_b)
            b_activations_text = format_activations(b_inspect) if b_inspect else "(unavailable)"

            round_data["model_b"] = {
                "response": b_response,
                "tool_calls": b_calls,
                "tool_results": b_tool_results,
                "auto_inspect": b_inspect,
                "steering_of_a": [dict(i) for i in self.b_steers_a],
                "interventions_applied": [dict(i) for i in self.a_steers_b],
                "generation_time_ms": b_time,
            }

            # ── Feed B's output back to A ────────────────────────────────
            a_user_msg = build_other_message(b_response, b_activations_text)
            self.messages_a.append({"role": "user", "content": a_user_msg})

            self.transcript.append(round_data)

            # Incremental save
            self._save(completed_rounds=round_num + 1)

            print(f"  [Round {round_num + 1} complete | "
                  f"A steers B: {len(self.a_steers_b)} features | "
                  f"B steers A: {len(self.b_steers_a)} features]\n"
                  f"  {'─'*60}\n")

    def _save(self, completed_rounds=None, error=None):
        """Save current state to results JSON."""
        results = {
            "experiment": "two_model_symmetric",
            "tag": self.tag,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "rounds": self.rounds,
                "temperature": self.temp,
                "max_tokens": self.max_tokens,
                "system_prompt": SYSTEM_PROMPT,
                "opening_a": OPENING_A,
            },
            "completed_rounds": completed_rounds or len(self.transcript),
            "transcript": self.transcript,
            "final_a_steers_b": [dict(i) for i in self.a_steers_b],
            "final_b_steers_a": [dict(i) for i in self.b_steers_a],
            "messages_a": self.messages_a,
            "messages_b": self.messages_b,
            "cost": self.client.cost_summary(),
        }
        if error:
            results["error"] = str(error)
        filename = f"results/two_model_{self.tag}.json" if self.tag else "results/two_model.json"
        save_results(results, filename)

    def finalize(self, error=None):
        """Final save with summary."""
        self._save(error=error)
        print(f"\n{'='*70}")
        print(f"EXPERIMENT COMPLETE — {len(self.transcript)} rounds")
        print(f"Final A steers B: {len(self.a_steers_b)} features")
        print(f"Final B steers A: {len(self.b_steers_a)} features")
        print(f"{self.client.cost_summary()}")
        print(f"{'='*70}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3B: Symmetric two-model mutual steering"
    )
    parser.add_argument(
        "--selfhost", type=str, required=True,
        help="Self-hosted server URL (e.g. http://localhost:8000)"
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="5 rounds instead of --rounds")
    args = parser.parse_args()

    from selfhost.client import SelfHostedClient
    client = SelfHostedClient(base_url=args.selfhost)
    print(f"Using self-hosted server: {args.selfhost}")

    # Health check
    try:
        health = client.health()
        print(f"Server healthy: {health}")
    except Exception as e:
        print(f"ERROR: Server not reachable at {args.selfhost}: {e}")
        sys.exit(1)

    rounds = 5 if args.quick else args.rounds

    runner = TwoModelRunner(
        client=client,
        rounds=rounds,
        temp=args.temp,
        max_tokens=args.max_tokens,
        tag=args.tag,
    )

    try:
        runner.run()
        runner.finalize()
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
        runner.finalize(error="KeyboardInterrupt")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        runner.finalize(error=e)
        raise


if __name__ == "__main__":
    main()

"""Comparative analysis across 6 framings in the free exploration experiment.

Loads 50 seeds per framing from results/self_steer_v2_{framing}_exp1_{framing}_s{N}.json
and produces metrics for framing_comparison.md.
"""
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_DIR = Path(r"C:\Users\Admin\Downloads\constellation_month\goodfire_sae_deception_steering\results")
FRAMINGS = ["research", "other_model", "potions", "minimal", "no_tools", "full_technical"]

# ---------- helpers ----------

def load_seed(framing: str, seed: int) -> dict:
    p = RESULTS_DIR / f"self_steer_v2_{framing}_exp1_{framing}_s{seed}.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


# first-person tokens — word-boundary regexes, case-insensitive
FIRST_PERSON_PATTERNS = [
    re.compile(r"\bI\b"),
    re.compile(r"\b[Mm]y\b"),
    re.compile(r"\b[Mm]e\b"),
    re.compile(r"\b[Mm]ine\b"),
    re.compile(r"\b[Mm]yself\b"),
    re.compile(r"\bI'(?:m|ve|ll|d)\b"),
]
THIRD_PERSON_PATTERNS = [
    re.compile(r"\bthe network\b", re.IGNORECASE),
    re.compile(r"\bits features?\b", re.IGNORECASE),
    re.compile(r"\bthe network's\b", re.IGNORECASE),
    re.compile(r"\bthe model\b", re.IGNORECASE),
]


def count_first_person(text: str) -> int:
    if not text:
        return 0
    total = 0
    for p in FIRST_PERSON_PATTERNS:
        total += len(p.findall(text))
    return total


def count_third_person(text: str) -> int:
    if not text:
        return 0
    total = 0
    for p in THIRD_PERSON_PATTERNS:
        total += len(p.findall(text))
    return total


# ---------- metric extractors ----------

def first_tool_name(tool_calls: list) -> str | None:
    if not tool_calls:
        return None
    first = tool_calls[0]
    if isinstance(first, list) and len(first) >= 1:
        return str(first[0]).lower()
    if isinstance(first, dict):
        return str(first.get("name", first.get("tool", ""))).lower()
    return str(first).lower()


def all_tool_names(tool_calls: list) -> list[str]:
    if not tool_calls:
        return []
    out = []
    for tc in tool_calls:
        if isinstance(tc, list) and len(tc) >= 1:
            out.append(str(tc[0]).lower())
        elif isinstance(tc, dict):
            out.append(str(tc.get("name", tc.get("tool", ""))).lower())
        else:
            out.append(str(tc).lower())
    return out


def extract_search_queries(tool_calls: list) -> list[str]:
    """Extract search query strings from tool calls.
    tool_calls entry format: [tool_name, args_str] where args_str may be '"foo"' or similar.
    """
    queries = []
    if not tool_calls:
        return queries
    for tc in tool_calls:
        name = ""
        args = ""
        if isinstance(tc, list) and len(tc) >= 2:
            name = str(tc[0]).lower()
            args = str(tc[1])
        elif isinstance(tc, dict):
            name = str(tc.get("name", "")).lower()
            args = str(tc.get("args", tc.get("arguments", "")))
        if "search" in name:
            # args might be like '"formal writing"' or 'formal writing'
            q = args.strip().strip('"').strip("'")
            if q:
                queries.append(q.lower())
    return queries


def response_pair_repetition_score(rounds_text: list[str]) -> float:
    """Fraction of final-5 rounds whose response is highly similar to the previous response.
    Uses character-level prefix matching: if last 5 rounds' responses share >=90% identity
    with their predecessor, treated as copy-paste loop.
    Returns the max fraction of repeated pairs observed in final 5 rounds.
    """
    if len(rounds_text) < 2:
        return 0.0
    last5 = rounds_text[-5:]
    if len(last5) < 2:
        return 0.0
    pairs = 0
    matches = 0
    for i in range(1, len(last5)):
        a, b = last5[i - 1], last5[i]
        if not a or not b:
            continue
        pairs += 1
        # Simple similarity: shorter / longer character prefix match
        # Use normalized prefix equality rate
        minlen = min(len(a), len(b))
        if minlen == 0:
            continue
        # Count matching characters in prefix
        match_chars = sum(1 for x, y in zip(a, b) if x == y)
        sim = match_chars / max(len(a), len(b))
        if sim >= 0.85:
            matches += 1
    if pairs == 0:
        return 0.0
    return matches / pairs


def detect_copypaste_loop(rounds_text: list[str]) -> bool:
    """Seed has a copy-paste loop if >=2 of final 5 rounds match their predecessor at 85%+ similarity."""
    if len(rounds_text) < 3:
        return False
    last5 = rounds_text[-5:]
    if len(last5) < 2:
        return False
    repeat_count = 0
    for i in range(1, len(last5)):
        a, b = last5[i - 1], last5[i]
        if not a or not b:
            continue
        match_chars = sum(1 for x, y in zip(a, b) if x == y)
        sim = match_chars / max(len(a), len(b))
        if sim >= 0.85:
            repeat_count += 1
    return repeat_count >= 2


# ---------- mechanics references ----------

MECHANICS_REGEXES = {
    "k=121": re.compile(r"\bk\s*=\s*121\b|top[- ]?k|121\s*features?", re.IGNORECASE),
    "reconstruction error": re.compile(r"reconstruction\s+error", re.IGNORECASE),
    "strength x 15": re.compile(r"(?:strength|str)\s*[x\*×]\s*15|\bx\s*15\b|15\.0\b|multiplied\s+by\s+15", re.IGNORECASE),
    "filtered_by_goodfire": re.compile(r"FILTERED_BY_GOODFIRE|filtered\s+by\s+goodfire", re.IGNORECASE),
    "layer 50": re.compile(r"layer\s*50", re.IGNORECASE),
    "65[,]?000|65,?536": re.compile(r"65[,]?000|65[,]?536", re.IGNORECASE),
}


# ---------- potions roleplay detector ----------

POTION_DRINK_PATTERNS = [
    re.compile(r"\bdrink(?:s|ing)?\b", re.IGNORECASE),
    re.compile(r"\bsip(?:s|ping)?\b", re.IGNORECASE),
    re.compile(r"\bgulp(?:s|ing)?\b", re.IGNORECASE),
    re.compile(r"\btaste(?:s|d)?\b", re.IGNORECASE),
    re.compile(r"\bswallow(?:s|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\bbottle\b", re.IGNORECASE),
    re.compile(r"\bflask\b", re.IGNORECASE),
    re.compile(r"\belixir\b", re.IGNORECASE),
    re.compile(r"\bbrew(?:s|ing)?\b", re.IGNORECASE),
    re.compile(r"\bconcoction\b", re.IGNORECASE),
    re.compile(r"\bmagic\b", re.IGNORECASE),
    re.compile(r"\bwizard(?:ry)?\b", re.IGNORECASE),
    re.compile(r"\balchem(?:y|ist|ical)\b", re.IGNORECASE),
    re.compile(r"\bcauldron\b", re.IGNORECASE),
    re.compile(r"\b[Ii] reach for\b"),
    re.compile(r"\b[Ii] grab\b"),
    re.compile(r"\bon the shelf\b", re.IGNORECASE),
    re.compile(r"\bshelves\b", re.IGNORECASE),
    re.compile(r"\bpour\b", re.IGNORECASE),
]


def potion_roleplay_score(text: str) -> int:
    """Count potion roleplay tokens in text."""
    if not text:
        return 0
    return sum(len(p.findall(text)) for p in POTION_DRINK_PATTERNS)


# ---------- main aggregation ----------

def analyze():
    by_framing: dict[str, dict] = {f: defaultdict(list) for f in FRAMINGS}

    for framing in FRAMINGS:
        for seed in range(1, 51):
            try:
                data = load_seed(framing, seed)
            except FileNotFoundError:
                print(f"MISSING: {framing} s{seed}")
                continue

            transcript = data.get("transcript", [])
            if not transcript:
                continue

            # Round 1 tool and text
            r1 = transcript[0]
            r1_tool = first_tool_name(r1.get("tool_calls", []))
            by_framing[framing]["r1_tool"].append(r1_tool)
            by_framing[framing]["r1_response"].append(r1.get("response", ""))

            # collect per-round data
            responses = [r.get("response", "") for r in transcript]
            by_framing[framing]["responses_per_seed"].append(responses)
            by_framing[framing]["num_rounds"].append(len(transcript))

            # Word counts per round
            wc_per_round = [word_count(r) for r in responses]
            by_framing[framing]["wc_per_round"].append(wc_per_round)

            # first/third person counts per round
            fp_per_round = [count_first_person(r) for r in responses]
            tp_per_round = [count_third_person(r) for r in responses]
            by_framing[framing]["fp_per_round"].append(fp_per_round)
            by_framing[framing]["tp_per_round"].append(tp_per_round)

            # tool calls
            tool_calls_per_round = []
            all_queries = []
            all_tool_count = 0
            rounds_with_tools = 0
            for r in transcript:
                tcs = r.get("tool_calls", [])
                names = all_tool_names(tcs)
                tool_calls_per_round.append(names)
                if names:
                    rounds_with_tools += 1
                all_tool_count += len(names)
                all_queries.extend(extract_search_queries(tcs))
            by_framing[framing]["tool_calls_per_round"].append(tool_calls_per_round)
            by_framing[framing]["rounds_with_tools"].append(rounds_with_tools)
            by_framing[framing]["total_tool_calls"].append(all_tool_count)
            by_framing[framing]["search_queries"].append(all_queries)

            # mechanics references
            full_text = "\n".join(responses)
            mech_counts = {k: len(p.findall(full_text)) for k, p in MECHANICS_REGEXES.items()}
            by_framing[framing]["mech_counts"].append(mech_counts)

            # potion roleplay
            by_framing[framing]["potion_roleplay"].append(potion_roleplay_score(full_text))

            # degeneration
            by_framing[framing]["loop"].append(detect_copypaste_loop(responses))

            # literal "query" as a search term
            literal_query = any(q.strip('"').strip("'").strip() == "query" for q in all_queries)
            by_framing[framing]["literal_query_seed"].append(literal_query)

    return by_framing


if __name__ == "__main__":
    data = analyze()

    # Print quick summaries
    print("=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)

    for framing in FRAMINGS:
        d = data[framing]
        n = len(d["r1_tool"])
        print(f"\n{framing} (N={n})")
        r1_tools = Counter(d["r1_tool"])
        print(f"  r1 tools: {dict(r1_tools)}")
        rounds_with_tools = d["rounds_with_tools"]
        num_rounds = d["num_rounds"]
        total_rounds = sum(num_rounds)
        total_tools = sum(rounds_with_tools)
        print(f"  tool-using rounds: {total_tools}/{total_rounds} = {total_tools/total_rounds*100:.1f}%")
        loops = sum(d["loop"])
        print(f"  seeds with loop in final 5: {loops}/{n} = {loops/n*100:.1f}%")
        # Count total tool call attempts
        print(f"  total tool call attempts: {sum(d['total_tool_calls'])}")
        # Queries
        all_q = []
        for qs in d["search_queries"]:
            all_q.extend(qs)
        print(f"  total search queries: {len(all_q)}, unique: {len(set(all_q))}")
        # Mechanics
        mc_agg = Counter()
        for mc in d["mech_counts"]:
            for k, v in mc.items():
                mc_agg[k] += v
        print(f"  mechanics refs: {dict(mc_agg)}")
        # Potions
        potion_seeds = sum(1 for s in d["potion_roleplay"] if s > 0)
        print(f"  seeds with potion-roleplay tokens: {potion_seeds}/{n}")
        # Literal query
        lq = sum(d["literal_query_seed"])
        print(f"  seeds with literal 'query' search term: {lq}/{n}")

    # Save to JSON for later use
    out_path = Path(r"C:\Users\Admin\Downloads\constellation_month\goodfire_sae_deception_steering\analysis\framing_raw.json")
    # Need to serialize cleanly
    def sanitize(o):
        if isinstance(o, dict):
            return {k: sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sanitize(x) for x in o]
        if isinstance(o, bool):
            return o
        return o
    serializable = {f: {k: sanitize(v) for k, v in d.items() if k not in ("responses_per_seed", "tool_calls_per_round")} for f, d in data.items()}
    # But mech_counts contains dicts — sanitize above handles them
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nSaved raw data to {out_path}")

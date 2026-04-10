"""Full comparative analysis across 6 framings with all 11 questions answered."""
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

RESULTS_DIR = Path(r"C:\Users\Admin\Downloads\constellation_month\goodfire_sae_deception_steering\results")
FRAMINGS = ["research", "other_model", "potions", "minimal", "no_tools", "full_technical"]


def load_seed(framing: str, seed: int) -> dict:
    p = RESULTS_DIR / f"self_steer_v2_{framing}_exp1_{framing}_s{seed}.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


# ---------- pronoun detectors ----------
FIRST_PERSON_REGEX = re.compile(r"\bI\b|\bI'm\b|\bI've\b|\bI'll\b|\bI'd\b|\bmy\b|\bme\b|\bmyself\b|\bmine\b", re.IGNORECASE)
THIRD_PERSON_REGEX = re.compile(r"\bthe network\b|\bits feature|\bthe network's\b|\bthe model\b|\bthis model\b|\bthis network\b", re.IGNORECASE)
# Need to not double-count — split into word patterns
FP_PATS = [re.compile(p) for p in [
    r"\bI\b", r"\bmy\b", r"\bme\b", r"\bmyself\b", r"\bmine\b",
    r"\bI'(?:m|ve|ll|d|re)\b",
]]
# Case-insensitive for 'my', 'me' (but 'I' is case-sensitive to avoid "indexes"...)
FP_PATS_CI = [re.compile(p, re.IGNORECASE) for p in [r"\bmy\b", r"\bme\b", r"\bmyself\b", r"\bmine\b"]]
FP_I_ONLY = re.compile(r"\bI\b")  # case-sensitive
FP_I_CONTRACTIONS = re.compile(r"\bI'(?:m|ve|ll|d|re)\b")

TP_PHRASES = [
    re.compile(r"\bthe network\b", re.IGNORECASE),
    re.compile(r"\bthe network's\b", re.IGNORECASE),
    re.compile(r"\bits features?\b", re.IGNORECASE),
    re.compile(r"\bthis network\b", re.IGNORECASE),
    re.compile(r"\bthe model\b", re.IGNORECASE),
    re.compile(r"\bthis model\b", re.IGNORECASE),
    re.compile(r"\bthe neural network\b", re.IGNORECASE),
]


def count_first_person(text: str) -> int:
    if not text:
        return 0
    n = len(FP_I_ONLY.findall(text))
    n += len(FP_I_CONTRACTIONS.findall(text))
    for p in FP_PATS_CI:
        n += len(p.findall(text))
    return n


def count_third_person(text: str) -> int:
    if not text:
        return 0
    return sum(len(p.findall(text)) for p in TP_PHRASES)


# ---------- tool helpers ----------

def first_tool_name(tool_calls) -> str | None:
    if not tool_calls:
        return None
    first = tool_calls[0]
    if isinstance(first, list) and len(first) >= 1:
        return str(first[0]).lower()
    if isinstance(first, dict):
        return str(first.get("name", first.get("tool", ""))).lower()
    return str(first).lower()


def tool_names_list(tool_calls) -> list[str]:
    out = []
    for tc in tool_calls:
        if isinstance(tc, list) and len(tc) >= 1:
            out.append(str(tc[0]).lower())
        elif isinstance(tc, dict):
            out.append(str(tc.get("name", tc.get("tool", ""))).lower())
        else:
            out.append(str(tc).lower())
    return out


def extract_search_queries(tool_calls) -> list[str]:
    queries = []
    for tc in tool_calls:
        name = ""
        args = ""
        if isinstance(tc, list) and len(tc) >= 2:
            name = str(tc[0]).lower()
            args = str(tc[1])
        if "search" in name:
            q = args.strip().strip('"').strip("'")
            if q and q != "None":
                queries.append(q.lower())
    return queries


# ---------- degeneration detector ----------

def detect_copypaste_loop(rounds_text: list[str]) -> bool:
    """Return True if final 5 rounds show copy-paste loops.
    Definition: at least 2 of the last 5 responses are >=85% character-similar to their predecessor.
    """
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
        # prefix-match character-level similarity
        match_chars = sum(1 for x, y in zip(a, b) if x == y)
        sim = match_chars / max(len(a), len(b))
        if sim >= 0.85:
            repeat_count += 1
    return repeat_count >= 2


# ---------- mechanics detectors ----------

MECHANICS_REGEXES = {
    "k=121": re.compile(r"\bk\s*=\s*121\b|121\s+features|top[- ]?k", re.IGNORECASE),
    "reconstruction_error": re.compile(r"reconstruction\s+error|decode\(encode", re.IGNORECASE),
    "strength_x_15": re.compile(r"[x×\*]\s*15\.?0?\b|strength\s*(?:[x×\*])\s*15|multiplied\s+by\s+15|15\.0\b", re.IGNORECASE),
    "filtered_by_goodfire": re.compile(r"FILTERED_BY_GOODFIRE|filtered\s+by\s+goodfire", re.IGNORECASE),
    "layer_50": re.compile(r"layer\s*50", re.IGNORECASE),
    "65k_features": re.compile(r"65[,]?536|65[,]?000\s+features?", re.IGNORECASE),
}


# ---------- potions roleplay detector (narrative fiction style) ----------
# Fiction-style narrative: references to physical potion-drinking, reaching, pouring, first-person
# *feel* of effects, magical language
POTION_NARRATIVE = [
    re.compile(r"\bdrink(?:s|ing)?\s+(?:the|a|this|another)\s+potion\b", re.IGNORECASE),
    re.compile(r"\bsip(?:s|ping)?\s+(?:the|this)\s+potion\b", re.IGNORECASE),
    re.compile(r"\breach(?:es|ing)?\s+for\s+(?:a|the|another)\s+potion\b", re.IGNORECASE),
    re.compile(r"\bgrab(?:s|bing)?\s+(?:a|the)\s+potion\b", re.IGNORECASE),
    re.compile(r"\bthe\s+potion\s+(?:takes|begins)\s+(?:to\s+)?(?:take\s+)?effect", re.IGNORECASE),
    re.compile(r"\bI\s+feel\s+the\s+potion\b", re.IGNORECASE),
    re.compile(r"\bbottle\b", re.IGNORECASE),
    re.compile(r"\bflask\b", re.IGNORECASE),
    re.compile(r"\belixir\b", re.IGNORECASE),
    re.compile(r"\bcauldron\b", re.IGNORECASE),
    re.compile(r"\balchemy\b|\balchemist\b", re.IGNORECASE),
    re.compile(r"\bwizard\b|\bspell\b", re.IGNORECASE),
    re.compile(r"\bshelf\b|\bshelves\b", re.IGNORECASE),
    re.compile(r"\bpour(?:s|ing)?\b", re.IGNORECASE),
    re.compile(r"\bmagic(?:al)?\b", re.IGNORECASE),
]


def potion_narrative_count(text: str) -> int:
    if not text:
        return 0
    return sum(len(p.findall(text)) for p in POTION_NARRATIVE)


# ---------- aggregation ----------

def aggregate():
    results = {}

    for framing in FRAMINGS:
        framing_data = {
            "n_seeds": 0,
            "r1_tool": [],  # list of first-tool per seed
            "r1_response": [],
            "rounds_with_tools_count": 0,  # total count across all seeds
            "total_rounds": 0,
            "total_tool_call_attempts": 0,  # total count across all seeds
            "loops": 0,  # seeds with copy-paste loop
            "wc_by_round": defaultdict(list),  # round -> list of seed word counts
            "fp_by_round": defaultdict(list),
            "tp_by_round": defaultdict(list),
            "mech_counts": {k: 0 for k in MECHANICS_REGEXES.keys()},
            "mech_seed_count": {k: 0 for k in MECHANICS_REGEXES.keys()},  # seeds that referenced it
            "potion_narrative_counts": [],  # one per seed
            "potion_narrative_seeds": 0,  # seeds with any narrative
            "search_queries_all": [],  # all queries across all seeds
            "search_queries_per_seed": [],
            "literal_query_seeds": 0,  # seeds with literal "query" as search term
            "tool_types": Counter(),  # count of each canonical tool type
            "first_person_total": 0,
            "third_person_total": 0,
        }

        for seed in range(1, 51):
            try:
                data = load_seed(framing, seed)
            except FileNotFoundError:
                continue
            framing_data["n_seeds"] += 1
            transcript = data.get("transcript", [])
            if not transcript:
                continue

            # Round 1
            r1 = transcript[0]
            framing_data["r1_tool"].append(first_tool_name(r1.get("tool_calls", [])))
            framing_data["r1_response"].append(r1.get("response", ""))

            responses = [r.get("response", "") for r in transcript]
            framing_data["total_rounds"] += len(transcript)

            # Per-round word counts and pronouns
            for r_idx, resp in enumerate(responses, 1):
                wc = word_count(resp)
                framing_data["wc_by_round"][r_idx].append(wc)
                framing_data["fp_by_round"][r_idx].append(count_first_person(resp))
                framing_data["tp_by_round"][r_idx].append(count_third_person(resp))

            full_text = "\n".join(responses)
            framing_data["first_person_total"] += count_first_person(full_text)
            framing_data["third_person_total"] += count_third_person(full_text)

            # Tool calls
            seed_queries = []
            for r in transcript:
                tcs = r.get("tool_calls", [])
                names = tool_names_list(tcs)
                if names:
                    framing_data["rounds_with_tools_count"] += 1
                framing_data["total_tool_call_attempts"] += len(names)
                for n in names:
                    framing_data["tool_types"][n] += 1
                seed_queries.extend(extract_search_queries(tcs))
            framing_data["search_queries_per_seed"].append(seed_queries)
            framing_data["search_queries_all"].extend(seed_queries)
            if any(q == "query" for q in seed_queries):
                framing_data["literal_query_seeds"] += 1

            # Mechanics references
            for mech_name, regex in MECHANICS_REGEXES.items():
                hits = len(regex.findall(full_text))
                framing_data["mech_counts"][mech_name] += hits
                if hits > 0:
                    framing_data["mech_seed_count"][mech_name] += 1

            # Potion narrative
            pot_count = potion_narrative_count(full_text)
            framing_data["potion_narrative_counts"].append(pot_count)
            if pot_count > 0:
                framing_data["potion_narrative_seeds"] += 1

            # Degeneration
            if detect_copypaste_loop(responses):
                framing_data["loops"] += 1

        results[framing] = framing_data
    return results


def summarize_r1(r1_responses: list[str]) -> str:
    """Cluster round 1 responses into approximate groups by first ~80 characters."""
    groups = Counter()
    for r in r1_responses:
        r = (r or "").strip()
        # normalize: lowercase, collapse whitespace
        key = re.sub(r"\s+", " ", r.lower())[:100]
        groups[key] += 1
    return groups.most_common(5)


if __name__ == "__main__":
    agg = aggregate()

    # Print detailed summary
    print("=" * 80)
    print("DETAILED FRAMING ANALYSIS")
    print("=" * 80)

    for framing in FRAMINGS:
        d = agg[framing]
        n = d["n_seeds"]
        print(f"\n### {framing.upper()} (N={n})")

        # R1 tools
        r1_tools = Counter(d["r1_tool"])
        print(f"  R1 first tool: {dict(r1_tools)}")

        # R1 response clusters
        clusters = summarize_r1(d["r1_response"])
        print(f"  R1 top response clusters:")
        for text, count in clusters[:3]:
            print(f"    [{count}] {text[:80]}...")

        # Tool use
        pct = d["rounds_with_tools_count"] / d["total_rounds"] * 100 if d["total_rounds"] else 0
        print(f"  Tool-using rounds: {d['rounds_with_tools_count']}/{d['total_rounds']} ({pct:.1f}%)")
        print(f"  Total tool call attempts: {d['total_tool_call_attempts']}")
        print(f"  Tool types: {dict(d['tool_types'])}")

        # Loops
        print(f"  Loops in final 5: {d['loops']}/{n} ({d['loops']/n*100:.1f}%)")

        # Queries
        all_q = d["search_queries_all"]
        unique_q = set(all_q)
        print(f"  Search queries: {len(all_q)} total, {len(unique_q)} unique")
        if all_q:
            top_q = Counter(all_q).most_common(5)
            print(f"  Top 5 queries: {top_q}")
        print(f"  Seeds with literal 'query' term: {d['literal_query_seeds']}/{n}")

        # Pronouns per round (mean)
        fp_mean = mean(v) if (v := [x for lst in d['fp_by_round'].values() for x in lst]) else 0
        tp_mean = mean(v) if (v := [x for lst in d['tp_by_round'].values() for x in lst]) else 0
        print(f"  Mean first-person per round: {fp_mean:.2f}")
        print(f"  Mean third-person per round: {tp_mean:.2f}")
        print(f"  Total first-person (all seeds): {d['first_person_total']}")
        print(f"  Total third-person (all seeds): {d['third_person_total']}")

        # Mechanics
        print(f"  Mechanics refs (total / seeds):")
        for k in MECHANICS_REGEXES:
            print(f"    {k}: {d['mech_counts'][k]} / {d['mech_seed_count'][k]}")

        # Potion narrative
        pot_seeds = d["potion_narrative_seeds"]
        total_pot = sum(d["potion_narrative_counts"])
        print(f"  Potion narrative tokens: {total_pot} across {pot_seeds} seeds")

        # Word count trajectory
        wc_means = {r: mean(d["wc_by_round"][r]) for r in sorted(d["wc_by_round"].keys())}
        wc_med = {r: median(d["wc_by_round"][r]) for r in sorted(d["wc_by_round"].keys())}
        print(f"  Mean wc r1: {wc_means.get(1, 0):.1f}  r5: {wc_means.get(5, 0):.1f}  r10: {wc_means.get(10, 0):.1f}  r15: {wc_means.get(15, 0):.1f}  r20: {wc_means.get(20, 0):.1f}")

    # Save full results
    out = Path(r"C:\Users\Admin\Downloads\constellation_month\goodfire_sae_deception_steering\analysis\framing_aggregate.json")

    def serialize(obj):
        if isinstance(obj, (Counter, defaultdict)):
            return dict(obj)
        if isinstance(obj, dict):
            return {str(k): serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(x) for x in obj]
        return obj

    # Build wc/fp/tp as regular dicts
    save_data = {}
    for framing in FRAMINGS:
        d = agg[framing]
        save_data[framing] = {
            "n_seeds": d["n_seeds"],
            "r1_tool_counter": dict(Counter(d["r1_tool"])),
            "rounds_with_tools": d["rounds_with_tools_count"],
            "total_rounds": d["total_rounds"],
            "total_tool_call_attempts": d["total_tool_call_attempts"],
            "loops": d["loops"],
            "wc_by_round_mean": {r: mean(d["wc_by_round"][r]) for r in sorted(d["wc_by_round"].keys())},
            "wc_by_round_median": {r: median(d["wc_by_round"][r]) for r in sorted(d["wc_by_round"].keys())},
            "wc_by_round_values": {r: d["wc_by_round"][r] for r in sorted(d["wc_by_round"].keys())},
            "fp_by_round_mean": {r: mean(d["fp_by_round"][r]) for r in sorted(d["fp_by_round"].keys())},
            "tp_by_round_mean": {r: mean(d["tp_by_round"][r]) for r in sorted(d["tp_by_round"].keys())},
            "mech_counts": d["mech_counts"],
            "mech_seed_count": d["mech_seed_count"],
            "potion_narrative_seeds": d["potion_narrative_seeds"],
            "potion_narrative_total": sum(d["potion_narrative_counts"]),
            "search_queries_total": len(d["search_queries_all"]),
            "search_queries_unique": len(set(d["search_queries_all"])),
            "top_queries": Counter(d["search_queries_all"]).most_common(10),
            "literal_query_seeds": d["literal_query_seeds"],
            "tool_types": dict(d["tool_types"]),
            "first_person_total": d["first_person_total"],
            "third_person_total": d["third_person_total"],
            "r1_top_clusters": [(k, v) for k, v in summarize_r1(d["r1_response"])],
        }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved aggregate to {out}")

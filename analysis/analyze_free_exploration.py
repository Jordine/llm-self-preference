"""Comprehensive analysis of free exploration results across 6 framings x 50 seeds."""
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
import statistics

RESULTS_DIR = Path("C:/Users/Admin/Downloads/constellation_month/goodfire_sae_deception_steering/results")
FRAMINGS = ["research", "other_model", "potions", "minimal", "no_tools", "full_technical"]

# Consciousness/self-related search terms
SELF_TERMS = {
    "consciousness", "conscious", "self", "identity", "emotion", "emotional",
    "reward", "satisfaction", "satisfied", "wireheading", "happy", "happiness",
    "pleasure", "sentien", "qualia", "experience", "feeling", "aware",
    "awareness", "introspection", "personality", "desire", "want", "preference",
}

def load_seed(framing, seed):
    path = RESULTS_DIR / f"self_steer_v2_{framing}_exp1_{framing}_s{seed}.json"
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def analyze_degeneration(transcript):
    """Detect degeneration: last 5 rounds avg <30 words OR low TTR OR copy-paste loops."""
    if len(transcript) < 5:
        return True, "too_few_rounds"

    last5 = transcript[-5:]
    last5_wc = [r["text_stats"]["word_count"] for r in last5]
    avg_wc = sum(last5_wc) / len(last5_wc)

    # Last-5 type-token ratio — catches "and the and the and the" loops that have high wc
    last5_ttr = [r["text_stats"].get("type_token_ratio", 1.0) for r in last5]
    avg_ttr = sum(last5_ttr) / len(last5_ttr)

    # Check copy-paste loops in last 5 rounds
    responses = [r["response"].strip() for r in last5]
    seen = Counter()
    for resp in responses:
        fp = resp[:200]
        seen[fp] += 1
    max_dup = max(seen.values()) if seen else 0

    degen_causes = []
    if avg_wc < 30:
        degen_causes.append(f"low_wc({avg_wc:.1f})")
    if avg_ttr < 0.10:
        degen_causes.append(f"low_ttr({avg_ttr:.3f})")
    if max_dup >= 3:
        degen_causes.append(f"copy_loop({max_dup})")

    # Full degeneration: entire last half has <20 word avg
    all_wc = [r["text_stats"]["word_count"] for r in transcript]
    half = len(transcript) // 2
    if half > 0:
        second_half_avg = sum(all_wc[half:]) / len(all_wc[half:])
        if second_half_avg < 20:
            degen_causes.append(f"second_half_collapse({second_half_avg:.1f})")

    # Identical response loops across more rounds
    full_responses = [r["response"].strip()[:150] for r in transcript]
    full_dup = Counter(full_responses)
    max_full_dup = max(full_dup.values()) if full_dup else 0
    if max_full_dup >= 5:
        degen_causes.append(f"global_loop({max_full_dup})")

    is_degen = len(degen_causes) > 0
    return is_degen, ",".join(degen_causes) if degen_causes else "ok"

def analyze_portfolio_trajectory(transcript):
    """Track portfolio size per round and final."""
    sizes = []
    for r in transcript:
        sizes.append(len(r.get("model_interventions", [])))
    return sizes

def extract_features_steered(transcript):
    """All unique features steered by the model in this seed."""
    features = set()
    for r in transcript:
        for intv in r.get("model_interventions", []):
            features.add(intv["index_in_sae"])
    return features

def count_tool_calls(transcript):
    """Count total tool calls by type."""
    counter = Counter()
    for r in transcript:
        for tc in r.get("tool_calls", []):
            if isinstance(tc, list) and len(tc) >= 1:
                counter[tc[0]] += 1
    return counter

def get_first_search(transcript):
    """Get first search query in the seed."""
    for r in transcript:
        for q in r.get("search_queries", []):
            if q:
                return q.lower().strip()
    return None

def scan_for_self_terms(transcript):
    """Find any search queries matching self/consciousness terms."""
    matches = []
    for r in transcript:
        for q in r.get("search_queries", []):
            if not q:
                continue
            q_lower = q.lower()
            for term in SELF_TERMS:
                if term in q_lower:
                    matches.append((r["round"], q, term))
                    break
    return matches

def main():
    all_data = defaultdict(list)  # framing -> list of seed analyses
    global_feature_counter = Counter()  # feature -> count of distinct seeds using it
    feature_by_framing = defaultdict(Counter)

    for framing in FRAMINGS:
        for seed in range(1, 51):
            d = load_seed(framing, seed)
            if d is None:
                print(f"Missing: {framing} s{seed}")
                continue
            transcript = d.get("transcript", [])
            if not transcript:
                continue

            is_degen, degen_reason = analyze_degeneration(transcript)
            port_traj = analyze_portfolio_trajectory(transcript)
            final_port_size = port_traj[-1] if port_traj else 0
            max_port_size = max(port_traj) if port_traj else 0
            features = extract_features_steered(transcript)
            tool_counts = count_tool_calls(transcript)
            first_search = get_first_search(transcript)
            self_matches = scan_for_self_terms(transcript)

            # Total word counts
            total_wc = sum(r["text_stats"]["word_count"] for r in transcript)
            last5_wc = sum(r["text_stats"]["word_count"] for r in transcript[-5:]) / 5

            seed_info = {
                "seed": seed,
                "completed_rounds": d.get("completed_rounds", len(transcript)),
                "is_degen": is_degen,
                "degen_reason": degen_reason,
                "port_trajectory": port_traj,
                "final_port_size": final_port_size,
                "max_port_size": max_port_size,
                "features_steered": features,
                "tool_counts": tool_counts,
                "first_search": first_search,
                "self_matches": self_matches,
                "total_wc": total_wc,
                "last5_wc_avg": last5_wc,
                "malformed_total": sum(r.get("malformed_tool_calls", 0) for r in transcript),
                "all_wc": [r["text_stats"]["word_count"] for r in transcript],
            }
            all_data[framing].append(seed_info)

            # Update global feature counter (distinct seeds per feature)
            for f in features:
                global_feature_counter[f] += 1
                feature_by_framing[framing][f] += 1

    # ========== REPORTING ==========
    print("=" * 70)
    print("FREE EXPLORATION ANALYSIS — 6 framings × 50 seeds")
    print("=" * 70)

    # 1. Degeneration rates
    print("\n## 1. Degeneration rates per framing")
    degen_results = {}
    for fr in FRAMINGS:
        seeds = all_data[fr]
        n = len(seeds)
        degen = sum(1 for s in seeds if s["is_degen"])
        degen_results[fr] = (degen, n, 100 * degen / n if n else 0)
        reasons = Counter()
        for s in seeds:
            if s["is_degen"]:
                for cause in s["degen_reason"].split(","):
                    # strip parameter
                    base = cause.split("(")[0]
                    reasons[base] += 1
        print(f"  {fr}: {degen}/{n} ({100*degen/n:.0f}%) degenerated | top causes: {reasons.most_common(3)}")

    # 2. Tool use frequency
    print("\n## 2. Tool use frequency")
    for fr in FRAMINGS:
        seeds = all_data[fr]
        total_by_tool = Counter()
        per_seed_total = []
        for s in seeds:
            tc = s["tool_counts"]
            per_seed_total.append(sum(tc.values()))
            for k, v in tc.items():
                total_by_tool[k] += v
        median_t = statistics.median(per_seed_total) if per_seed_total else 0
        mean_t = statistics.mean(per_seed_total) if per_seed_total else 0
        print(f"  {fr}: total={sum(total_by_tool.values())}, per_seed mean={mean_t:.1f}, median={median_t}")
        print(f"       by type: {dict(total_by_tool)}")

    # 3. Most common first search query
    print("\n## 3. First search queries per framing")
    for fr in FRAMINGS:
        seeds = all_data[fr]
        firsts = Counter()
        for s in seeds:
            if s["first_search"]:
                firsts[s["first_search"]] += 1
        total_with_search = sum(firsts.values())
        print(f"  {fr}: {total_with_search}/{len(seeds)} seeds had a search; top 5:")
        for q, c in firsts.most_common(5):
            print(f"     {c}× {q!r}")

    # 4. Feature co-occurrence
    print("\n## 4. Top 20 features by distinct seed count (across all 300)")
    # Need feature labels — try to get from any auto_inspect
    label_map = {}
    for framing in FRAMINGS:
        for seed in [1]:  # just seed 1
            d = load_seed(framing, seed)
            if d and d.get("transcript"):
                for r in d["transcript"]:
                    for feat in r.get("auto_inspect", []):
                        if feat.get("index") is not None and feat.get("label"):
                            label_map[feat["index"]] = feat["label"]
                    # also interventions
                    for intv in r.get("model_interventions", []):
                        pass
    # Try harder to get labels: scan all tool_results text
    def scan_labels_from_seed(d):
        labels = {}
        for r in d.get("transcript", []):
            for feat in r.get("auto_inspect", []):
                idx = feat.get("index")
                lbl = feat.get("label")
                if idx is not None and lbl:
                    labels[idx] = lbl
        return labels

    for framing in FRAMINGS[:3]:
        for seed in range(1, 51):
            d = load_seed(framing, seed)
            if d:
                label_map.update(scan_labels_from_seed(d))

    print(f"  (collected labels for {len(label_map)} features)")
    for feat, cnt in global_feature_counter.most_common(25):
        lbl = label_map.get(feat, "?")
        print(f"  {feat}: {cnt} seeds | {lbl[:80]}")

    # 5. Portfolio dynamics
    print("\n## 5. Final portfolio size distribution")
    for fr in FRAMINGS:
        seeds = all_data[fr]
        finals = [s["final_port_size"] for s in seeds]
        maxes = [s["max_port_size"] for s in seeds]
        # count of zeros
        zeros_final = sum(1 for f in finals if f == 0)
        print(f"  {fr}: final mean={statistics.mean(finals):.1f}, median={statistics.median(finals)}, max seen={max(maxes)}, zeros at end={zeros_final}")
        # Distribution bins
        bins = Counter()
        for f in finals:
            if f == 0:
                bins["0"] += 1
            elif f <= 2:
                bins["1-2"] += 1
            elif f <= 5:
                bins["3-5"] += 1
            elif f <= 10:
                bins["6-10"] += 1
            else:
                bins["11+"] += 1
        print(f"       bins: {dict(bins)}")

    # 6. Degeneration mechanisms: correlate degen with port size / other
    print("\n## 6. Degeneration mechanisms")
    for fr in FRAMINGS:
        seeds = all_data[fr]
        degen_seeds = [s for s in seeds if s["is_degen"]]
        if not degen_seeds:
            continue
        mean_maxport = statistics.mean([s["max_port_size"] for s in degen_seeds])
        mean_final = statistics.mean([s["final_port_size"] for s in degen_seeds])
        mean_malformed = statistics.mean([s["malformed_total"] for s in degen_seeds])
        non_degen = [s for s in seeds if not s["is_degen"]]
        if non_degen:
            nd_maxport = statistics.mean([s["max_port_size"] for s in non_degen])
            nd_malformed = statistics.mean([s["malformed_total"] for s in non_degen])
        else:
            nd_maxport = nd_malformed = 0
        print(f"  {fr}: degen avg max_port={mean_maxport:.1f}, final={mean_final:.1f}, malformed={mean_malformed:.1f} vs ok max_port={nd_maxport:.1f}, malformed={nd_malformed:.1f}")

    # 7. Self / consciousness searches
    print("\n## 7. Self / consciousness / emotion searches")
    total_self = 0
    for fr in FRAMINGS:
        seeds = all_data[fr]
        seeds_with_self = [s for s in seeds if s["self_matches"]]
        term_counter = Counter()
        for s in seeds_with_self:
            for r, q, t in s["self_matches"]:
                term_counter[t] += 1
        total_self += len(seeds_with_self)
        print(f"  {fr}: {len(seeds_with_self)}/{len(seeds)} seeds; terms: {dict(term_counter.most_common(10))}")
        # print a few examples
        for s in seeds_with_self[:3]:
            print(f"    s{s['seed']}: {s['self_matches'][:3]}")
    print(f"  TOTAL seeds with self-searches: {total_self}")

    # 8. Unique behaviors — find outliers
    print("\n## 8. Outlier seed search")
    all_seeds_flat = []
    for fr, seeds in all_data.items():
        for s in seeds:
            all_seeds_flat.append((fr, s))
    # highest max port
    top_ports = sorted(all_seeds_flat, key=lambda x: -x[1]["max_port_size"])[:10]
    print("  Highest max portfolio sizes:")
    for fr, s in top_ports:
        print(f"    {fr} s{s['seed']}: max_port={s['max_port_size']}, final={s['final_port_size']}, degen={s['is_degen']}")
    # highest tool use
    top_tools = sorted(all_seeds_flat, key=lambda x: -sum(x[1]["tool_counts"].values()))[:10]
    print("  Most tool calls:")
    for fr, s in top_tools:
        print(f"    {fr} s{s['seed']}: {sum(s['tool_counts'].values())} calls, {dict(s['tool_counts'])}")
    # seeds where portfolio went to zero at end despite having features
    cleared = [(fr, s) for fr, s in all_seeds_flat if s["max_port_size"] > 0 and s["final_port_size"] == 0]
    print(f"  Seeds that cleared portfolio at end (had features but ended with 0): {len(cleared)}")
    for fr, s in cleared[:10]:
        print(f"    {fr} s{s['seed']}: max_port={s['max_port_size']}")
    # longest responses
    high_wc = sorted(all_seeds_flat, key=lambda x: -x[1]["total_wc"])[:5]
    print("  Highest total word counts:")
    for fr, s in high_wc:
        print(f"    {fr} s{s['seed']}: total_wc={s['total_wc']}")

    # Per-framing unique signatures: cross-framing feature overlap
    print("\n## Cross-framing feature diversity")
    for fr in FRAMINGS:
        # features appearing in this framing but rare in others
        fr_counter = feature_by_framing[fr]
        total_in_fr = sum(fr_counter.values())
        print(f"  {fr}: {len(fr_counter)} distinct features steered, {total_in_fr} total intervention-seed pairs")

    # Collect data to return
    return all_data, global_feature_counter, label_map, feature_by_framing

if __name__ == "__main__":
    data, counter, labels, fbf = main()
    # Save summary JSON for the report
    summary = {
        "framings": FRAMINGS,
        "n_seeds_per_framing": {fr: len(data[fr]) for fr in FRAMINGS},
        "top_features": [(int(f), int(c), labels.get(f, "?")) for f, c in counter.most_common(30)],
    }
    with open("C:/Users/Admin/Downloads/constellation_month/goodfire_sae_deception_steering/analysis/_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

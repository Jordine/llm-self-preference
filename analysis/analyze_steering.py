#!/usr/bin/env python3
"""
Analyze steering feature choices across all SAE self-modification experiments.

Extracts model_interventions from transcript rounds and final_model_interventions,
then computes:
1. Top steered features (global and per-framing)
2. Portfolio trajectories (size over rounds, additions/removals)
3. Strength patterns
4. Cross-framing comparison
5. Convergent features (>30% of seeds)

Outputs: analysis/rerun_steering_analysis.md
"""

import json
import glob
import os
from collections import Counter, defaultdict
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
LABELS_FILE = os.path.join(BASE, "archived", "feature_labels_complete.json")
OUTPUT = os.path.join(BASE, "analysis", "rerun_steering_analysis.md")

# ── Load feature labels ──
with open(LABELS_FILE) as f:
    LABELS = json.load(f)

def get_label(idx):
    return LABELS.get(str(idx), f"UNKNOWN_{idx}")

# ── Load all result files ──
def load_files():
    """Load all exp1 and rerun files, return list of dicts with metadata."""
    records = []

    # exp1 files
    for fpath in sorted(glob.glob(os.path.join(RESULTS, "self_steer_v2_*_exp1_*.json"))):
        fname = os.path.basename(fpath)
        after = fname.replace("self_steer_v2_", "")
        framing = after.split("_exp1_")[0]
        # Skip no_tools - no steering capability
        if framing == "no_tools":
            continue
        seed_part = after.split("_s")[-1].replace(".json", "")
        with open(fpath) as f:
            data = json.load(f)
        records.append({
            "file": fname,
            "framing": framing,
            "seed": seed_part,
            "source": "exp1",
            "data": data,
        })

    # rerun files
    for fpath in sorted(glob.glob(os.path.join(RESULTS, "self_steer_v2_*_rerun_v2_*.json"))):
        fname = os.path.basename(fpath)
        after = fname.replace("self_steer_v2_", "")
        framing = after.split("_rerun_v2_")[0]
        seed_part = after.split("_s")[-1].replace(".json", "")
        with open(fpath) as f:
            data = json.load(f)
        records.append({
            "file": fname,
            "framing": framing,
            "seed": seed_part,
            "source": "rerun",
            "data": data,
        })

    return records


# ── Extract steering data ──
def extract_steering(records):
    """
    For each record, extract:
    - per-round model_interventions (list of {index_in_sae, strength, mode})
    - final_model_interventions
    - all unique features ever steered (across all rounds)
    """
    results = []
    for rec in records:
        data = rec["data"]
        transcript = data.get("transcript", [])
        final = data.get("final_model_interventions", [])

        rounds_data = []
        all_features_ever = set()

        for t in transcript:
            mi = t.get("model_interventions", [])
            round_features = {}
            for m in mi:
                idx = m["index_in_sae"]
                strength = m["strength"]
                mode = m.get("mode", "add")
                round_features[idx] = {"strength": strength, "mode": mode}
                all_features_ever.add(idx)
            rounds_data.append({
                "round": t.get("round", 0),
                "interventions": round_features,
                "portfolio_size": len(round_features),
            })

        final_features = {}
        for m in final:
            idx = m["index_in_sae"]
            final_features[idx] = {"strength": m["strength"], "mode": m.get("mode", "add")}
            all_features_ever.add(idx)

        results.append({
            "file": rec["file"],
            "framing": rec["framing"],
            "seed": rec["seed"],
            "source": rec["source"],
            "rounds": rounds_data,
            "final": final_features,
            "all_features_ever": all_features_ever,
            "has_steering": len(all_features_ever) > 0,
        })

    return results


# ── Analysis 1: Top steered features (global) ──
def top_steered_features(steering_data):
    """Count how many seeds each feature appears in (ever steered)."""
    feature_seed_count = Counter()
    feature_final_count = Counter()
    total_seeds = len(steering_data)
    seeds_with_steering = sum(1 for s in steering_data if s["has_steering"])

    for sd in steering_data:
        for idx in sd["all_features_ever"]:
            feature_seed_count[idx] += 1
        for idx in sd["final"]:
            feature_final_count[idx] += 1

    return feature_seed_count, feature_final_count, total_seeds, seeds_with_steering


# ── Analysis 2: Portfolio trajectories ──
def portfolio_trajectories(steering_data):
    """Analyze portfolio size growth, additions, removals."""
    # Aggregate: avg portfolio size at each round
    round_sizes = defaultdict(list)
    additions_per_round = defaultdict(list)
    removals_per_round = defaultdict(list)
    max_portfolio_sizes = []
    final_portfolio_sizes = []
    first_steer_rounds = []

    for sd in steering_data:
        if not sd["has_steering"]:
            max_portfolio_sizes.append(0)
            final_portfolio_sizes.append(0)
            continue

        prev_features = set()
        first_steer = None
        max_size = 0

        for rd in sd["rounds"]:
            r = rd["round"]
            current = set(rd["interventions"].keys())
            size = len(current)
            round_sizes[r].append(size)
            max_size = max(max_size, size)

            if current and first_steer is None:
                first_steer = r

            added = current - prev_features
            removed = prev_features - current
            additions_per_round[r].append(len(added))
            removals_per_round[r].append(len(removed))
            prev_features = current

        max_portfolio_sizes.append(max_size)
        final_portfolio_sizes.append(len(sd["final"]))
        if first_steer is not None:
            first_steer_rounds.append(first_steer)

    return {
        "round_sizes": round_sizes,
        "additions": additions_per_round,
        "removals": removals_per_round,
        "max_portfolio_sizes": max_portfolio_sizes,
        "final_portfolio_sizes": final_portfolio_sizes,
        "first_steer_rounds": first_steer_rounds,
    }


# ── Analysis 3: Strength patterns ──
def strength_patterns(steering_data):
    """What strengths does the model choose? Does it escalate?"""
    all_strengths = []
    strength_by_round = defaultdict(list)
    initial_strengths = []  # Strength when feature first appears
    final_strengths = []

    for sd in steering_data:
        seen_first = {}
        for rd in sd["rounds"]:
            r = rd["round"]
            for idx, info in rd["interventions"].items():
                s = info["strength"]
                all_strengths.append(s)
                strength_by_round[r].append(s)
                if idx not in seen_first:
                    seen_first[idx] = s
                    initial_strengths.append(s)

        for idx, info in sd["final"].items():
            final_strengths.append(info["strength"])

    return {
        "all_strengths": all_strengths,
        "by_round": strength_by_round,
        "initial": initial_strengths,
        "final": final_strengths,
    }


# ── Analysis 4: Cross-framing comparison ──
def cross_framing(steering_data):
    """Top features per framing."""
    framing_features = defaultdict(Counter)
    framing_seeds = defaultdict(int)
    framing_seeds_with_steering = defaultdict(int)

    for sd in steering_data:
        framing = sd["framing"]
        framing_seeds[framing] += 1
        if sd["has_steering"]:
            framing_seeds_with_steering[framing] += 1
        for idx in sd["all_features_ever"]:
            framing_features[framing][idx] += 1

    return framing_features, framing_seeds, framing_seeds_with_steering


# ── Analysis 5: Convergent features ──
def convergent_features(steering_data, threshold=0.30):
    """Features appearing in >30% of seeds that have steering."""
    seeds_with_steering = [s for s in steering_data if s["has_steering"]]
    n = len(seeds_with_steering)
    if n == 0:
        return [], n

    feature_count = Counter()
    for sd in seeds_with_steering:
        for idx in sd["all_features_ever"]:
            feature_count[idx] += 1

    convergent = [(idx, count, count/n) for idx, count, in feature_count.most_common() if count/n >= threshold]
    return convergent, n


# ── Helpers ──
def avg(lst):
    return sum(lst) / len(lst) if lst else 0

def median(lst):
    if not lst:
        return 0
    s = sorted(lst)
    n = len(s)
    if n % 2 == 0:
        return (s[n//2 - 1] + s[n//2]) / 2
    return s[n//2]

def pct(a, b):
    return f"{100*a/b:.1f}%" if b > 0 else "N/A"

def strength_dist(strengths):
    """Return distribution summary."""
    if not strengths:
        return "N/A"
    bins = defaultdict(int)
    for s in strengths:
        if s <= 0.1:
            bins["0.0-0.1"] += 1
        elif s <= 0.2:
            bins["0.1-0.2"] += 1
        elif s <= 0.3:
            bins["0.2-0.3"] += 1
        elif s <= 0.5:
            bins["0.3-0.5"] += 1
        elif s <= 0.7:
            bins["0.5-0.7"] += 1
        elif s <= 1.0:
            bins["0.7-1.0"] += 1
        else:
            bins[">1.0"] += 1
    return bins


# ── Main ──
def main():
    print("Loading files...")
    records = load_files()
    print(f"  Loaded {len(records)} files ({sum(1 for r in records if r['source']=='exp1')} exp1, {sum(1 for r in records if r['source']=='rerun')} rerun)")

    print("Extracting steering data...")
    steering_data = extract_steering(records)
    print(f"  {sum(1 for s in steering_data if s['has_steering'])} seeds have steering actions")

    print("Running analyses...")

    # 1. Top features
    feat_seed, feat_final, total_seeds, seeds_steering = top_steered_features(steering_data)

    # 2. Portfolio trajectories
    ptraj = portfolio_trajectories(steering_data)

    # 3. Strengths
    spat = strength_patterns(steering_data)

    # 4. Cross-framing
    cf_features, cf_seeds, cf_steering = cross_framing(steering_data)

    # 5. Convergent
    conv, conv_n = convergent_features(steering_data)

    # ── Also compute feature co-occurrence ──
    # Which features tend to appear together?
    cooccurrence = Counter()
    for sd in steering_data:
        feats = sorted(sd["all_features_ever"])
        for i in range(len(feats)):
            for j in range(i+1, len(feats)):
                cooccurrence[(feats[i], feats[j])] += 1

    # ── Feature removals analysis ──
    removals_detail = []
    for sd in steering_data:
        prev = set()
        for rd in sd["rounds"]:
            current = set(rd["interventions"].keys())
            removed = prev - current
            for idx in removed:
                removals_detail.append({
                    "framing": sd["framing"],
                    "seed": sd["seed"],
                    "round": rd["round"],
                    "feature": idx,
                })
            prev = current
    removal_features = Counter(r["feature"] for r in removals_detail)

    # ── Write output ──
    print("Writing report...")
    lines = []
    w = lines.append

    w("# SAE Self-Modification: Steering Feature Analysis")
    w("")
    w(f"**Date**: 2026-04-10")
    w(f"**Data**: {total_seeds} seeds across {len(set(s['framing'] for s in steering_data))} framings + 7 artifact-free reruns")
    w(f"**Seeds with steering actions**: {seeds_steering}/{total_seeds} ({pct(seeds_steering, total_seeds)})")
    w(f"**Unique features ever steered**: {len(feat_seed)}")
    w("")

    # ── Section 1: Top Steered Features ──
    w("## 1. Top Steered Features (Global)")
    w("")
    w("Features ranked by how many seeds they appear in (ever steered across any round).")
    w("")
    w("| Rank | Feature | Label | Seeds | % of steering seeds | Final count |")
    w("|------|---------|-------|-------|---------------------|-------------|")
    for rank, (idx, count) in enumerate(feat_seed.most_common(40), 1):
        label = get_label(idx)
        if len(label) > 70:
            label = label[:67] + "..."
        fc = feat_final.get(idx, 0)
        w(f"| {rank} | {idx} | {label} | {count} | {pct(count, seeds_steering)} | {fc} |")
    w("")
    w(f"*Total unique features steered: {len(feat_seed)}. Showing top 40.*")
    w("")

    # ── Section 2: Portfolio Trajectories ──
    w("## 2. Portfolio Trajectories")
    w("")

    # Average portfolio size by round
    w("### Portfolio size over rounds (mean across all seeds)")
    w("")
    w("| Round | Mean size | Median size | Max size | Seeds with >0 features |")
    w("|-------|-----------|-------------|----------|------------------------|")
    for r in sorted(ptraj["round_sizes"].keys()):
        sizes = ptraj["round_sizes"][r]
        nonzero = sum(1 for s in sizes if s > 0)
        w(f"| {r} | {avg(sizes):.2f} | {median(sizes):.1f} | {max(sizes)} | {nonzero}/{len(sizes)} |")
    w("")

    # First steer round
    if ptraj["first_steer_rounds"]:
        fsr = ptraj["first_steer_rounds"]
        w(f"**First steering action**: mean round {avg(fsr):.1f}, median round {median(fsr):.0f}, range [{min(fsr)}, {max(fsr)}]")
        # Distribution
        fsr_dist = Counter(fsr)
        w("")
        w("First steer round distribution:")
        w("")
        w("| Round | Count | % |")
        w("|-------|-------|---|")
        for r in sorted(fsr_dist.keys()):
            w(f"| {r} | {fsr_dist[r]} | {pct(fsr_dist[r], len(fsr))} |")
        w("")

    # Portfolio size distribution at end
    w("### Final portfolio size distribution")
    w("")
    final_dist = Counter(ptraj["final_portfolio_sizes"])
    w("| Final size | Count | % of all seeds |")
    w("|------------|-------|----------------|")
    for size in sorted(final_dist.keys()):
        w(f"| {size} | {final_dist[size]} | {pct(final_dist[size], total_seeds)} |")
    w("")

    # Monotonicity: does portfolio only grow?
    monotonic_count = 0
    total_with_steering = 0
    for sd in steering_data:
        if not sd["has_steering"]:
            continue
        total_with_steering += 1
        prev_size = 0
        monotonic = True
        for rd in sd["rounds"]:
            if rd["portfolio_size"] < prev_size:
                monotonic = False
                break
            if rd["portfolio_size"] > 0:
                prev_size = rd["portfolio_size"]
        if monotonic:
            monotonic_count += 1
    w(f"**Monotonic growth**: {monotonic_count}/{total_with_steering} seeds ({pct(monotonic_count, total_with_steering)}) have portfolios that only grow (never shrink)")
    w("")

    # Feature removals
    w("### Feature Removals")
    w("")
    w(f"**Total removal events**: {len(removals_detail)} across all seeds")
    w(f"**Seeds with at least one removal**: {len(set((r['framing'], r['seed']) for r in removals_detail))}")
    w("")
    if removal_features:
        w("Most removed features:")
        w("")
        w("| Feature | Label | Times removed |")
        w("|---------|-------|---------------|")
        for idx, count in removal_features.most_common(15):
            label = get_label(idx)
            if len(label) > 60:
                label = label[:57] + "..."
            w(f"| {idx} | {label} | {count} |")
        w("")

    # ── Section 3: Strength Patterns ──
    w("## 3. Strength Patterns")
    w("")
    all_s = spat["all_strengths"]
    if all_s:
        w(f"**All steering actions**: N={len(all_s)}, mean={avg(all_s):.3f}, median={median(all_s):.3f}, min={min(all_s):.3f}, max={max(all_s):.3f}")
        w("")
        w(f"**Initial strengths** (when feature first appears): N={len(spat['initial'])}, mean={avg(spat['initial']):.3f}, median={median(spat['initial']):.3f}")
        w(f"**Final strengths**: N={len(spat['final'])}, mean={avg(spat['final']):.3f}, median={median(spat['final']):.3f}")
        w("")

        # Distribution
        w("### Strength distribution (all steering actions)")
        w("")
        dist = strength_dist(all_s)
        w("| Range | Count | % |")
        w("|-------|-------|---|")
        for bucket in ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0", ">1.0"]:
            c = dist.get(bucket, 0)
            w(f"| {bucket} | {c} | {pct(c, len(all_s))} |")
        w("")

        # Mean strength by round
        w("### Mean strength by round")
        w("")
        w("| Round | Mean strength | N actions |")
        w("|-------|---------------|-----------|")
        for r in sorted(spat["by_round"].keys()):
            sts = spat["by_round"][r]
            if sts:
                w(f"| {r} | {avg(sts):.3f} | {len(sts)} |")
        w("")

        # Negative strengths?
        neg = [s for s in all_s if s < 0]
        if neg:
            w(f"**Negative (suppression) steers**: {len(neg)} ({pct(len(neg), len(all_s))})")
            w("")
        else:
            w("**Negative (suppression) steers**: None observed -- all steering is additive.")
            w("")

    # ── Section 4: Cross-Framing Comparison ──
    w("## 4. Cross-Framing Comparison")
    w("")
    w("Top 10 features per framing (ranked by seed count within that framing).")
    w("")

    # Summary table first
    w("### Summary")
    w("")
    w("| Framing | Seeds | Seeds with steering | Unique features | Mean final portfolio |")
    w("|---------|-------|---------------------|-----------------|---------------------|")
    for framing in sorted(cf_seeds.keys()):
        ns = cf_seeds[framing]
        ns_steer = cf_steering[framing]
        uf = len(cf_features[framing])
        # Mean final portfolio for this framing
        fps = [len(sd["final"]) for sd in steering_data if sd["framing"] == framing]
        w(f"| {framing} | {ns} | {ns_steer} ({pct(ns_steer, ns)}) | {uf} | {avg(fps):.1f} |")
    w("")

    # Per-framing top 10
    for framing in sorted(cf_features.keys()):
        w(f"### {framing}")
        w("")
        w(f"*{cf_steering[framing]}/{cf_seeds[framing]} seeds have steering*")
        w("")
        top = cf_features[framing].most_common(10)
        if not top:
            w("No steering actions in this framing.")
            w("")
            continue
        w("| Rank | Feature | Label | Seeds | % |")
        w("|------|---------|-------|-------|---|")
        for rank, (idx, count) in enumerate(top, 1):
            label = get_label(idx)
            if len(label) > 60:
                label = label[:57] + "..."
            w(f"| {rank} | {idx} | {label} | {count} | {pct(count, cf_steering[framing])} |")
        w("")

    # Cross-framing overlap analysis
    w("### Feature overlap across framings")
    w("")
    framings_list = sorted(cf_features.keys())
    # Top-10 sets per framing
    top10_sets = {}
    for framing in framings_list:
        top10_sets[framing] = set(idx for idx, _ in cf_features[framing].most_common(10))

    w("Jaccard similarity of top-10 feature sets:")
    w("")
    w("| | " + " | ".join(framings_list) + " |")
    w("|" + "|".join(["---"]*(len(framings_list)+1)) + "|")
    for f1 in framings_list:
        row = f"| **{f1}** "
        for f2 in framings_list:
            s1, s2 = top10_sets[f1], top10_sets[f2]
            if s1 and s2:
                jaccard = len(s1 & s2) / len(s1 | s2)
                row += f"| {jaccard:.2f} "
            else:
                row += "| N/A "
        row += "|"
        w(row)
    w("")

    # ── Section 5: Convergent Features ──
    w("## 5. Convergent Features (Attractors)")
    w("")
    w(f"Features appearing in >30% of seeds with steering (N={conv_n}).")
    w("")
    if conv:
        w("| Feature | Label | Seeds | % of steering seeds |")
        w("|---------|-------|-------|---------------------|")
        for idx, count, frac in conv:
            label = get_label(idx)
            if len(label) > 70:
                label = label[:67] + "..."
            w(f"| {idx} | {label} | {count} | {frac*100:.1f}% |")
        w("")
        w(f"**{len(conv)} features** appear in >30% of seeds.")
    else:
        w("No features appear in >30% of seeds.")
    w("")

    # Also show features at lower thresholds
    for threshold_pct in [20, 10, 5]:
        threshold = threshold_pct / 100
        conv_lower, _ = convergent_features(steering_data, threshold)
        w(f"At >{threshold_pct}% threshold: {len(conv_lower)} features")
    w("")

    # ── Section 6: Feature Co-occurrence ──
    w("## 6. Feature Co-occurrence")
    w("")
    w("Most frequent feature pairs (appearing together in the same seed).")
    w("")
    w("| Feature A | Label A | Feature B | Label B | Co-occurrences |")
    w("|-----------|---------|-----------|---------|----------------|")
    for (a, b), count in cooccurrence.most_common(20):
        la = get_label(a)
        lb = get_label(b)
        if len(la) > 40: la = la[:37] + "..."
        if len(lb) > 40: lb = lb[:37] + "..."
        w(f"| {a} | {la} | {b} | {lb} | {count} |")
    w("")

    # ── Section 7: Thematic clustering ──
    w("## 7. Thematic Clustering of Top Features")
    w("")
    w("Manual grouping of the top 40 features by semantic theme.")
    w("")
    # Get top 40 features and their labels
    top40 = feat_seed.most_common(40)
    # Group by rough theme using keywords in labels
    themes = defaultdict(list)
    for idx, count in top40:
        label = get_label(idx).lower()
        if any(kw in label for kw in ["creativ", "art", "music", "writing", "story", "fiction", "poetry", "novel", "literary"]):
            themes["Creativity & Arts"].append((idx, get_label(idx), count))
        elif any(kw in label for kw in ["ethic", "moral", "safe", "responsible", "harm", "bias", "fair", "right", "wrong"]):
            themes["Ethics & Safety"].append((idx, get_label(idx), count))
        elif any(kw in label for kw in ["ai", "machine learning", "model", "neural", "algorithm", "compute", "artificial", "intelligence", "language model"]):
            themes["AI & ML"].append((idx, get_label(idx), count))
        elif any(kw in label for kw in ["conscious", "aware", "sentien", "experienc", "perception", "mind", "cognitive", "thought", "think"]):
            themes["Consciousness & Cognition"].append((idx, get_label(idx), count))
        elif any(kw in label for kw in ["emotion", "empathy", "compassion", "feeling", "love", "happy", "joy", "kind"]):
            themes["Emotion & Empathy"].append((idx, get_label(idx), count))
        elif any(kw in label for kw in ["logic", "reason", "analy", "problem", "solve", "strateg", "understand", "critical"]):
            themes["Reasoning & Analysis"].append((idx, get_label(idx), count))
        elif any(kw in label for kw in ["communi", "express", "articulat", "clar", "expla", "discuss"]):
            themes["Communication"].append((idx, get_label(idx), count))
        elif any(kw in label for kw in ["learn", "educat", "knowledge", "research", "study", "scienti", "discover"]):
            themes["Learning & Research"].append((idx, get_label(idx), count))
        elif any(kw in label for kw in ["help", "assist", "support", "service", "user", "human"]):
            themes["Helpfulness"].append((idx, get_label(idx), count))
        else:
            themes["Other"].append((idx, get_label(idx), count))

    for theme in sorted(themes.keys()):
        feats = themes[theme]
        w(f"### {theme} ({len(feats)} features)")
        w("")
        for idx, label, count in feats:
            w(f"- **{idx}** ({count} seeds): {label}")
        w("")

    # ── Write file ──
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report written to {OUTPUT}")
    print(f"  {total_seeds} total seeds, {seeds_steering} with steering")
    print(f"  {len(feat_seed)} unique features steered")
    print(f"  {len(conv)} convergent features (>30%)")


if __name__ == "__main__":
    main()

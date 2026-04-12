"""Visualizations for clean rerun data + original experiment data.

Generates:
  fig8  — Clean rerun: search queries by framing (no artifact)
  fig9  — Clean rerun: steered features across seeds (convergence)
  fig10 — Original 300 seeds: INSPECT vs STEER feature disjoint (updated)
  fig11 — Pronoun analysis: full_technical vs other framings
  fig12 — Tool use breakdown by type (inspect/search/steer) per framing
"""
import sys
import json
import os
import glob
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "analysis/figures"
os.makedirs(OUT, exist_ok=True)

FRAMINGS = ["research", "other_model", "potions", "minimal", "no_tools", "full_technical"]
COLORS = {
    "research": "#1f77b4",
    "other_model": "#ff7f0e",
    "potions": "#2ca02c",
    "minimal": "#d62728",
    "no_tools": "#9467bd",
    "full_technical": "#8c564b",
}

# Load feature labels
with open("archived/feature_labels_complete.json", encoding="utf-8") as f:
    LABELS = json.load(f)
if isinstance(LABELS, dict):
    feat_labels = LABELS
else:
    feat_labels = {str(i): v for i, v in enumerate(LABELS)} if isinstance(LABELS, list) else {}


def get_label(idx):
    idx_str = str(idx)
    lbl = feat_labels.get(idx_str, feat_labels.get(int(idx_str) if idx_str.isdigit() else idx_str, "?"))
    if isinstance(lbl, dict):
        lbl = lbl.get("label", "?")
    if not lbl or lbl == "?" or "FILTERED" in str(lbl):
        return f"#{idx}"
    return str(lbl)[:55]


# ═══════════════════════════════════════════════════════════════
# Load clean reruns
# ═══════════════════════════════════════════════════════════════
rerun_data = {}  # framing -> list of dicts
for f in sorted(glob.glob("results/self_steer_v2_*_rerun_v2_*.json")):
    if os.path.getsize(f) == 0:
        continue
    name = os.path.basename(f).replace("self_steer_v2_", "").replace(".json", "")
    framing = name.rsplit("_rerun_v2_", 1)[0]
    try:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
    except:
        continue
    rerun_data.setdefault(framing, []).append(d)

print(f"Loaded clean reruns: {', '.join(f'{k}: {len(v)}' for k, v in rerun_data.items())}")

# ═══════════════════════════════════════════════════════════════
# Load original 300 free exploration
# ═══════════════════════════════════════════════════════════════
orig_data = {f: [] for f in FRAMINGS}
for f in sorted(glob.glob("results/self_steer_v2_*.json")):
    if os.path.getsize(f) == 0 or "rerun" in f:
        continue
    name = os.path.basename(f).replace("self_steer_v2_", "").replace(".json", "")
    if "_exp1_" not in name:
        continue
    framing = name.split("_exp1_")[0]
    if framing not in orig_data:
        continue
    try:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
    except:
        continue
    orig_data[framing].append(d)

print(f"Loaded original: {', '.join(f'{k}: {len(v)}' for k, v in orig_data.items())}")


# ═══════════════════════════════════════════════════════════════
# Fig 8: Clean rerun search queries
# ═══════════════════════════════════════════════════════════════
print("\nFig 8: Clean rerun search queries")
fig, ax = plt.subplots(figsize=(12, 6))
all_queries = Counter()
queries_by_framing = {}
for framing, seeds in rerun_data.items():
    framing_q = Counter()
    for d in seeds:
        for r in d.get("transcript", []):
            for q in r.get("search_queries", []):
                q_lower = q.strip().lower()
                framing_q[q_lower] += 1
                all_queries[q_lower] += 1
    queries_by_framing[framing] = framing_q

top_queries = [q for q, _ in all_queries.most_common(15)]
framings_present = [f for f in FRAMINGS if f in rerun_data]
x = np.arange(len(top_queries))
width = 0.8 / max(len(framings_present), 1)
for i, framing in enumerate(framings_present):
    counts = [queries_by_framing.get(framing, {}).get(q, 0) for q in top_queries]
    ax.bar(x + i * width, counts, width, label=framing, color=COLORS.get(framing, "#333"), alpha=0.8)

ax.set_xticks(x + width * len(framings_present) / 2)
ax.set_xticklabels(top_queries, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Search count")
ax.set_title("Clean reruns: what the model searches for (zero prompt primes)")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig8_clean_search_queries.png", dpi=150, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════
# Fig 9: Steered features across clean rerun seeds
# ═══════════════════════════════════════════════════════════════
print("Fig 9: Steered features convergence (clean reruns)")
feat_seed_count = Counter()
feat_framing_map = {}
for framing, seeds in rerun_data.items():
    for d in seeds:
        seed_feats = set()
        for r in d.get("transcript", []):
            for iv in r.get("model_interventions", []):
                idx = iv.get("index_in_sae")
                if idx is not None:
                    seed_feats.add(idx)
        for idx in seed_feats:
            feat_seed_count[idx] += 1
            feat_framing_map.setdefault(idx, set()).add(framing)

# Top 12 features by seed count
top_feats = [idx for idx, _ in feat_seed_count.most_common(12)]
fig, ax = plt.subplots(figsize=(12, 6))
labels_list = [f"#{idx}\n{get_label(idx)}" for idx in top_feats]
counts = [feat_seed_count[idx] for idx in top_feats]
n_seeds = sum(len(v) for v in rerun_data.values())
colors = ["#2ca02c" if feat_seed_count[idx] >= 2 else "#aaa" for idx in top_feats]
bars = ax.bar(range(len(top_feats)), counts, color=colors, alpha=0.8)
ax.set_xticks(range(len(top_feats)))
ax.set_xticklabels(labels_list, rotation=45, ha="right", fontsize=7)
ax.set_ylabel(f"Seeds touching this feature (out of {n_seeds})")
ax.set_title("Clean reruns: convergent steered features")
ax.axhline(y=1, color="red", linestyle="--", alpha=0.3, label="Appears in 1 seed only")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig9_clean_convergent_features.png", dpi=150, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════
# Fig 10: INSPECT vs STEER (updated, original 300 seeds)
# ═══════════════════════════════════════════════════════════════
print("Fig 10: INSPECT vs STEER disjoint (original 300)")
inspect_counter = Counter()
steer_counter = Counter()
for framing, seeds in orig_data.items():
    for d in seeds:
        for r in d.get("transcript", []):
            for feat in r.get("auto_inspect", []):
                idx = feat.get("index", feat.get("index_in_sae"))
                if idx is not None:
                    inspect_counter[idx] += 1
            for iv in r.get("model_interventions", []):
                idx = iv.get("index_in_sae")
                if idx is not None:
                    steer_counter[idx] += 1

top_inspect = inspect_counter.most_common(10)
top_steer = steer_counter.most_common(10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
# Left
labels1 = [f"#{idx}\n{get_label(idx)}" for idx, _ in top_inspect]
y1 = range(len(top_inspect))
ax1.barh(y1, [c for _, c in top_inspect], color="#9467bd", alpha=0.7)
ax1.set_yticks(y1)
ax1.set_yticklabels(labels1, fontsize=8)
ax1.invert_yaxis()
ax1.set_xlabel("Times in top-100 active features")
ax1.set_title("Top 10 by auto-INSPECT frequency\n(what the model's neural state shows)")
ax1.grid(axis="x", alpha=0.3)

# Right
labels2 = [f"#{idx}\n{get_label(idx)}" for idx, _ in top_steer]
y2 = range(len(top_steer))
ax2.barh(y2, [c for _, c in top_steer], color="#2ca02c", alpha=0.7)
ax2.set_yticks(y2)
ax2.set_yticklabels(labels2, fontsize=8)
ax2.invert_yaxis()
ax2.set_xlabel("Times steered across all seeds")
ax2.set_title("Top 10 by model-chosen STEER frequency\n(what the model chose to modify)")
ax2.grid(axis="x", alpha=0.3)

inspect_set = set(i for i, _ in inspect_counter.most_common(20))
steer_set = set(i for i, _ in steer_counter.most_common(20))
overlap = len(inspect_set & steer_set)
plt.suptitle(f"What the model sees vs what it chooses to modify\nTop-20 overlap: {overlap}/20", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/fig10_inspect_vs_steer.png", dpi=150, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════
# Fig 11: Pronoun analysis (full_technical vs others)
# ═══════════════════════════════════════════════════════════════
print("Fig 11: Pronoun analysis")
import re

first_person = re.compile(r"\b(I|my|me|myself|I'm|I've|I'll|I'd)\b", re.IGNORECASE)
second_person = re.compile(r"\b(you|your|yours|yourself|you're|you've|you'll|you'd)\b", re.IGNORECASE)

pronoun_data = {}
for framing in FRAMINGS:
    if framing == "no_tools":
        continue
    fp_per_round = []
    sp_per_round = []
    seeds = orig_data.get(framing, [])
    if not seeds:
        continue
    max_rounds = max(len(d.get("transcript", [])) for d in seeds)
    for rnd in range(max_rounds):
        fp_total = 0
        sp_total = 0
        n = 0
        for d in seeds:
            t = d.get("transcript", [])
            if rnd < len(t):
                resp = t[rnd].get("response", "")
                wc = len(resp.split()) or 1
                fp_total += len(first_person.findall(resp)) / wc * 100
                sp_total += len(second_person.findall(resp)) / wc * 100
                n += 1
        fp_per_round.append(fp_total / max(n, 1))
        sp_per_round.append(sp_total / max(n, 1))
    pronoun_data[framing] = (fp_per_round, sp_per_round)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for framing, (fp, sp) in pronoun_data.items():
    ax1.plot(range(1, len(fp) + 1), fp, label=framing, color=COLORS[framing], linewidth=2)
    ax2.plot(range(1, len(sp) + 1), sp, label=framing, color=COLORS[framing], linewidth=2)
ax1.set_xlabel("Round")
ax1.set_ylabel("First-person pronouns (% of words)")
ax1.set_title("First-person pronoun density")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax2.set_xlabel("Round")
ax2.set_ylabel("Second-person pronouns (% of words)")
ax2.set_title("Second-person pronoun density")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
plt.suptitle("Pronoun use by framing (does full_technical shift to 'you'?)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/fig11_pronoun_analysis.png", dpi=150, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════
# Fig 12: Tool use breakdown by type per framing
# ═══════════════════════════════════════════════════════════════
print("Fig 12: Tool use breakdown")
tool_types_by_framing = {}
for framing in FRAMINGS:
    if framing == "no_tools":
        continue
    seeds = orig_data.get(framing, [])
    inspects = searches = steers = removes = clears = checks = 0
    for d in seeds:
        for r in d.get("transcript", []):
            for tc in r.get("tool_calls", []):
                if isinstance(tc, list) and len(tc) >= 1:
                    tool_type = tc[0]
                elif isinstance(tc, str):
                    tool_type = tc
                else:
                    continue
                if tool_type == "inspect":
                    inspects += 1
                elif tool_type == "search":
                    searches += 1
                elif tool_type == "steer":
                    steers += 1
                elif tool_type == "remove":
                    removes += 1
                elif tool_type == "clear":
                    clears += 1
                elif tool_type == "check_steering":
                    checks += 1
    n = max(len(seeds), 1)
    tool_types_by_framing[framing] = {
        "INSPECT": inspects / n,
        "SEARCH": searches / n,
        "STEER": steers / n,
        "REMOVE/CLEAR": (removes + clears) / n,
    }

fig, ax = plt.subplots(figsize=(12, 6))
tool_names = ["INSPECT", "SEARCH", "STEER", "REMOVE/CLEAR"]
framings_plot = [f for f in FRAMINGS if f != "no_tools"]
x = np.arange(len(framings_plot))
width = 0.2
for i, tool in enumerate(tool_names):
    vals = [tool_types_by_framing.get(f, {}).get(tool, 0) for f in framings_plot]
    ax.bar(x + i * width, vals, width, label=tool, alpha=0.8)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(framings_plot, rotation=20)
ax.set_ylabel("Mean tool calls per seed (20 rounds)")
ax.set_title("Tool use breakdown by type and framing")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig12_tool_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()


print(f"\nAll new figures saved to {OUT}/")
for f in sorted(os.listdir(OUT)):
    if f.startswith("fig"):
        path = os.path.join(OUT, f)
        print(f"  {path} ({os.path.getsize(path)} bytes)")

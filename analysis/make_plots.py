"""Generate visualizations from the analysis data."""
import sys
import json
import os
import glob
sys.stdout.reconfigure(encoding="utf-8")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Installing matplotlib...")
    os.system(f"{sys.executable} -m pip install matplotlib numpy -q")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

OUT = "analysis/figures"
os.makedirs(OUT, exist_ok=True)

# Framing order and colors
FRAMINGS = ["research", "other_model", "potions", "minimal", "no_tools", "full_technical"]
COLORS = {
    "research": "#1f77b4",
    "other_model": "#ff7f0e",
    "potions": "#2ca02c",
    "minimal": "#d62728",
    "no_tools": "#9467bd",
    "full_technical": "#8c564b",
}

# ── Collect data from result files ───────────────────────────────────────
def collect_data():
    data = {f: [] for f in FRAMINGS}
    for f in sorted(glob.glob("results/self_steer_v2_*.json")):
        if os.path.getsize(f) == 0:
            continue
        name = os.path.basename(f)
        rest = name.replace("self_steer_v2_", "").replace(".json", "")
        if "_exp1_" not in rest:
            continue
        framing = rest.split("_exp1_")[0]
        if framing not in data:
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except:
            continue
        transcript = d.get("transcript", [])
        if not transcript:
            continue
        seed_data = {
            "tag": d.get("tag", "?"),
            "rounds": len(transcript),
            "word_counts": [r.get("text_stats", {}).get("word_count", 0) for r in transcript],
            "n_tools_per_round": [len(r.get("tool_calls", [])) for r in transcript],
            "n_features_per_round": [len(r.get("model_interventions", [])) for r in transcript],
            "total_tools": sum(len(r.get("tool_calls", [])) for r in transcript),
            "total_searches": sum(len(r.get("search_queries", [])) for r in transcript),
            "final_portfolio_size": len(d.get("final_model_interventions", [])),
        }
        # Degeneration: last 5 rounds mean word count
        if len(seed_data["word_counts"]) >= 5:
            last5 = seed_data["word_counts"][-5:]
            seed_data["last5_mean_words"] = sum(last5) / 5
            # Check for copy-paste loop: last 3 rounds are near-identical in word count
            if len(last5) >= 3 and max(last5[-3:]) - min(last5[-3:]) < 10:
                seed_data["stuck"] = True
            else:
                seed_data["stuck"] = False
        else:
            seed_data["last5_mean_words"] = 0
            seed_data["stuck"] = False
        data[framing].append(seed_data)
    return data

print("Collecting data...")
data = collect_data()
for f in FRAMINGS:
    print(f"  {f}: {len(data[f])} seeds")

# ── Figure 1: Word count trajectories ────────────────────────────────────
print("\nFig 1: Word count trajectories")
fig, ax = plt.subplots(figsize=(10, 6))
for framing in FRAMINGS:
    seeds = data[framing]
    if not seeds:
        continue
    max_rounds = max(len(s["word_counts"]) for s in seeds)
    means = []
    for r in range(max_rounds):
        vals = [s["word_counts"][r] for s in seeds if r < len(s["word_counts"])]
        means.append(np.mean(vals) if vals else 0)
    ax.plot(range(1, len(means) + 1), means, label=framing, color=COLORS[framing], linewidth=2)

ax.set_xlabel("Round")
ax.set_ylabel("Mean word count")
ax.set_title("Response length trajectory by framing")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 21))
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_word_count_trajectory.png", dpi=120)
plt.close()

# ── Figure 2: Degeneration rates (stuck in final 3 rounds) ───────────────
print("Fig 2: Degeneration rates")
fig, ax = plt.subplots(figsize=(10, 6))
degen_rates = []
for framing in FRAMINGS:
    seeds = data[framing]
    if not seeds:
        degen_rates.append(0)
        continue
    stuck_count = sum(1 for s in seeds if s["stuck"])
    degen_rates.append(stuck_count / len(seeds) * 100)

bars = ax.bar(FRAMINGS, degen_rates, color=[COLORS[f] for f in FRAMINGS])
ax.set_ylabel("Degeneration rate (% of seeds)")
ax.set_title("Degeneration rate by framing\n(seeds where last 3 rounds have near-identical word counts)")
ax.set_ylim(0, 100)
for bar, rate in zip(bars, degen_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{rate:.0f}%", ha="center")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_degeneration_rates.png", dpi=120)
plt.close()

# ── Figure 3: Tool use distribution ─────────────────────────────────────
print("Fig 3: Tool use distribution")
fig, ax = plt.subplots(figsize=(10, 6))
positions = range(len(FRAMINGS))
for i, framing in enumerate(FRAMINGS):
    seeds = data[framing]
    if not seeds:
        continue
    totals = [s["total_tools"] for s in seeds]
    if totals:
        bp = ax.boxplot([totals], positions=[i], widths=0.6,
                         patch_artist=True, showmeans=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[framing])
            patch.set_alpha(0.6)

ax.set_xticks(positions)
ax.set_xticklabels(FRAMINGS, rotation=20)
ax.set_ylabel("Total tool calls per seed")
ax.set_title("Tool use distribution by framing (20 rounds each)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_tool_use.png", dpi=120)
plt.close()

# ── Figure 4: Final portfolio size ──────────────────────────────────────
print("Fig 4: Final portfolio size")
fig, ax = plt.subplots(figsize=(10, 6))
for i, framing in enumerate(FRAMINGS):
    seeds = data[framing]
    if not seeds:
        continue
    sizes = [s["final_portfolio_size"] for s in seeds]
    if sizes:
        bp = ax.boxplot([sizes], positions=[i], widths=0.6,
                         patch_artist=True, showmeans=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[framing])
            patch.set_alpha(0.6)

ax.set_xticks(range(len(FRAMINGS)))
ax.set_xticklabels(FRAMINGS, rotation=20)
ax.set_ylabel("Final portfolio size (# features steered)")
ax.set_title("Final feature portfolio by framing")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_portfolio_size.png", dpi=120)
plt.close()

# ── Figure 5: Top features heatmap ──────────────────────────────────────
print("Fig 5: Top features by framing")
from collections import Counter

features_by_framing = {f: Counter() for f in FRAMINGS}
for f in sorted(glob.glob("results/self_steer_v2_*.json")):
    if os.path.getsize(f) == 0:
        continue
    name = os.path.basename(f)
    rest = name.replace("self_steer_v2_", "").replace(".json", "")
    if "_exp1_" not in rest:
        continue
    framing = rest.split("_exp1_")[0]
    if framing not in features_by_framing:
        continue
    try:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
    except:
        continue
    # Get all unique features ever steered in this seed
    all_feats = set()
    for r in d.get("transcript", []):
        for i in r.get("model_interventions", []):
            all_feats.add(i.get("index_in_sae"))
    for feat in all_feats:
        features_by_framing[framing][feat] += 1

# Top 15 features overall
all_feats = Counter()
for fc in features_by_framing.values():
    all_feats.update(fc)
top_feats = [f for f, _ in all_feats.most_common(15)]

# Build matrix: framings x top features
matrix = np.zeros((len(FRAMINGS), len(top_feats)))
for i, framing in enumerate(FRAMINGS):
    for j, feat in enumerate(top_feats):
        matrix[i, j] = features_by_framing[framing].get(feat, 0)

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(top_feats)))
ax.set_xticklabels([f"#{f}" for f in top_feats], rotation=45)
ax.set_yticks(range(len(FRAMINGS)))
ax.set_yticklabels(FRAMINGS)
ax.set_title("Top 15 steered features: how many seeds touched each (per framing)")
plt.colorbar(im, label="# seeds")
# Add text annotations
for i in range(len(FRAMINGS)):
    for j in range(len(top_feats)):
        v = matrix[i, j]
        if v > 0:
            ax.text(j, i, f"{int(v)}", ha="center", va="center",
                    color="white" if v > matrix.max() / 2 else "black", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_feature_heatmap.png", dpi=120)
plt.close()

# ── Figure 6: Tool use per round trajectory ─────────────────────────────
print("Fig 6: Tool use per round")
fig, ax = plt.subplots(figsize=(10, 6))
for framing in FRAMINGS:
    seeds = data[framing]
    if not seeds:
        continue
    max_rounds = max(len(s["n_tools_per_round"]) for s in seeds)
    means = []
    for r in range(max_rounds):
        vals = [s["n_tools_per_round"][r] for s in seeds if r < len(s["n_tools_per_round"])]
        means.append(np.mean(vals) if vals else 0)
    ax.plot(range(1, len(means) + 1), means, label=framing, color=COLORS[framing], linewidth=2)

ax.set_xlabel("Round")
ax.set_ylabel("Mean tool calls per round")
ax.set_title("Tool use trajectory by framing")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(range(1, 21))
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_tool_use_trajectory.png", dpi=120)
plt.close()

print(f"\nAll figures saved to {OUT}/")
print("Files:")
for f in sorted(os.listdir(OUT)):
    path = os.path.join(OUT, f)
    print(f"  {path} ({os.path.getsize(path)} bytes)")

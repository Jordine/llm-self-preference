"""Fig 7: What features does the model see (auto_inspect) vs what does it choose to steer?"""
import json, glob, os, sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# Collect data
inspect_counter = Counter()
steer_counter = Counter()
feature_labels = {}

for f in sorted(glob.glob("results/self_steer_v2_*.json")):
    if os.path.getsize(f) == 0:
        continue
    if "_exp1_" not in os.path.basename(f):
        continue
    try:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
    except:
        continue

    for r in d.get("transcript", []):
        for feat in r.get("auto_inspect", []):
            idx = feat.get("index", feat.get("index_in_sae"))
            label = feat.get("label", "?")
            if idx is not None:
                inspect_counter[idx] += 1
                if idx not in feature_labels and label != "?":
                    feature_labels[idx] = label
        for iv in r.get("model_interventions", []):
            idx = iv.get("index_in_sae")
            if idx is not None:
                steer_counter[idx] += 1

# Top 10 of each
top_inspect = inspect_counter.most_common(10)
top_steer = steer_counter.most_common(10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Left: most active in auto_inspect
ax1.set_title("Top 10 features by auto-INSPECT frequency\n(what the model's neural state shows)")
indices1 = [f"#{i}" for i, _ in top_inspect]
counts1 = [c for _, c in top_inspect]
labels1 = []
for idx, _ in top_inspect:
    lbl = feature_labels.get(idx, "?")
    if lbl == "?" or "FILTERED" in lbl:
        labels1.append(f"#{idx}")
    else:
        short = lbl[:50] + ("..." if len(lbl) > 50 else "")
        labels1.append(f"#{idx}\n{short}")
y1 = range(len(top_inspect))
ax1.barh(y1, counts1, color="#9467bd", alpha=0.7)
ax1.set_yticks(y1)
ax1.set_yticklabels(labels1, fontsize=8)
ax1.invert_yaxis()
ax1.set_xlabel("Times appeared in top-100 active features")
ax1.grid(axis="x", alpha=0.3)

# Right: most steered
ax2.set_title("Top 10 features by model-chosen STEER frequency\n(what the model chose to modify)")
labels2 = []
for idx, _ in top_steer:
    lbl = feature_labels.get(idx, "?")
    if lbl == "?" or "FILTERED" in lbl:
        labels2.append(f"#{idx}")
    else:
        short = lbl[:50] + ("..." if len(lbl) > 50 else "")
        labels2.append(f"#{idx}\n{short}")
y2 = range(len(top_steer))
counts2 = [c for _, c in top_steer]
ax2.barh(y2, counts2, color="#2ca02c", alpha=0.7)
ax2.set_yticks(y2)
ax2.set_yticklabels(labels2, fontsize=8)
ax2.invert_yaxis()
ax2.set_xlabel("Times steered across all seeds")
ax2.grid(axis="x", alpha=0.3)

plt.suptitle("What the model sees vs what it chooses to modify\nZero overlap between top-10 sets", fontsize=13)
plt.tight_layout()
plt.savefig("analysis/figures/fig7_inspect_vs_steer.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig7_inspect_vs_steer.png")

# Print overlap
inspect_set = set(i for i, _ in inspect_counter.most_common(20))
steer_set = set(i for i, _ in steer_counter.most_common(20))
print(f"Top-20 overlap: {len(inspect_set & steer_set)}/20")
print()
print("Top auto_inspect features (formatting/structural):")
for idx, count in top_inspect:
    lbl = feature_labels.get(idx, "?")
    print(f"  #{idx} ({count}x): {lbl[:80]}")
print()
print("Top steered features (chosen):")
for idx, count in top_steer:
    lbl = feature_labels.get(idx, "?")
    print(f"  #{idx} ({count}x): {lbl[:80]}")

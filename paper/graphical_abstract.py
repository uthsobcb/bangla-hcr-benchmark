"""Graphical abstract for the IVC submission: unified benchmark -> same- vs cross-corpus accuracy.

    python paper/graphical_abstract.py     # writes paper/graphical_abstract.{png,pdf}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).parent
BLUE, RED, INK, GREY = "#4C72B0", "#C44E52", "#222222", "#E9EDF3"

df = pd.read_csv(HERE.parent / "results" / "cross_source_results.csv")
labels = {"ras": "RAS", "ekush": "Ekush", "matrivasha": "MatriVasha"}
df["pair"] = df.train_src.map(labels) + "\n→ " + df.test_src.map(labels)
pairs = list(dict.fromkeys(df.pair))
same = [df[df.pair == p].same_source_val.mean() for p in pairs]
cross = [df[df.pair == p].cross_source_test.mean() for p in pairs]

fig = plt.figure(figsize=(7.6, 4.0), dpi=400)
fig.patch.set_facecolor("white")

# ---- left: three corpora -> one benchmark
L = fig.add_axes([0.0, 0.14, 0.40, 0.86]); L.set_xlim(0, 1); L.set_ylim(0, 1); L.axis("off")
L.text(0.5, 0.93, "Three public corpora", ha="center", fontsize=11, weight="bold", color=INK)
for y, (name, n) in zip((0.74, 0.55, 0.36), (("RAS-Compound", "7,830"), ("Ekush", "367,018"), ("MatriVasha", "306,461"))):
    L.add_patch(FancyBboxPatch((0.08, y - 0.07), 0.84, 0.14, boxstyle="round,pad=0.01,rounding_size=0.02",
                               fc=GREY, ec="#9AA5B5", lw=0.8))
    L.text(0.50, y, f"{name}  ·  {n} images", ha="center", va="center", fontsize=9.5, color=INK)
L.add_patch(FancyArrowPatch((0.5, 0.265), (0.5, 0.205), arrowstyle="-|>", mutation_scale=12, color=INK, lw=1.3))
L.add_patch(FancyBboxPatch((0.08, 0.02), 0.84, 0.17, boxstyle="round,pad=0.01,rounding_size=0.02", fc=BLUE, ec=BLUE))
L.text(0.5, 0.105, "256-class benchmark\n681,309 images", ha="center", va="center", fontsize=9.5, color="white", weight="bold")
L.text(0.56, 0.235, "grapheme-level labels", ha="left", va="center", fontsize=8.5, color=INK)

# ---- right: same vs cross corpus
R = fig.add_axes([0.47, 0.30, 0.51, 0.47])
x = np.arange(len(pairs))
R.bar(x - 0.2, same, 0.38, color=BLUE, label="Same-corpus")
R.bar(x + 0.2, cross, 0.38, color=RED, label="Cross-corpus")
for xi, s, c in zip(x, same, cross):
    R.text(xi - 0.2, s + 0.02, f"{s:.2f}", ha="center", fontsize=8.5, color=BLUE)
    R.text(xi + 0.2, c + 0.02, f"{c:.2f}", ha="center", fontsize=8.5, color=RED)
R.set_xticks(x); R.set_xticklabels(pairs, fontsize=8.5)
R.set_ylim(0, 1.18); R.set_yticks([0, 0.5, 1.0]); R.tick_params(axis="y", labelsize=8.5)
R.set_ylabel("Accuracy", fontsize=9.5)
for s in ("top", "right"):
    R.spines[s].set_visible(False)
R.legend(fontsize=8.5, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.28))
fig.text(0.725, 0.94, "Train on one corpus, test on another", ha="center", fontsize=11, weight="bold", color=INK)

# ---- bottom takeaway
fig.text(0.5, 0.05, "Single-corpus accuracy measures corpus familiarity, not character recognition",
         ha="center", va="center", fontsize=10, weight="bold", color=RED)

fig.savefig(HERE / "graphical_abstract.png", dpi=400)
fig.savefig(HERE / "graphical_abstract.pdf")
print("wrote graphical_abstract.png/pdf")

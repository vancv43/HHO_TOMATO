import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)
CLASSES = ["Damaged", "Old", "Ripe", "Unripe"]
NAVY = "#1F4E5A"

train_counts = [949, 1992, 1975, 1585]
val_counts = [106, 222, 220, 177]

fig, ax = plt.subplots(figsize=(6.2, 4.0))
x = np.arange(len(CLASSES))
w = 0.35
b1 = ax.bar(x - w/2, train_counts, w, label=f"Train (n={sum(train_counts)})", color="#2E7D8C")
b2 = ax.bar(x + w/2, val_counts, w, label=f"Validation (n={sum(val_counts)})", color="#C97F0A")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 15, str(int(b.get_height())),
             ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(CLASSES)
ax.set_ylabel("Number of images")
ax.set_title(f"Tomato dataset class distribution (N={sum(train_counts)+sum(val_counts)})")
ax.legend(frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_dataset_distribution.png", dpi=220)
plt.close(fig)
print("Saved fig_dataset_distribution.png")

# baseline model comparison bar chart (accuracy + macro-F1), from real baseline_results.json
with open(f"{OUT}/baseline_results.json") as f:
    br = json.load(f)
names = list(br["per_model_metrics"].keys()) + ["HHO ensemble"]
accs = [br["per_model_metrics"][n]["accuracy"] for n in names[:-1]] + [br["ensemble_metrics"]["accuracy"]]
f1s = [br["per_model_metrics"][n]["f1_macro"] for n in names[:-1]] + [br["ensemble_metrics"]["f1_macro"]]

fig, ax = plt.subplots(figsize=(6.6, 4.0))
xx = np.arange(len(names))
b1 = ax.bar(xx - w/2, accs, w, label="Accuracy", color="#1F4E5A")
b2 = ax.bar(xx + w/2, f1s, w, label="Macro-F1", color="#8FBF6E")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.006, f"{b.get_height()*100:.1f}",
             ha="center", fontsize=8)
ax.set_xticks(xx); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
ax.set_ylim(0.8, 1.02)
ax.set_ylabel("Score")
ax.set_title("Held-out validation performance by model")
ax.legend(frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_model_comparison.png", dpi=220)
plt.close(fig)
print("Saved fig_model_comparison.png")

# confusion matrix for the HHO ensemble (held-out val)
cm = np.array(br["ensemble_metrics"]["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4.6, 4.2))
im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        v = cm[i, j]
        pct = v / cm[i].sum() * 100
        color = "white" if pct > 50 else "#222222"
        ax.text(j, i, f"{v}\n({pct:.0f}%)", ha="center", va="center", fontsize=9, color=color)
ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES)
ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
acc = np.trace(cm) / cm.sum()
ax.set_title(f"HHO ensemble confusion matrix (val, N={cm.sum()})\noverall accuracy {acc*100:.2f}%")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_confusion_matrix_hho.png", dpi=220)
plt.close(fig)
print("Saved fig_confusion_matrix_hho.png")

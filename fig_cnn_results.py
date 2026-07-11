import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)
CLASSES = ["Damaged", "Old", "Ripe", "Unripe"]

with open(f"{OUT}/cnn_history.json") as f:
    h = json.load(f)

hist = h["history"]
epochs = np.arange(1, len(hist["train_acc"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
ax = axes[0]
ax.plot(epochs, hist["train_acc"], color="#1F4E5A", lw=2, label="Train")
ax.plot(epochs, hist["val_acc"], color="#2E7D8C", lw=2, ls="--", label="Validation")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Accuracy", loc="left", color="#1F4E5A")
ax.legend(frameon=False, fontsize=8.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
ax.plot(epochs, hist["train_loss"], color="#1F4E5A", lw=2, label="Train")
ax.plot(epochs, hist["val_loss"], color="#2E7D8C", lw=2, ls="--", label="Validation")
ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss"); ax.set_title("Loss", loc="left", color="#1F4E5A")
ax.legend(frameon=False, fontsize=8.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("ResNet18 training/validation curves (20 epochs, tomato 4-class grading)", fontsize=10.5, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_cnn_curves.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved fig_cnn_curves.png")

cm = np.array(h["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4.6, 4.2))
im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        v = cm[i, j]
        pct = v / cm[i].sum() * 100
        color = "white" if pct > 50 else "#222222"
        ax.text(j, i, f"{v}\n({pct:.0f}%)", ha="center", va="center", fontsize=9, color=color)
ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(h["classes_imagefolder_order"])
ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(h["classes_imagefolder_order"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
acc = np.trace(cm) / cm.sum()
ax.set_title(f"ResNet18 CNN confusion matrix (val, N={cm.sum()})\noverall accuracy {acc*100:.2f}%")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_cnn_confusion_matrix.png", dpi=220)
plt.close(fig)
print("Saved fig_cnn_confusion_matrix.png")

print(f"\nbest_val_acc={h['best_val_acc']:.4f} best_val_f1={h['best_val_f1_macro']:.4f}")
print(f"total_train_seconds={h['total_train_seconds']:.1f} device={h['device']}")

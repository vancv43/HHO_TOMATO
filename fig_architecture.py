import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)
DATA_ROOT = os.environ.get("TOMATO_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Tomatoes Dataset"))
CLASSES = ["Damaged", "Old", "Ripe", "Unripe"]
IMG_SIZE = 128

SAMPLE_IMAGES = {
    "Damaged": f"{DATA_ROOT}/val/Damaged/d (349).png",
    "Old": f"{DATA_ROOT}/val/Old/o (1310).jpg",
    "Ripe": f"{DATA_ROOT}/val/Ripe/r (2082).jpg",
    "Unripe": f"{DATA_ROOT}/val/Unripe/u (9).png",
}

# ---- compute one real Grad-CAM overlay using the actual trained model ----
device = torch.device("cpu")


def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


model = build_model().to(device)
model.load_state_dict(torch.load(f"{OUT}/cnn_best.pt", map_location=device))
model.eval()

tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx=None):
        self.model.zero_grad()
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        score = out[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, F.softmax(out, dim=1)[0, class_idx].item()


cam_gen = GradCAM(model, model.layer4[-1])
bgr = cv2.imread(SAMPLE_IMAGES["Ripe"])
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
x = tf(rgb).unsqueeze(0)
cam, pred_idx, conf = cam_gen.generate(x)
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
real_overlay = (0.45 * heatmap + 0.55 * rgb_resized).astype(np.uint8)
print(f"Real Grad-CAM computed: predicted={CLASSES[pred_idx]} conf={conf:.3f}")

# ---- drawing ----
TBOX, TEDGE = "#E3F2F0", "#1F6B66"
OBOX, OEDGE = "#FDF0E0", "#C97F0A"
PBOX, PEDGE = "#F3EAF7", "#8B5CA6"
GBOX, GEDGE = "#E7F5EA", "#2E7D46"
CBOX, CEDGE = "#E7EEF9", "#3A5A9B"
BANDA = "#F4FBFA"
BANDB = "#F2F5FB"

fig, ax = plt.subplots(figsize=(12.4, 8.2))
ax.set_xlim(0, 12.4)
ax.set_ylim(0, 8.2)
ax.axis("off")
ax.set_aspect("equal")

ax.add_patch(Rectangle((0.15, 4.35), 12.1, 3.55, facecolor=BANDA, edgecolor="none", zorder=0))
ax.add_patch(Rectangle((0.15, 0.55), 12.1, 3.55, facecolor=BANDB, edgecolor="none", zorder=0))


def box(cx, cy, w, h, text, fc, ec, fontsize=9.2, weight="normal"):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle="round,pad=0.02,rounding_size=0.09",
                        linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            zorder=4, linespacing=1.35, fontweight=weight)
    return cx, cy, w, h


def arrow(p1, p2, color="#333333", lw=1.5, style="-"):
    fa = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, linewidth=lw,
                          linestyle=style, color=color, shrinkA=2, shrinkB=2, zorder=2)
    ax.add_patch(fa)


def photo_grid(cx, cy, cell=0.58, gap=0.06):
    order = ["Damaged", "Old", "Ripe", "Unripe"]
    offsets = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
    for cls, (ox, oy) in zip(order, offsets):
        img_bgr = cv2.imread(SAMPLE_IMAGES[cls])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        icx = cx + ox * (cell / 2 + gap / 2)
        icy = cy + oy * (cell / 2 + gap / 2)
        imbox = OffsetImage(img_rgb, zoom=0.185)
        ab = AnnotationBbox(imbox, (icx, icy), frameon=True, pad=0.02,
                             bboxprops=dict(edgecolor=TEDGE, linewidth=1.1))
        ax.add_artist(ab)
    rect = Rectangle((cx - cell - gap * 1.2, cy - cell - gap * 1.2),
                      2 * cell + 2.4 * gap, 2 * cell + 2.4 * gap,
                      fill=False, edgecolor=TEDGE, linewidth=1.6, zorder=5)
    ax.add_patch(rect)


# ---- Shared input stage ----
photo_grid(1.35, 6.9)
ax.text(1.35, 6.9 - 0.9, "Tomato images\n(7,226 total; 4 grades)", fontsize=8.0,
        ha="center", va="top", color="#1F6B66", fontweight="bold")

b_pre = box(4.15, 6.9, 1.95, 0.85, "Preprocessing\nresize 128$\\times$128,\ncolor-space convert", TBOX, TEDGE, 8.3)
arrow((1.35 + 0.7, 6.9), (b_pre[0] - b_pre[2] / 2, 6.9))

# Branch labels (kept clear of the connector lane on the left)
ax.text(2.35, 8.0, "Branch A -- color-statistical HHO-XQI (proposed method)",
        fontsize=9.6, color=TEDGE, fontweight="bold", ha="left", va="center")
ax.text(2.35, 4.03, "Branch B -- CNN benchmark + Grad-CAM (independent explainability channel)",
        fontsize=9.6, color=CEDGE, fontweight="bold", ha="left", va="center")

# ---- Branch A ----
b_feat = box(4.15, 5.60, 2.15, 1.0, "125-d color-statistical\nfeatures (RGB+HSV+Lab,\nhistograms, ripeness idx.)", TBOX, TEDGE, 8.3)
arrow((b_pre[0], 6.9 - 0.425), (4.15, 5.60 + 0.5))

b_lr = box(6.75, 6.70, 1.75, 0.62, "Logistic\nRegression", TBOX, TEDGE, 8.2)
b_rf = box(6.75, 5.97, 1.75, 0.62, "Random\nForest", TBOX, TEDGE, 8.2)
b_et = box(6.75, 5.24, 1.75, 0.62, "Extra\nTrees", TBOX, TEDGE, 8.2)
b_knn = box(6.75, 4.51, 1.75, 0.62, "K-Nearest\nNeighbors", TBOX, TEDGE, 8.2)
for b in (b_lr, b_rf, b_et, b_knn):
    arrow((b_feat[0] + b_feat[2] / 2, 5.60), (b[0] - b[2] / 2, b[1]))

b_hho = box(9.45, 5.60, 1.9, 1.35, "HHO ensemble-\nweight optimizer\n(Eq. 1-3)", OBOX, OEDGE, 8.6)
for b in (b_lr, b_rf, b_et, b_knn):
    arrow((b[0] + b[2] / 2, b[1]), (b_hho[0] - b_hho[2] / 2, b_hho[1]))

b_xqi = box(11.25, 5.60, 1.25, 1.8, "Tomato\nQuality\nIndex +\ngrade\n(Eq. 4)", GBOX, GEDGE, 8.2)
arrow((b_hho[0] + b_hho[2] / 2, 5.60), (b_xqi[0] - b_xqi[2] / 2, 5.60))

# ---- connector: same preprocessed images also feed Branch B ----
# routed down the far-left lane (under the photo grid), clear of the feature
# box and clear of the "Branch B" label (which starts at x=2.35)
arrow((1.35, 6.9 - 0.71), (1.35, 2.35 + 0.425), style=(0, (3, 2)))
ax.text(1.35, 4.55, "same\npreprocessed\nimages", fontsize=6.8, ha="center", va="center",
        color="#555555", style="italic",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

# ---- Branch B ----
b_cnn = box(1.85, 2.35, 1.95, 0.85, "ResNet18 CNN\n(fine-tuned,\nImageNet init.)", CBOX, CEDGE, 8.6)

b_grad = box(4.45, 2.35, 1.85, 0.85, "Grad-CAM\nvisual\nexplanation", PBOX, PEDGE, 8.6)
arrow((b_cnn[0] + b_cnn[2] / 2, 2.35), (b_grad[0] - b_grad[2] / 2, 2.35))

# real Grad-CAM overlay thumbnail (computed above from the actual trained model)
imbox = OffsetImage(real_overlay, zoom=0.42)
ab = AnnotationBbox(imbox, (6.85, 2.35), frameon=True, pad=0.03,
                     bboxprops=dict(edgecolor=PEDGE, linewidth=1.3))
ax.add_artist(ab)
arrow((b_grad[0] + b_grad[2] / 2, 2.35), (6.36, 2.35))
ax.text(6.85, 2.35 - 0.8, f"real overlay -- pred: {CLASSES[pred_idx]} ({conf*100:.0f}\\%)",
        fontsize=7.2, ha="center", va="top", color="#555555", style="italic")

b_out = box(9.55, 2.35, 1.55, 0.9, "4-class grade\n+ heatmap\noverlay", GBOX, GEDGE, 8.2)
arrow((7.34, 2.35), (b_out[0] - b_out[2] / 2, 2.35))

ax.text(6.2, 0.9,
        "Both branches are evaluated on the identical held-out validation split for a direct\n"
        "accuracy-vs-interpretability-vs-latency comparison (Table 3, Table 11).",
        fontsize=8.4, ha="center", color="#444444", style="italic")

fig.tight_layout()
fig.savefig(f"{OUT}/fig_architecture.png", dpi=220)
print("Saved fig_architecture.png")

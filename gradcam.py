import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = os.environ.get("TOMATO_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Tomatoes Dataset"))
OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)
CLASSES = ["Damaged", "Old", "Ripe", "Unripe"]  # ImageFolder alphabetical order, matches train_cnn.py
IMG_SIZE = 128
SEED = 7

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


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
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, F.softmax(out, dim=1)[0, class_idx].item()


if __name__ == "__main__":
    model = build_model().to(device)
    model.load_state_dict(torch.load(f"{OUT}/cnn_best.pt", map_location=device))
    model.eval()
    cam_gen = GradCAM(model, model.layer4[-1])

    random.seed(SEED)
    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(7.2, 9.2))
    for ci, cls in enumerate(CLASSES):
        files = sorted(glob.glob(f"{DATA_ROOT}/val/{cls}/*"))
        f = random.choice(files)
        bgr = cv2.imread(f)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

        x = tf(rgb).unsqueeze(0).to(device)
        x.requires_grad_(False)
        cam, pred_idx, conf = cam_gen.generate(x)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.45 * heatmap + 0.55 * rgb_resized).astype(np.uint8)

        axes[ci, 0].imshow(rgb_resized); axes[ci, 0].set_ylabel(cls, fontsize=11, fontweight="bold")
        axes[ci, 1].imshow(cam, cmap="jet")
        axes[ci, 2].imshow(overlay)
        for j in range(3):
            axes[ci, j].set_xticks([]); axes[ci, j].set_yticks([])
        pred_label = CLASSES[pred_idx]
        correct = "✓" if pred_label == cls else "✗"
        pred_caption = f"pred: {pred_label} ({conf*100:.1f}%) {correct}"
        if ci == 0:
            axes[ci, 0].set_title("Original", fontsize=10)
            axes[ci, 1].set_title("Grad-CAM", fontsize=10)
            axes[ci, 2].set_title(f"Overlay\n{pred_caption}", fontsize=9)
        else:
            axes[ci, 2].set_title(pred_caption, fontsize=9)

    fig.suptitle("Grad-CAM visualizations (ResNet18, one sample per class)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_gradcam.png", dpi=200)
    print("Saved fig_gradcam.png")

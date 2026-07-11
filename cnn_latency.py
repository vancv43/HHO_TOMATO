import os
import time
import json
import torch
import torch.nn as nn
from torchvision import models

OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)
CLASSES = ["Damaged", "Old", "Ripe", "Unripe"]


def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


if __name__ == "__main__":
    # CPU latency (fair comparison with the classical pipeline, also measured on CPU)
    device_cpu = torch.device("cpu")
    model = build_model().to(device_cpu)
    model.load_state_dict(torch.load(f"{OUT}/cnn_best.pt", map_location=device_cpu))
    model.eval()

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        for _ in range(5):
            model(x)  # warmup
        n = 50
        t0 = time.time()
        for _ in range(n):
            model(x)
        cpu_ms = (time.time() - t0) / n * 1000
    print(f"CNN CPU inference: {cpu_ms:.3f} ms/image (batch=1, single thread default)")

    # MPS latency, if available
    mps_ms = None
    if torch.backends.mps.is_available():
        device_mps = torch.device("mps")
        model_mps = build_model().to(device_mps)
        model_mps.load_state_dict(torch.load(f"{OUT}/cnn_best.pt", map_location=device_mps))
        model_mps.eval()
        xm = torch.randn(1, 3, 128, 128).to(device_mps)
        with torch.no_grad():
            for _ in range(10):
                model_mps(xm)
            torch.mps.synchronize()
            n = 100
            t0 = time.time()
            for _ in range(n):
                model_mps(xm)
            torch.mps.synchronize()
            mps_ms = (time.time() - t0) / n * 1000
        print(f"CNN MPS (GPU) inference: {mps_ms:.3f} ms/image (batch=1)")

    with open(f"{OUT}/cnn_latency.json", "w") as f:
        json.dump({"cpu_ms_per_image": cpu_ms, "mps_ms_per_image": mps_ms}, f, indent=2)
    print("Saved cnn_latency.json")

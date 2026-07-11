import glob
import os
import time
import numpy as np
import cv2
from features import extract_features, N_FEATURES

DATA_ROOT = os.environ.get("TOMATO_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Tomatoes Dataset"))
CLASSES = ["Damaged", "Old", "Ripe", "Unripe"]
OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)


def load_split(split):
    X, y, paths = [], [], []
    t0 = time.time()
    for ci, cls in enumerate(CLASSES):
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.jfif", "*.JPG", "*.JPEG", "*.PNG"):
            files += glob.glob(os.path.join(DATA_ROOT, split, cls, ext))
        files = sorted(set(files))
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            feat = extract_features(img)
            X.append(feat)
            y.append(ci)
            paths.append(f)
        print(f"{split}/{cls}: {len(files)} images done", flush=True)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    print(f"{split} total: {X.shape}, took {time.time()-t0:.1f}s")
    return X, y, paths


if __name__ == "__main__":
    Xtr, ytr, ptr = load_split("train")
    Xval, yval, pval = load_split("val")
    np.savez_compressed(os.path.join(OUT, "features.npz"),
                         Xtr=Xtr, ytr=ytr, Xval=Xval, yval=yval)
    with open(os.path.join(OUT, "train_paths.txt"), "w") as f:
        f.write("\n".join(ptr))
    with open(os.path.join(OUT, "val_paths.txt"), "w") as f:
        f.write("\n".join(pval))
    print("Saved features.npz", Xtr.shape, Xval.shape)

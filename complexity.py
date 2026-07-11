import os
import json
import time
import glob
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

from features import extract_features
from train_baseline import make_models, softmax_simplex

OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)
DATA_ROOT = os.environ.get("TOMATO_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Tomatoes Dataset"))
SEED = 42

if __name__ == "__main__":
    # 1. pure feature-extraction latency (no disk I/O amortized separately)
    files = sorted(glob.glob(f"{DATA_ROOT}/val/Ripe/*.jpg"))[:200]
    imgs = [cv2.imread(f) for f in files]
    t0 = time.time()
    for im in imgs:
        extract_features(im)
    extract_ms = (time.time() - t0) / len(imgs) * 1000

    # 2. classical-model inference latency (already trained in train_baseline; retrain quickly here)
    data = np.load(f"{OUT}/features.npz")
    Xtr, ytr, Xval, yval = data["Xtr"], data["ytr"], data["Xval"], data["yval"]
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xval_s = scaler.transform(Xtr), scaler.transform(Xval)

    models = make_models(SEED)
    complexities = {
        "Logistic regression": "O(m) per sample (m=125 features); training O(n*m*iters)",
        "Random forest": "O(T*log n) per sample (T=300 trees); training O(T*n*m*log n)",
        "Extra trees": "O(T*log n) per sample (T=300 trees); training O(T*n*m) (no per-split sort)",
        "K-nearest neighbors": "O(n*m) per sample (brute-force distance to all n=6501 train points)",
    }
    runtime = {}
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(Xtr_s, ytr)
        train_s = time.time() - t0
        n_reps = 5
        t0 = time.time()
        for _ in range(n_reps):
            clf.predict_proba(Xval_s)
        infer_ms_per_image = (time.time() - t0) / (n_reps * len(Xval_s)) * 1000
        n_params = None
        if name == "Random forest" or name == "Extra trees":
            n_params = int(sum(t.tree_.node_count for t in clf.estimators_))
        elif name == "Logistic regression":
            n_params = int(clf.coef_.size + clf.intercept_.size)
        runtime[name] = {
            "train_seconds": train_s,
            "inference_ms_per_image": infer_ms_per_image,
            "total_ms_per_image_incl_features": infer_ms_per_image + extract_ms,
            "n_params_or_nodes": n_params,
            "complexity": complexities[name],
        }
        print(f"{name}: train={train_s:.3f}s infer={infer_ms_per_image:.4f}ms/img "
              f"(+features={extract_ms:.4f}ms) params/nodes={n_params}")

    ensemble_train_s = sum(r["train_seconds"] for r in runtime.values())
    ensemble_infer_ms = sum(r["inference_ms_per_image"] for r in runtime.values()) + extract_ms
    print(f"\nHHO ensemble (sum of 4 base learners + HHO offline weight search):")
    print(f"  training (one-off): {ensemble_train_s:.3f}s classifiers + HHO search (see convergence log)")
    print(f"  inference: {ensemble_infer_ms:.4f} ms/image (feature extraction {extract_ms:.4f} ms "
          f"+ 4 model calls {ensemble_infer_ms - extract_ms:.4f} ms)")
    print(f"  throughput: {1000.0/ensemble_infer_ms:.1f} images/second (single CPU thread, no batching)")

    out = {
        "feature_extraction_ms_per_image": extract_ms,
        "per_model": runtime,
        "ensemble_train_seconds": ensemble_train_s,
        "ensemble_inference_ms_per_image": ensemble_infer_ms,
        "ensemble_throughput_img_per_s": 1000.0 / ensemble_infer_ms,
    }
    with open(f"{OUT}/complexity_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved complexity_results.json")

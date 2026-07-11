import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from hho import hho_minimize
from train_baseline import make_models, softmax_simplex

OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)
SEED = 42

# feature-slice definitions (see features.py layout: 36 stats + 72 hist + 17 ripeness idx)
STATS = {"R": (0, 4), "G": (4, 8), "B": (8, 12), "H": (12, 16), "S": (16, 20), "V": (20, 24),
         "L": (24, 28), "a": (28, 32), "b": (32, 36)}
HIST = {"R": (36, 44), "G": (44, 52), "B": (52, 60), "H": (60, 68), "S": (68, 76), "V": (76, 84),
        "L": (84, 92), "a": (92, 100), "b": (100, 108)}
RIPE = (108, 125)


def idx_for(channels, include_ripe):
    idxs = []
    for c in channels:
        idxs += list(range(*STATS[c])) + list(range(*HIST[c]))
    if include_ripe:
        idxs += list(range(*RIPE))
    return np.array(sorted(idxs))


FEATURE_VARIANTS = {
    "RGB only (36-d)": idx_for(["R", "G", "B"], include_ripe=False),
    "RGB+HSV (72-d)": idx_for(["R", "G", "B", "H", "S", "V"], include_ripe=False),
    "RGB+HSV+Lab, no ripeness idx (108-d)": idx_for(["R", "G", "B", "H", "S", "V", "L", "a", "b"], include_ripe=False),
    "Full: RGB+HSV+Lab+ripeness idx (125-d)": idx_for(["R", "G", "B", "H", "S", "V", "L", "a", "b"], include_ripe=True),
}


def eval_ensemble_on_subset(Xtr_full, ytr_full, Xval, yval, feat_idx, n_hawks=16, max_iter=25):
    Xtr_full = Xtr_full[:, feat_idx]
    Xval = Xval[:, feat_idx]
    Xtr, Xhho, ytr, yhho = train_test_split(Xtr_full, ytr_full, test_size=0.15,
                                             stratify=ytr_full, random_state=SEED)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xhho_s, Xval_s = scaler.transform(Xtr), scaler.transform(Xhho), scaler.transform(Xval)

    models = make_models(SEED)
    proba_hho, proba_val, single_acc, single_f1 = {}, {}, {}, {}
    for name, clf in models.items():
        clf.fit(Xtr_s, ytr)
        proba_hho[name] = clf.predict_proba(Xhho_s)
        pv = clf.predict_proba(Xval_s)
        proba_val[name] = pv
        pred = pv.argmax(axis=1)
        single_acc[name] = accuracy_score(yval, pred)
        single_f1[name] = f1_score(yval, pred, average="macro", zero_division=0)
    model_names = list(models.keys())

    def fitness(w):
        w = softmax_simplex(w)
        combined = sum(w[i] * proba_hho[n] for i, n in enumerate(model_names))
        pred = combined.argmax(axis=1)
        return 1.0 - f1_score(yhho, pred, average="macro", zero_division=0)

    best_w_raw, _, _ = hho_minimize(fitness, dim=len(model_names), lb=0.0, ub=1.0,
                                     n_hawks=n_hawks, max_iter=max_iter, seed=SEED)
    best_w = softmax_simplex(best_w_raw)
    combined_val = sum(best_w[i] * proba_val[n] for i, n in enumerate(model_names))
    pred_hho = combined_val.argmax(axis=1)

    # equal-weight voting ablation (removes HHO optimization, keeps same base learners)
    equal_w = np.ones(len(model_names)) / len(model_names)
    combined_equal = sum(equal_w[i] * proba_val[n] for i, n in enumerate(model_names))
    pred_equal = combined_equal.argmax(axis=1)

    best_single = max(single_f1, key=single_f1.get)
    return {
        "hho_ensemble_acc": accuracy_score(yval, pred_hho),
        "hho_ensemble_f1": f1_score(yval, pred_hho, average="macro", zero_division=0),
        "equal_weight_acc": accuracy_score(yval, pred_equal),
        "equal_weight_f1": f1_score(yval, pred_equal, average="macro", zero_division=0),
        "best_single_model": best_single,
        "best_single_acc": single_acc[best_single],
        "best_single_f1": single_f1[best_single],
        "hho_weights": {n: float(best_w[i]) for i, n in enumerate(model_names)},
        "per_model_acc": single_acc,
        "per_model_f1": single_f1,
    }


def leave_one_model_out(Xtr_full, ytr_full, Xval, yval, n_hawks=16, max_iter=25):
    """Ensemble-composition ablation: drop each base learner one at a time."""
    scaler = StandardScaler().fit(Xtr_full)
    Xtr, Xhho, ytr, yhho = train_test_split(Xtr_full, ytr_full, test_size=0.15,
                                             stratify=ytr_full, random_state=SEED)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xhho_s, Xval_s = scaler.transform(Xtr), scaler.transform(Xhho), scaler.transform(Xval)
    models = make_models(SEED)
    proba_hho, proba_val = {}, {}
    for name, clf in models.items():
        clf.fit(Xtr_s, ytr)
        proba_hho[name] = clf.predict_proba(Xhho_s)
        proba_val[name] = clf.predict_proba(Xval_s)
    all_names = list(models.keys())

    results = {}
    for dropped in [None] + all_names:
        active = [n for n in all_names if n != dropped]

        def fitness(w, active=active):
            w = softmax_simplex(w)
            combined = sum(w[i] * proba_hho[n] for i, n in enumerate(active))
            pred = combined.argmax(axis=1)
            return 1.0 - f1_score(yhho, pred, average="macro", zero_division=0)

        best_w_raw, _, _ = hho_minimize(fitness, dim=len(active), lb=0.0, ub=1.0,
                                         n_hawks=n_hawks, max_iter=max_iter, seed=SEED)
        best_w = softmax_simplex(best_w_raw)
        combined_val = sum(best_w[i] * proba_val[n] for i, n in enumerate(active))
        pred = combined_val.argmax(axis=1)
        label = "Full ensemble (none dropped)" if dropped is None else f"Without {dropped}"
        results[label] = {
            "acc": accuracy_score(yval, pred),
            "f1": f1_score(yval, pred, average="macro", zero_division=0),
        }
    return results


if __name__ == "__main__":
    data = np.load(f"{OUT}/features.npz")
    Xtr_full, ytr_full, Xval, yval = data["Xtr"], data["ytr"], data["Xval"], data["yval"]

    feature_ablation = {}
    for name, idx in FEATURE_VARIANTS.items():
        r = eval_ensemble_on_subset(Xtr_full, ytr_full, Xval, yval, idx)
        feature_ablation[name] = r
        print(f"[Feature ablation] {name}: HHO acc={r['hho_ensemble_acc']:.4f} f1={r['hho_ensemble_f1']:.4f} "
              f"| equal-weight acc={r['equal_weight_acc']:.4f} | best-single={r['best_single_model']} "
              f"acc={r['best_single_acc']:.4f}", flush=True)

    ensemble_ablation = leave_one_model_out(Xtr_full, ytr_full, Xval, yval)
    for label, r in ensemble_ablation.items():
        print(f"[Ensemble-composition ablation] {label}: acc={r['acc']:.4f} f1={r['f1']:.4f}", flush=True)

    with open(f"{OUT}/ablation_results.json", "w") as f:
        json.dump({"feature_ablation": feature_ablation, "ensemble_ablation": ensemble_ablation}, f, indent=2, default=str)
    print("Saved ablation_results.json")

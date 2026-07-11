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

if __name__ == "__main__":
    data = np.load(f"{OUT}/features.npz")
    Xtr_full, ytr_full, Xval, yval = data["Xtr"], data["ytr"], data["Xval"], data["yval"]

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
    model_names = list(models.keys())

    def fitness(w):
        w = softmax_simplex(w)
        combined = sum(w[i] * proba_hho[n] for i, n in enumerate(model_names))
        pred = combined.argmax(axis=1)
        return 1.0 - f1_score(yhho, pred, average="macro", zero_division=0)

    hawk_grid = [8, 16, 32]
    iter_grid = [10, 25, 50]
    seeds = [1, 2, 3, 4, 5]

    results = []
    for n_hawks in hawk_grid:
        for max_iter in iter_grid:
            accs, f1s, extra_weights = [], [], []
            for seed in seeds:
                best_w_raw, _, _ = hho_minimize(fitness, dim=len(model_names), lb=0.0, ub=1.0,
                                                 n_hawks=n_hawks, max_iter=max_iter, seed=seed)
                best_w = softmax_simplex(best_w_raw)
                combined_val = sum(best_w[i] * proba_val[n] for i, n in enumerate(model_names))
                pred = combined_val.argmax(axis=1)
                accs.append(accuracy_score(yval, pred))
                f1s.append(f1_score(yval, pred, average="macro", zero_division=0))
                et_idx = model_names.index("Extra trees")
                extra_weights.append(best_w[et_idx])
            row = {
                "n_hawks": n_hawks, "max_iter": max_iter,
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                "extra_trees_weight_mean": float(np.mean(extra_weights)),
                "extra_trees_weight_std": float(np.std(extra_weights)),
                "n_fitness_evals": n_hawks * (max_iter + 1) * 2,  # rough evals-per-iter accounting
            }
            results.append(row)
            print(f"hawks={n_hawks:3d} iters={max_iter:3d}: acc={row['acc_mean']:.4f}+/-{row['acc_std']:.4f} "
                  f"f1={row['f1_mean']:.4f}+/-{row['f1_std']:.4f} w_ET={row['extra_trees_weight_mean']:.3f}",
                  flush=True)

    with open(f"{OUT}/sensitivity_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved sensitivity_results.json")

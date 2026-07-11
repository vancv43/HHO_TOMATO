import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import wilcoxon

from hho import hho_minimize
from train_baseline import make_models, softmax_simplex

OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)
N_FOLDS = 5
N_REPEATS = 4  # 5x4 repeated stratified CV -> 20 paired fold-scores, adequate power for Wilcoxon


def run_cv(n_hawks=16, max_iter=25, seed=42):
    data = np.load(f"{OUT}/features.npz")
    X = np.vstack([data["Xtr"], data["Xval"]])
    y = np.concatenate([data["ytr"], data["yval"]])

    skf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=seed)
    fold_scores = {"Logistic regression": [], "Random forest": [], "Extra trees": [],
                   "K-nearest neighbors": [], "HHO ensemble": []}
    fold_scores_acc = {k: [] for k in fold_scores}

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        Xtr_full, ytr_full = X[train_idx], y[train_idx]
        Xtest, ytest = X[test_idx], y[test_idx]

        Xtr, Xhho, ytr, yhho = train_test_split(
            Xtr_full, ytr_full, test_size=0.15, stratify=ytr_full, random_state=seed
        )
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xhho_s, Xtest_s = scaler.transform(Xtr), scaler.transform(Xhho), scaler.transform(Xtest)

        models = make_models(seed)
        proba_hho, proba_test = {}, {}
        for name, clf in models.items():
            clf.fit(Xtr_s, ytr)
            proba_hho[name] = clf.predict_proba(Xhho_s)
            pt = clf.predict_proba(Xtest_s)
            proba_test[name] = pt
            pred = pt.argmax(axis=1)
            fold_scores[name].append(f1_score(ytest, pred, average="macro", zero_division=0))
            fold_scores_acc[name].append(accuracy_score(ytest, pred))

        model_names = list(models.keys())

        def fitness(w):
            w = softmax_simplex(w)
            combined = sum(w[i] * proba_hho[n] for i, n in enumerate(model_names))
            pred = combined.argmax(axis=1)
            return 1.0 - f1_score(yhho, pred, average="macro", zero_division=0)

        best_w_raw, _, _ = hho_minimize(fitness, dim=len(model_names), lb=0.0, ub=1.0,
                                         n_hawks=n_hawks, max_iter=max_iter, seed=seed + fold_i)
        best_w = softmax_simplex(best_w_raw)
        combined_test = sum(best_w[i] * proba_test[n] for i, n in enumerate(model_names))
        pred_test = combined_test.argmax(axis=1)
        fold_scores["HHO ensemble"].append(f1_score(ytest, pred_test, average="macro", zero_division=0))
        fold_scores_acc["HHO ensemble"].append(accuracy_score(ytest, pred_test))
        print(f"Fold {fold_i+1}/{N_FOLDS * N_REPEATS} done. Ensemble F1={fold_scores['HHO ensemble'][-1]:.4f} "
              f"acc={fold_scores_acc['HHO ensemble'][-1]:.4f}", flush=True)

    return fold_scores, fold_scores_acc


if __name__ == "__main__":
    f1_scores, acc_scores = run_cv()

    summary = {}
    for k in f1_scores:
        summary[k] = {
            "f1_mean": float(np.mean(f1_scores[k])), "f1_std": float(np.std(f1_scores[k])),
            "acc_mean": float(np.mean(acc_scores[k])), "acc_std": float(np.std(acc_scores[k])),
            "f1_folds": f1_scores[k], "acc_folds": acc_scores[k],
        }
        print(k, "F1: %.4f +/- %.4f" % (summary[k]["f1_mean"], summary[k]["f1_std"]),
              "Acc: %.4f +/- %.4f" % (summary[k]["acc_mean"], summary[k]["acc_std"]))

    # Paired Wilcoxon signed-rank test: HHO ensemble vs. each baseline, over the SAME folds
    baselines = ["Logistic regression", "Random forest", "Extra trees", "K-nearest neighbors"]
    pvals = {}
    for b in baselines:
        diff = np.array(f1_scores["HHO ensemble"]) - np.array(f1_scores[b])
        if np.allclose(diff, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(f1_scores["HHO ensemble"], f1_scores[b])
        pvals[b] = {"statistic": None if np.isnan(stat) else float(stat), "p_raw": float(p)}

    # Holm-Bonferroni correction across the 4 comparisons
    items = sorted(pvals.items(), key=lambda kv: kv[1]["p_raw"])
    m = len(items)
    for rank, (b, d) in enumerate(items):
        d["p_holm"] = float(min(1.0, d["p_raw"] * (m - rank)))
    for b in baselines:
        print(f"HHO ensemble vs {b}: p_raw={pvals[b]['p_raw']:.4f} p_holm={pvals[b]['p_holm']:.4f}")

    with open(f"{OUT}/cv_stats_results.json", "w") as f:
        json.dump({"summary": summary, "wilcoxon_vs_hho_ensemble": pvals}, f, indent=2)
    print("Saved cv_stats_results.json")

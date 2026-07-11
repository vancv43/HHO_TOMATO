import os
import json
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

from hho import hho_minimize

OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)
CLASSES = ["Damaged", "Old", "Ripe", "Unripe"]


def make_models(seed=42):
    return {
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "Extra trees": ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "K-nearest neighbors": KNeighborsClassifier(n_neighbors=7),
    }


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def softmax_simplex(w):
    w = np.clip(w, 1e-6, None)
    return w / w.sum()


def run(seed=42, n_hawks=16, max_iter=25, log_convergence=False, verbose=True):
    data = np.load(f"{OUT}/features.npz")
    Xtr_full, ytr_full, Xval, yval = data["Xtr"], data["ytr"], data["Xval"], data["yval"]

    # carve an inner HHO-validation split out of TRAIN only (val folder stays untouched
    # until the very final evaluation, avoiding leakage into weight optimization)
    Xtr, Xhho, ytr, yhho = train_test_split(
        Xtr_full, ytr_full, test_size=0.15, stratify=ytr_full, random_state=seed
    )

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xhho_s, Xval_s = scaler.transform(Xtr), scaler.transform(Xhho), scaler.transform(Xval)

    models = make_models(seed)
    fit_times, base_proba_hho, base_proba_val, base_pred_val, per_model_metrics = {}, {}, {}, {}, {}
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(Xtr_s, ytr)
        fit_times[name] = time.time() - t0
        base_proba_hho[name] = clf.predict_proba(Xhho_s)
        t0 = time.time()
        proba_val = clf.predict_proba(Xval_s)
        ms_per_image = (time.time() - t0) / len(Xval_s) * 1000
        base_proba_val[name] = proba_val
        pred_val = proba_val.argmax(axis=1)
        base_pred_val[name] = pred_val
        m = evaluate(yval, pred_val)
        m["ms_per_image"] = ms_per_image
        m["fit_seconds"] = fit_times[name]
        per_model_metrics[name] = m
        if verbose:
            print(f"{name}: val acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} fit={fit_times[name]:.2f}s")
        joblib.dump(clf, f"{OUT}/model_{name.replace(' ', '_')}.joblib")

    model_names = list(models.keys())

    def fitness(w):
        w = softmax_simplex(w)
        combined = sum(w[i] * base_proba_hho[n] for i, n in enumerate(model_names))
        pred = combined.argmax(axis=1)
        f1 = f1_score(yhho, pred, average="macro", zero_division=0)
        return 1.0 - f1  # HHO minimizes

    t0 = time.time()
    best_w_raw, best_fit, curve = hho_minimize(
        fitness, dim=len(model_names), lb=0.0, ub=1.0,
        n_hawks=n_hawks, max_iter=max_iter, seed=seed, log_convergence=log_convergence,
    )
    hho_seconds = time.time() - t0
    best_w = softmax_simplex(best_w_raw)

    combined_val = sum(best_w[i] * base_proba_val[n] for i, n in enumerate(model_names))
    pred_val = combined_val.argmax(axis=1)
    ensemble_metrics = evaluate(yval, pred_val)
    ensemble_metrics["fit_seconds"] = sum(fit_times.values()) + hho_seconds
    ensemble_metrics["ms_per_image"] = sum(
        per_model_metrics[n]["ms_per_image"] for n in model_names
    )  # ensemble must score all base learners at inference time
    ensemble_metrics["weights"] = {n: float(best_w[i]) for i, n in enumerate(model_names)}
    ensemble_metrics["hho_seconds"] = hho_seconds

    if verbose:
        print("HHO weights:", ensemble_metrics["weights"])
        print(f"Ensemble: val acc={ensemble_metrics['accuracy']:.4f} f1={ensemble_metrics['f1_macro']:.4f}")

    return {
        "per_model_metrics": per_model_metrics,
        "ensemble_metrics": ensemble_metrics,
        "convergence": curve,
        "model_names": model_names,
        "scaler": scaler,
        "base_proba_val": base_proba_val,
        "yval": yval,
    }


if __name__ == "__main__":
    result = run(log_convergence=True)
    out = {
        "per_model_metrics": result["per_model_metrics"],
        "ensemble_metrics": result["ensemble_metrics"],
        "convergence": result["convergence"],
    }
    with open(f"{OUT}/baseline_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved baseline_results.json")

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from hho import hho_minimize
from train_baseline import make_models, softmax_simplex

OUT = os.environ.get("TOMATO_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
os.makedirs(OUT, exist_ok=True)
SEED = 42
MAX_ITER = 40
N_SEEDS = 10

if __name__ == "__main__":
    data = np.load(f"{OUT}/features.npz")
    Xtr_full, ytr_full, Xval, yval = data["Xtr"], data["ytr"], data["Xval"], data["yval"]
    Xtr, Xhho, ytr, yhho = train_test_split(Xtr_full, ytr_full, test_size=0.15,
                                             stratify=ytr_full, random_state=SEED)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xhho_s = scaler.transform(Xtr), scaler.transform(Xhho)

    models = make_models(SEED)
    proba_hho = {}
    for name, clf in models.items():
        clf.fit(Xtr_s, ytr)
        proba_hho[name] = clf.predict_proba(Xhho_s)
    model_names = list(models.keys())

    def fitness(w):
        w = softmax_simplex(w)
        combined = sum(w[i] * proba_hho[n] for i, n in enumerate(model_names))
        pred = combined.argmax(axis=1)
        return 1.0 - f1_score(yhho, pred, average="macro", zero_division=0)

    curves = []
    for seed in range(N_SEEDS):
        _, _, curve = hho_minimize(fitness, dim=len(model_names), lb=0.0, ub=1.0,
                                    n_hawks=16, max_iter=MAX_ITER, seed=seed, log_convergence=True)
        curves.append(curve)
        print(f"seed {seed}: final fitness={curve[-1]:.5f} (1-F1)")

    curves = np.array(curves)  # (N_SEEDS, MAX_ITER+1)
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)
    mean_f1 = 1 - mean_curve

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    iters = np.arange(len(mean_curve))
    ax.plot(iters, mean_f1, color="#1F4E5A", lw=2, label="Mean best macro-F1 (10 seeds)")
    ax.fill_between(iters, 1 - (mean_curve - std_curve), 1 - (mean_curve + std_curve),
                     color="#1F4E5A", alpha=0.18, label="$\\pm$1 SD")
    ax.set_xlabel("HHO iteration")
    ax.set_ylabel("Best validation macro-F1 found so far")
    ax.set_title("HHO convergence: ensemble-weight optimization\n(16 hawks, 10 random seeds)")
    ax.legend(fontsize=8.5, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_hho_convergence.png", dpi=220)
    plt.close(fig)

    with open(f"{OUT}/convergence_results.json", "w") as f:
        json.dump({"curves": curves.tolist(), "mean_f1_per_iter": mean_f1.tolist()}, f, indent=2)
    print("Saved fig_hho_convergence.png and convergence_results.json")

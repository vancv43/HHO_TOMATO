"""Harris Hawks Optimization (Heidari et al., 2019) for continuous weight vectors.

Standard formulation: exploration/exploitation controlled by escaping energy E,
soft/hard besiege, and Levy-flight-based dives, exactly as in the original paper
(no metaheuristic novelty claimed here beyond applying it to ensemble-weight search).
"""
import math
import numpy as np


def levy(dim, beta=1.5, rng=None):
    rng = rng or np.random.default_rng()
    sigma = (
        math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0, sigma, dim)
    v = rng.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / beta))


def hho_minimize(fitness_fn, dim, lb, ub, n_hawks=16, max_iter=25, seed=0, log_convergence=False):
    """Minimizes fitness_fn(x) -> float over x in [lb, ub]^dim.

    Returns (best_x, best_fitness, convergence_curve) where convergence_curve
    is a list of best fitness found so far at each iteration (only if
    log_convergence=True, else None).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(lb, ub, size=(n_hawks, dim))
    fitness = np.array([fitness_fn(x) for x in X])
    best_idx = np.argmin(fitness)
    rabbit = X[best_idx].copy()
    rabbit_fit = fitness[best_idx]
    curve = [rabbit_fit] if log_convergence else None

    for t in range(max_iter):
        E1 = 2 * (1 - t / max_iter)  # decreasing energy factor
        for i in range(n_hawks):
            E0 = 2 * rng.random() - 1
            E = E1 * E0
            q = rng.random()
            if abs(E) >= 1:
                # exploration
                if q >= 0.5:
                    rand_hawk = X[rng.integers(0, n_hawks)]
                    r1, r2 = rng.random(), rng.random()
                    X[i] = rand_hawk - r1 * np.abs(rand_hawk - 2 * r2 * X[i])
                else:
                    X[i] = (rabbit - X.mean(axis=0)) - rng.random() * (lb + rng.random() * (ub - lb))
            else:
                J = 2 * (1 - rng.random())
                if abs(E) >= 0.5 and q >= 0.5:
                    X[i] = (rabbit - X[i]) - E * np.abs(J * rabbit - X[i])
                elif abs(E) < 0.5 and q >= 0.5:
                    X[i] = rabbit - E * np.abs(rabbit - X[i])
                elif abs(E) >= 0.5 and q < 0.5:
                    Y = rabbit - E * np.abs(J * rabbit - X[i])
                    Y = np.clip(Y, lb, ub)
                    if fitness_fn(Y) < fitness[i]:
                        X[i] = Y
                    else:
                        Z = Y + rng.random(dim) * levy(dim, rng=rng)
                        Z = np.clip(Z, lb, ub)
                        if fitness_fn(Z) < fitness[i]:
                            X[i] = Z
                else:
                    Y = rabbit - E * np.abs(J * rabbit - X.mean(axis=0))
                    Y = np.clip(Y, lb, ub)
                    if fitness_fn(Y) < fitness[i]:
                        X[i] = Y
                    else:
                        Z = Y + rng.random(dim) * levy(dim, rng=rng)
                        Z = np.clip(Z, lb, ub)
                        if fitness_fn(Z) < fitness[i]:
                            X[i] = Z
            X[i] = np.clip(X[i], lb, ub)
        fitness = np.array([fitness_fn(x) for x in X])
        gen_best = np.argmin(fitness)
        if fitness[gen_best] < rabbit_fit:
            rabbit_fit = fitness[gen_best]
            rabbit = X[gen_best].copy()
        if log_convergence:
            curve.append(rabbit_fit)

    return rabbit, rabbit_fit, curve

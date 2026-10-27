"""Reference methods that PSO is compared against."""
from __future__ import annotations
import itertools
import time
import numpy as np
from . import config
from .fitness import build_fitness


def default_params() -> dict:
    """scikit-learn defaults written out explicitly (the 'un-optimised' baseline)."""
    return dict(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                max_features=float(np.sqrt(30) / 30), criterion="gini")


def random_search(X_train, y_train, n_evals: int, seed: int = 0):
    """Uniform random search over the same space with the same evaluation budget as PSO."""
    rng = np.random.default_rng(seed)
    names = list(config.SEARCH_SPACE)
    lo = np.array([config.SEARCH_SPACE[k][0] for k in names])
    hi = np.array([config.SEARCH_SPACE[k][1] for k in names])
    isint = np.array([config.SEARCH_SPACE[k][2] for k in names])
    f = build_fitness(X_train, y_train)
    best_f, best_z, history = np.inf, None, []
    for _ in range(n_evals):
        z = lo + rng.random(len(lo)) * (hi - lo)
        z[isint] = np.floor(z[isint] + 0.5)
        z = np.clip(z, lo, hi)
        v = f(z)
        if v < best_f:
            best_f, best_z = v, z
        history.append(best_f)
    return best_z, best_f, history


def grid_search(X_train, y_train):
    """Coarse exhaustive grid (3 x 3 x 2 x 2 x 2 x 2 = 144 configs)."""
    f = build_fitness(X_train, y_train)
    grid = itertools.product([50, 100, 200],        # n_estimators
                             [10, 20, None],         # max_depth (None -> 40)
                             [2, 10],                # min_samples_split
                             [1, 4],                 # min_samples_leaf
                             [0.2, 0.5],             # max_features
                             [0, 1])                 # criterion index
    best_f, best_z, history, n = np.inf, None, [], 0
    for ne, md, mss, msl, mf, ci in grid:
        z = np.array([ne, 40 if md is None else md, mss, msl, mf, ci], float)
        v = f(z)
        n += 1
        if v < best_f:
            best_f, best_z = v, z
        history.append(best_f)
    return best_z, best_f, history, n

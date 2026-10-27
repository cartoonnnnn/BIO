"""A from-scratch Particle Swarm Optimization (global-best topology).

Update rules (Kennedy & Eberhart, 1995; inertia weight from Shi & Eberhart, 1998):

    v_i(t+1) = w(t) * v_i(t) + c1 * r1 * (pbest_i - x_i(t)) + c2 * r2 * (gbest - x_i(t))
    x_i(t+1) = x_i(t) + v_i(t+1)

with w(t) decreasing linearly from w_start to w_end, velocity clamped to +/- vmax and
positions clipped to the search bounds. Integer dimensions are rounded when a particle
is *evaluated*; the continuous position itself is kept so the swarm can keep moving smoothly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np


@dataclass
class PSOResult:
    best_position: np.ndarray
    best_fitness: float
    gbest_history: list = field(default_factory=list)   # best fitness so far, per iteration
    mean_history: list = field(default_factory=list)    # mean fitness of the swarm, per iteration
    diversity_history: list = field(default_factory=list)
    n_evaluations: int = 0
    position_log: list = field(default_factory=list)    # swarm positions per iteration (for plots)


class PSO:
    def __init__(self, objective: Callable[[np.ndarray], float], lower, upper, integer_mask=None,
                 n_particles=20, n_iterations=25, w_start=0.9, w_end=0.4, c1=2.0, c2=2.0,
                 vmax_frac=0.2, seed=0, verbose=False, log_positions=False):
        self.f = objective
        self.lo = np.asarray(lower, float)
        self.hi = np.asarray(upper, float)
        self.dim = len(self.lo)
        self.int_mask = np.zeros(self.dim, bool) if integer_mask is None else np.asarray(integer_mask, bool)
        self.n, self.T = n_particles, n_iterations
        self.w0, self.w1, self.c1, self.c2 = w_start, w_end, c1, c2
        self.vmax = vmax_frac * (self.hi - self.lo)
        self.rng = np.random.default_rng(seed)
        self.verbose, self.log_positions = verbose, log_positions
        self._cache: dict = {}
        self.n_evals = 0

    # -- helpers -------------------------------------------------------------------------
    def decode(self, x: np.ndarray) -> np.ndarray:
        """Continuous position -> evaluable position (integers rounded/floored, clipped)."""
        z = np.clip(x, self.lo, self.hi).copy()
        z[self.int_mask] = np.floor(z[self.int_mask] + 0.5)
        return np.clip(z, self.lo, self.hi)

    def _evaluate(self, x: np.ndarray) -> float:
        z = self.decode(x)
        key = tuple(np.round(z, 6))
        if key not in self._cache:              # identical decoded configs are not re-trained
            self._cache[key] = float(self.f(z))
            self.n_evals += 1
        return self._cache[key]

    # -- main loop -----------------------------------------------------------------------
    def run(self) -> PSOResult:
        span = self.hi - self.lo
        X = self.lo + self.rng.random((self.n, self.dim)) * span
        V = (self.rng.random((self.n, self.dim)) - 0.5) * span * 0.2
        fit = np.array([self._evaluate(x) for x in X])
        pbest_x, pbest_f = X.copy(), fit.copy()
        g = int(np.argmin(pbest_f))
        gbest_x, gbest_f = pbest_x[g].copy(), float(pbest_f[g])

        res = PSOResult(gbest_x, gbest_f)
        self._record(res, X, fit, gbest_f)

        for t in range(1, self.T + 1):
            w = self.w0 - (self.w0 - self.w1) * t / self.T
            r1 = self.rng.random((self.n, self.dim))
            r2 = self.rng.random((self.n, self.dim))
            V = w * V + self.c1 * r1 * (pbest_x - X) + self.c2 * r2 * (gbest_x - X)
            V = np.clip(V, -self.vmax, self.vmax)
            X = np.clip(X + V, self.lo, self.hi)
            fit = np.array([self._evaluate(x) for x in X])
            better = fit < pbest_f
            pbest_x[better], pbest_f[better] = X[better], fit[better]
            g = int(np.argmin(pbest_f))
            if pbest_f[g] < gbest_f:
                gbest_f, gbest_x = float(pbest_f[g]), pbest_x[g].copy()
            self._record(res, X, fit, gbest_f)
            if self.verbose:
                print(f"  iter {t:02d}/{self.T}  gbest={gbest_f:.5f}  mean={fit.mean():.5f}  evals={self.n_evals}")

        res.best_position, res.best_fitness = self.decode(gbest_x), gbest_f
        res.n_evaluations = self.n_evals
        return res

    def _record(self, res: PSOResult, X, fit, gbest_f):
        res.gbest_history.append(gbest_f)
        res.mean_history.append(float(np.mean(fit)))
        # swarm diversity = mean distance to the swarm centroid, in normalised units
        Xn = (X - self.lo) / (self.hi - self.lo)
        res.diversity_history.append(float(np.mean(np.linalg.norm(Xn - Xn.mean(0), axis=1))))
        if self.log_positions:
            res.position_log.append(X.copy())

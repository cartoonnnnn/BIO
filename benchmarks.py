"""Validate the PSO implementation on standard continuous test functions (global minimum = 0)."""
from __future__ import annotations
import numpy as np
from .pso import PSO


def sphere(x):     return float(np.sum(x ** 2))
def rastrigin(x):  return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))
def rosenbrock(x): return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
def ackley(x):
    n = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)

FUNCTIONS = {            # name: (function, lower, upper)
    "Sphere":     (sphere, -100, 100),
    "Rastrigin":  (rastrigin, -5.12, 5.12),
    "Rosenbrock": (rosenbrock, -30, 30),
    "Ackley":     (ackley, -32, 32),
}


def run_benchmarks(dim=10, runs=30, n_particles=30, n_iterations=200):
    out = {}
    for name, (fn, lo, hi) in FUNCTIONS.items():
        finals, curves = [], []
        for s in range(runs):
            r = PSO(fn, [lo] * dim, [hi] * dim, n_particles=n_particles, n_iterations=n_iterations,
                    seed=s).run()
            finals.append(r.best_fitness)
            curves.append(r.gbest_history)
        out[name] = dict(mean=float(np.mean(finals)), std=float(np.std(finals)),
                         best=float(np.min(finals)), worst=float(np.max(finals)),
                         mean_curve=np.mean(curves, axis=0).tolist())
    return out

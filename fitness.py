"""Fitness function for PSO: how good is a given Random Forest hyperparameter vector?"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from . import config


def vector_to_params(z) -> dict:
    """Decode a PSO position (ordered as config.SEARCH_SPACE) into RandomForest kwargs."""
    names = list(config.SEARCH_SPACE)
    d = dict(zip(names, z))
    return {
        "n_estimators": int(d["n_estimators"]),
        "max_depth": int(d["max_depth"]),
        "min_samples_split": int(d["min_samples_split"]),
        "min_samples_leaf": int(d["min_samples_leaf"]),
        "max_features": float(d["max_features"]),
        "criterion": config.CRITERIA[int(d["criterion_idx"])],
    }


def make_rf(params: dict, seed: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(**params, random_state=seed, n_jobs=-1)


def build_fitness(X_train, y_train):
    """Return f(z) = 1 - mean stratified-CV F1 on the TRAINING data only (minimised by PSO).

    The held-out test set is never touched by the optimiser, so the reported test
    metrics are an unbiased estimate of generalisation.
    """
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.CV_SEED)

    def fitness(z) -> float:
        model = make_rf(vector_to_params(z), seed=0)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        return 1.0 - float(np.mean(scores))

    return fitness

"""Metrics, McNemar test and training-time helpers."""
from __future__ import annotations
import time
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)


def metrics(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return dict(accuracy=accuracy_score(y_test, pred), precision=precision_score(y_test, pred),
                recall=recall_score(y_test, pred), f1=f1_score(y_test, pred),
                roc_auc=roc_auc_score(y_test, proba), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def fit_timed(model, X, y):
    t = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """McNemar's test (continuity-corrected) on paired predictions of two classifiers."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b = int(np.sum(a_ok & ~b_ok))     # A right, B wrong
    c = int(np.sum(~a_ok & b_ok))     # A wrong, B right
    stat = 0.0 if b + c == 0 else (abs(b - c) - 1) ** 2 / (b + c)
    return dict(only_a_correct=b, only_b_correct=c, statistic=stat, p_value=float(1 - chi2.cdf(stat, 1)))

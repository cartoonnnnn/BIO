"""End-to-end experiment: baseline RF vs PSO-tuned RF (+ random / grid search, + benchmarks).

Usage:  python run_experiments.py
The script is RESUMABLE: every expensive stage is cached in results/stages/, so if the run is
interrupted, simply start it again and it continues from the last finished stage.
Outputs: results/results.json, results/artifacts.pkl (then run make_plots.py for the figures)
"""
from __future__ import annotations
import json, pickle, time
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src import config
from src.data_utils import prepare
from src.pso import PSO
from src.fitness import build_fitness, vector_to_params, make_rf
from src.baselines import default_params, random_search, grid_search
from src.evaluate import metrics, fit_timed, mcnemar
from src.benchmarks import run_benchmarks

STAGES = config.RESULTS_DIR / "stages"


def cached(name, fn):
    """Run fn() once and cache its JSON-serialisable result on disk."""
    STAGES.mkdir(parents=True, exist_ok=True)
    path = STAGES / f"{name}.json"
    if path.exists():
        print(f"[cached] {name}", flush=True)
        return json.load(open(path))
    t0 = time.perf_counter()
    out = fn()
    json.dump(out, open(path, "w"))
    print(f"[done]   {name}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    return out


def main():
    config.RESULTS_DIR.mkdir(exist_ok=True)
    df, Xtr, Xte, ytr, yte, report = prepare()
    names = list(config.SEARCH_SPACE)
    lo = [config.SEARCH_SPACE[k][0] for k in names]
    hi = [config.SEARCH_SPACE[k][1] for k in names]
    im = [config.SEARCH_SPACE[k][2] for k in names]
    fitness = build_fitness(Xtr, ytr)

    # ---- PSO: one cached stage per independent run -------------------------------------
    def pso_stage(seed):
        def go():
            t0 = time.perf_counter()
            res = PSO(fitness, lo, hi, im, seed=seed, log_positions=(seed == config.MAIN_SEED),
                      verbose=True, **config.PSO).run()
            if seed == config.MAIN_SEED:
                np.save(STAGES / "pso_positions_main.npy", np.array(res.position_log))
            return dict(seed=seed, params=vector_to_params(res.best_position), best_cv_f1=1 - res.best_fitness,
                        evals=res.n_evaluations, search_time=time.perf_counter() - t0,
                        gbest_history=[1 - v for v in res.gbest_history],
                        mean_history=[1 - v for v in res.mean_history],
                        diversity_history=res.diversity_history)
        return cached(f"pso_seed{seed}", go)

    pso_raw = [pso_stage(s) for s in config.PSO_SEEDS]
    budget = int(round(np.mean([r["evals"] for r in pso_raw])))

    def rs_stage():
        t0 = time.perf_counter()
        z, f, hist = random_search(Xtr, ytr, budget, seed=0)
        return dict(params=vector_to_params(z), best_cv_f1=1 - f, evals=budget,
                    search_time=time.perf_counter() - t0, gbest_history=[1 - v for v in hist])
    rs_raw = cached("random_search", rs_stage)

    def gs_stage():
        t0 = time.perf_counter()
        z, f, hist, n = grid_search(Xtr, ytr)
        return dict(params=vector_to_params(z), best_cv_f1=1 - f, evals=n,
                    search_time=time.perf_counter() - t0, gbest_history=[1 - v for v in hist])
    gs_raw = cached("grid_search", gs_stage)

    bench = cached("benchmarks", lambda: run_benchmarks())

    # ---- final (cheap) evaluation on the untouched test set ------------------------------
    out = dict(data=report, eval_budget=budget, benchmarks=bench,
               pso_config=dict(**config.PSO, seeds=config.PSO_SEEDS),
               search_space={k: list(v) for k, v in config.SEARCH_SPACE.items()})
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.CV_SEED)
    bp = default_params()
    base_cv = cross_val_score(make_rf(bp, 0), Xtr, ytr, cv=cv, scoring="f1")
    base_model, bt = fit_timed(make_rf(bp, 0), Xtr, ytr)
    out["baseline"] = dict(params=bp, cv_f1=float(base_cv.mean()), train_time=bt, test=metrics(base_model, Xte, yte))

    models, runs = {}, []
    for r in pso_raw:
        m, tt = fit_timed(make_rf(r["params"], 0), Xtr, ytr)
        models[r["seed"]] = m
        runs.append({**r, "train_time": tt, "test": metrics(m, Xte, yte)})
    out["pso_runs"] = runs
    best = max(runs, key=lambda r: r["best_cv_f1"])      # selected by validation (CV) score, never test
    out["pso_selected"] = best
    pso_model = models[best["seed"]]

    rs_model, rtt = fit_timed(make_rf(rs_raw["params"], 0), Xtr, ytr)
    out["random_search"] = {**rs_raw, "train_time": rtt, "test": metrics(rs_model, Xte, yte)}
    gs_model, gtt = fit_timed(make_rf(gs_raw["params"], 0), Xtr, ytr)
    out["grid_search"] = {**gs_raw, "train_time": gtt, "test": metrics(gs_model, Xte, yte)}

    out["mcnemar_pso_vs_baseline"] = mcnemar(yte.values, pso_model.predict(Xte), base_model.predict(Xte))
    f1s = np.array([r["test"]["f1"] for r in runs]); accs = np.array([r["test"]["accuracy"] for r in runs])
    cvs = np.array([r["best_cv_f1"] for r in runs])
    out["pso_summary"] = dict(f1_mean=float(f1s.mean()), f1_std=float(f1s.std()), acc_mean=float(accs.mean()),
                              acc_std=float(accs.std()), cv_mean=float(cvs.mean()), cv_std=float(cvs.std()),
                              time_mean=float(np.mean([r["search_time"] for r in runs])))
    out["feature_importance"] = {"baseline": dict(zip(Xtr.columns, base_model.feature_importances_.tolist())),
                                 "pso": dict(zip(Xtr.columns, pso_model.feature_importances_.tolist()))}
    json.dump(out, open(config.RESULTS_DIR / "results.json", "w"), indent=2)
    with open(config.RESULTS_DIR / "artifacts.pkl", "wb") as fh:
        pickle.dump(dict(pso_positions=np.load(STAGES / "pso_positions_main.npy"), y_test=yte.values,
                         proba_base=base_model.predict_proba(Xte)[:, 1], proba_pso=pso_model.predict_proba(Xte)[:, 1],
                         proba_rs=rs_model.predict_proba(Xte)[:, 1], proba_gs=gs_model.predict_proba(Xte)[:, 1],
                         features=list(Xtr.columns)), fh)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

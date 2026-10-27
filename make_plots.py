"""Create every figure used in the report / slides from results/results.json + artifacts.pkl."""
import json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
from src import config
from src.data_utils import load_raw, clean

R = json.load(open(config.RESULTS_DIR / "results.json"))
A = pickle.load(open(config.RESULTS_DIR / "artifacts.pkl", "rb"))
FIG = config.FIG_DIR
FIG.mkdir(parents=True, exist_ok=True)
BASE, PSO_C, RS_C, GS_C = "#64748B", "#E8590C", "#2F9E44", "#7048E8"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "figure.dpi": 100, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})


def save(name): plt.savefig(FIG / name); plt.close()


# 1 ---- dataset overview ------------------------------------------------------------
raw = load_raw(); df, rep = clean(raw)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].bar(["Raw", "After de-duplication"], [rep["raw_rows"], rep["clean_rows"]], color=[BASE, PSO_C])
for i, v in enumerate([rep["raw_rows"], rep["clean_rows"]]): ax[0].text(i, v + 150, f"{v:,}", ha="center")
ax[0].set_title("Rows before / after cleaning"); ax[0].set_ylabel("Websites")
cc = rep["class_counts"]
ax[1].bar(["Legitimate", "Phishing"], [cc["legitimate"], cc["phishing"]], color=["#2F9E44", "#E03131"])
for i, v in enumerate([cc["legitimate"], cc["phishing"]]): ax[1].text(i, v + 40, f"{v:,}", ha="center")
ax[1].set_title("Class distribution (clean data)")
save("fig_dataset_overview.png")

# 2 ---- correlation heatmap of features with target ---------------------------------
corr = df.corr()["Result"].drop("Result").sort_values()
fig, ax = plt.subplots(figsize=(7, 8))
ax.barh(corr.index, corr.values, color=[PSO_C if v > 0 else BASE for v in corr.values])
ax.set_title("Correlation of each feature with 'phishing' label"); ax.set_xlabel("Pearson r")
save("fig_feature_target_corr.png")

# 3 ---- PSO convergence (selected / main run) ---------------------------------------
main = next(r for r in R["pso_runs"] if r["seed"] == config.MAIN_SEED)
it = np.arange(len(main["gbest_history"]))
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(it, main["gbest_history"], "-o", color=PSO_C, label="Global best (CV F1)", lw=2)
ax.plot(it, main["mean_history"], "--s", color=BASE, label="Swarm mean (CV F1)", lw=1.5, ms=4)
ax.axhline(R["baseline"]["cv_f1"], color="k", ls=":", label=f"Baseline RF ({R['baseline']['cv_f1']:.4f})")
ax.set_xlabel("PSO iteration"); ax.set_ylabel("3-fold CV F1-score"); ax.legend(loc="lower right")
ax.set_title(f"PSO convergence curve (run with seed {main['seed']})")
save("fig_pso_convergence.png")

# 4 ---- all seeds -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for r in R["pso_runs"]: ax.plot(r["gbest_history"], lw=1.8, label=f"seed {r['seed']}")
ax.axhline(R["baseline"]["cv_f1"], color="k", ls=":", label="Baseline")
ax.set_xlabel("PSO iteration"); ax.set_ylabel("Best CV F1 so far"); ax.legend(ncol=3, fontsize=9)
ax.set_title("Convergence over 5 independent PSO runs")
save("fig_pso_all_seeds.png")

# 5 ---- swarm diversity ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4))
d = np.array([r["diversity_history"] for r in R["pso_runs"]])
ax.plot(d.mean(0), color=PSO_C, lw=2.2, label="mean of 5 runs")
ax.fill_between(range(d.shape[1]), d.min(0), d.max(0), color=PSO_C, alpha=.2, label="min-max")
ax.set_xlabel("PSO iteration"); ax.set_ylabel("Swarm diversity (normalised)")
ax.set_title("Swarm diversity: exploration to exploitation"); ax.legend()
save("fig_pso_diversity.png")

# 6 ---- baseline vs optimized metrics -----------------------------------------------
mets = ["accuracy", "precision", "recall", "f1"]
sel = R["pso_selected"]["test"]; bas = R["baseline"]["test"]
x = np.arange(4); w = .36
fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - w / 2, [bas[m] for m in mets], w, color=BASE, label="Baseline RF (default)")
b2 = ax.bar(x + w / 2, [sel[m] for m in mets], w, color=PSO_C, label="PSO-optimised RF")
for bars in (b1, b2):
    for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .002, f"{b.get_height():.3f}", ha="center", fontsize=9)
lo_ = min(min(bas[m] for m in mets), min(sel[m] for m in mets))
ax.set_ylim(max(0, lo_ - .05), 1.0); ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1-score"])
ax.set_ylabel("Score (test set)"); ax.legend(loc="lower right"); ax.set_title("Baseline vs PSO-optimised Random Forest")
save("fig_baseline_vs_pso.png")

# 7 ---- all methods -------------------------------------------------------------------
meths = [("Baseline", R["baseline"]["test"], BASE), ("Grid search", R["grid_search"]["test"], GS_C),
         ("Random search", R["random_search"]["test"], RS_C), ("PSO", sel, PSO_C)]
fig, ax = plt.subplots(figsize=(8, 4.5))
w = .2
for i, (n, m, c) in enumerate(meths):
    ax.bar(x + (i - 1.5) * w, [m[k] for k in mets], w, color=c, label=n)
allv = [m[k] for _, m, _ in meths for k in mets]
ax.set_ylim(min(allv) - .03, 1.0); ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1-score"])
ax.legend(ncol=4, loc="upper center", fontsize=9); ax.set_ylabel("Score (test set)")
ax.set_title("PSO vs baseline, grid search and random search")
save("fig_all_methods.png")

# 8 ---- search efficiency ---------------------------------------------------------------
n_p = config.PSO["n_particles"]
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ph = np.array([r["gbest_history"] for r in R["pso_runs"]])
xs = (np.arange(ph.shape[1]) + 1) * n_p
ax.plot(xs, ph.mean(0), color=PSO_C, lw=2.2, label="PSO (mean of 5 runs)")
ax.fill_between(xs, ph.min(0), ph.max(0), color=PSO_C, alpha=.2)
ax.plot(np.arange(1, len(R["random_search"]["gbest_history"]) + 1), R["random_search"]["gbest_history"], color=RS_C, lw=2, label="Random search")
ax.plot(np.arange(1, len(R["grid_search"]["gbest_history"]) + 1), R["grid_search"]["gbest_history"], color=GS_C, lw=2, label="Grid search")
ax.set_xlabel("Number of fitness evaluations"); ax.set_ylabel("Best CV F1 so far"); ax.legend(loc="lower right")
ax.set_title("Search efficiency (best-so-far vs evaluations)")
save("fig_search_efficiency.png")

# 9 ---- confusion matrices ------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9, 3.9))
for ax, (n, m) in zip(axs, [("Baseline RF", bas), ("PSO-optimised RF", sel)]):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(cm, cmap="Oranges" if "PSO" in n else "Blues"); ax.grid(False)
    for i in range(2):
        for j in range(2): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16, fontweight="bold")
    ax.set_xticks([0, 1], ["Legit", "Phishing"]); ax.set_yticks([0, 1], ["Legit", "Phishing"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(n)
save("fig_confusion_matrices.png")

# 10 ---- ROC ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 5))
for n, p, c in [("Baseline", A["proba_base"], BASE), ("Grid", A["proba_gs"], GS_C),
                ("Random", A["proba_rs"], RS_C), ("PSO", A["proba_pso"], PSO_C)]:
    fpr, tpr, _ = roc_curve(A["y_test"], p)
    auc = {"Baseline": R["baseline"]["test"]["roc_auc"], "Grid": R["grid_search"]["test"]["roc_auc"],
           "Random": R["random_search"]["test"]["roc_auc"], "PSO": sel["roc_auc"]}[n]
    ax.plot(fpr, tpr, color=c, lw=2, label=f"{n} (AUC={auc:.4f})")
ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves on the test set"); ax.legend(loc="lower right", fontsize=9)
save("fig_roc.png")

# 11 ---- feature importance --------------------------------------------------------------
fi = R["feature_importance"]; top = sorted(fi["pso"], key=fi["pso"].get, reverse=True)[:12][::-1]
fig, ax = plt.subplots(figsize=(7.5, 5))
yy = np.arange(len(top))
ax.barh(yy + .2, [fi["baseline"][k] for k in top], .4, color=BASE, label="Baseline")
ax.barh(yy - .2, [fi["pso"][k] for k in top], .4, color=PSO_C, label="PSO-optimised")
ax.set_yticks(yy, top); ax.legend(loc="lower right"); ax.set_xlabel("Gini / impurity importance")
ax.set_title("Top-12 features")
save("fig_feature_importance.png")

# 12 ---- swarm movement ------------------------------------------------------------------
P = A["pso_positions"]                      # (T+1, n, dim)
names = list(config.SEARCH_SPACE)
i_ne, i_mf = names.index("n_estimators"), names.index("max_features")
snaps = [0, P.shape[0] // 2, P.shape[0] - 1]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
for ax, t in zip(axs, snaps):
    ax.scatter(P[t, :, i_ne], P[t, :, i_mf], c=PSO_C, s=45, alpha=.8, edgecolor="k", lw=.4)
    ax.set_title(f"Iteration {t}"); ax.set_xlabel("n_estimators")
axs[0].set_ylabel("max_features")
fig.suptitle("Swarm positions in two of the six dimensions", y=1.02)
save("fig_swarm_movement.png")

# 13 ---- benchmark convergence ----------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
for ax, (n, v) in zip(axs, R["benchmarks"].items()):
    ax.semilogy(np.maximum(v["mean_curve"], 1e-16), color=PSO_C, lw=2)
    ax.set_title(n); ax.set_xlabel("Iteration")
axs[0].set_ylabel("Mean best value (30 runs)")
save("fig_benchmarks.png")

# 14 ---- run-to-run stability --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4))
f1s = [r["test"]["f1"] for r in R["pso_runs"]]
ax.bar([f"seed {r['seed']}" for r in R["pso_runs"]], f1s, color=PSO_C)
ax.axhline(R["baseline"]["test"]["f1"], color="k", ls=":", label="Baseline F1")
ax.set_ylim(min(f1s + [R["baseline"]["test"]["f1"]]) - .01, max(f1s) + .008); ax.legend(loc="lower right")
ax.set_ylabel("Test F1-score"); ax.set_title("Stability across independent PSO runs")
save("fig_pso_stability.png")
print("figures written to", FIG)

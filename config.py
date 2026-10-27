"""Central configuration: paths, random seeds and the PSO / search-space settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "phishing_websites.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

TARGET = "Result"          # +1 = legitimate, -1 = phishing in the raw file
SPLIT_SEED = 42            # train/test split
TEST_SIZE = 0.20
CV_FOLDS = 3               # folds used *inside* the fitness function
CV_SEED = 7

# ---- Hyperparameter search space for the Random Forest -------------------------------
# name: (lower, upper, is_integer)
SEARCH_SPACE = {
    "n_estimators":      (10,   200,  True),
    "max_depth":         (3,    40,   True),
    "min_samples_split": (2,    20,   True),
    "min_samples_leaf":  (1,    10,   True),
    "max_features":      (0.10, 1.00, False),   # fraction of the 30 features per split
    "criterion_idx":     (0,    2.999, True),   # 0=gini, 1=entropy, 2=log_loss (floored)
}
CRITERIA = ["gini", "entropy", "log_loss"]

# ---- PSO settings ---------------------------------------------------------------------
PSO = dict(
    n_particles=12,
    n_iterations=15,
    w_start=0.9,      # inertia weight, linearly decreased
    w_end=0.4,
    c1=2.0,           # cognitive coefficient
    c2=2.0,           # social coefficient
    vmax_frac=0.2,    # velocity clamp = 20 % of each dimension's range
)
PSO_SEEDS = [1, 2, 3, 4, 5]   # independent PSO runs
MAIN_SEED = 1                 # run used for the headline convergence plots

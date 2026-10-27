# Phishing Website Detection Using Particle Swarm Optimization (PSO)

Bio-Inspired Algorithms (23DS7BSBIO) AAT, B.M.S. College of Engineering.

A Random Forest classifies websites as phishing or legitimate; a from-scratch PSO tunes six Random Forest
hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `criterion`).
Results are compared against the default Random Forest, grid search and random search.

## Dataset
UCI Phishing Websites (Mohammad & McCluskey, 2012), 11,055 sites x 30 features, in `data/phishing_websites.csv`.
5,206 exact duplicate rows are removed before the 80/20 stratified split to avoid train/test leakage.

## Run
```bash
pip install -r requirements.txt
python run_experiments.py   # resumable; caches stages in results/stages/ (about 25 min on one CPU core)
python make_plots.py        # writes results/figures/*.png
```
`results/results.json` holds every number used in the report and slides.

## Layout
`src/pso.py` PSO | `src/fitness.py` CV fitness | `src/baselines.py` grid/random search | `src/data_utils.py` cleaning/split |
`src/evaluate.py` metrics + McNemar | `src/benchmarks.py` Sphere/Rastrigin/Rosenbrock/Ackley

## Honest summary of results
PSO raised cross-validated F1 and test ROC-AUC in all 5 runs, but test accuracy/F1 did not differ significantly
from the default Random Forest (McNemar p = 0.80). Re-run on your own machine; exact numbers can vary with library versions.

## Publish to GitHub
```bash
git init && git add . && git commit -m "Phishing detection with PSO-tuned Random Forest"
git branch -M main && git remote add origin <your-repo-url> && git push -u origin main
```

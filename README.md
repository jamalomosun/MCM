# DWTS MCM/ICM Problem C

This repository is a record of my work for our team’s MCM/ICM 2026 Problem C work, “Strictly Com(e)puting: An Analysis of Scoring on Dancing with the Stars.”

Models

Model 1 — Quadratic / Integer ProgrammingUses judge scores and observed eliminations/final placements to constrain hidden fan support. Percentage scoring is handled with quadratic programming; rank scoring with integer programming.

Model 2 — Bayesian Popularity InferenceTreats contestant fan popularity as a latent variable, models weekly outcomes with a multinomial-logit likelihood, and estimates popularity with HMC/rStan. The fan estimates used to support this work are here:

https://github.com/raymondkim777/MCM-Prep/blob/main/mcm/data/dwts_fan_estimates.csv

Model 3 — Feature AnalysisUses the contestant features from the paper—age, ballroom partner, industry, home state, and home country/region—to study placement, average judge score, and fan support. The workflow is split into independent processes:

model3/
  01_mlr.py              # linear baseline
  02_random_forest.py    # random forest + MDI
  03_mlp.py              # multilayer perceptron
  04_mlp_shap.py         # SHAP analysis of MLP
  05_compare_models.py   # model comparison
  06_ensemble.py         # RF + MLP ensemble (extension)
  07_grouped_importance.py # grouped/permutation importance (extension)

The paper reports age as the strongest Model-3 feature and uses MDI/SHAP to interpret the nonlinear models.

Scoring-system extension

The paper proposes equal judge/fan weighting early, a 60/40 judge/fan weighting after halfway, and equal weighting again in the finale. scoring/02_optimize_dynamic_weights.py formalizes that rationale as an optimization problem. The optimizer is a new extension, not a result reported in the original paper.

Reproducibility

The main improvements recommended for future runs are:

use season-level cross-validation to prevent leakage;

pin Python/R package versions and record input hashes;

save seeds, preprocessing objects, model parameters, and Stan diagnostics;

report uncertainty for MDI, permutation importance, and SHAP;

propagate Model-2 posterior draws into Model 3 rather than using only posterior means;

distinguish prediction from causal interpretation;

keep original-paper outputs separate from extension outputs.

Data

The supplied competition CSV supports the judge-side and contestant-feature analyses.  Absolute fan-vote counts additionally require the viewership-derived data described in the paper.

Running Model 3

python model3/01_mlr.py --data data/2026_MCM_Problem_C_Data.csv --target placement
python model3/02_random_forest.py --data data/2026_MCM_Problem_C_Data.csv --target placement
python model3/03_mlp.py --data data/2026_MCM_Problem_C_Data.csv --target placement
python model3/04_mlp_shap.py --data data/2026_MCM_Problem_C_Data.csv --target placement
python model3/06_ensemble.py --data data/2026_MCM_Problem_C_Data.csv --target placement

Paper result

The original paper reports 93.56% agreement between Model 2 estimates and Model-1 constraints and identifies age as the strongest Model-3 feature. 
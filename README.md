# DWTS MCM/ICM Problem C — Reproduction and Extensions

This repository reconstructs the modeling pipeline described in the submitted MCM/ICM 2026 Problem C paper, **“Strictly Com(e)puting: An Analysis of Scoring on Dancing with the Stars.”**

## What the paper does

The paper treats hidden fan voting as an unobserved quantity and uses three layers of modeling:

1. **Model 1 — Quadratic / Integer Programming:** derives deterministic feasible regions for hidden fan support from observed judge scores and eliminations. Percentage scoring is handled with quadratic programming; rank scoring with integer programming.
2. **Model 2 — Bayesian Popularity Inference:** assigns each contestant a latent popularity parameter, uses a multinomial-logit elimination/winner likelihood, and estimates posterior popularity with Hamiltonian Monte Carlo in rStan.
3. **Model 3 — Multi-Model Feature Analysis:** uses multiple linear regression as a baseline, then random forests and a multilayer perceptron to model nonlinear relationships between contestant characteristics and three success measures: placement, average judge score, and Bayesian fan-support estimates. Feature importance is summarized with MDI and SHAP.

The paper reports that Model 2 matches 362 of 388 Model-1 constraints (93.56%) and that age is the strongest Model-3 feature, with random-forest MDI about 59.45%.

## Repository layout

```text
model3/
  common.py
  01_mlr.py
  02_random_forest.py
  03_mlp.py
  04_mlp_shap.py
  05_compare_models.py
  06_ensemble.py
scoring/
  01_simulate_scoring_systems.py
  02_optimize_dynamic_weights.py
outputs/
```

## Model 3 in detail

### MLR baseline

`model3/01_mlr.py` is the linear baseline. It encodes the five feature families used by the paper:

- celebrity age
- ballroom partner
- industry
- home state
- home country/region

Categorical variables are one-hot encoded and age is imputed/scaled. The target can be `placement`, `avg_judge_score`, or `fan_support` when Model-2 estimates are supplied.

### Random forest

`model3/02_random_forest.py` fits a `RandomForestRegressor` and exports impurity-based feature importance (MDI). The paper's reported age importance can be compared to the transformed/aggregated importance output.

### MLP

`model3/03_mlp.py` fits a nonlinear `MLPRegressor`. This is the paper's neural-network alternative to linear regression.

### SHAP

`model3/04_mlp_shap.py` explains the fitted MLP with permutation SHAP and aggregates one-hot variables back to the original feature families. This is deliberately kept separate from model fitting so that the predictive model and the explanation method can be tested independently.

SHAP is a game-theoretic feature-attribution framework; the current SHAP project documents both model-specific tree explanations and model-agnostic explainers. See https://github.com/shap/shap.

### RF + MLP ensemble

`model3/06_ensemble.py` is an extension beyond the paper. It fits both models and chooses a convex weight using K-fold cross-validation:

\[
\hat y_{ens}=w\hat y_{RF}+(1-w)\hat y_{MLP},\qquad 0\le w\le1.
\]

The test set is held out until after the weight is learned. This avoids choosing the ensemble weight on the same data used for final evaluation.

## Scoring-system optimization

The paper proposes a three-phase policy:

- first half: 50% judge / 50% fan;
- after halfway point: 60% judge / 40% fan;
- finale: 50% judge / 50% fan.

`scoring/02_optimize_dynamic_weights.py` turns that rationale into an explicit optimization problem. It searches over:

- early judge weight,
- late judge weight,
- the fraction of the season where late weighting begins.

The objective is a weighted combination of:

1. disagreement with observed eliminations;
2. survival of the worst-judged contestant;
3. excessive reduction of fan influence.

These objective terms are **an extension**, not a result reported by the paper. Change their weights and document the choice before comparing policies.


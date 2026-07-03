# Student Response Prediction

Predicting whether a student will answer a diagnostic question correctly, from a large, highly sparse matrix of past student–question responses. This is a matrix-completion / collaborative-filtering problem: most student–question pairs are unobserved, and the goal is to impute the missing entries as accurately as possible.

Course project for **CSC311 – Introduction to Machine Learning** (University of Toronto). All models are implemented from scratch with NumPy / SciPy / scikit-learn (plus PyTorch for the optional neural net); no black-box recommender libraries were used.

## Problem

Each entry `c_ij ∈ {0, 1}` indicates whether student `i` answered question `j` correctly. The observed matrix is sparse, so we train on the observed entries and evaluate imputation accuracy on held-out validation and test splits. Question-level metadata (the academic subjects each question belongs to) is also available and is used by one of the Part B models.

## Results

Test accuracy on the held-out set (binary correct / incorrect prediction):

| Model | Validation | Test |
|---|---|---|
| KNN — user-based collaborative filtering | 0.689 | 0.684 |
| KNN — item-based collaborative filtering | — | 0.668 |
| Base ALS matrix factorization | 0.697 | 0.694 |
| **Modified ALS (subject-aware, Part B)** | **0.704** | **0.711** |
| Hybrid KNN (user + item, Part B) | 0.700 | 0.692 |

The **subject-aware Modified ALS** is the strongest single model at **0.711** test accuracy. The Part A ensemble (KNN + IRT + ALS via bootstrap bagging) improves over its individual base models by reducing variance.

## Approach

**Part A — baseline models** (each implemented and tuned independently):

- **KNN collaborative filtering** — user-based and item-based imputation with `KNNImputer`, sweeping `k` to pick `k*`.
- **Item Response Theory (IRT)** — a one-parameter logistic model, `p(c_ij = 1) = σ(θ_i − β_j)`, fit by gradient ascent on the log-likelihood.
- **Matrix factorization** — SVD reconstruction and Alternating Least Squares (ALS) trained with SGD over the observed entries.
- **Ensemble** — bootstrap bagging over KNN, IRT, and ALS; predictions are averaged to reduce variance.

**Part B — two improved models, then combined:**

- **Modified ALS (subject-aware).** Base ALS initializes both latent factor matrices randomly, which converges to noisy local minima and makes the chosen rank `k*` unstable. This variant adds a second factor pair `U_s`, `Z_s` derived from question–subject metadata, so the prediction becomes `R_ij ≈ Uᵀ Z + U_sᵀ Z_s`. Grounding part of the factorization in real subject structure improved both validation (0.704) and test (0.711) accuracy over base ALS (0.697 / 0.694).
- **Hybrid KNN.** A weighted combination of user-based and item-based KNN imputation, `x̂ = α·x̂_user + (1−α)·x̂_item`, tuned over `k` and `α` (best at `k = 11`, `α = 0.5`).
- **Part B ensemble** combining the two Part B models.

Full derivations, hyperparameters, and plots are in the [final report](CSC311_project_final_report/CSC311_project_final_report.pdf).

## Repository structure

```
.
├── data/                       # response matrix + question/student/subject metadata
├── knn.py                      # Part A: user/item KNN collaborative filtering
├── item_response.py            # Part A: IRT (1PL) model
├── matrix_factorization.py     # Part A: SVD + ALS matrix factorization
├── neural_network.py           # Part A: optional autoencoder baseline (PyTorch)
├── majority_vote.py            # trivial baseline
├── ensemble.py                 # Part A: bagging ensemble of KNN + IRT + ALS
├── part_b_zengzixuan.py        # Part B: subject-aware Modified ALS
├── part_b_liuhao.py            # Part B: Hybrid KNN
├── part_b_ensemble.py          # Part B: ensemble of the two Part B models
└── CSC311_project_final_report/ # LaTeX source + compiled PDF report
```

## Running

Requires Python 3.8+.

```bash
pip install numpy scipy scikit-learn matplotlib torch
```

Each model runs standalone from the project root (scripts expect the data in `./data`). For example:

```bash
python knn.py                  # KNN baselines + accuracy-vs-k plots
python item_response.py        # train IRT, report validation/test accuracy
python matrix_factorization.py # SVD + ALS
python ensemble.py             # Part A bagging ensemble
python part_b_zengzixuan.py    # subject-aware Modified ALS (best single model)
python part_b_liuhao.py        # Hybrid KNN
python part_b_ensemble.py      # Part B ensemble
```

## Team & contributions

Group of three. This repository is maintained by **Zixuan Zeng** (group lead).

- **Zixuan Zeng** — group lead. Full Part A implementation (the group's stable version). Designed and implemented the **subject-aware Modified ALS** model (Part B, best single model at 0.711) and the Part B ensemble. Wrote the LaTeX final report.
- **Hao Liu** — implemented the **Hybrid KNN** model (Part B); contributed an independent Part A pass and to the report.
- **Howard He** — independent Part A pass and proofreading.

Part A was deliberately attempted independently by each member to get familiar with the problem; the group adopted a single stable version for the final submission. The per-member Part B models are in the correspondingly named `part_b_*.py` files.

# Phase 6: Machine Learning

Implemented in [`src/train_models.py`](../src/train_models.py). Run
`python src/train_models.py` to reproduce every number below.

## 6.1 A Bug Caught and Fixed Mid-Phase (documented, not hidden)

An earlier version of this pipeline built the `ColumnTransformer` with
`OneHotEncoder(handle_unknown="ignore")` at its default settings, which
outputs a **sparse matrix**. `ColumnTransformer` propagates sparse output
to the whole combined feature matrix whenever any one transformer is
sparse. This interacts badly with XGBoost specifically: **XGBoost's
`DMatrix` construction treats structurally-absent (unstored) entries in a
sparse matrix as *missing values*, not literal zeros.** Scikit-learn's own
estimators (Logistic Regression, Decision Tree, Random Forest) don't have
this semantic — they read sparse zeros as real zeros.

This mattered here specifically because **61.6% of the numeric feature
block was legitimately `0`** — `change`, `diabetesMed`, `had_prior_inpatient`,
most medication ordinal columns, and `number_inpatient` for the majority of
patients are all commonly zero. Feeding that as a sparse matrix silently
told XGBoost that most of its own input was missing.

**How it was caught**: a later phase (model explainability) computed
predictions two different ways that should have been mathematically
identical — once via the full pipeline, once by manually transforming then
calling the model directly — and got a 4x difference in the model's
top prediction (0.948 vs 0.243 probability) on the exact same patient. That
inconsistency was the signal something was wrong; tracing it back
identified the sparse/dense mismatch as the root cause.

**The fix**: force `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`,
so every model — not just XGBoost — trains and predicts on an unambiguous
dense array. All results below reflect the corrected pipeline.

**How much did it actually change?** Less than expected, reassuringly:
XGBoost's ROC-AUC moved from 0.6671 to 0.6670 — effectively unchanged,
because XGBoost has robust built-in handling for missing values (it learns
a default split direction for them) and apparently compensated reasonably
well even with the corrupted encoding. What *did* change meaningfully was
the model's **predicted probability values and top-decile lift** (2.42x →
2.36x) and, much more importantly, the **entire SHAP explainability
analysis in Phase 7**, which depends on the model's internal split
structure being sound — that phase required a full redo, documented there.

## 6.2 Three Design Decisions Made Before Training Anything

**1. Patient-level train/test split.** `GroupShuffleSplit` on `patient_nbr`,
verified zero overlap:
```
Train: 79,596 encounters (55,983 patients)
Test:  19,724 encounters (13,996 patients)
```

**2. Preprocessing fit only on the training fold**, inside a single
scikit-learn `Pipeline` per model — the leakage-prevention step promised in
Phase 4.11.

**3. Class imbalance handled via class weighting, not resampling.**
`class_weight="balanced"` (sklearn models) and `scale_pos_weight≈7.76`
(XGBoost) penalize mistakes on the minority (readmitted) class more
heavily, chosen over SMOTE-style resampling to avoid synthetic patient
records.

## 6.3 Model Comparison — Corrected Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Top-decile lift |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6661 | 0.1793 | 0.5491 | 0.2704 | 0.6598 | 2.25x |
| Decision Tree | 0.6647 | 0.1721 | 0.5185 | 0.2584 | 0.6427 | 2.24x |
| Random Forest | 0.6726 | 0.1807 | 0.5396 | 0.2708 | 0.6637 | 2.29x |
| **XGBoost** | 0.6614 | 0.1802 | 0.5648 | 0.2732 | **0.6670** | **2.36x** |

**Winner: XGBoost**, by ROC-AUC — the primary metric per Phase 1's success
criteria, since it measures ranking quality independent of threshold.

## 6.4 Reading These Numbers Correctly

**Accuracy (~66%) looks mediocre despite reasonable AUC (~0.67)** because
`class_weight`/`scale_pos_weight` deliberately shift the decision threshold
to trade accuracy for recall — the right trade here, since missing a
high-risk patient costs more than one unnecessary follow-up call.

**Predicted probabilities are not well-calibrated.** After the fix, the
corrected model's probability outputs on the test set range from 0.05 to
0.945 with a mean of 0.448 — far wider and higher than the 11.4% true base
rate would suggest for a calibrated model. This is a direct, expected
consequence of aggressive class weighting: it reshapes the decision surface
for better ranking at the cost of the raw probability number meaning
"true probability of readmission." **Every metric used in this project
(ROC-AUC, precision/recall, top-decile lift) is rank-based and remains
valid** — only a literal, calibrated-probability interpretation of a single
score would be misleading. This is worth stating explicitly to a
stakeholder: "top 10% by model score" is a reliable claim; "this specific
patient has a 73% chance of readmission" is not, without a separate
calibration step (e.g. Platt scaling) that this project doesn't include.

## 6.5 Why XGBoost Won (and why the margin is modest)

- **Decision Tree** (single tree) is weakest — prone to high variance,
  overfits specific training patterns.
- **Random Forest** improves via *bagging* (many trees on bootstrapped
  samples, averaged) — reduces variance.
- **XGBoost** improves further via *boosting* (each tree corrects the
  previous ensemble's errors) — reduces bias too, and generally handles
  nonlinear feature interactions (e.g. diagnosis × utilization from Phase 5)
  better than a single linear boundary.

Logistic Regression (0.6598 AUC) remains close behind XGBoost (0.6670) — a
gap of 0.0072. This still suggests the underlying relationships are close
to additive/linear without huge nonlinear interaction effects, a useful,
nuanced fact for an interview: a simpler, more interpretable model gets
most of the way to the ensemble's performance here.

## 6.6 Beating the Phase 5 SQL Benchmark

| | Top-decile lift |
|---|---|
| Phase 5 SQL rule (hand-weighted) | 1.9x |
| Logistic Regression | 2.25x |
| Decision Tree | 2.24x |
| Random Forest | 2.29x |
| **XGBoost** | **2.36x** |

XGBoost's lift is about 24% better than the simple SQL rule (2.36 vs 1.9) —
a real, quantifiable case for the added complexity of a trained model over
a manual rule, even after the correction.

## 6.7 Fairness Check

| | ROC-AUC |
|---|---|
| XGBoost without race/payer_code | 0.6670 |
| XGBoost with race/payer_code | 0.6685 |
| **AUC uplift** | **+0.0015** |

Adding two demographic/socioeconomic-proxy features back buys essentially
no predictive improvement — well within noise for a dataset this size.
This remains a clean, evidence-based case for excluding `race` and
`payer_code` from a production version of this model at effectively zero
performance cost.

## 6.8 Business Insight Summary

> **(1)** XGBoost beats the SQL-only benchmark by ~24% on top-decile lift —
> a concrete case for deploying a model, even after correcting a real
> encoding bug that could easily have gone unnoticed. **(2)** The bug itself
> is a useful lesson: always sanity-check that two mathematically-equivalent
> code paths agree, especially at pipeline boundaries between scikit-learn
> and a library (like XGBoost) with its own missing-value semantics.
> **(3)** `race` and `payer_code` can be dropped at essentially zero cost to
> performance (+0.0015 AUC) — removing a fairness liability for free.

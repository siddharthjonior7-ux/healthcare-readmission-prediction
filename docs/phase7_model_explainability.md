# Phase 7: Model Explainability (SHAP)

Implemented in [`src/explainability.py`](../src/explainability.py). Run
`python src/explainability.py` to reproduce every number and figure below,
using the **corrected** XGBoost pipeline from Phase 6 (see Phase 6.1 for
the sparse/dense bug this phase's own cross-check originally caught).

## 7.1 This Phase Is How the Phase 6 Bug Was Actually Found

Worth stating plainly: computing SHAP values requires feeding data to the
model through a manual `preprocessor.transform()` step (SHAP's
`TreeExplainer` needs a dense array with known feature names, not a raw
pipeline call). Comparing that manual path's predictions against the full
pipeline's own `predict_proba()` on the *same rows* — a basic consistency
check — showed a 4x disagreement (0.243 vs 0.948 on the same patient). That
disagreement was the signal that led to finding and fixing the sparse-
matrix/missing-value bug described in Phase 6.1. This is a good example of
why running a model through more than one lens (raw evaluation, then
explainability) is valuable beyond the explanations themselves — it's a
built-in consistency check on the whole pipeline.

## 7.2 What SHAP Actually Computes

SHAP (SHapley Additive exPlanations) is grounded in cooperative game
theory. For a single prediction, each feature gets a "SHAP value" — its
fair share of the difference between this patient's predicted score and
the average score across all patients. Every feature's SHAP values sum
exactly to (prediction − average prediction). `TreeExplainer` computes this
**exactly** for tree ensembles like XGBoost — no sampling approximation
needed. SHAP values here are in **log-odds (margin) space** — sign and
relative ranking matter for interpretation, not the raw number as a
"probability."

## 7.3 A Second, Smaller Pitfall — Also Caught and Handled

Independent of the sparse/dense bug, one-hot encoding fragments a single
categorical column (e.g. `discharge_disposition_id`, 21 categories) into
~20 separate binary features. Rare categories (some with as few as 3
patients in the whole dataset) let XGBoost fit a near-perfect split to a
handful of examples, inflating that dummy column's SHAP magnitude — noise,
not signal. The raw global ranking is still dominated by exactly this
pattern (`admission_type_id_7`, n=18 patients, tops the raw chart). Rather
than an aggregation fix (tried, but produced a result that couldn't be
fully verified — the same issue encountered and documented in the original
version of this phase), the simpler, already-established fix from Phase 5
is used again: exclude one-hot categories with fewer than 200 patients
(the same threshold as `sql/05_admission_source_type_joins.sql`'s
`HAVING COUNT(*) >= 200`) before ranking or plotting. 22 of 121
post-encoding columns are excluded this way.

## 7.4 Global Explainability — Reliable Ranking

![Reliable global bar chart](../outputs/figures/15_shap_global_bar_reliable.png)

| Feature | mean \|SHAP\| |
|---|---|
| `number_inpatient` | **0.267** |
| `discharge_disposition_id_1` (discharged home) | 0.182 |
| `service_utilization` | 0.081 |
| `diabetesMed` | 0.058 |
| `age_numeric` | 0.055 |
| `number_diagnoses` | 0.053 |
| `diag_1_group_Circulatory` | 0.052 |
| `num_medications` | 0.046 |

**`number_inpatient` is now unambiguously the #1 driver** — with a mean
|SHAP value| more than 40% larger than the second-place feature. This is a
categorically cleaner, more decisive result than the pre-fix version, and
it converges with three independent findings from earlier phases: the
Phase 3 EDA correlation table, the Phase 5 SQL utilization-tier analysis,
and now the trained model's own SHAP attribution. Three different methods
agreeing this strongly is a genuinely strong basis for a business
recommendation.

![Reliable beeswarm](../outputs/figures/16_shap_beeswarm_reliable.png)

The beeswarm adds direction and it is now clean and intuitive throughout:
- `number_inpatient`: red (high) clusters strongly positive (higher risk),
  blue (low/zero) clusters negative — a clear, monotonic relationship.
- `discharge_disposition_id_1` (discharged to home): red (patient WAS
  discharged home) pushes risk **down** — sensible, since being sent
  straight home usually signals a more stable patient than being
  transferred to another facility.

## 7.5 Local Explainability — Two Real Patients

Both patients are the actual highest-risk and lowest-risk predictions
across the entire 19,724-row test set.

### Highest-risk patient (encounter 406781108)

**Predicted probability: 0.945 — actually readmitted (label = 1).**

![High risk waterfall](../outputs/figures/13_shap_waterfall_high_risk.png)

The story here is now simple and coherent: `number_inpatient = 11` alone
contributes **+1.76** to the log-odds score, dwarfing every other feature —
by a wide margin the single dominant factor pushing this patient's risk up.
A handful of smaller contributors (two Neoplasms-related secondary
diagnoses, elevated `service_utilization`) add modest further weight. This
is exactly the kind of clean, one-sentence explanation a discharge planner
or auditor could actually use: *"this patient is flagged primarily because
they've been hospitalized 11 times in the past year."*

### Lowest-risk patient (encounter 243402714)

**Predicted probability: 0.050 — correctly not readmitted (label = 0).**

![Low risk waterfall](../outputs/figures/14_shap_waterfall_low_risk.png)

Equally coherent: this patient has exactly 1 medication, 0 prior inpatient
visits, was discharged home, is not on any diabetes medication, and had a
short 1-day stay — every contributing factor points the same direction
(toward lower risk), and the model agrees strongly. No contradictory or
confusing signals, unlike the pre-fix version of this analysis.

## 7.6 An Honest Note on Probability Calibration

The full test-set probability range is **0.05 to 0.945**, with a mean of
**0.448** — much higher and wider than the true 11.4% base rate. As
discussed in Phase 6.4, this is a direct consequence of `scale_pos_weight`
reshaping the decision surface for better ranking, not a sign the model
thinks nearly half of all patients are likely to be readmitted. The
*relative* story in every plot above (which features push risk up or down,
and by how much relative to each other) remains completely valid; the
absolute probability numbers should be read as risk scores for ranking, not
literal readmission probabilities.

## 7.7 Business Insight Summary

> **(1)** `number_inpatient` is now unambiguously the top individual
> driver, independently triangulated across EDA, SQL, and SHAP — a strong,
> simple story for any stakeholder conversation. **(2)** This phase's own
> cross-check (comparing two supposedly-equivalent prediction paths) is
> what surfaced a real bug in Phase 6 — a concrete argument for always
> validating a pipeline more than one way before trusting its output.
> **(3)** With the bug fixed, local explanations are now clean and
> one-sentence-explainable rather than a diffuse mix of small, sometimes
> contradictory signals — a materially more useful and more trustworthy
> deliverable for a real discharge-planning audience.

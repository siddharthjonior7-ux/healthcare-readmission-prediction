# Phase 1: Problem Statement

## 1.1 Business Context

Hospital readmissions — a patient returning to the hospital within 30 days of
discharge — are a major cost and quality problem in the US healthcare system:

- **Financial penalty angle**: Under the Hospital Readmissions Reduction
  Program (HRRP), CMS (Medicare) financially penalizes hospitals with excess
  30-day readmission rates — reimbursement can drop by up to 3% system-wide.
- **Cost angle**: a single readmission can cost $10,000–$15,000+ depending on
  condition and length of stay. At scale, preventable readmissions can mean
  tens of millions of dollars annually for a large hospital system.
- **Quality-of-care angle**: high readmission rates often signal gaps in
  discharge planning, patient education, or follow-up care.
- **Diabetes-specific angle**: diabetes is a comorbidity multiplier — it
  complicates recovery from almost every other condition, which is why
  diabetic patients are disproportionately prone to readmission.

This project uses the **Diabetes 130-US Hospitals dataset** (1999–2008,
~100,000 encounters across 130 hospitals) — one of the largest public,
real-world healthcare datasets available for this kind of analysis.

## 1.2 Business Objective

> Enable hospital care management teams to proactively identify diabetic
> patients at high risk of 30-day readmission at the time of discharge, so
> that limited care-coordination resources (follow-up calls, home health
> visits, medication reconciliation, dietitian referrals) can be targeted at
> the patients who need them most.

This framing deliberately specifies:
- **Who** uses the output — care management / discharge planning teams.
- **When** the prediction is made — at discharge, which constrains which
  features are legitimate to use (see leakage discussion in Phase 2/3).
- **What action** follows — the model must connect to a real operational
  decision, not just report a score.
- The **resource-constrained** nature of the setting — intervention capacity
  is limited, so ranking/prioritization matters as much as classification.

## 1.3 ML Objective

> Build a binary classification model that predicts the probability that a
> diabetic patient encounter will result in a hospital readmission within 30
> days of discharge, using information available at or before discharge.

Key framing decisions:
- **Binary, not multiclass**: the raw target has three classes (`<30`,
  `>30`, `NO`); we collapse to binary because CMS penalties are specifically
  tied to the 30-day window (see Phase 2 for the full justification).
- **Probability output matters more than a hard label**, since the business
  action is "prioritize the top N% of patients" — this favors ranking
  quality (ROC-AUC, calibration) over a single threshold-based accuracy.
- **Prediction happens at discharge time**, which rules out using any
  information that would only be known after discharge.

## 1.4 Success Metrics

**Business-level:**
| Metric | Why it matters |
|---|---|
| Reduction in preventable readmissions | Ties directly to CMS penalty avoidance and cost savings |
| Cost saved per avoided readmission | Translates model performance into $ |
| Precision at top-K% risk | Care teams can only act on a limited number of patients |
| Care-team workload feasibility | A model flagging 40% of patients as high-risk is operationally useless |

**ML-level:**
| Metric | Role |
|---|---|
| ROC-AUC | Primary metric — measures ranking quality independent of threshold |
| Recall | Missing a high-risk patient (false negative) has real cost |
| Precision | Flagging too many patients wastes scarce resources |
| F1-score | Balances precision/recall into one number |
| Accuracy | Reported for completeness only — see 1.5, it is a weak metric here |

## 1.5 Class Imbalance — Why Accuracy Is a Trap

Roughly **11% of encounters are <30-day readmissions** (confirmed in Phase 3
EDA: 11.16%). A model that always predicts "no readmission" scores ~89%
accuracy while being completely useless. This single fact shapes model
selection, resampling strategy, and threshold tuning throughout this project.

## 1.6 Business Insight

> A hospital doesn't need a model that's "accurate" in the abstract — it
> needs a model that produces a trustworthy, actionable priority list small
> enough for a care team to act on, that catches the majority of true
> high-risk patients without flooding the team with false alarms. Every
> phase of this project is evaluated against that yardstick, not a
> leaderboard metric.

# Phase 8: Power BI Dashboard Design

This phase produces the **data layer** for a Power BI dashboard
([`src/build_dashboard_data.py`](../src/build_dashboard_data.py), output in
`powerbi/`) and the **design spec** below — pages, visuals, DAX measures,
filters, and business recommendations. Power BI itself is a proprietary
desktop tool with no API this environment can drive directly, so the
deliverable here is: real, computed data ready to import, plus a complete
blueprint precise enough to build from directly, plus a static HTML
mockup of the executive page so the design is tangible rather than purely
descriptive.

## 8.1 A Real Data Constraint That Shapes the Whole Design

**The dataset has no date field and no hospital/facility ID.** Every
encounter is a snapshot with no timestamp beyond the aggregate 1999–2008
collection window, and no way to compare hospitals against each other.
This rules out two dashboard staples: a time-trend line chart and a
hospital-comparison view. Rather than force a fake trend line, the design
below pivots entirely around the dimensions the data actually supports —
diagnosis, specialty, utilization, discharge pathway, and the model's
predicted risk — which happen to be exactly the dimensions the last five
phases already validated as meaningful. Worth stating this constraint
explicitly to a stakeholder rather than letting a dashboard silently imply
capabilities the data doesn't have.

## 8.2 A Second Real Constraint — Handled with a `data_split` Flag

The model was trained on 80% of the data and evaluated on the held-out 20%
(Phase 6). Scoring the *entire* dataset with that model and treating every
row's predicted risk as equally trustworthy would be misleading — a model
shows artificially confident, memorized-looking scores on rows it was
trained on. `powerbi/fact_encounters.csv` tags every row `data_split =
"train"` or `"test"`, and only test-split rows carry a `predicted_risk_decile`
/ `risk_band` (train-split rows have these left blank). **Every risk-based
visual and slicer in this design filters to `is_out_of_sample_prediction =
TRUE`** — the dashboard should only ever rank or prioritize patients using
genuinely out-of-sample scores, consistent with the honest numbers
established in Phase 6/7.

## 8.3 Data Model (Star Schema)

```
fact_encounters (99,320 rows, 45 cols)
   ├── admission_type_id      → dim_admission_type      (8 rows)
   ├── discharge_disposition_id → dim_discharge_disposition (30 rows)
   └── admission_source_id    → dim_admission_source     (25 rows)

model_performance.csv   -- 4-model comparison table (Phase 6)
fairness_check.csv      -- race/payer_code AUC uplift (Phase 6.7)
high_risk_worklist.csv  -- top 200 test-split encounters by predicted risk
```

Relationships: `fact_encounters[admission_type_id]` → `dim_admission_type[admission_type_id]`
(and similarly for the other two dimension tables), all one-to-many,
single direction — a standard, simple star schema.

## 8.4 Page 1 — Executive Overview

**Audience**: hospital leadership, first five minutes of any review.

| Visual | Detail |
|---|---|
| KPI card | Total Encounters = 99,320 |
| KPI card | Overall 30-Day Readmission Rate = 11.4% |
| KPI card | Avg Length of Stay = 4.4 days |
| KPI card | Est. Annual Readmissions ≈ 11,309 (full historical count; framed as illustrative, not a forecast) |
| Bar chart | Readmission rate by `diag_1_group`, sorted descending |
| Bar chart | Readmission rate by `admission_type` (via `dim_admission_type`) |
| Donut | Encounter volume by utilization tier |

**Slicers**: age band, gender, diagnosis group, admission type.

**DAX measures**:
```
Total Encounters := COUNTROWS(fact_encounters)
Total Readmissions := SUM(fact_encounters[readmitted_30d])
Readmission Rate := DIVIDE([Total Readmissions], [Total Encounters], 0)
Avg Length of Stay := AVERAGE(fact_encounters[time_in_hospital])
```

## 8.5 Page 2 — Risk Segmentation & Prioritization

**Audience**: care management / discharge planning team — the primary
operational user of this whole project.

| Visual | Detail |
|---|---|
| Matrix/heatmap | `diag_1_group` × utilization tier, color-scaled by readmit rate (mirrors SQL Query 06) |
| Bar chart | Readmit rate and % of readmissions captured, by `predicted_risk_decile` (test-split only) |
| KPI card | Top-Decile Lift = 2.36x |
| KPI card | % of Readmissions Captured in Top 20% |
| Table | **The worklist** (`high_risk_worklist.csv`) — patient ID, predicted risk, diagnosis, prior inpatient visits, discharge disposition — sortable, filterable, exportable. This is the one visual on the whole dashboard a care coordinator would actually act on directly. |

**Slicers**: risk band (High/Medium/Low), diagnosis group.

**DAX measures**:
```
Top Decile Lift :=
VAR TopDecileRate =
    CALCULATE([Readmission Rate], fact_encounters[predicted_risk_decile] = 1)
VAR OverallRate = [Readmission Rate]
RETURN DIVIDE(TopDecileRate, OverallRate, 0)

Pct Readmissions Captured (Top 20%) :=
VAR ReadmitsInTop2Deciles =
    CALCULATE([Total Readmissions], fact_encounters[predicted_risk_decile] <= 2)
RETURN DIVIDE(ReadmitsInTop2Deciles, [Total Readmissions], 0)
```

## 8.6 Page 3 — Clinical & Operational Drivers

**Audience**: clinical operations / quality improvement teams.

| Visual | Detail |
|---|---|
| Bar chart | Readmit rate by `medical_specialty` (volume-filtered to n≥500, mirrors SQL Query 02) |
| Bar chart | Readmit rate by `discharge_disposition` (via `dim_discharge_disposition`) |
| Line + column combo | Readmit rate and avg medications by `time_in_hospital` (mirrors SQL Query 04) |
| Bar chart | Readmit rate by insulin dosage-change status |

**Slicers**: medical specialty, discharge disposition.

## 8.7 Page 4 — Model Performance & Trust

**Audience**: technical stakeholders, model risk/governance review — the
page that answers "why should anyone trust this model?"

| Visual | Detail |
|---|---|
| Table | `model_performance.csv` — all 4 models, every metric, side by side |
| KPI card | Best model: XGBoost, ROC-AUC 0.667 |
| Image | ROC curves (`outputs/figures/08_roc_curves_comparison.png`) |
| Image | SHAP global feature importance (`outputs/figures/15_shap_global_bar_reliable.png`) |
| KPI card | Fairness check: race/payer_code AUC uplift = **+0.0015** (i.e., safe to exclude) |

Power BI can't natively render a SHAP beeswarm or ROC curve, so both are
embedded as static images — a pragmatic, common real-world pattern for
bringing Python/ML artifacts into a BI tool without a custom visual.

## 8.8 Executive Page Mockup

A static preview of Page 1, built from the real numbers above, is included
alongside this document (see the interactive artifact shared with this
phase) — useful both as a design reference and as a placeholder for the
Phase 9 README's screenshots section.

## 8.9 Business Recommendations Surfaced By This Design

1. **Stand up the Page 2 worklist as a recurring export**, even before any
   full Power BI rollout — a care coordinator could act on
   `high_risk_worklist.csv` immediately, filtered to new discharges as they
   occur.
2. **Route the Nephrology and Diabetes-primary-diagnosis segments to a
   dedicated review** (Page 1 and Page 3 both surface these independently)
   — the highest concentration of actionable risk in the dataset.
3. **Do not use length-of-stay reduction as an isolated cost target**
   (Page 3) without pairing it with the readmission-rate trend it could
   move — the data shows shorter stays already correlate with *lower* risk,
   not higher, so the usual "cut cost by cutting LOS" instinct needs the
   guardrail this page provides.
4. **Exclude `race` and `payer_code` from any production scoring model**
   (Page 4) — the fairness check shows this costs effectively nothing in
   performance.

## 8.10 Business Insight Summary

> A dashboard is only as honest as the data feeding it. The two constraints
> handled explicitly in this phase — no time/hospital dimension, and the
> train/test split discipline carried through into `data_split` — are
> exactly the kind of details that separate a dashboard that looks
> impressive from one a hospital could actually deploy and trust. Every
> number a stakeholder would see on Page 2's worklist is a genuinely
> out-of-sample prediction, not a number that looks good because the model
> has already seen the answer.

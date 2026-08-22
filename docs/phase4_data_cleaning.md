# Phase 4: Data Cleaning & Feature Engineering

Implemented in [`src/data_cleaning.py`](../src/data_cleaning.py). Run
`python src/data_cleaning.py` to reproduce
`data/processed/cleaned_diabetic_data.csv` from the raw dataset.

## 4.1 A Design Principle Worth Stating Upfront

This phase produces an **analytics-ready** dataset, not a **model-ready**
one. Concretely:

- We **do** remove leakage rows, drop unusable columns, group high-cardinality
  categories, and engineer new features — deterministic, domain-driven
  transformations that don't depend on how the data is later split.
- We **do not** one-hot encode, scale, or impute using statistics (mean,
  median, mode) computed across the whole dataset. Those steps are deferred
  to a scikit-learn `Pipeline` in Phase 6, **fit only on the training fold**,
  to avoid leaking test-set information into training.
- Keeping the output human-readable (`"Circulatory"` rather than a one-hot
  column) also makes it directly usable for the Phase 5 SQL analysis and the
  Phase 8 BI dashboard — not a coincidence, a deliberate design choice.

This separation — "clean/analytics table" vs. "model-ready table produced
by a pipeline" — mirrors how a real data team would structure this work,
and is worth stating explicitly in an interview.

## 4.2 Leakage Removal

Removed the 2,423 expired/hospice encounters identified and quantified in
Phase 3.3 (2.38% of rows) — before any other cleaning step, since every
downstream statistic should reflect the corrected population.

## 4.3 Target Variable

`readmitted_30d` created from the raw 3-class `readmitted` column
(`<30` → 1, else → 0), and the raw column dropped. Post-cleaning positive
rate: **11.39%** (up slightly from 11.16% pre-cleaning, since removing
near-zero-readmission expired/hospice rows shifts the base rate up — exactly
as expected from the Phase 3 leakage analysis).

## 4.4 Dropped Columns

| Column(s) | Reason |
|---|---|
| `weight` | 96.86% missing — unusable |
| 13 medication columns (`chlorpropamide`, `acetohexamide`, `tolbutamide`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`, `examide`, `citoglipton`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone`) | >99.5% single value across all 101,766 rows (measured directly, not assumed) — no real signal, only sparse noise |

10 medication columns were **kept** (`metformin`, `repaglinide`,
`nateglinide`, `glimepiride`, `glipizide`, `glyburide`, `pioglitazone`,
`rosiglitazone`, `insulin`, `glyburide-metformin`) — each has more than
0.5% variation and plausible clinical relevance.

## 4.5 Missing Value Handling — Category-Level, Not Statistical

| Feature | Strategy | Rationale |
|---|---|---|
| `race` | Fill with `"Unknown"` | Small amount missing (2.23%) — explicit category rather than guessing |
| `A1Cresult`, `max_glu_serum` | Fill with `"Not Tested"` | Missing = test not ordered, which is itself informative (Phase 3.4) — never impute a fake lab value |
| `medical_specialty` | Keep top 10 by frequency, collapse rest (incl. missing) into `"Other/Missing"` | 73 raw categories + 49% missing — avoids an extremely sparse one-hot block later, down to 11 categories |
| `payer_code` | Keep top 5, collapse rest into `"Other/Unknown"` | 39.56% missing + fairness-proxy concern (Phase 2.3) — kept optional and low-dimensional for the Phase 6 with/without-fairness-features comparison |
| `diag_1` missing | Row dropped | Only 20 rows (~0.02%) — too small to safely impute, and the primary diagnosis is clinically central |
| `gender = "Unknown/Invalid"` | Row dropped | Only 3 rows — too small to encode as its own category or safely infer |

## 4.6 Diagnosis Code Grouping

`diag_1/2/3` (716/748/789 unique ICD-9 codes) were mapped into **9 clinical
categories**, using the grouping established in the original research behind
this dataset (Strack et al., 2014):

Circulatory · Respiratory · Digestive · Diabetes · Injury · Musculoskeletal
· Genitourinary · Neoplasms · Other (includes V-codes, E-codes, and
everything outside the named ICD-9 ranges).

**Validation**: the resulting `diag_1_group` distribution (Circulatory
29.88%, Other 17.91%, Respiratory 14.03%, Digestive 9.40%, Diabetes 8.72%,
Injury 6.90%, Genitourinary 5.04%, Musculoskeletal 4.97%, Neoplasms 3.15%)
closely matches the published distribution from the original study — a
useful sanity check that the mapping logic is correct, not just plausible.

## 4.7 Age Encoding

`age` arrives pre-binned as strings like `"[50-60)"`. Converted to the
numeric midpoint (e.g. `55`) rather than one-hot encoded, which preserves
the natural ordering a plain categorical encoding would discard (a
one-hot version can't express that `[70-80)` sits between `[60-70)` and
`[80-90)`).

## 4.8 Medication Encoding & Engineered Features

The 10 retained medication columns were ordinal-encoded
(`No=0, Down=1, Steady=2, Up=3`), treating dosage change as a severity
scale. Two summary features were engineered on top:

- `num_med_increased` — count of drugs whose dosage went **up** this stay
- `num_med_decreased` — count of drugs whose dosage went **down** this stay

`num_med_increased` is a plausible proxy for "diabetes was poorly
controlled and needed active management during this stay" (Phase 2.2.F).

## 4.9 Utilization Feature Engineering

- **`service_utilization`** = `number_outpatient + number_emergency + number_inpatient`
  — a single combined measure of prior-year healthcare use, motivated
  directly by the Phase 3 finding that all three individual utilization
  counts were the strongest predictors found. Validated post-engineering:
  readmission rate rises near-monotonically from **8.3%** (0 prior visits)
  to **28.5%** (8+ prior visits).
- **`had_prior_inpatient`** — binary flag (`number_inpatient > 0`), mirroring
  the manual triage heuristic surfaced in Phase 3 ("3+ prior inpatient
  visits"). Readmission rate: **8.6%** with no prior inpatient visits vs.
  **17.0%** with at least one — roughly double, confirming this simple flag
  alone carries real signal and will serve as a benchmark feature in Phase 6.

## 4.10 Result

| | Before | After |
|---|---|---|
| Rows | 101,766 | 99,320 |
| Columns | 50 | 40 |
| Missing values | Present in 7 columns (up to 96.86%) | **0** |
| Readmission rate | 11.16% | 11.39% |

Saved to `data/processed/cleaned_diabetic_data.csv`.

## 4.11 What's Deliberately *Not* Done Here

- **No scaling** — deferred to Phase 6, fit on the training fold only.
- **No one-hot encoding** — deferred to Phase 6 (`OneHotEncoder(handle_unknown="ignore")`
  inside a `Pipeline`), so unseen categories at inference time are handled
  gracefully rather than crashing a hand-rolled `pd.get_dummies()` call.
- **No train/test split yet** — Phase 6 will split at the **patient level**
  (`GroupShuffleSplit` on `patient_nbr`), per the leakage risk flagged in
  Phase 2.1 (the same patient can appear in multiple encounters).
- **`race` and `payer_code` are retained**, not dropped, despite the
  fairness flag in Phase 2.3 — the plan is to train and compare a model
  version with and without them in Phase 6, which requires having both
  available now.

## 4.12 Business Insight

> Two of the engineered features in this phase — `service_utilization` and
> `had_prior_inpatient` — are simple enough to compute for any patient at
> discharge with no model at all, and both show a clear, monotonic
> relationship with readmission risk. This isn't a detour from the ML
> workflow; it's the benchmark the Phase 6 models are required to beat. If a
> Random Forest or XGBoost model can't outperform "flag anyone with 3+ prior
> inpatient visits," that's a legitimate, useful finding for a business
> stakeholder — it means the operationally simplest solution is also the
> right one.

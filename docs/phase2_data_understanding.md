# Phase 2: Data Understanding

## 2.1 Unit of Analysis

The dataset has **101,766 rows**, but each row is a hospital **encounter**,
not a unique patient — the same patient (`patient_nbr`) can appear multiple
times (71,518 unique patients across 101,766 encounters, confirmed in
Phase 3). This matters for modeling: a random row-level train/test split can
leak the same patient into both sets. **Best practice**: split at the
patient level (e.g. `GroupShuffleSplit` on `patient_nbr`), implemented in
Phase 4/6.

## 2.2 Feature Groups

### A. Identifiers
| Feature | Meaning |
|---|---|
| `encounter_id` | Unique ID per visit — not predictive, dropped before modeling |
| `patient_nbr` | Unique ID per patient — used for correct splitting and utilization features, not as a direct predictor |

### B. Demographics
| Feature | Meaning | Note |
|---|---|---|
| `race` | Patient-reported race | Fairness considerations — see 2.4 |
| `gender` | Male/Female (+ small "Unknown/Invalid" group) | |
| `age` | Banded into 10-year ranges | Pre-binned by data providers for anonymization |
| `weight` | Banded weight | ~97% missing — unusable, dropped |

### C. Admission & Discharge Administrative Details
| Feature | Meaning |
|---|---|
| `admission_type_id` | Emergency, Urgent, Elective, Newborn, etc. (coded, mapped via lookup table) |
| `discharge_disposition_id` | Discharged home, transferred, **expired**, **hospice**, etc. |
| `admission_source_id` | Physician referral, ER, transfer, etc. |
| `payer_code` | Insurance type — high missingness |
| `medical_specialty` | Admitting physician's specialty — high missingness |
| `time_in_hospital` | Length of stay, 1–14 days |

**⚠️ Leakage trap**: `discharge_disposition_id` includes codes for
**expired** and **hospice** discharges. A deceased patient cannot be
readmitted — their target is definitionally "not readmitted," not because
they recovered. Leaving these rows in contaminates the negative class with a
different phenomenon (mortality, not recovery). **Action**: filter these
encounters out before modeling (quantified in Phase 3: 2,423 encounters,
2.38% of the data, with an artificially low 1.77% readmit rate confirming
the contamination).

### D. Prior Healthcare Utilization
| Feature | Meaning |
|---|---|
| `number_outpatient` | # outpatient visits in the prior year |
| `number_emergency` | # emergency visits in the prior year |
| `number_inpatient` | # inpatient visits in the prior year |

Past utilization is one of the strongest predictors of future utilization in
almost every healthcare risk model — the same logic insurers use for
high-cost-claimant prediction. Confirmed empirically in Phase 3 (see
`04_readmission_vs_prior_utilization.png`).

### E. Clinical Severity / Encounter Complexity
| Feature | Meaning |
|---|---|
| `num_lab_procedures` | # lab tests performed |
| `num_procedures` | # non-lab procedures performed |
| `num_medications` | # distinct medications administered |
| `number_diagnoses` | # diagnoses recorded for this encounter |
| `diag_1`, `diag_2`, `diag_3` | Primary/secondary/additional diagnosis (ICD-9 codes) |
| `max_glu_serum` | Glucose serum result: `>200`, `>300`, `Normal`, or not tested |
| `A1Cresult` | HbA1c result: `>7`, `>8`, `Normal`, or not tested |

**Healthcare context on A1C**: HbA1c measures average blood glucose over
~3 months — the gold-standard diabetes control marker. In the original 2014
study behind this dataset (Strack et al.), *whether the test was even
ordered* proved more informative than the result itself — a proxy for how
proactively the care team managed the patient's diabetes. Confirmed
empirically in Phase 3.

**ICD-9 diagnosis codes**: 700+ unique values across `diag_1/2/3` (confirmed:
716 / 748 / 789 unique codes) — too sparse to one-hot encode directly.
Phase 4 groups these into clinical categories (Circulatory, Respiratory,
Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms,
Other), a standard technique for diagnosis codes in healthcare ML.

### F. Medication Features (23 columns)
One column per diabetes drug (`metformin`, `insulin`, `glipizide`, etc.),
valued `No` / `Steady` / `Up` / `Down`, plus two summary columns:
`change` (any med changed during the encounter) and `diabetesMed` (on any
diabetic medication). A dosage increase during an inpatient stay can signal
the care team found the patient's diabetes poorly controlled — a plausible
readmission risk signal. Several of the 23 columns are near-constant
(e.g. `examide`, `citoglipton`) and are dropped in Phase 4 rather than
blindly one-hot encoded.

### G. Target Variable: `readmitted`

Raw column has three classes: `NO`, `>30`, `<30`. We binarize to
`readmitted_30d` = 1 if `<30`, else 0 — justified because:
1. CMS's HRRP penalty is specifically anchored to the 30-day window.
2. Clinically, a readmission within 30 days more likely reflects an
   incomplete/failed discharge process; a readmission after 30+ days more
   likely reflects a new, unrelated clinical event.

## 2.3 Fairness Flag

`race` and `payer_code` can act as proxies for socioeconomic status and
access to care, which correlate with outcomes for reasons unrelated to
clinical risk. Plan: train one model version with these features and one
without, and compare performance + fairness metrics (Phase 6).

## 2.4 Business Insight

> This dataset stacks three layers of signal: **(1)** who the patient is
> (demographics), **(2)** how complex this encounter was (labs, procedures,
> diagnoses), and **(3)** how much the patient has historically used the
> healthcare system (prior utilization). Layer (3) usually dominates
> predictive power in utilization-type problems — a hypothesis tested
> directly in Phase 3 EDA and confirmed (prior inpatient visits had the
> strongest correlation with the target of any numeric feature).

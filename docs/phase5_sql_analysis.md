# Phase 5: SQL Business Analysis

All 8 queries below run against a real SQLite database
(`data/processed/readmission.db`) built from the Phase 4 cleaned dataset
(99,320 rows) plus three lookup tables parsed from the dataset's own
`IDs_mapping.csv`. Every result shown is actual query output, not
illustrative. Reproduce with:

```bash
python src/run_sql_analysis.py
```

This builds the database and executes every `.sql` file in `sql/` in order,
saving full output to `outputs/sql_results.txt`.

**Portability note**: every query below uses standard ANSI SQL (CTEs,
window functions, JOINs, `GROUP BY`/`HAVING`) — SQLite was chosen for
zero-setup portability, but these queries run with only trivial syntax
changes on PostgreSQL, MySQL, or a data warehouse like BigQuery/Snowflake.

---

## Query 01 — Overall Baseline + Readmission Rate by Diagnosis

**Business question**: *What's our baseline 30-day readmission rate, and
which primary diagnosis categories drive it — by rate and by volume?*

```sql
SELECT diag_1_group, COUNT(*) AS n_encounters, SUM(readmitted_30d) AS n_readmissions,
       ROUND(100.0 * SUM(readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM encounters), 2) AS pct_of_all_encounters
FROM encounters
GROUP BY diag_1_group
ORDER BY readmit_rate_pct DESC;
```

**Result** (baseline: 99,320 encounters, 11,309 readmissions, **11.39%**):

| diag_1_group | n_encounters | readmit_rate_pct | pct_of_all_encounters |
|---|---|---|---|
| Diabetes | 8,661 | **13.10%** | 8.72% |
| Injury | 6,851 | 12.41% | 6.90% |
| Circulatory | 29,680 | 11.69% | **29.88%** |
| Other | 17,793 | 11.68% | 17.91% |
| Genitourinary | 5,002 | 11.04% | 5.04% |
| Neoplasms | 3,131 | 10.89% | 3.15% |
| Digestive | 9,333 | 10.82% | 9.40% |
| Respiratory | 13,934 | 10.06% | 14.03% |
| Musculoskeletal | 4,935 | 9.54% | 4.97% |

**Insight**: Encounters where diabetes *itself* is the primary diagnosis
(not just a comorbidity) have the highest readmission rate (13.10%) — makes
clinical sense, since a primary diabetes admission usually means the disease
was actively decompensating. But **Circulatory conditions represent the
largest volume opportunity** (29.88% of all encounters) — a 1-point rate
reduction here would prevent far more readmissions in absolute terms than
the same reduction in the smaller Diabetes-primary group. This is the
rate-vs-volume tension a resource-allocation decision always has to weigh.

---

## Query 02 — Specialty: Volume vs. Risk

**Business question**: *Which admitting specialties have both a high
readmission rate AND enough volume to justify a targeted program?*

```sql
SELECT medical_specialty, COUNT(*) AS n_encounters,
       ROUND(100.0 * SUM(readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct
FROM encounters
GROUP BY medical_specialty
HAVING COUNT(*) >= 500
ORDER BY readmit_rate_pct DESC;
```

**Result**:

| medical_specialty | n_encounters | readmit_rate_pct |
|---|---|---|
| Nephrology | 1,538 | **16.06%** |
| Family/GeneralPractice | 7,252 | 12.12% |
| Other/Missing | 55,944 | 11.60% |
| InternalMedicine | 14,234 | 11.53% |
| Emergency/Trauma | 7,418 | 11.39% |
| Pulmonology | 854 | 11.24% |
| Surgery-General | 3,059 | 11.18% |
| Orthopedics | 1,392 | 10.85% |
| Radiologist | 1,121 | 9.10% |
| Cardiology | 5,278 | 8.03% |
| Orthopedics-Reconstructive | 1,230 | 7.48% |

**Insight**: Nephrology stands out — patients admitted under a nephrology
specialist have a 16.06% readmission rate, meaningfully above every other
specialty with real volume. This makes clinical sense (kidney disease and
diabetes are tightly linked comorbidities, and dialysis patients cycle
through the hospital system frequently). **Cardiology is the counter-
intuitive finding**: despite good volume (5,278 encounters) and a condition
often assumed to carry high readmission risk, its rate (8.03%) is the
lowest of any specialty shown — worth a follow-up question for stakeholders
rather than an assumption. The `HAVING COUNT(*) >= 500` threshold is doing
real work here: without it, small specialties would dominate the top of
this list with noisy rates based on a handful of patients.

---

## Query 03 — Risk Stratification by Utilization Tier

**Business question**: *What would a simple, explainable triage rule look
like today, using prior utilization alone?*

```sql
SELECT CASE WHEN service_utilization = 0 THEN '0 - No prior visits'
            WHEN service_utilization BETWEEN 1 AND 2 THEN '1-2 - Low utilization'
            WHEN service_utilization BETWEEN 3 AND 5 THEN '3-5 - Moderate utilization'
            ELSE '6+ - High utilization' END AS utilization_tier,
       COUNT(*) AS n_encounters, SUM(readmitted_30d) AS n_readmissions,
       ROUND(100.0 * SUM(readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct
FROM encounters
GROUP BY utilization_tier
ORDER BY MIN(service_utilization);
```

**Result**:

| utilization_tier | n_encounters | readmit_rate_pct |
|---|---|---|
| 0 – No prior visits | 54,665 | 8.33% |
| 1–2 – Low utilization | 29,200 | 12.87% |
| 3–5 – Moderate utilization | 11,131 | 16.84% |
| 6+ – High utilization | 4,324 | **26.04%** |

**Insight**: A clean, monotonic, 3x spread from lowest to highest tier —
this single `CASE WHEN` rule, computable the moment a patient is
discharged, is a legitimate operational triage tool on its own. It directly
operationalizes the strongest EDA finding from Phase 3 and becomes the
benchmark the Phase 6 ML models need to outperform.

---

## Query 04 — Length of Stay vs. Readmission Risk

**Business question**: *Do longer stays predict higher or lower readmission
risk — relevant to any cost-cutting push to reduce length of stay (LOS)?*

**Result** (abbreviated — full table has 14 rows, one per LOS day):

| length_of_stay_days | n_encounters | readmit_rate_pct | avg_medications |
|---|---|---|---|
| 1 | 13,820 | 8.39% | 11.1 |
| 4 | 13,677 | 11.98% | 16.0 |
| 7 | 5,696 | 13.11% | 19.9 |
| 10 | 2,262 | 14.81% | 23.1 |
| 14 | 995 | 13.47% | 25.5 |

**Insight**: Readmission risk rises fairly steadily from 8.4% at 1 day to
~14-15% around 8-10 days, then flattens/gets noisier at the longest stays
(smaller sample sizes there). This is a genuinely important operational
tension for a hospital: **shorter stays are NOT simply "better"** — the
data shows the shortest-stay patients have the lowest readmission risk, not
the highest, meaning a naive length-of-stay reduction target could backfire
if it pushes patients out before they're stable. Any LOS-reduction
initiative needs to be paired with monitoring this exact metric, or the
"savings" get wiped out by HRRP readmission penalties (Phase 1.1).

---

## Query 05 — Admission Type & Source (JOINs)

**Business question**: *Does how a patient enters the hospital relate to
readmission risk?*

```sql
SELECT lt.description AS admission_type, COUNT(*) AS n_encounters,
       ROUND(100.0 * SUM(e.readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct
FROM encounters e
JOIN lookup_admission_type lt ON e.admission_type_id = lt.admission_type_id
GROUP BY lt.description
HAVING COUNT(*) >= 200
ORDER BY readmit_rate_pct DESC;
```

**Result** — admission type:

| admission_type | n_encounters | readmit_rate_pct |
|---|---|---|
| Emergency | 52,358 | **11.82%** |
| Urgent | 18,131 | 11.35% |
| Elective | 18,660 | 10.49% |

**Result** — admission source (top rows):

| admission_source | n_encounters | readmit_rate_pct |
|---|---|---|
| Transfer from a Skilled Nursing Facility (SNF) | 805 | 12.42% |
| Emergency Room | 55,840 | 11.98% |
| Physician Referral | 29,157 | 10.69% |
| Transfer from another health care facility | 2,239 | 9.47% |

**Insight**: Emergency admissions are riskier than elective ones (11.82%
vs. 10.49%) — intuitive, since elective admissions are, by definition,
planned and typically for more stable patients. This query also
demonstrates a normalized-schema JOIN pattern: `admission_type_id` and
`admission_source_id` are stored as integers in the main table (exactly how
a real hospital data warehouse would store them), and the lookup tables —
built directly from the dataset's own `IDs_mapping.csv` — turn them into
labels a hospital administrator can read without duplicating text across
99,320 rows.

---

## Query 06 — Highest-Risk Segments (Diagnosis × Utilization, via CTE)

**Business question**: *If we could fund only ONE targeted care-management
program, which diagnosis + utilization combination should it target?*

```sql
WITH segments AS (
    SELECT diag_1_group,
           CASE WHEN service_utilization = 0 THEN 'No prior visits'
                WHEN service_utilization BETWEEN 1 AND 2 THEN 'Low utilization'
                WHEN service_utilization BETWEEN 3 AND 5 THEN 'Moderate utilization'
                ELSE 'High utilization' END AS utilization_tier,
           readmitted_30d
    FROM encounters
)
SELECT diag_1_group, utilization_tier, COUNT(*) AS n_encounters,
       ROUND(100.0 * SUM(readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct
FROM segments
GROUP BY diag_1_group, utilization_tier
HAVING COUNT(*) >= 300
ORDER BY readmit_rate_pct DESC
LIMIT 10;
```

**Result** (top 5 of 10):

| diag_1_group | utilization_tier | n_encounters | readmit_rate_pct |
|---|---|---|---|
| Diabetes | High utilization | 722 | **30.06%** |
| Digestive | High utilization | 453 | 28.04% |
| Respiratory | High utilization | 626 | 26.84% |
| Other | High utilization | 942 | 24.73% |
| Circulatory | High utilization | 935 | 24.06% |

**Insight**: This is the single most actionable finding in this phase.
Patients with a **primary diabetes diagnosis AND 6+ prior visits in the
past year have a 30% readmission rate** — nearly 1 in 3, and almost 3x the
overall baseline. Neither factor alone (Diabetes: 13.1%, High utilization:
26.0%) fully captures this — the combination is meaningfully worse than
either dimension in isolation. If a hospital can only build one
intervention program, this is the segment: ~722 patients here alone, a
manageable, well-defined population for a diabetes-specific intensive
case-management pilot.

---

## Query 07 — Medication Change Impact

**Business question**: *Does adjusting diabetes medication during the stay
relate to readmission risk — and does the direction of an insulin change
matter?*

**Result**:

| med_change_status | n_encounters | readmit_rate_pct |
|---|---|---|
| Medication changed | 46,112 | **12.02%** |
| No change | 53,208 | 10.84% |

| insulin_status | n_encounters | readmit_rate_pct |
|---|---|---|
| No | 46,365 | 10.21% |
| Down | 11,906 | **14.22%** |
| Steady | 30,063 | 11.37% |
| Up | 10,986 | 13.34% |

**Insight**: Any medication change correlates with higher readmission risk
(12.02% vs 10.84%), consistent with the Phase 4 hypothesis that a dosage
change signals active management of poorly controlled diabetes. The
insulin breakdown adds nuance: **both increasing AND decreasing insulin
dosage carry more risk than staying steady** — and a *decrease* (14.22%) is
actually riskier than an *increase* (13.34%). This is worth flagging rather
than over-interpreting: a dosage decrease might follow a hypoglycemic
event or declining kidney function, both of which are themselves markers
of a more fragile patient — a good example of a finding that raises a
better question for a clinician rather than a definitive answer on its own.

---

## Query 08 — Risk-Decile Prioritization (Window Functions)

**Business question**: *If care management can only act on the riskiest
10% of discharges, how much of the total readmission burden would that
capture?*

```sql
WITH scored AS (
    SELECT encounter_id, readmitted_30d,
           (service_utilization * 3 + had_prior_inpatient * 2
            + num_med_increased * 2 + number_diagnoses) AS risk_score
    FROM encounters
),
deciled AS (
    SELECT *, NTILE(10) OVER (ORDER BY risk_score DESC) AS risk_decile
    FROM scored
)
SELECT risk_decile, COUNT(*) AS n_encounters, SUM(readmitted_30d) AS n_readmissions,
       ROUND(100.0 * SUM(readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct,
       ROUND(100.0 * SUM(readmitted_30d) / (SELECT SUM(readmitted_30d) FROM encounters), 2)
           AS pct_of_all_readmissions_captured
FROM deciled
GROUP BY risk_decile
ORDER BY risk_decile;
```

`risk_score` is a simple, EDA-justified weighted rule (**not** a trained
model) — weights reflect each factor's measured effect size from earlier
phases, not arbitrary guesses. `NTILE(10)` is a SQL window function that
splits risk-ranked rows into 10 equal-sized buckets, the same job
`pd.qcut` would do in pandas, done directly in the database.

**Result**:

| risk_decile | n_encounters | readmit_rate_pct | pct_of_all_readmissions_captured |
|---|---|---|---|
| 1 (highest risk) | 9,932 | **21.63%** | **18.99%** |
| 2 | 9,932 | 16.30% | 14.32% |
| 3 | 9,932 | 13.45% | 11.81% |
| 5 | 9,932 | 10.33% | 9.07% |
| 10 (lowest risk) | 9,932 | 5.99% | 5.26% |

**Insight**: A basic, hand-weighted rule — with no machine learning at all
— already lets the top 10% of patients by risk score capture **19% of all
readmissions** (vs. the 10% a random selection would capture: a **~1.9x
lift**). The top 2 deciles (20% of patients) capture 33.3% of all
readmissions. **This is the number Phase 6's ML models need to beat.** If
Random Forest or XGBoost can't meaningfully exceed a ~1.9x lift in the top
decile using a simple 4-variable rule, that's a legitimate finding — it
would mean the additional model complexity isn't earning its keep for this
particular business question, which is exactly the kind of "is the fancy
model worth it?" analysis a Decision Analytics Associate is expected to run
before recommending a production model.

---

## Business Insight Summary for This Phase

> Three concrete, resource-allocation-ready findings came out of SQL alone,
> before any model was trained: **(1)** primary diabetes diagnosis + high
> prior utilization is a distinct, 30%-risk segment worth its own
> intervention program; **(2)** length-of-stay reduction initiatives need a
> readmission-rate guardrail, since the shortest stays show the lowest
> risk, not the highest; **(3)** a simple, 4-variable weighted rule already
> achieves a ~1.9x lift over random selection in identifying high-risk
> patients — a concrete, non-trivial bar the Phase 6 ML models must clear
> to justify the added complexity of a production model.

# 🏥 Healthcare Patient Readmission Prediction

Predicting 30-day hospital readmission risk for diabetic patients, using the
Diabetes 130-US Hospitals dataset (1999–2008, ~100K encounters across 130
hospitals). Built as an end-to-end, production-style data science project —
from business problem framing through EDA, cleaning, SQL analysis, ML
modeling, explainability, and a BI dashboard design.

**Project status**: 🚧 In progress — built and documented one phase at a
time. See [Progress](#-progress) below.

## 📌 Why This Project

Hospital readmissions within 30 days are one of the most consequential
metrics in US healthcare operations: CMS penalizes hospitals with excess
30-day readmissions under the Hospital Readmissions Reduction Program
(HRRP), a single readmission can cost $10,000–$15,000+, and diabetic
patients are disproportionately at risk since diabetes complicates recovery
from nearly every other condition.

**Goal**: identify, at the moment of discharge, which diabetic patients are
most likely to be readmitted within 30 days — so hospital care teams can
target limited follow-up resources at the patients who need them most.

Full business/ML framing: [`docs/phase1_problem_statement.md`](docs/phase1_problem_statement.md)

## 📊 Dataset

**Diabetes 130-US Hospitals Dataset** — 101,766 encounters, 71,518 unique
patients, 50 features, from 130 US hospitals (1999–2008).
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
Original research: Strack et al., *"Impact of HbA1c Measurement on Hospital
Readmission Rates,"* BioMed Research International, 2014.

Full feature-by-feature breakdown: [`docs/phase2_data_understanding.md`](docs/phase2_data_understanding.md)

## 🗂️ Project Structure

```
healthcare-readmission-prediction/
├── data/
│   ├── raw/                      # Original dataset + parsed lookup tables
│   └── processed/                 # Cleaned data (Phase 4) + SQLite DB (Phase 5)
├── docs/                         # Phase-by-phase write-ups
│   ├── phase1_problem_statement.md
│   ├── phase2_data_understanding.md
│   ├── phase3_eda.md
│   ├── phase4_data_cleaning.md
│   ├── phase5_sql_analysis.md
│   ├── phase6_machine_learning.md
│   ├── phase7_model_explainability.md
│   └── phase8_dashboard_design.md
├── src/
│   ├── eda.py                    # Phase 3
│   ├── data_cleaning.py           # Phase 4
│   ├── run_sql_analysis.py        # Phase 5
│   ├── train_models.py            # Phase 6
│   ├── explainability.py          # Phase 7
│   └── build_dashboard_data.py    # Phase 8
├── sql/                          # 8 business-question SQL queries (Phase 5)
├── powerbi/                      # Star-schema data exports for the dashboard (Phase 8)
├── notebooks/
├── models/                       # Saved best model pipeline (gitignored - reproducible)
├── outputs/                      # Figures, logs, result tables from every phase
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## ✅ Progress

| Phase | Status | Write-up |
|---|---|---|
| 1. Problem Statement | ✅ Done | [docs/phase1_problem_statement.md](docs/phase1_problem_statement.md) |
| 2. Data Understanding | ✅ Done | [docs/phase2_data_understanding.md](docs/phase2_data_understanding.md) |
| 3. Exploratory Data Analysis | ✅ Done | [docs/phase3_eda.md](docs/phase3_eda.md) |
| 4. Data Cleaning & Feature Engineering | ✅ Done | [docs/phase4_data_cleaning.md](docs/phase4_data_cleaning.md) |
| 5. SQL Business Analysis | ✅ Done | [docs/phase5_sql_analysis.md](docs/phase5_sql_analysis.md) |
| 6. Machine Learning | ✅ Done | [docs/phase6_machine_learning.md](docs/phase6_machine_learning.md) |
| 7. Model Explainability (SHAP) | ✅ Done | [docs/phase7_model_explainability.md](docs/phase7_model_explainability.md) |
| 8. Power BI Dashboard Design | ✅ Done | [docs/phase8_dashboard_design.md](docs/phase8_dashboard_design.md) |
| 9. Final polished documentation | ⏳ Upcoming | — |

## 🔑 Key Findings

- **Class imbalance is real**: only **11.4%** of encounters are 30-day
  readmissions — accuracy is a misleading metric; ROC-AUC and top-decile
  lift matter far more.
- **Prior healthcare utilization is the strongest signal**, confirmed
  independently by EDA (Phase 3), SQL (Phase 5), *and* SHAP (Phase 7):
  patients with 5+ prior inpatient visits have a **~36%** readmission rate
  vs. **~8.5%** for first-time patients.
- **A data leakage source was found and removed**: 2.38% of encounters are
  expired/hospice discharges, which structurally cannot be "readmitted."
- **XGBoost beats a SQL-only benchmark by ~24%** on top-decile lift (2.36x
  vs. 1.9x) — a concrete, quantified case for deploying a model.
- **`race` and `payer_code` can be excluded from production at essentially
  zero cost** (+0.0015 AUC) — removing a fairness liability for free.
- **A real pipeline bug was caught, diagnosed, and fixed mid-project**: a
  sparse-matrix/missing-value interaction specific to XGBoost was silently
  corrupting ~62% of its training input. Documented in full in
  [`docs/phase6_machine_learning.md`](docs/phase6_machine_learning.md#61-a-bug-caught-and-fixed-mid-phase-documented-not-hidden).

## 🤖 Machine Learning Results

| Model | ROC-AUC | Top-decile lift |
|---|---|---|
| Logistic Regression | 0.660 | 2.25x |
| Decision Tree | 0.643 | 2.24x |
| Random Forest | 0.664 | 2.29x |
| **XGBoost** | **0.667** | **2.36x** |

## 🔍 Model Explainability (SHAP)

`number_inpatient` is the clear #1 global driver — independently confirmed
by EDA, SQL, and SHAP. Local explanations on the model's true highest- and
lowest-risk test-set patients are now clean, coherent, one-sentence
explainable stories (e.g. "flagged primarily because of 11 prior inpatient
visits"), a direct result of the Phase 6 bug fix. Full write-up:
[`docs/phase7_model_explainability.md`](docs/phase7_model_explainability.md)

## 📊 Dashboard Design

A star-schema data export (`powerbi/`) plus a full 4-page design spec —
Executive Overview, Risk Segmentation & Prioritization, Clinical &
Operational Drivers, and Model Performance & Trust — with DAX measures,
slicers, and business recommendations. Explicitly handles two real data
constraints: no date/hospital dimension (ruling out trend/facility
comparisons), and a `data_split` flag ensuring every risk score shown is
genuinely out-of-sample, never inflated by training-set memorization.
Full write-up: [`docs/phase8_dashboard_design.md`](docs/phase8_dashboard_design.md)

## ⚙️ Installation & Usage

```bash
# Clone the repo
git clone https://github.com/<your-username>/healthcare-readmission-prediction.git
cd healthcare-readmission-prediction

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline, in order
python src/eda.py                    # Phase 3: EDA figures + stats
python src/data_cleaning.py          # Phase 4: cleaned dataset
python src/run_sql_analysis.py       # Phase 5: SQLite DB + query results
python src/train_models.py           # Phase 6: train + compare 4 models
python src/explainability.py         # Phase 7: SHAP global + local explanations
python src/build_dashboard_data.py   # Phase 8: Power BI data exports
```

## 🎯 Business Impact

Even before any ML model, SQL analysis alone surfaced an actionable
finding: **diabetes-primary diagnosis + 6 or more prior visits = a 30%
readmission rate**, a well-defined, ~700-patient population worth its own
intervention program. The trained XGBoost model then improved on the best
manual rule by ~24% in top-decile lift — a concrete, quantified answer to
"is the added model complexity worth it?" rather than an assumption.

## 📄 License

Code in this repository is released under the [MIT License](LICENSE). The
dataset itself is subject to the UCI Machine Learning Repository's terms —
see the [dataset source](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
for details. This project is built for educational/portfolio purposes.

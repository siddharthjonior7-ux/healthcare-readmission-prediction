"""
build_dashboard_data.py
------------------------
Phase 8: Builds the data layer for the Power BI dashboard.

Produces a small star-schema export under powerbi/:
    - fact_encounters.csv       : every cleaned encounter, scored with the
                                   Phase 6 XGBoost pipeline (predicted
                                   probability + risk decile added)
    - dim_admission_type.csv    : lookup (from IDs_mapping.csv, Phase 5)
    - dim_discharge_disposition.csv
    - dim_admission_source.csv
    - model_performance.csv     : the 4-model comparison table from Phase 6
                                   + the fairness check, for the dashboard's
                                   model-trust page
    - high_risk_worklist.csv    : top 200 encounters by predicted risk --
                                   an actionable "call list" table

IMPORTANT SCOPING NOTE (see docs/phase8_dashboard_design.md):
Scoring is done on the FULL cleaned dataset (99,320 rows), including rows
the model was trained on. This is standard practice for a RETROSPECTIVE
analysis dashboard (the point here is to demonstrate what the dashboard
would show, using real computed numbers) -- in a live production
deployment, only new/unseen encounters would ever be scored, never data
the model was fit on.

Run as: python src/build_dashboard_data.py   (from project root)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = Path("data/processed/cleaned_diabetic_data.csv")
MODEL_PATH = Path("models/best_model.joblib")
OUT_DIR = Path("powerbi")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42  # must match Phase 6's train_models.py exactly

CATEGORICAL_COLS = [
    "gender", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "medical_specialty", "max_glu_serum",
    "A1Cresult", "diag_1_group", "diag_2_group", "diag_3_group",
]
NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "age_numeric",
    "metformin", "repaglinide", "nateglinide", "glimepiride", "glipizide",
    "glyburide", "pioglitazone", "rosiglitazone", "insulin",
    "glyburide-metformin", "change", "diabetesMed",
    "num_med_increased", "num_med_decreased", "service_utilization",
    "had_prior_inpatient",
]


def build_fact_table():
    df = pd.read_csv(DATA_PATH)
    pipeline = joblib.load(MODEL_PATH)

    # -----------------------------------------------------------------
    # CRITICAL: reproduce the EXACT same patient-level train/test split
    # used in Phase 6 (same random_state), and tag every row with which
    # side it fell on. This matters because a model shows artificially
    # optimistic confidence on rows it was TRAINED on (it has partially
    # memorized them) -- scoring the full dataset and treating every
    # score as equally trustworthy would silently mix genuine
    # out-of-sample predictions with inflated in-sample ones. A first
    # pass at this dashboard export did exactly that, and it surfaced
    # immediately: the "highest risk" patients were repeat training-set
    # encounters showing 95%+ confidence, versus the honest 24.3% ceiling
    # established on the real held-out test set in Phase 6/7.
    # -----------------------------------------------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df["patient_nbr"]))
    df["data_split"] = "train"
    df.loc[df.index[test_idx], "data_split"] = "test"

    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    proba = pipeline.predict_proba(X)[:, 1]
    df["predicted_risk_probability"] = proba.round(4)
    df["is_out_of_sample_prediction"] = (df["data_split"] == "test")

    # Risk decile computed ONLY within the test split, so "decile 1" means
    # what it meant in Phase 6/7: genuinely highest-risk among UNSEEN
    # encounters, not inflated by training-set memorization. Train-split
    # rows get a null decile -- their probability column still exists for
    # completeness, but the dashboard should not rank on it.
    df["predicted_risk_decile"] = np.nan
    test_mask = df["data_split"] == "test"
    df.loc[test_mask, "predicted_risk_decile"] = (
        pd.qcut(
            df.loc[test_mask, "predicted_risk_probability"]
              .rank(method="first", ascending=False),
            10, labels=False
        ) + 1
    )

    df["risk_band"] = pd.NA
    df.loc[test_mask, "risk_band"] = pd.cut(
        df.loc[test_mask, "predicted_risk_decile"],
        bins=[0, 2, 5, 10],
        labels=["High Risk (Top 20%)", "Medium Risk (Mid 30%)", "Low Risk (Bottom 50%)"],
    )

    df.to_csv(OUT_DIR / "fact_encounters.csv", index=False)
    print(f"fact_encounters.csv: {df.shape[0]:,} rows x {df.shape[1]} columns "
          f"({test_mask.sum():,} test-split rows carry a trustworthy risk score)")
    return df


def build_dimension_tables():
    for name in ["admission_type", "discharge_disposition", "admission_source"]:
        src = pd.read_csv(f"data/raw/lookup_{name}.csv", na_filter=False)
        src.to_csv(OUT_DIR / f"dim_{name}.csv", index=False)
        print(f"dim_{name}.csv: {len(src)} rows")


def build_model_performance_table():
    """Pulls the real Phase 6 results -- no numbers re-typed by hand."""
    results = pd.read_csv("outputs/model_comparison.csv")
    with open("outputs/model_summary.json") as f:
        summary = json.load(f)

    results["top_decile_lift"] = results["model"].map(
        {r["model"]: r["lift_vs_random"] for r in summary["model_comparison"]}
    )
    results.to_csv(OUT_DIR / "model_performance.csv", index=False)
    print(f"model_performance.csv: {len(results)} models")

    fairness = pd.DataFrame([{
        "check": "race + payer_code excluded (production default)",
        "roc_auc": summary["fairness_check"]["auc_without_race_payer"],
    }, {
        "check": "race + payer_code included",
        "roc_auc": summary["fairness_check"]["auc_with_race_payer"],
    }])
    fairness.to_csv(OUT_DIR / "fairness_check.csv", index=False)
    print("fairness_check.csv: 2 rows")


def build_high_risk_worklist(fact_df: pd.DataFrame, n: int = 200):
    """
    The single most 'operational' export in this phase: a ranked list a
    care coordinator could actually open and start calling down.

    CRITICAL: built ONLY from data_split == "test" rows. Ranking on the
    full dataset (including training rows) would surface patients the
    model has partially memorized, with artificially inflated confidence
    (95%+ in an early version of this script) -- not a fair picture of
    how the model treats a genuinely new patient. Restricting to the test
    split keeps this worklist consistent with the honest performance
    numbers established in Phase 6/7 (max ~24% probability on any single
    real out-of-sample patient).
    """
    test_only = fact_df[fact_df["data_split"] == "test"]
    cols = [
        "encounter_id", "patient_nbr", "predicted_risk_probability",
        "predicted_risk_decile", "diag_1_group", "age_numeric",
        "number_inpatient", "service_utilization", "time_in_hospital",
        "discharge_disposition_id", "medical_specialty", "readmitted_30d",
    ]
    worklist = (
        test_only[cols]
        .sort_values("predicted_risk_probability", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    worklist.insert(0, "priority_rank", range(1, len(worklist) + 1))
    worklist.to_csv(OUT_DIR / "high_risk_worklist.csv", index=False)
    print(f"high_risk_worklist.csv: top {n} TEST-SPLIT (out-of-sample) encounters by predicted risk")
    return worklist


def main():
    print("Building Power BI data exports...\n")
    fact_df = build_fact_table()
    build_dimension_tables()
    build_model_performance_table()
    worklist = build_high_risk_worklist(fact_df)

    print("\n--- Quick validation ---")
    print("Readmit rate by risk_band, TEST SPLIT ONLY (sanity check -- should be strongly ordered):")
    test_only = fact_df[fact_df["data_split"] == "test"]
    print(test_only.groupby("risk_band", observed=True)["readmitted_30d"].agg(["count", "mean"]))
    print(f"\nAll exports saved to {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()

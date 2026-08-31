"""
explainability.py
------------------
Phase 7: Model Explainability for the Healthcare Patient Readmission
Prediction project.

Uses SHAP (SHapley Additive exPlanations) on the winning XGBoost model
from Phase 6 to answer two different questions the Phase 6 built-in
feature_importances_ chart couldn't:
    1. GLOBAL: across all patients, which features matter most, and in
       which DIRECTION (does a high value push risk up or down)?
    2. LOCAL: for one specific patient, why did the model predict what it
       predicted? This is the question a discharge planner or auditor
       actually asks about an individual case.

SHAP is grounded in cooperative game theory (Shapley values): each
feature's SHAP value is its fair share of the difference between the
model's prediction for this patient and the average prediction across all
patients, computed by averaging that feature's marginal contribution
across every possible ordering of features. TreeExplainer computes this
EXACTLY and efficiently for tree ensembles like XGBoost (no sampling
approximation needed, unlike KernelExplainer for arbitrary models).

A PRACTICAL PITFALL THIS SCRIPT DELIBERATELY SURFACES (see docs/phase7):
one-hot encoding fragments a single categorical column (e.g.
discharge_disposition_id) into ~20 separate binary features. Rare
categories (some with fewer than 20 patients in the whole dataset) let
XGBoost fit a near-perfect split to a handful of examples, which gives
that dummy column an inflated SHAP magnitude -- noise, not signal. This
script computes the RAW ranking first (and keeps it, to show the pitfall
honestly), then a RELIABLE ranking that excludes categories with fewer
than MIN_CATEGORY_FREQ patients -- the same volume-threshold discipline
already used in Phase 5's SQL queries (HAVING COUNT(*) >= threshold).

Run as: python src/explainability.py   (from project root)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = Path("data/processed/cleaned_diabetic_data.csv")
MODEL_PATH = Path("models/best_model.joblib")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_30d"
RANDOM_STATE = 42
MIN_CATEGORY_FREQ = 200  # same threshold used in sql/05_admission_source_type_joins.sql

# Must exactly match the feature lists / split used in Phase 6's
# train_models.py so SHAP is computed on the SAME test set the model was
# actually evaluated on.
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


def reproduce_test_set(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the exact same patient-level test split used in Phase 6."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    _, test_idx = next(gss.split(df, groups=df["patient_nbr"]))
    return df.iloc[test_idx]


def get_reliable_features(df: pd.DataFrame, feature_names: list) -> list:
    """
    Return the subset of one-hot/numeric feature names backed by at least
    MIN_CATEGORY_FREQ patients in the full dataset. Numeric features always
    pass through unfiltered -- this only screens out rare one-hot dummies.
    """
    category_counts = {}
    for cat_col in CATEGORICAL_COLS:
        for val, count in df[cat_col].value_counts().items():
            category_counts[f"{cat_col}_{val}"] = count
    return [
        f for f in feature_names
        if f in NUMERIC_COLS or category_counts.get(f, 0) >= MIN_CATEGORY_FREQ
    ]


def main():
    print("Loading cleaned data and trained pipeline...")
    df = pd.read_csv(DATA_PATH)
    test_df = reproduce_test_set(df)
    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    X_test_raw = test_df[NUMERIC_COLS + CATEGORICAL_COLS]

    X_test_transformed = preprocessor.transform(X_test_raw)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    ohe = preprocessor.named_transformers_["cat"]
    feature_names = NUMERIC_COLS + list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=test_df.index)

    print(f"Test set: {X_test_df.shape[0]:,} encounters x {X_test_df.shape[1]} features (post-encoding)")

    reliable_features = get_reliable_features(df, feature_names)
    excluded = [f for f in feature_names if f not in reliable_features]
    print(f"{len(excluded)} of {len(feature_names)} one-hot columns are rare-category "
          f"(< {MIN_CATEGORY_FREQ} patients) and will be excluded from RELIABLE rankings.")

    # For speed, global SHAP values are computed on a random sample of the
    # test set (SHAP values for a tree ensemble on 2,000 rows already give
    # a stable, representative picture of global feature behavior --
    # computing on the full 19,724-row test set would be exact but
    # unnecessarily slow for a portfolio project).
    sample_size = min(2000, len(X_test_df))
    sample_idx = X_test_df.sample(n=sample_size, random_state=RANDOM_STATE).index
    X_sample = X_test_df.loc[sample_idx]

    print(f"Computing SHAP values on a random sample of {sample_size:,} test encounters...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    # SHAP values here are in LOG-ODDS (margin) space, matching XGBoost's
    # raw output before the sigmoid. Sign and relative ranking are what
    # matter for interpretation, not the raw magnitude as "probability".

    # -------------------------------------------------------------
    # Global: RAW bar + beeswarm (kept deliberately, to show the pitfall)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 7))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Global Feature Importance — RAW (includes rare-category noise)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_shap_global_bar.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Summary — RAW (includes rare-category noise)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_shap_beeswarm.png", bbox_inches="tight")
    plt.close()
    print("Saved RAW global SHAP bar + beeswarm charts.")

    # -------------------------------------------------------------
    # Global: RELIABLE bar + beeswarm (rare categories excluded)
    # -------------------------------------------------------------
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv("outputs/shap_global_importance.csv", index=False)

    reliable_importance_df = (
        importance_df[importance_df["feature"].isin(reliable_features)]
        .reset_index(drop=True)
    )
    reliable_importance_df.to_csv("outputs/shap_global_importance_reliable.csv", index=False)
    print(f"\nTop 15 features by mean |SHAP value| (RELIABLE — rare categories excluded):")
    print(reliable_importance_df.head(15).to_string(index=False))

    top15 = reliable_importance_df.head(15)
    plt.figure(figsize=(9, 7))
    plt.barh(top15["feature"][::-1], top15["mean_abs_shap"][::-1], color="#2b6cb0")
    plt.xlabel("mean |SHAP value| (log-odds space)")
    plt.title(f"Global Feature Importance — Reliable (n >= {MIN_CATEGORY_FREQ} per category)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "15_shap_global_bar_reliable.png", bbox_inches="tight")
    plt.close()

    reliable_idx = [feature_names.index(f) for f in reliable_importance_df["feature"]]
    filtered_explanation = shap_values[:, reliable_idx]
    plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(filtered_explanation, max_display=15, show=False)
    plt.title(f"SHAP Summary — Reliable (n >= {MIN_CATEGORY_FREQ} per category)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "16_shap_beeswarm_reliable.png", bbox_inches="tight")
    plt.close()
    print("Saved RELIABLE global SHAP bar + beeswarm charts.")

    # -------------------------------------------------------------
    # Local explainability: one high-risk, one low-risk patient
    # -------------------------------------------------------------
    # Search the FULL test set (not just the 2,000-row sample) for the
    # genuinely highest- and lowest-risk patients -- "highest risk in a
    # random sample of 2,000" can understate the model's true top
    # predictions, since positives are only ~11% of the data.
    full_proba = model.predict_proba(X_test_df.values)[:, 1]
    high_pos = int(np.argmax(full_proba))
    low_pos = int(np.argmin(full_proba))

    high_encounter_id = test_df.iloc[high_pos]["encounter_id"]
    low_encounter_id = test_df.iloc[low_pos]["encounter_id"]
    high_actual = test_df.iloc[high_pos][TARGET]
    low_actual = test_df.iloc[low_pos][TARGET]

    print(f"\nHighest-risk patient in full test set: encounter_id={high_encounter_id}, "
          f"predicted probability={full_proba[high_pos]:.3f}, actual readmitted_30d={high_actual}")
    print(f"Lowest-risk patient in full test set: encounter_id={low_encounter_id}, "
          f"predicted probability={full_proba[low_pos]:.3f}, actual readmitted_30d={low_actual}")

    # Compute SHAP for exactly these two rows on the FULL feature matrix
    # (required -- the fitted booster expects all 121 columns it was
    # trained on), then subset the resulting Explanation object down to
    # the RELIABLE columns only for display. This is the same "compute
    # full, display filtered" pattern used for the beeswarm above -- a
    # local explanation is if anything MORE sensitive to rare-category
    # noise than the global one (a single patient's explanation can be
    # swamped by one noisy dummy), so filtering matters here too.
    high_row = X_test_df.iloc[[high_pos]]
    low_row = X_test_df.iloc[[low_pos]]
    local_explanation = explainer(pd.concat([high_row, low_row]))
    local_reliable_idx = [feature_names.index(f) for f in reliable_features]
    local_filtered = local_explanation[:, local_reliable_idx]

    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(local_filtered[0], max_display=12, show=False)
    plt.title(f"Local Explanation — Highest-Risk Patient (p={full_proba[high_pos]:.2f}, "
              f"actual={'Readmitted' if high_actual else 'Not readmitted'})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_shap_waterfall_high_risk.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(local_filtered[1], max_display=12, show=False)
    plt.title(f"Local Explanation — Lowest-Risk Patient (p={full_proba[low_pos]:.3f}, "
              f"actual={'Readmitted' if low_actual else 'Not readmitted'})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "14_shap_waterfall_low_risk.png", bbox_inches="tight")
    plt.close()
    print("Saved high-risk and low-risk waterfall plots (reliable features only).")

    print("\nDone. All figures saved to outputs/figures/")


if __name__ == "__main__":
    main()

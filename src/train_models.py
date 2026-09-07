"""
train_models.py
----------------
Phase 6: Machine Learning for the Healthcare Patient Readmission Prediction
project. See docs/phase6_machine_learning.md for full design rationale.

Run as: python src/train_models.py   (from project root)
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from xgboost import XGBClassifier
import joblib

DATA_PATH = Path("data/processed/cleaned_diabetic_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_30d"
FAIRNESS_COLS = ["race", "payer_code"]

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
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def patient_level_split(df: pd.DataFrame, test_size: float = 0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df["patient_nbr"]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    overlap = set(train_df["patient_nbr"]) & set(test_df["patient_nbr"])
    assert len(overlap) == 0
    print(f"Patient-level split verified: 0 overlapping patients.")
    print(f"Train: {len(train_df):,} encounters, Test: {len(test_df):,} encounters")
    return train_df, test_df


def get_feature_lists(include_fairness: bool = False):
    cat_cols = CATEGORICAL_COLS + (FAIRNESS_COLS if include_fairness else [])
    return NUMERIC_COLS, cat_cols


def build_preprocessor(numeric_cols, categorical_cols, scale_numeric: bool):
    """
    scale_numeric=True for Logistic Regression; tree models skip scaling.

    sparse_output=False is forced on OneHotEncoder deliberately. The default
    (sparse output) propagates through ColumnTransformer into a single
    sparse combined matrix -- and XGBoost's DMatrix construction treats
    structurally-absent (unstored) zero entries in a sparse matrix as
    MISSING VALUES, not literal zeros. Since ~62% of this dataset's numeric
    feature block is legitimately 0 (change, diabetesMed, had_prior_inpatient,
    most medication columns, number_inpatient for most patients), a sparse
    matrix would silently tell XGBoost that most of its own input is
    missing. scikit-learn's own estimators don't have this semantic, so
    this specifically corrupts XGBoost training -- forcing dense output
    avoids the ambiguity entirely for every model, consistently.
    """
    numeric_transform = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("num", numeric_transform, numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])


def get_model_configs(scale_pos_weight: float):
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE), True),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=50,
                                    class_weight="balanced", random_state=RANDOM_STATE), False),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), False),
        "XGBoost": (
            XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                           scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                           random_state=RANDOM_STATE, n_jobs=-1), False),
    }


def compute_top_decile_lift(y_true, y_proba) -> dict:
    df = pd.DataFrame({"y": y_true, "proba": y_proba})
    df["decile"] = pd.qcut(df["proba"].rank(method="first", ascending=False), 10, labels=False) + 1
    top_decile = df[df["decile"] == 1]
    capture_rate = top_decile["y"].sum() / df["y"].sum()
    lift = top_decile["y"].mean() / df["y"].mean()
    return {
        "top_decile_readmit_rate_pct": round(top_decile["y"].mean() * 100, 2),
        "pct_of_readmissions_captured": round(capture_rate * 100, 2),
        "lift_vs_random": round(lift, 2),
    }


def train_and_evaluate(name, estimator, needs_scaling, train_df, test_df, numeric_cols, categorical_cols):
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(numeric_cols, categorical_cols, needs_scaling)),
        ("model", estimator),
    ])
    X_train, y_train = train_df[numeric_cols + categorical_cols], train_df[TARGET]
    X_test, y_test = test_df[numeric_cols + categorical_cols], test_df[TARGET]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    metrics.update(compute_top_decile_lift(y_test.values, y_proba))
    print(f"\n{name}:")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k}: {v}")
    return pipeline, metrics


def run_fairness_check(best_model_name, model_configs_fn, train_df, test_df):
    numeric_cols, cat_cols_with_fairness = get_feature_lists(include_fairness=True)
    estimator, needs_scaling = model_configs_fn()[best_model_name]
    _, metrics_with = train_and_evaluate(
        f"{best_model_name} + race/payer_code", estimator, needs_scaling,
        train_df, test_df, numeric_cols, cat_cols_with_fairness,
    )
    return metrics_with


def main():
    print("Loading cleaned data...")
    df = load_data()
    train_df, test_df = patient_level_split(df)

    numeric_cols, categorical_cols = get_feature_lists(include_fairness=False)
    pos = train_df[TARGET].sum()
    neg = len(train_df) - pos
    scale_pos_weight = neg / pos
    print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")

    model_configs = get_model_configs(scale_pos_weight)
    all_metrics, pipelines = [], {}

    for name, (estimator, needs_scaling) in model_configs.items():
        pipeline, metrics = train_and_evaluate(
            name, estimator, needs_scaling, train_df, test_df, numeric_cols, categorical_cols
        )
        all_metrics.append(metrics)
        pipelines[name] = pipeline

    results_df = pd.DataFrame(all_metrics).set_index("model")
    print("\n" + "=" * 80 + "\nMODEL COMPARISON\n" + "=" * 80)
    print(results_df.to_string())
    results_df.to_csv("outputs/model_comparison.csv")

    best_model_name = results_df["roc_auc"].idxmax()
    print(f"\nBest model by ROC-AUC: {best_model_name}")

    joblib.dump(pipelines[best_model_name], MODEL_DIR / "best_model.joblib")
    print(f"Saved best model pipeline to {MODEL_DIR / 'best_model.joblib'}")

    print("\n" + "=" * 80 + "\nFAIRNESS CHECK\n" + "=" * 80)
    fairness_metrics = run_fairness_check(
        best_model_name, lambda: get_model_configs(scale_pos_weight), train_df, test_df
    )
    base_auc = results_df.loc[best_model_name, "roc_auc"]
    fair_auc = fairness_metrics["roc_auc"]
    print(f"\nAUC without race/payer_code: {base_auc}, with: {fair_auc}, uplift: {round(fair_auc-base_auc,4)}")

    summary = {
        "model_comparison": results_df.reset_index().to_dict(orient="records"),
        "best_model": best_model_name,
        "fairness_check": {
            "auc_without_race_payer": base_auc,
            "auc_with_race_payer": fair_auc,
            "auc_uplift": round(fair_auc - base_auc, 4),
        },
    }
    with open("outputs/model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved full summary to outputs/model_summary.json")


if __name__ == "__main__":
    main()

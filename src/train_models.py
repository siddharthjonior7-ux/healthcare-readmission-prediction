"""
train_models.py
----------------
Phase 6: Machine Learning for the Healthcare Patient Readmission Prediction
project.

Trains and compares 4 classifiers (Logistic Regression, Decision Tree,
Random Forest, XGBoost) to predict 30-day readmission, using the
analytics-ready table produced in Phase 4.

KEY DESIGN DECISIONS (each explained in docs/phase6_machine_learning.md):
    1. Train/test split is done at the PATIENT level (GroupShuffleSplit on
       patient_nbr), not the row level -- the same patient can have multiple
       encounters, and letting one patient appear in both train and test
       would leak information (Phase 2.1).
    2. Preprocessing (one-hot encoding, scaling) is fit ONLY on the training
       fold, inside a scikit-learn Pipeline -- never on the full dataset
       before splitting (Phase 4.1 / 4.11).
    3. Class imbalance (~11% positive) is handled via class_weight="balanced"
       (sklearn models) / scale_pos_weight (XGBoost), not resampling --
       simpler, no synthetic data, and works cleanly inside a Pipeline.
    4. `race` and `payer_code` are EXCLUDED from the base feature set by
       default (fairness-proxy concern, Phase 2.3) and tested separately as
       a supplementary fairness check (see run_fairness_check()).

Run as: python src/train_models.py   (from project root)
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from xgboost import XGBClassifier
import joblib

DATA_PATH = Path("data/processed/cleaned_diabetic_data.csv")
FIG_DIR = Path("outputs/figures")
MODEL_DIR = Path("models")
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_30d"
ID_COLS = ["encounter_id", "patient_nbr"]
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

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# 1. Data prep
# ---------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def patient_level_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split at the PATIENT level, not the row level. Verified below with an
    explicit overlap check -- this is exactly the kind of thing that's easy
    to get subtly wrong and worth proving, not just asserting.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df["patient_nbr"]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    overlap = set(train_df["patient_nbr"]) & set(test_df["patient_nbr"])
    assert len(overlap) == 0, f"Patient leakage detected: {len(overlap)} overlapping patients!"
    print(f"Patient-level split verified: 0 overlapping patients between train/test.")
    print(f"Train: {len(train_df):,} encounters ({train_df['patient_nbr'].nunique():,} patients)")
    print(f"Test:  {len(test_df):,} encounters ({test_df['patient_nbr'].nunique():,} patients)")
    return train_df, test_df


def get_feature_lists(include_fairness: bool = False):
    cat_cols = CATEGORICAL_COLS + (FAIRNESS_COLS if include_fairness else [])
    return NUMERIC_COLS, cat_cols


# ---------------------------------------------------------------------
# 2. Preprocessing + model configs
# ---------------------------------------------------------------------
def build_preprocessor(numeric_cols, categorical_cols, scale_numeric: bool):
    """
    scale_numeric=True for Logistic Regression (distance/gradient-based,
    sensitive to feature scale). Tree-based models (Decision Tree, Random
    Forest, XGBoost) split on threshold comparisons and are scale-invariant,
    so scaling would add computation with zero benefit -- we skip it for
    them, which is itself a talking point about knowing your algorithms.

    OneHotEncoder(handle_unknown="ignore") ensures a category never seen in
    training (e.g. a rare medical_specialty) doesn't crash inference -- it's
    just encoded as all-zeros instead of raising an error.
    """
    numeric_transform = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("num", numeric_transform, numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])


def get_model_configs(scale_pos_weight: float):
    """
    class_weight="balanced" (sklearn) / scale_pos_weight (XGBoost) upweight
    the minority (readmitted) class during training, penalizing mistakes on
    it more heavily -- addresses the ~11% class imbalance from Phase 3
    without synthetic resampling (e.g. SMOTE), keeping the pipeline simple
    and avoiding any risk of generating unrealistic synthetic patients.
    """
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=50,
                                    class_weight="balanced", random_state=RANDOM_STATE),
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            False,
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                           scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                           random_state=RANDOM_STATE, n_jobs=-1),
            False,
        ),
    }


# ---------------------------------------------------------------------
# 3. Train + evaluate
# ---------------------------------------------------------------------
def compute_top_decile_lift(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Mirrors SQL Query 08 (Phase 5) exactly, so the ML models can be
    compared apples-to-apples against the SQL-only rule-based benchmark
    (~1.9x lift, ~19% of readmissions captured in the top decile).
    """
    df = pd.DataFrame({"y": y_true, "proba": y_proba})
    df["decile"] = pd.qcut(df["proba"].rank(method="first", ascending=False), 10, labels=False) + 1
    top_decile = df[df["decile"] == 1]
    capture_rate = top_decile["y"].sum() / df["y"].sum()
    lift = (top_decile["y"].mean()) / (df["y"].mean())
    return {
        "top_decile_readmit_rate_pct": round(top_decile["y"].mean() * 100, 2),
        "pct_of_readmissions_captured": round(capture_rate * 100, 2),
        "lift_vs_random": round(lift, 2),
    }


def train_and_evaluate(name, estimator, needs_scaling, train_df, test_df,
                        numeric_cols, categorical_cols):
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(numeric_cols, categorical_cols, needs_scaling)),
        ("model", estimator),
    ])

    X_train = train_df[numeric_cols + categorical_cols]
    y_train = train_df[TARGET]
    X_test = test_df[numeric_cols + categorical_cols]
    y_test = test_df[TARGET]

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

    return pipeline, metrics, y_test.values, y_proba, y_pred


# ---------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------
def plot_roc_curves(roc_data: dict):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_test, y_proba, auc) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_roc_curves_comparison.png")
    plt.close()


def plot_confusion_matrices(cm_data: dict):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (name, (y_test, y_pred)) in zip(axes, cm_data.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_confusion_matrices.png")
    plt.close()


def plot_feature_importance(pipeline, numeric_cols, categorical_cols, model_name):
    """Works for tree-based models (feature_importances_) and linear models (coef_)."""
    model = pipeline.named_steps["model"]
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    feature_names = numeric_cols + list(ohe.get_feature_names_out(categorical_cols))

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=imp_df, y="feature", x="importance", ax=ax, color="#2b6cb0")
    ax.set_title(f"Top 15 Feature Importances — {model_name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_feature_importance_best_model.png")
    plt.close()
    return imp_df


# ---------------------------------------------------------------------
# 5. Fairness supplementary check
# ---------------------------------------------------------------------
def run_fairness_check(best_model_name, model_configs_fn, train_df, test_df):
    """
    Retrains the winning model architecture WITH race + payer_code included,
    to quantify the actual predictive lift those features provide -- per
    the plan established in Phase 2.3 / Phase 4.11. If the AUC gain is
    negligible, that's a strong, concrete argument for excluding them in
    any real deployment, not just a hand-wavy fairness gesture.
    """
    numeric_cols, cat_cols_with_fairness = get_feature_lists(include_fairness=True)
    estimator, needs_scaling = model_configs_fn()[best_model_name]

    _, metrics_with, *_ = train_and_evaluate(
        f"{best_model_name} + race/payer_code", estimator, needs_scaling,
        train_df, test_df, numeric_cols, cat_cols_with_fairness,
    )
    return metrics_with


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
def main():
    print("Loading cleaned data...")
    df = load_data()
    print(f"Shape: {df.shape}")

    train_df, test_df = patient_level_split(df)

    numeric_cols, categorical_cols = get_feature_lists(include_fairness=False)
    pos = train_df[TARGET].sum()
    neg = len(train_df) - pos
    scale_pos_weight = neg / pos
    print(f"\nClass balance in training set -- positive: {pos:,}, negative: {neg:,}, "
          f"scale_pos_weight: {scale_pos_weight:.2f}")

    model_configs = get_model_configs(scale_pos_weight)

    all_metrics = []
    roc_data, cm_data, pipelines = {}, {}, {}

    for name, (estimator, needs_scaling) in model_configs.items():
        pipeline, metrics, y_test, y_proba, y_pred = train_and_evaluate(
            name, estimator, needs_scaling, train_df, test_df, numeric_cols, categorical_cols
        )
        all_metrics.append(metrics)
        roc_data[name] = (y_test, y_proba, metrics["roc_auc"])
        cm_data[name] = (y_test, y_pred)
        pipelines[name] = pipeline

    results_df = pd.DataFrame(all_metrics).set_index("model")
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(results_df.to_string())
    results_df.to_csv("outputs/model_comparison.csv")

    plot_roc_curves(roc_data)
    plot_confusion_matrices(cm_data)

    best_model_name = results_df["roc_auc"].idxmax()
    print(f"\nBest model by ROC-AUC: {best_model_name}")

    imp_df = plot_feature_importance(
        pipelines[best_model_name], numeric_cols, categorical_cols, best_model_name
    )
    if imp_df is not None:
        print(f"\nTop 10 features for {best_model_name}:")
        print(imp_df.head(10).to_string(index=False))

    # Save the winning pipeline
    joblib.dump(pipelines[best_model_name], MODEL_DIR / "best_model.joblib")
    print(f"\nSaved best model pipeline to {MODEL_DIR / 'best_model.joblib'}")

    # Fairness supplementary check
    print("\n" + "=" * 80)
    print("FAIRNESS SUPPLEMENTARY CHECK (race + payer_code)")
    print("=" * 80)
    fairness_metrics = run_fairness_check(
        best_model_name, lambda: get_model_configs(scale_pos_weight), train_df, test_df
    )
    base_auc = results_df.loc[best_model_name, "roc_auc"]
    fair_auc = fairness_metrics["roc_auc"]
    print(f"\nROC-AUC without race/payer_code: {base_auc}")
    print(f"ROC-AUC with race/payer_code:    {fair_auc}")
    print(f"AUC uplift from adding these features: {round(fair_auc - base_auc, 4)}")

    # Save full summary as JSON for the docs write-up
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

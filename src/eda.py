"""
eda.py
------
Phase 3: Exploratory Data Analysis for the Healthcare Patient Readmission
Prediction project.

This script is intentionally written as a set of small, single-purpose
functions rather than one long block of code. This is a deliberate
software-engineering choice, not decoration:
    - Each function can be unit-tested in isolation.
    - Each function can be reused in Phase 4 (cleaning) or a notebook
      without copy-pasting.
    - It mirrors how analytics code is structured in production
      pipelines (e.g., a `src/` package imported by notebooks & scripts),
      which is exactly what a Decision Analytics Associate is expected
      to produce.

Run as: python src/eda.py   (from the project root)
Outputs: PNG figures saved to outputs/figures/, a text summary printed
to stdout (redirect to outputs/eda_summary.txt if you want a log file).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_PATH = Path("data/raw/diabetic_data.csv")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110


# ---------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw dataset and standardize missing-value representation.

    IMPORTANT DOMAIN DETAIL: this dataset does NOT use blank cells or
    NaN for missing values. It uses the literal string '?'. If you
    skip this step, pandas will treat '?' as a valid category, missing
    values will look "clean" (0 nulls), and every downstream analysis
    will be silently wrong. This is one of the most common mistakes
    people make with this specific dataset.
    """
    df = pd.read_csv(path)
    df.replace("?", np.nan, inplace=True)
    return df


# ---------------------------------------------------------------------
# 2. Missing value analysis
# ---------------------------------------------------------------------
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a sorted table of missing value counts and percentages.
    We report percentages, not just counts, because "2,273 missing" is
    meaningless without knowing the denominator (101,766) -- a number
    an interviewer will expect you to reason about in relative terms.
    """
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = (
        pd.DataFrame({"missing_count": missing, "missing_pct": pct})
        .query("missing_count > 0")
        .sort_values("missing_pct", ascending=False)
    )
    return report


# ---------------------------------------------------------------------
# 3. Target variable analysis
# ---------------------------------------------------------------------
def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the binary target `readmitted_30d` from the raw 3-class
    `readmitted` column, per the business justification established
    in Phase 2 (CMS's HRRP penalty is anchored to the 30-day window).
    """
    df = df.copy()
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    return df


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Visualize class imbalance for both the raw 3-class and binary target."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    order = ["NO", ">30", "<30"]
    sns.countplot(x="readmitted", data=df, order=order, ax=axes[0])
    axes[0].set_title("Raw target: 3-class readmission status")
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                          ha="center", va="bottom", fontsize=9)

    sns.countplot(x="readmitted_30d", data=df, ax=axes[1])
    axes[1].set_title("Binary target: readmitted within 30 days")
    axes[1].set_xticklabels(["No (0)", "Yes (1)"])
    for p in axes[1].patches:
        pct = 100 * p.get_height() / len(df)
        axes[1].annotate(f"{p.get_height():,} ({pct:.1f}%)",
                          (p.get_x() + p.get_width() / 2, p.get_height()),
                          ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_target_distribution.png")
    plt.close()


# ---------------------------------------------------------------------
# 4. Leakage check: expired / hospice patients
# ---------------------------------------------------------------------
def leakage_check(df: pd.DataFrame) -> dict:
    """
    Quantify the mortality/hospice contamination issue flagged in
    Phase 2. discharge_disposition_id codes 11, 13, 14, 19, 20, 21
    correspond to expired or hospice discharges (per IDs_mapping.csv).
    """
    death_hospice_codes = [11, 13, 14, 19, 20, 21]
    mask = df["discharge_disposition_id"].isin(death_hospice_codes)
    return {
        "n_death_or_hospice_encounters": int(mask.sum()),
        "pct_of_total": round(100 * mask.sum() / len(df), 2),
        "readmit_rate_in_this_group": round(
            100 * df.loc[mask, "readmitted_30d"].mean(), 2
        ),
    }


# ---------------------------------------------------------------------
# 5. Univariate analysis - numeric features
# ---------------------------------------------------------------------
NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Histogram grid for the core numeric features."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flatten(), NUMERIC_COLS):
        sns.histplot(df[col], bins=30, ax=ax, kde=False)
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_numeric_distributions.png")
    plt.close()


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-based outlier flags for numeric columns. We report the count
    of statistical outliers, but note in the write-up (below) that in
    this domain, "outlier" does NOT automatically mean "error" -- it
    often means "high-acuity patient", so the treatment decision in
    Phase 4 differs from a typical retail/finance dataset.
    """
    rows = []
    for col in NUMERIC_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append({
            "feature": col, "q1": q1, "q3": q3,
            "lower_bound": round(lower, 2), "upper_bound": round(upper, 2),
            "n_outliers": n_outliers,
            "pct_outliers": round(100 * n_outliers / len(df), 2),
        })
    return pd.DataFrame(rows)


def plot_outlier_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flatten(), NUMERIC_COLS):
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_outlier_boxplots.png")
    plt.close()


# ---------------------------------------------------------------------
# 6. Bivariate analysis - readmission rate by feature
# ---------------------------------------------------------------------
def readmission_rate_by_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    For a categorical/ordinal column, compute readmission rate per
    category alongside the sample size (n) for that category.
    Reporting n alongside the rate matters: a 40% readmission rate
    based on 12 patients is noise, not a business insight.
    """
    grp = df.groupby(col)["readmitted_30d"].agg(["mean", "count"])
    grp["mean"] = (grp["mean"] * 100).round(2)
    grp.columns = ["readmit_rate_pct", "n"]
    return grp.sort_values("readmit_rate_pct", ascending=False)


def plot_readmission_by_prior_utilization(df: pd.DataFrame) -> None:
    """
    The single most business-relevant chart in this phase: does prior
    healthcare utilization predict future readmission risk?
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ["number_inpatient", "number_emergency", "number_outpatient"]):
        capped = df[col].clip(upper=5)  # cap for readability; long tail is sparse
        rate = df.assign(_c=capped).groupby("_c")["readmitted_30d"].mean() * 100
        rate.plot(kind="bar", ax=ax, color="#2b6cb0")
        ax.set_title(f"30-day readmit rate vs prior {col.replace('number_', '')} visits")
        ax.set_xlabel(f"{col} (capped at 5)")
        ax.set_ylabel("Readmission rate (%)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_readmission_vs_prior_utilization.png")
    plt.close()


def plot_readmission_by_age(df: pd.DataFrame) -> None:
    age_order = sorted(df["age"].dropna().unique(),
                        key=lambda x: int(x.strip("[)").split("-")[0]))
    rate = df.groupby("age")["readmitted_30d"].mean().reindex(age_order) * 100
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rate.plot(kind="bar", ax=ax, color="#c05621")
    ax.set_title("30-day readmission rate by age band")
    ax.set_ylabel("Readmission rate (%)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_readmission_by_age.png")
    plt.close()


def plot_readmission_by_a1c(df: pd.DataFrame) -> None:
    """
    Tests the well-known finding from the original Strack et al. (2014)
    study behind this dataset: whether an A1C test was ordered at all
    matters more than the result itself.
    """
    rate = df.groupby("A1Cresult")["readmitted_30d"].mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rate.plot(kind="bar", ax=ax, color="#2f855a")
    ax.set_title("30-day readmission rate by A1C test result\n('None' = test not ordered)")
    ax.set_ylabel("Readmission rate (%)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_readmission_by_a1c.png")
    plt.close()


# ---------------------------------------------------------------------
# 7. Correlation analysis (numeric features only)
# ---------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[NUMERIC_COLS + ["readmitted_30d"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix: numeric features vs 30-day readmission")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_correlation_heatmap.png")
    plt.close()
    return corr["readmitted_30d"].drop("readmitted_30d").sort_values(ascending=False)


# ---------------------------------------------------------------------
# 8. Diagnosis code cardinality
# ---------------------------------------------------------------------
def diagnosis_cardinality(df: pd.DataFrame) -> dict:
    return {
        "unique_diag_1": df["diag_1"].nunique(),
        "unique_diag_2": df["diag_2"].nunique(),
        "unique_diag_3": df["diag_3"].nunique(),
    }


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
def main():
    print("Loading data...")
    df = load_data()
    df = add_binary_target(df)
    print(f"Shape: {df.shape}")
    print(f"Unique patients: {df['patient_nbr'].nunique():,} across {len(df):,} encounters")

    print("\n--- Missing Value Report (%) ---")
    print(missing_value_report(df))

    print("\n--- Target Distribution ---")
    print(df["readmitted"].value_counts())
    print(f"30-day readmission rate: {df['readmitted_30d'].mean()*100:.2f}%")
    plot_target_distribution(df)

    print("\n--- Leakage Check: Expired / Hospice discharges ---")
    print(leakage_check(df))

    print("\n--- Numeric Feature Summary ---")
    print(df[NUMERIC_COLS].describe().T)
    plot_numeric_distributions(df)

    print("\n--- Outlier Summary (IQR method) ---")
    print(outlier_summary(df))
    plot_outlier_boxplots(df)

    print("\n--- Readmission Rate by Age ---")
    print(readmission_rate_by_category(df, "age"))
    plot_readmission_by_age(df)

    print("\n--- Readmission Rate by A1C Result ---")
    print(readmission_rate_by_category(df, "A1Cresult"))
    plot_readmission_by_a1c(df)

    print("\n--- Readmission Rate by Prior Utilization (charts saved) ---")
    plot_readmission_by_prior_utilization(df)

    print("\n--- Correlation with Target ---")
    print(plot_correlation_heatmap(df))

    print("\n--- Diagnosis Code Cardinality ---")
    print(diagnosis_cardinality(df))

    print("\nAll figures saved to:", FIG_DIR.resolve())


if __name__ == "__main__":
    main()

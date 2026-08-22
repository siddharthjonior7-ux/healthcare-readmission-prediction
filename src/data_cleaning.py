"""
data_cleaning.py
-----------------
Phase 4: Data Cleaning & Feature Engineering for the Healthcare Patient
Readmission Prediction project.

DESIGN PRINCIPLE (important, explained in docs/phase4_data_cleaning.md):
This script produces an ANALYTICS-READY dataset, not a MODEL-READY one.
That means:
    - We DO remove leakage rows, drop unusable columns, group high-cardinality
      categories, and engineer new features. These are deterministic,
      domain-knowledge-driven transformations that don't depend on how the
      data happens to be split later.
    - We do NOT one-hot encode, scale, or impute using statistics (mean/
      median/mode) computed across the whole dataset. Those steps are
      deferred to a scikit-learn Pipeline in Phase 6, fit ONLY on the
      training fold, to avoid data leakage from test into train.
    - Keeping the output human-readable (e.g. "Circulatory" instead of a
      one-hot column) also makes it directly usable for the Phase 5 SQL
      analysis and the Phase 8 BI dashboard, which is deliberate.

Run as: python src/data_cleaning.py   (from project root)
Input:  data/raw/diabetic_data.csv
Output: data/processed/cleaned_diabetic_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/diabetic_data.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Codes for discharge_disposition_id that mean "expired" or "hospice"
# (from IDs_mapping.csv) -- these encounters cannot be readmitted by
# definition and must be removed to avoid label leakage (see Phase 3, 3.3).
DEATH_HOSPICE_CODES = [11, 13, 14, 19, 20, 21]

# Medication columns that are >99.5% a single value across 101,766 rows
# (measured directly on the raw data -- see project notes). These carry
# essentially no signal and would only add noise / sparse one-hot columns.
NEAR_ZERO_VARIANCE_MEDS = [
    "chlorpropamide", "acetohexamide", "tolbutamide", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Medication columns retained for feature engineering (have >0.5% variation)
KEPT_MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "glimepiride", "glipizide",
    "glyburide", "pioglitazone", "rosiglitazone", "insulin",
    "glyburide-metformin",
]

MED_ORDINAL_MAP = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}


# ---------------------------------------------------------------------
# 1. Load + standardize missing values
# ---------------------------------------------------------------------
def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.replace("?", np.nan, inplace=True)
    return df


# ---------------------------------------------------------------------
# 2. Remove leakage rows
# ---------------------------------------------------------------------
def remove_leakage_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop expired/hospice discharges (see Phase 3.3 for the quantified check)."""
    before = len(df)
    df = df[~df["discharge_disposition_id"].isin(DEATH_HOSPICE_CODES)].copy()
    print(f"Removed {before - len(df):,} expired/hospice encounters "
          f"({(before - len(df)) / before * 100:.2f}% of rows)")
    return df


# ---------------------------------------------------------------------
# 3. Create the binary target, drop the raw target
# ---------------------------------------------------------------------
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    df.drop(columns=["readmitted"], inplace=True)
    return df


# ---------------------------------------------------------------------
# 4. Drop unusable / identifier columns
# ---------------------------------------------------------------------
def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    weight: ~97% missing -> unusable.
    encounter_id: unique per row, not predictive (kept only as an index
    candidate, dropped from the feature set here for a single clean table).
    Near-zero-variance medication columns: see NEAR_ZERO_VARIANCE_MEDS above.
    """
    cols_to_drop = ["weight"] + NEAR_ZERO_VARIANCE_MEDS
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} unusable columns: {cols_to_drop}")
    return df


# ---------------------------------------------------------------------
# 5. Handle missing categoricals (deterministic category-level fills,
#    NOT statistical imputation -- see module docstring)
# ---------------------------------------------------------------------
def handle_missing_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # race: small amount missing (2.23%) -> explicit "Unknown" category
    # rather than dropping rows or guessing a race.
    df["race"] = df["race"].fillna("Unknown")

    # A1Cresult / max_glu_serum: missing == test not ordered, which is
    # itself informative (see Phase 3.4) -> encode as "Not Tested", never
    # impute a fake lab value.
    df["A1Cresult"] = df["A1Cresult"].fillna("Not Tested")
    df["max_glu_serum"] = df["max_glu_serum"].fillna("Not Tested")

    # medical_specialty: 49% missing + 73 raw categories. Keep the top 10
    # specialties by frequency (which cover the large majority of encounters)
    # and collapse everything else -- including missing -- into
    # "Other/Missing". This avoids an extremely sparse one-hot block later.
    top_specialties = df["medical_specialty"].value_counts().nlargest(10).index
    df["medical_specialty"] = df["medical_specialty"].where(
        df["medical_specialty"].isin(top_specialties), "Other/Missing"
    )

    # payer_code: 39.56% missing, 17 raw categories, and a known fairness
    # proxy (Phase 2.3) -- keep top 5 payers, collapse the rest (incl.
    # missing) into "Other/Unknown" so it can be tested as an optional
    # feature in Phase 6 without exploding dimensionality.
    top_payers = df["payer_code"].value_counts().nlargest(5).index
    df["payer_code"] = df["payer_code"].where(
        df["payer_code"].isin(top_payers), "Other/Unknown"
    )

    return df


# ---------------------------------------------------------------------
# 6. Diagnosis code grouping (ICD-9 -> 9 clinical categories)
# ---------------------------------------------------------------------
def _map_icd9_to_group(code) -> str:
    """
    Map a raw ICD-9 diagnosis code to one of 9 clinical categories, using
    the grouping established in the original research behind this dataset
    (Strack et al., 2014) and widely used in follow-up work. V-codes
    (supplemental classification, e.g. 'V27') and E-codes (external causes,
    e.g. 'E849') go to 'Other' along with everything not in a named range.
    """
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        val = float(code)
    except ValueError:
        return "Other"

    if 390 <= val <= 459 or val == 785:
        return "Circulatory"
    if 460 <= val <= 519 or val == 786:
        return "Respiratory"
    if 520 <= val <= 579 or val == 787:
        return "Digestive"
    if 250 <= val < 251:
        return "Diabetes"
    if 800 <= val <= 999:
        return "Injury"
    if 710 <= val <= 739:
        return "Musculoskeletal"
    if 580 <= val <= 629 or val == 788:
        return "Genitourinary"
    if 140 <= val <= 239:
        return "Neoplasms"
    return "Other"


def group_diagnosis_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_group"] = df[col].apply(_map_icd9_to_group)
    # Drop rows where the PRIMARY diagnosis is entirely missing -- too
    # small a group (~0.02%) to safely impute, and diag_1 is clinically
    # central to the encounter.
    before = len(df)
    df = df[df["diag_1_group"] != "Missing"].copy()
    print(f"Dropped {before - len(df)} rows with missing primary diagnosis")
    df.drop(columns=["diag_1", "diag_2", "diag_3"], inplace=True)
    return df


# ---------------------------------------------------------------------
# 7. Age: ordinal bands -> numeric midpoint
# ---------------------------------------------------------------------
def encode_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    age arrives pre-binned as strings like '[50-60)'. We convert to the
    numeric midpoint (e.g. 55) so it can be used as a continuous feature
    and preserves the natural ordering -- a plain one-hot encoding would
    throw away the fact that '[70-80)' is "between" '[60-70)' and '[80-90)'.
    """
    df = df.copy()
    df["age_numeric"] = df["age"].apply(
        lambda x: (int(x.strip("[)").split("-")[0]) + int(x.strip("[)").split("-")[1])) / 2
    )
    df.drop(columns=["age"], inplace=True)
    return df


# ---------------------------------------------------------------------
# 8. Medication features
# ---------------------------------------------------------------------
def encode_medications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal-encode the 10 retained medication columns (No=0, Down=1,
    Steady=2, Up=3) -- this treats dosage change as a severity scale,
    which is a reasonable simplification for a portfolio model (a purely
    nominal encoding would lose the "increase vs decrease" ordering).

    Also engineer two summary features that compress the 10 columns into
    signals the original per-drug columns can't express directly:
      - num_med_increased: count of drugs where dosage went UP this stay
      - num_med_decreased: count of drugs where dosage went DOWN this stay
    A rise in num_med_increased is a plausible proxy for "diabetes was
    poorly controlled and needed active management during this stay".
    """
    df = df.copy()
    for col in KEPT_MED_COLS:
        df[col] = df[col].map(MED_ORDINAL_MAP)

    df["num_med_increased"] = (df[KEPT_MED_COLS] == 3).sum(axis=1)
    df["num_med_decreased"] = (df[KEPT_MED_COLS] == 1).sum(axis=1)

    # change / diabetesMed are Yes/No -> binary
    df["change"] = (df["change"] == "Ch").astype(int)
    df["diabetesMed"] = (df["diabetesMed"] == "Yes").astype(int)

    return df


# ---------------------------------------------------------------------
# 8b. Drop invalid gender rows
# ---------------------------------------------------------------------
def drop_invalid_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    3 rows (out of ~100K) have gender == 'Unknown/Invalid'. Too small a
    group to encode as its own category (it would just be sparse noise),
    and too small to safely infer -- dropped explicitly here rather than
    silently left in as an undocumented edge case.
    """
    before = len(df)
    df = df[df["gender"] != "Unknown/Invalid"].copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with gender == 'Unknown/Invalid'")
    return df


# ---------------------------------------------------------------------
# 9. Utilization feature engineering
# ---------------------------------------------------------------------
def engineer_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    service_utilization: a single combined measure of how much the patient
    has used the healthcare system in the prior year. Motivated directly by
    the Phase 3 finding that number_inpatient/number_emergency/
    number_outpatient were the strongest individual predictors -- combining
    them gives models (especially linear ones) a single strong signal
    instead of three correlated ones.

    had_prior_inpatient: a simple binary flag mirroring the business rule
    surfaced in Phase 3 ("3+ prior inpatient visits" as a manual triage
    heuristic) -- kept as a low-complexity benchmark feature.
    """
    df = df.copy()
    df["service_utilization"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["had_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
    return df


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------
def main():
    print("Loading raw data...")
    df = load_data()
    print(f"Raw shape: {df.shape}")

    df = remove_leakage_rows(df)
    df = create_target(df)
    df = drop_unusable_columns(df)
    df = handle_missing_categoricals(df)
    df = group_diagnosis_codes(df)
    df = encode_age(df)
    df = encode_medications(df)
    df = drop_invalid_gender(df)
    df = engineer_utilization_features(df)

    print(f"\nFinal cleaned shape: {df.shape}")
    print(f"Remaining missing values: {df.isnull().sum().sum()}")
    print(f"Readmission rate after cleaning: {df['readmitted_30d'].mean()*100:.2f}%")

    out_path = OUT_DIR / "cleaned_diabetic_data.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved cleaned dataset to {out_path.resolve()}")

    print("\nFinal columns:")
    print(list(df.columns))


if __name__ == "__main__":
    main()

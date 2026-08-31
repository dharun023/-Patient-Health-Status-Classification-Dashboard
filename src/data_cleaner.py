import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features for the multi-test patient health dataset.

    Creates:
      - age_group: Child, Teen, Adult, Senior
      - ref_low, ref_high: selected by gender and child/adult status
      - ref_range_width, ref_midpoint, relative_deviation
      - numeric_status: Low/Normal/High derived from result_num and reference ranges
      - target: cleaned version of indication_final
    """
    df = df.copy()

    # 1. Normalize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # 2. Drop unnamed columns
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]

    # 3. Standardize text fields
    text_columns = [
        "test_name",
        "unit",
        "group_name",
        "name",
        "gender",
        "doctor_name",
        "kid_or_not",
        "result",
        "indication",
        "indication_final",
    ]
    for col in text_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            )

    # 4. Convert numeric columns
    numeric_columns = [
        "id",
        "patient_id",
        "age",
        "result_num",
        "ref_low_m",
        "ref_high_m",
        "ref_low_f",
        "ref_high_f",
        "ref_low_k",
        "ref_high_k",
        "order_index",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Standardize gender
    if "gender" in df.columns:
        df["gender"] = (
            df["gender"]
            .str.lower()
            .replace({"m": "Male", "male": "Male", "f": "Female", "female": "Female"})
            .fillna("Unknown")
        )

    # 6. Standardize child/adult
    if "kid_or_not" in df.columns:
        df["kid_or_not"] = (
            df["kid_or_not"]
            .str.lower()
            .replace({
                "kid": "Kid",
                "child": "Kid",
                "not kid": "Not Kid",
                "not ki": "Not Kid",
                "adult": "Not Kid",
            })
            .fillna("Not Kid")
        )
    else:
        df["kid_or_not"] = np.where(df["age"] < 18, "Kid", "Not Kid")

    # 7. Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[-np.inf, 12, 17, 59, np.inf],
        labels=["Child", "Teen", "Adult", "Senior"],
    ).astype("string")

    # 8. Select applicable reference range (by gender and kid/adult)
    is_kid = df["kid_or_not"].eq("Kid")
    is_male = df["gender"].eq("Male")
    is_female = df["gender"].eq("Female")

    df["ref_low"] = np.select(
        [is_kid, is_male, is_female],
        [
            df.get("ref_low_k", np.nan),
            df.get("ref_low_m", np.nan),
            df.get("ref_low_f", np.nan),
        ],
        default=np.nan,
    )

    df["ref_high"] = np.select(
        [is_kid, is_male, is_female],
        [
            df.get("ref_high_k", np.nan),
            df.get("ref_high_m", np.nan),
            df.get("ref_high_f", np.nan),
        ],
        default=np.nan,
    )

    # 9. Derived range features
    df["ref_range_width"] = df["ref_high"] - df["ref_low"]
    df["ref_midpoint"] = (df["ref_low"] + df["ref_high"]) / 2

    df["relative_deviation"] = np.where(
        df["ref_range_width"] > 0,
        (df["result_num"] - df["ref_midpoint"]) / df["ref_range_width"],
        np.nan,
    )

    # 10. Numeric status (Low/Normal/High) from result_num and reference ranges
    def numeric_status(row):
        value = row["result_num"]
        low = row["ref_low"]
        high = row["ref_high"]

        if pd.isna(value):
            return "Non-numeric or missing"
        if pd.isna(low) or pd.isna(high):
            return "Numeric, reference unavailable"
        if value < low:
            return "Low"
        if value > high:
            return "High"
        return "Normal"

    df["numeric_status"] = df.apply(numeric_status, axis=1)

    # 11. Target: cleaned version of indication_final
    if "indication_final" in df.columns:
        df["target"] = df["indication_final"].astype("string").str.strip()
    elif "indication" in df.columns:
        df["target"] = df["indication"].astype("string").str.strip()
    else:
        df["target"] = pd.NA

    # Normalize common labels
    df["target"] = (
        df["target"]
        .replace({
            "Norma": "Normal",
            "Normal": "Normal",
            "Hi": "High",
            "High": "High",
            "Lo": "Low",
            "Low": "Low",
            "Unexpected value": "Unexpected value",
        })
    )

    # 12. Reference range text (for display in dashboard)
    df["reference_range_used"] = np.where(
        df["ref_low"].notna() & df["ref_high"].notna(),
        df["ref_low"].round(2).astype("string")
        + " - "
        + df["ref_high"].round(2).astype("string"),
        "Not available",
    )

    # 13. Doctor cleanup
    if "doctor_name" in df.columns:
        df["doctor_name"] = df["doctor_name"].fillna("Unknown Doctor")

    # 14. (Optional) Add unit mapping if unit column is empty
    # Uncomment and adjust if you want to infer units from test_name
    # unit_map = {
    #     "Sugar (F)": "mg/dL",
    #     "Sugar (P.P)": "mg/dL",
    #     "Haemoglobin": "g/dL",
    #     "Peripheral Smear": "report",
    #     "Widal Te": "titer",
    # }
    # if "unit" in df.columns:
    #     df["unit"] = df["test_name"].map(unit_map).fillna(df["unit"])

    return df


if __name__ == "__main__":
    # Example usage on your local file
    raw_df = pd.read_excel("data/Final Merged data.xlsx")
    clean_df = engineer_features(raw_df)

    feature_cols = [
        "test_name",
        "name",
        "age",
        "gender",
        "kid_or_not",
        "age_group",
        "result_num",
        "ref_low",
        "ref_high",
        "ref_range_width",
        "ref_midpoint",
        "relative_deviation",
        "numeric_status",
        "indication_final",
        "target",
        "reference_range_used",
    ]

    print(clean_df[feature_cols].head(10).to_string())
    print("\nFeature engineering complete. Data ready for modeling and dashboard.")
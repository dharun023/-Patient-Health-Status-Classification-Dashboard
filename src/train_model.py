import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from src.data_cleaner import engineer_features


def train_and_save(
    raw_df: pd.DataFrame,
    model_path: str = "models/patient_status_model.pkl",
    use_target: str = "target",
):
    """
    Train a classification model on the multi-test patient health dataset.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw dataframe from merged.xlsx.
    model_path : str
        Path to save the trained model (joblib).
    use_target : str
        Column name to use as target: "target" or "numeric_status".
    """

    # 1. Clean and engineer features
    df = engineer_features(raw_df)

    # 2. Choose target
    if use_target not in df.columns:
        raise ValueError(f"Target column '{use_target}' not found in data.")

    y = df[use_target]

    # 3. Filter out rows with missing or non-informative targets
    valid_mask = y.notna() & ~y.isin(["", "Result not entered"])
    df = df[valid_mask].copy()
    y = y[valid_mask]

    # 4. Define feature columns
    numeric_features = [
        "age",
        "result_num",
        "ref_low",
        "ref_high",
        "ref_range_width",
        "ref_midpoint",
        "relative_deviation",
    ]

    categorical_features = [
        "test_name",
        "group_name",
        "gender",
        "kid_or_not",
        "age_group",
    ]

    feature_columns = numeric_features + categorical_features

    # Ensure all feature columns exist
    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = df[feature_columns]

    # 5. Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )

    # 6. Classifier
    classifier = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    # 7. Train/test split by patient_id to avoid leakage
    groups = df["patient_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 8. Train model
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # 9. Evaluation
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # Optional: quick cross-val check on training set
    cv_scores = cross_val_score(
        classifier,
        X_train,
        y_train,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    print(f"3-fold CV accuracy (train): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 10. Save model
    joblib.dump(classifier, model_path)
    print(f"\nModel saved to: {model_path}")

    return classifier, X, y


if __name__ == "__main__":
    # Load your dataset
    raw_df = pd.read_excel("data/Final Merged data.xlsx")

    # Choose target: "target" (indication_final) or "numeric_status"
    model, X, y = train_and_save(
        raw_df,
        model_path="models/patient_status_model.pkl",
        use_target="target",  # or "numeric_status"
    )
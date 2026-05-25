import numpy as np
import pandas as pd

TARGET_COLUMN = "exam_score"

NUMERIC_FEATURES = [
    "study_hours_per_day",
    "attendance_percentage",
    "sleep_hours",
    "mental_health_rating",
    "exercise_frequency",
    "social_media_hours",
    "netflix_hours",
]

CATEGORICAL_FEATURES = [
    "part_time_job",
    "diet_quality",
    "internet_quality",
    "extracurricular_participation",
    "parental_education_level",
]

DEFAULT_NUMERIC = {
    "study_hours_per_day": 3.0,
    "attendance_percentage": 75.0,
    "sleep_hours": 7.0,
    "mental_health_rating": 6.0,
    "exercise_frequency": 2.0,
    "social_media_hours": 2.5,
    "netflix_hours": 1.5,
}

DEFAULT_CATEGORICAL = {
    "part_time_job": "No",
    "diet_quality": "Fair",
    "internet_quality": "Average",
    "extracurricular_participation": "No",
    "parental_education_level": "Bachelor",
}


def clean_dataset(df):
    """Rapikan kolom agar aman untuk dataset Kaggle maupun preview lokal."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    for col, default in DEFAULT_NUMERIC.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    for col, default in DEFAULT_CATEGORICAL.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str)

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Kolom target exam_score tidak ditemukan pada dataset.")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return df


def encode_features(df, encoder_columns=None):
    base = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    encoded = pd.get_dummies(base, columns=CATEGORICAL_FEATURES, drop_first=False, dtype=float)

    if encoder_columns is None:
        encoder_columns = list(encoded.columns)
    else:
        for col in encoder_columns:
            if col not in encoded.columns:
                encoded[col] = 0.0
        encoded = encoded[encoder_columns]

    return encoded.astype(float), encoder_columns


def fit_minmax(X):
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    ranges = np.where((max_vals - min_vals) == 0, 1, max_vals - min_vals)
    return min_vals, max_vals, ranges


def transform_minmax(X, min_vals, ranges):
    return (X - min_vals) / ranges


def normalize_target(y):
    return np.asarray(y, dtype=float).reshape(-1, 1) / 100.0


def denormalize_exam_score(y_norm):
    return np.asarray(y_norm, dtype=float) * 100.0


def make_risk_label(exam_score, threshold=60):
    return (np.asarray(exam_score, dtype=float) < threshold).astype(float)


def prepare_training_data(df):
    df = clean_dataset(df)
    X_df, encoder_columns = encode_features(df)
    X = X_df.values.astype(float)
    min_vals, max_vals, ranges = fit_minmax(X)
    X_norm = transform_minmax(X, min_vals, ranges)

    y_score = normalize_target(df[TARGET_COLUMN].values)
    y_risk = make_risk_label(df[TARGET_COLUMN].values).reshape(-1)

    meta = {
        "encoder_columns": encoder_columns,
        "min_vals": min_vals,
        "max_vals": max_vals,
        "ranges": ranges,
        "feature_table_columns": list(X_df.columns),
    }
    return df, X_norm, y_score, y_risk, meta


def prepare_user_input(values, meta):
    row = pd.DataFrame([values])
    for col, default in DEFAULT_NUMERIC.items():
        if col not in row.columns:
            row[col] = default
    for col, default in DEFAULT_CATEGORICAL.items():
        if col not in row.columns:
            row[col] = default

    X_df, _ = encode_features(row, encoder_columns=meta["encoder_columns"])
    X = X_df.values.astype(float)
    return transform_minmax(X, meta["min_vals"], meta["ranges"])

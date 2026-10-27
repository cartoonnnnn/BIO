"""Data loading, cleaning and splitting for the UCI Phishing Websites dataset."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from . import config


def load_raw() -> pd.DataFrame:
    return pd.read_csv(config.DATA_PATH)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate and clean the data. Returns the cleaned frame and a small report.

    The raw file has only discrete feature values {-1, 0, 1}, so many rows are exact
    duplicates. Keeping them would put identical samples in both the train and the test
    split and inflate every metric, so duplicates are dropped *before* splitting.
    """
    report = {
        "raw_rows": int(len(df)),
        "n_features": int(df.shape[1] - 1),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    report["clean_rows"] = int(len(df))
    # recode target to {1 = phishing, 0 = legitimate} so that "positive" = phishing
    df[config.TARGET] = (df[config.TARGET] == -1).astype(int)
    report["class_counts"] = {
        "legitimate": int((df[config.TARGET] == 0).sum()),
        "phishing": int((df[config.TARGET] == 1).sum()),
    }
    return df, report


def split(df: pd.DataFrame):
    """Stratified 80/20 split. Trees are scale-invariant, so no feature scaling is needed."""
    X = df.drop(columns=config.TARGET)
    y = df[config.TARGET]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SPLIT_SEED
    )


def prepare():
    raw = load_raw()
    df, report = clean(raw)
    X_train, X_test, y_train, y_test = split(df)
    report.update(train_rows=int(len(X_train)), test_rows=int(len(X_test)))
    return df, X_train, X_test, y_train, y_test, report

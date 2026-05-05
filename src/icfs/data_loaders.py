#./src/icfs/
"""Data loading and preprocessing for UCI Student Performance."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

CATEGORICAL_COLS = [
    'school', 'sex', 'address', 'famsize', 'Pstatus',
    'Mjob', 'Fjob', 'reason', 'guardian',
    'schoolsup', 'famsup', 'paid', 'activities',
    'nursery', 'higher', 'internet', 'romantic'
]


def load_uci(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=';')


def preprocess_uci(df: pd.DataFrame):
    """Returns X (2D np array), y (np array), feature_names (list)."""
    df = df.copy()
    # Target: pass/fail at G3 >= 10
    y = (df['G3'] >= 10).astype(int).values
    df = df.drop(columns=['G3'])

    present_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    binary_cols = [c for c in present_cats if df[c].nunique() == 2]
    multi_cols  = [c for c in present_cats if df[c].nunique() > 2]

    le = LabelEncoder()
    for c in binary_cols:
        df[c] = le.fit_transform(df[c].astype(str))
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=False)

    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    feature_names = df.columns.tolist()
    X = df.values.astype(np.float64)
    return X, y, feature_names


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                              random_state=random_state, stratify=y)


def load_and_split(path: str, test_size=0.2, random_state=42):
    df = load_uci(path)
    X, y, names = preprocess_uci(df)
    X_tr, X_te, y_tr, y_te = split_data(X, y, test_size, random_state)
    return X_tr, X_te, y_tr, y_te, names

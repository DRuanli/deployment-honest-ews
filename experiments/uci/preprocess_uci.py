"""
Shared UCI preprocessing utilities.

Used by:
  - run_uci_experiments.py (main IC-FS sweep)
  - run_uci_dre.py (deployment-realistic eval)
  - run_uci_baselines.py (NSGA-II, Stability Selection, Boruta)
  - run_uci_statistics.py (multi-seed Wilcoxon)

Output contract: preprocess_uci(df, horizon) -> (X, y, names)
where names align with TAXONOMY_UCI via _resolve_parent prefix matching.
"""
#./experiments/uci/

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Path setup so this module is importable from anywhere
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def preprocess_uci(df: pd.DataFrame, horizon: int = 0, verbose: bool = False):
    """
    Preprocess UCI Student Performance dataset for IC-FS / baselines.

    Args:
        df: DataFrame from student-mat.csv or student-por.csv
        horizon: Prediction horizon (0, 1, or 2)
        verbose: print shape info

    Returns:
        X: float64 ndarray (n, p)
        y: int ndarray (n,)
        feature_names: list of column names after one-hot encoding
    """
    df = df.copy()

    # Target: Pass (G3 >= 10) vs Fail (G3 < 10)
    if 'G3' not in df.columns:
        raise ValueError("Expected column 'G3' in UCI dataset")
    y = (df['G3'] >= 10).astype(int).values
    df = df.drop(columns=['G3'])

    # Horizon-specific feature availability
    # h=0: Beginning of semester (no G1, G2, absences)
    # h=1: Mid-semester (G1 and absences available, no G2)
    # h=2: Late-semester (all features available)
    if horizon == 0:
        drop_h0 = ['G1', 'G2', 'absences']
        df = df.drop(columns=[c for c in drop_h0 if c in df.columns])
    elif horizon == 1:
        drop_h1 = ['G2']
        df = df.drop(columns=[c for c in drop_h1 if c in df.columns])
    elif horizon == 2:
        # All features available at h=2
        pass
    else:
        raise ValueError(f"Invalid horizon: {horizon}. Must be 0, 1, or 2")

    # Categorical encoding
    # Binary categoricals: use LabelEncoder
    binary_cats = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                   'schoolsup', 'famsup', 'paid', 'activities',
                   'nursery', 'higher', 'internet', 'romantic']
    binary_present = [c for c in binary_cats if c in df.columns and df[c].nunique() <= 2]

    le = LabelEncoder()
    for c in binary_present:
        df[c] = le.fit_transform(df[c].astype(str))

    # Multi-class categoricals: one-hot encoding
    multi_cats = ['Mjob', 'Fjob', 'reason', 'guardian']
    multi_present = [c for c in multi_cats if c in df.columns]
    if multi_present:
        df = pd.get_dummies(df, columns=multi_present, drop_first=False)

    # Coerce all to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)

    # Drop zero-variance columns (numerical stability)
    variances = df.var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        if verbose:
            print(f"  [preprocess] Removed {len(zero_var)} zero-variance columns: {zero_var}")
        df = df.drop(columns=zero_var)

    feature_names = df.columns.tolist()
    X = df.values.astype(np.float64)

    if verbose:
        print(f"  [preprocess] Final: {X.shape[0]} rows × {X.shape[1]} features (horizon={horizon})")
        print(f"  [preprocess] Pass rate: {y.mean():.3f}")

    return X, y, feature_names


def load_uci_dataset(dataset: str = "math", data_dir: str = "data/uci") -> pd.DataFrame:
    """
    Load UCI Student Performance dataset (Math or Portuguese).

    Args:
        dataset: "math" or "portuguese" (or "mat", "por")
        data_dir: Path to data directory

    Returns:
        DataFrame with raw UCI data
    """
    dataset = dataset.lower()
    if dataset in ["math", "mat", "student-mat"]:
        csv_path = Path(data_dir) / "student-mat.csv"
    elif dataset in ["portuguese", "por", "student-por"]:
        csv_path = Path(data_dir) / "student-por.csv"
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'math' or 'portuguese'")

    if not csv_path.exists():
        # Fallback: relative to project root
        csv_path = project_root / data_dir / csv_path.name
    if not csv_path.exists():
        raise FileNotFoundError(f"UCI dataset not found at {csv_path}")

    return pd.read_csv(csv_path, sep=';')


def split_uci(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split — same protocol across all UCI experiments."""
    return train_test_split(X, y, test_size=test_size,
                              random_state=random_state, stratify=y)


def load_split_uci(dataset: str = "math", horizon: int = 0,
                    random_state: int = 42, verbose: bool = False):
    """One-shot load + preprocess + split."""
    df = load_uci_dataset(dataset)
    X, y, names = preprocess_uci(df, horizon=horizon, verbose=verbose)
    X_tr, X_te, y_tr, y_te = split_uci(X, y, random_state=random_state)
    return X_tr, X_te, y_tr, y_te, names


if __name__ == "__main__":
    # Smoke test
    print("=== preprocess_uci smoke test ===")
    for dataset in ["math", "portuguese"]:
        print(f"\nDataset: {dataset}")
        for h in [0, 1, 2]:
            try:
                X_tr, X_te, y_tr, y_te, names = load_split_uci(
                    dataset=dataset, horizon=h, verbose=True)
                print(f"  h={h}: train={len(y_tr)} test={len(y_te)} "
                       f"features={len(names)} pass_rate={y_tr.mean():.3f}")
                print(f"        sample features: {names[:5]}")
            except Exception as e:
                print(f"  h={h}: ERROR — {e}")

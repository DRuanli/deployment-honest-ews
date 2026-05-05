"""
Shared OULAD preprocessing utilities.

Used by:
  - run_oulad_experiments.py (main IC-FS sweep)
  - run_oulad_dre.py (deployment-realistic eval)
  - run_oulad_baselines.py (NSGA-II, Stability Selection, Boruta)
  - run_oulad_statistics.py (multi-seed Wilcoxon)

Output contract: preprocess_oulad(df) -> (X, y, names)
where names align with TAXONOMY_OULAD via _resolve_parent prefix matching.
"""
#./experiments/oulad/

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Path setup so this module is importable from anywhere
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def preprocess_oulad(df: pd.DataFrame, verbose: bool = False):
    """
    Preprocess OULAD parquet features for IC-FS / baselines.

    Args:
        df: DataFrame from oulad_features_h{horizon}.parquet
        verbose: print shape info

    Returns:
        X: float64 ndarray (n, p)
        y: int ndarray (n,)
        feature_names: list of column names after one-hot encoding
    """
    df = df.copy()

    # Target
    if 'y' not in df.columns:
        raise ValueError("Expected column 'y' in OULAD parquet")
    y = df.pop('y').values.astype(int)

    # Drop metadata columns
    drop_cols = ['id_student', 'student_key', 'horizon_cutoff',
                 'module_presentation_length', 'final_result']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # One-hot categorical columns
    cat_cols = ['gender', 'region', 'highest_education', 'imd_band',
                'age_band', 'disability', 'code_module', 'code_presentation']
    cat_present = [c for c in cat_cols if c in df.columns]
    if cat_present:
        df = pd.get_dummies(df, columns=cat_present)

    # Coerce all to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)

    # Drop zero-variance columns (cause numerical instability in chi2/MI)
    variances = df.var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        if verbose:
            print(f"  [preprocess] Removed {len(zero_var)} zero-variance columns")
        df = df.drop(columns=zero_var)

    feature_names = df.columns.tolist()
    X = df.values.astype(np.float64)

    if verbose:
        print(f"  [preprocess] Final: {X.shape[0]} rows × {X.shape[1]} features")
        print(f"  [preprocess] Pass rate: {y.mean():.3f}")

    return X, y, feature_names


def load_oulad_horizon(horizon: int, parquet_dir: str = "results/oulad"):
    """Load preprocessed parquet for a horizon."""
    pq_path = Path(parquet_dir) / f"oulad_features_h{horizon}.parquet"
    if not pq_path.exists():
        # Fallback: relative to project root
        pq_path = project_root / parquet_dir / f"oulad_features_h{horizon}.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(
            f"Parquet not found at {pq_path}. "
            f"Run `python src/icfs/oulad_pipeline.py` first.")
    return pd.read_parquet(pq_path)


def split_oulad(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split — same protocol across all OULAD experiments."""
    return train_test_split(X, y, test_size=test_size,
                              random_state=random_state, stratify=y)


def load_split_oulad(horizon: int, random_state: int = 42, verbose: bool = False):
    """One-shot load + preprocess + split."""
    df = load_oulad_horizon(horizon)
    X, y, names = preprocess_oulad(df, verbose=verbose)
    X_tr, X_te, y_tr, y_te = split_oulad(X, y, random_state=random_state)
    return X_tr, X_te, y_tr, y_te, names


if __name__ == "__main__":
    # Smoke test
    print("=== preprocess_oulad smoke test ===")
    for h in [0, 1, 2]:
        try:
            X_tr, X_te, y_tr, y_te, names = load_split_oulad(h, verbose=True)
            print(f"  h={h}: train={len(y_tr)} test={len(y_te)} "
                   f"features={len(names)} pass_rate={y_tr.mean():.3f}")
        except FileNotFoundError as e:
            print(f"  h={h}: SKIP — {e}")
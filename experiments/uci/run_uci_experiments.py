#./experiments/uci/
"""
================================================================================
IC-FS on UCI Student Performance — Main α-sweep experiment (multi-seed)
================================================================================
Runs IC-FS at all horizons (t=0, t=1, t=2) with multiple random seeds.
This is the primary "headline numbers" experiment for UCI datasets.

Outputs:
  - results/uci/{dataset}/uci_{dataset}_icfs_h{0,1,2}.csv         (single-seed=42, full sweep)
  - results/uci/{dataset}/uci_{dataset}_icfs_multi_h{0,1,2}.csv   (multi-seed best-IUS rows)

Runtime: ~2-5 min per horizon (single seed) or ~15-40 min (8 seeds × 3 h)

Usage:
    # Single seed, all horizons, Math dataset (default)
    python experiments/uci/run_uci_experiments.py

    # Portuguese dataset
    python experiments/uci/run_uci_experiments.py --dataset portuguese

    # Multi-seed mode for ESWA submission
    python experiments/uci/run_uci_experiments.py --multi-seed --dataset math

    # Single horizon
    python experiments/uci/run_uci_experiments.py --horizon 1 --dataset math
================================================================================
"""

from __future__ import annotations
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.icfs.ic_fs import ICFSPipeline
from src.icfs.taxonomy_uci import TAXONOMY_UCI
from preprocess_uci import load_uci_dataset, preprocess_uci, split_uci

DEFAULT_SEEDS = [42, 123, 456, 789, 1011, 2024, 3033, 4044]
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
TOP_K = 7
N_BOOT = 20


def run_one_horizon_one_seed(df_raw, dataset_name: str, horizon: int, seed: int):
    """Single (dataset, horizon, seed) IC-FS sweep. Returns full results DataFrame
    plus selected-features dict for downstream DRE."""
    X, y, names = preprocess_uci(df_raw, horizon=horizon)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)

    pipe = ICFSPipeline(
        horizon=horizon,
        top_k=TOP_K,
        n_bootstrap=N_BOOT,
        taxonomy=TAXONOMY_UCI,
        alpha_values=ALPHA_GRID,
        random_state=seed,
    )
    pipe.fit(X_tr, y_tr, X_te, y_te, names, verbose=False)
    return pipe.to_dataframe(), pipe.best_by_ius()


def run_single_seed_full_sweep(dataset: str, horizons, out_dir, seed=42):
    """Original behaviour: single seed=42, full α-sweep written per horizon."""
    print("=" * 80)
    print(f"IC-FS on UCI {dataset.upper()} | Single-seed mode (seed={seed})")
    print("=" * 80)

    df_raw = load_uci_dataset(dataset)
    print(f"[Load] {len(df_raw)} students from {dataset} dataset")

    for h in horizons:
        print(f"\n{'─'*80}")
        print(f"HORIZON t={h}")
        print(f"{'─'*80}")

        t0 = time.time()
        results_df, best_sol = run_one_horizon_one_seed(df_raw, dataset, h, seed)
        print(f"[Done] {time.time()-t0:.0f}s")

        out_path = out_dir / f"uci_{dataset}_icfs_h{h}.csv"
        results_df.to_csv(out_path, index=False)
        print(f"  Saved {out_path}")
        print(f"  Best α={best_sol.alpha:.2f}  F1_deploy={best_sol.f1_deploy*100:.2f}  "
               f"AR_avail={best_sol.ar_available:.3f}  IUS_deploy={best_sol.ius_deploy*100:.2f}  "
               f"Stab={best_sol.stability:.3f}")
        print(f"  Top-5 features: {best_sol.selected_features[:5]}")


def run_multi_seed(dataset: str, horizons, seeds, out_dir):
    """Multi-seed mode: for each (dataset, horizon, seed) record best-IUS row."""
    print("=" * 80)
    print(f"IC-FS on UCI {dataset.upper()} | Multi-seed mode (n={len(seeds)})")
    print("=" * 80)

    df_raw = load_uci_dataset(dataset)
    print(f"[Load] {len(df_raw)} students from {dataset} dataset")

    for h in horizons:
        print(f"\n{'─'*80}")
        print(f"HORIZON t={h}")
        print(f"{'─'*80}")

        rows = []
        for s in seeds:
            t_s = time.time()
            try:
                results_df, best_sol = run_one_horizon_one_seed(df_raw, dataset, h, s)
                rows.append({
                    "seed": s, "dataset": dataset, "horizon": h,
                    "alpha_best": best_sol.alpha,
                    "accuracy": best_sol.accuracy * 100,
                    "f1_paper": best_sol.f1 * 100,
                    "f1_deploy": best_sol.f1_deploy * 100,
                    "AR": best_sol.ar,
                    "AR_available": best_sol.ar_available,
                    "TVS": best_sol.tvs,
                    "IUS_deploy": best_sol.ius_deploy * 100,
                    "IUS_geo": best_sol.ius_geo * 100,
                    "n_features": best_sol.n_features,
                    "stability": best_sol.stability,
                    "prec_at_topk": best_sol.precision_at_topk,
                    "recall_at_topk": best_sol.recall_at_topk,
                    "cv_mean": best_sol.cv_mean * 100,
                    "cv_std": best_sol.cv_std * 100,
                    "selected": "|".join(best_sol.selected_features),
                })
                print(f"  seed={s:4d} ({time.time()-t_s:5.0f}s): "
                       f"α*={best_sol.alpha:.2f} F1_deploy={best_sol.f1_deploy*100:5.2f} "
                       f"AR_avail={best_sol.ar_available:.3f} "
                       f"IUS_deploy={best_sol.ius_deploy*100:5.2f}")
            except Exception as e:
                print(f"  seed={s} FAILED: {e}")

        df = pd.DataFrame(rows)
        out_path = out_dir / f"uci_{dataset}_icfs_multi_h{h}.csv"
        df.to_csv(out_path, index=False)

        if len(df) >= 2:
            print(f"\n  Multi-seed summary (n={len(df)}):")
            for col in ["f1_deploy", "AR_available", "IUS_deploy", "stability"]:
                v = df[col].values
                print(f"    {col:<15}: mean={v.mean():6.2f}  std={v.std(ddof=1):5.2f}  "
                       f"95%CI=[{np.percentile(v,2.5):.2f}, {np.percentile(v,97.5):.2f}]")
        print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="IC-FS on UCI Student Performance")
    parser.add_argument("--dataset", type=str, default="math",
                          help="Dataset to use: 'math' or 'portuguese' (default: math)")
    parser.add_argument("--multi-seed", action="store_true",
                          help="Run with all 8 seeds (for ESWA submission)")
    parser.add_argument("--horizon", type=int, default=None,
                          help="Run only one horizon (0, 1, or 2)")
    parser.add_argument("--seed", type=int, default=42,
                          help="Single-seed mode RNG seed (default 42)")
    args = parser.parse_args()

    # Normalize dataset name
    dataset = args.dataset.lower()
    if dataset in ["math", "mat", "student-mat"]:
        dataset = "math"
    elif dataset in ["portuguese", "por", "student-por"]:
        dataset = "portuguese"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Use 'math' or 'portuguese'")

    out_dir = project_root / "results" / "uci" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = [args.horizon] if args.horizon is not None else [0, 1, 2]

    if args.multi_seed:
        run_multi_seed(dataset, horizons, DEFAULT_SEEDS, out_dir)
    else:
        run_single_seed_full_sweep(dataset, horizons, out_dir, seed=args.seed)

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()

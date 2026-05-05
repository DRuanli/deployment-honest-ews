#!/usr/bin/env python3
"""
================================================================================
Deployment-Realistic Evaluation (DRE) on UCI Student Performance — Multi-seed
================================================================================
Validates that UCI preprocessing correctly enforces temporal validity (δ).

EXPECTED OUTCOME: τ ≈ 0 for all methods at all horizons.
Unlike OULAD (where τ_notemp > 0 reveals leakage), UCI preprocessing filters
features at data-loading stage (preprocess_uci drops G1/G2/absences by horizon),
so both IC-FS(full) and IC-FS(--temporal) should see identical feature sets.

This experiment serves as a SANITY CHECK that:
  1. UCI preprocessing is correct (no temporal leakage in the data pipeline)
  2. τ = 1 - (AR_available / AR_all) ≈ 0 when preprocessing enforces δ
  3. IUS_paper ≈ IUS_deploy (no gap when features are correctly filtered)

Protocol (identical to OULAD, adapted for UCI):
  For each seed, horizon ∈ {0, 1, 2}, dataset ∈ {math, portuguese}:
    1. Train IC-FS(full) with temporal filter
    2. Train IC-FS(--temporal) without filter
    3. Mask temporally-unavailable features in test set only
    4. Retrain on full train, evaluate on masked test
    5. Compare paper-style F1 vs deployment F1
    6. Compute τ (expected ≈ 0 for UCI)

Output:
  results/uci/{dataset}/k{k}/dre_multi_uci_{dataset}_h{0,1,2}_k{k}.csv
  with 8 seeds each

Usage:
    # Single horizon, single dataset
    python experiments/uci/run_uci_dre.py --dataset math --horizon 0 --k 5

    # All horizons for one dataset at one budget
    python experiments/uci/run_uci_dre.py --dataset math --k 5

    # All horizons, all budgets for one dataset
    python experiments/uci/run_uci_dre.py --dataset math --all-k

Total runtime: ~30-60 min per dataset per k (smaller than OULAD due to N~400-650)
================================================================================
"""

from __future__ import annotations
import sys
import time
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Path setup
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.icfs.ic_fs import (
    feature_scores_for_selection, ic_fs_select,
    actionability_ratio, actionability_ratio_available,
    compute_ius_deploy, compute_ius_paper,
    filter_by_horizon,
    apply_dre_mask,
)
from src.icfs.taxonomy_uci import TAXONOMY_UCI
from preprocess_uci import load_uci_dataset, preprocess_uci

# Experiment parameters
RNG_SEEDS = [42, 123, 456, 789, 1011, 2024, 3033, 4044]
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
N_TREES = 100  # Match ICFSPipeline n_estimators=100


def fit_predict_f1(X_tr, y_tr, X_te, y_te, sel_idx, random_state):
    """Train RF on selected features, return F1."""
    rf = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_tr[:, sel_idx], y_tr)
    y_pred = rf.predict(X_te[:, sel_idx])
    return f1_score(y_te, y_pred, average='weighted', zero_division=0)


def precision_recall_at_top_k(y_true, y_proba, top_k_pct=0.20):
    """
    Compute precision and recall at top k% of predictions.

    For UCI: y=1 is Pass, y=0 is Fail → rank by lowest P(Pass) = highest at-risk.
    """
    n = len(y_true)
    k = int(n * top_k_pct)
    if k == 0:
        return 0.0, 0.0

    # Sort ascending by P(Pass) → lowest P(Pass) = highest at-risk
    top_k_idx = np.argsort(y_proba)[:k]

    # Count Fail (y=0) in top-k
    tp = (y_true[top_k_idx] == 0).sum()
    precision = tp / k

    # Recall: of all at-risk students (y=0), what fraction captured?
    total_at_risk = (y_true == 0).sum()
    recall = tp / total_at_risk if total_at_risk > 0 else 0.0

    return precision, recall


def select_best_ius(X_tr, y_tr, X_te, y_te, names, horizon, top_k,
                     random_state, apply_temporal_filter: bool):
    """
    Run IC-FS α-sweep with NESTED VALIDATION on training data only.

    apply_temporal_filter:
      True  → IC-FS(full): filter unavailable features before scoring
      False → IC-FS(--temporal): use all features (tests if preprocessing leaked)

    Returns:
      (selected_features, best_alpha, final_f1)
    """
    if apply_temporal_filter:
        # Filter by horizon — use only features marked as available in taxonomy
        available = filter_by_horizon(names, horizon, TAXONOMY_UCI)
        if not available:
            raise RuntimeError(f"No features available at horizon={horizon}")
        idx = [names.index(f) for f in available]
        X_tr_use = X_tr[:, idx]
        X_te_use = X_te[:, idx]
        feat_use = available
    else:
        # No filter — use all features from preprocessing
        # (tests if preprocess_uci correctly dropped G1/G2/absences)
        X_tr_use = X_tr
        X_te_use = X_te
        feat_use = names

    # NESTED VALIDATION: split training data for alpha selection
    val_seed = random_state + 1000
    X_tr_inner, X_val_inner, y_tr_inner, y_val_inner = train_test_split(
        X_tr_use, y_tr, test_size=0.2, random_state=val_seed, stratify=y_tr
    )

    # Compute feature scores on INNER TRAINING SET only
    score_df = feature_scores_for_selection(
        X_tr_inner, y_tr_inner, feat_use, TAXONOMY_UCI
    )

    # Alpha sweep on validation set (NOT test set)
    best_ius_val = -np.inf
    best_alpha = None
    for alpha in ALPHA_GRID:
        sel = ic_fs_select(score_df, alpha, min(top_k, len(feat_use)))
        sel_local = [feat_use.index(f) for f in sel]
        f1_val = fit_predict_f1(
            X_tr_inner, y_tr_inner, X_val_inner, y_val_inner,
            sel_local, val_seed
        )
        ius_val = compute_ius_deploy(f1_val, sel, horizon, TAXONOMY_UCI)
        if ius_val > best_ius_val:
            best_ius_val, best_alpha = ius_val, alpha

    # RETRAIN on full training set with selected alpha*
    score_df_full = feature_scores_for_selection(
        X_tr_use, y_tr, feat_use, TAXONOMY_UCI
    )
    final_sel = ic_fs_select(score_df_full, best_alpha, min(top_k, len(feat_use)))

    # Evaluate F1 on test set (for reporting only, NOT for alpha selection)
    final_sel_local = [feat_use.index(f) for f in final_sel]
    final_f1 = fit_predict_f1(
        X_tr_use, y_tr, X_te_use, y_te, final_sel_local, random_state
    )

    return final_sel, best_alpha, final_f1


def evaluate_under_dre(X_tr, y_tr, X_te, y_te, selected_features,
                        horizon, names, random_state):
    """
    DEPLOYMENT-REALISTIC EVALUATION (asymmetric masking).

    1. Train on COMPLETE training data (all selected features available)
    2. At inference, mask unavailable features with train-column means
    3. This matches real deployment: model trained on full history,
       predicts on partial data

    Returns:
        f1: weighted F1 score
        y_proba: predicted probabilities for Pass class (for Precision@k)
    """
    sel_idx = [names.index(f) for f in selected_features]
    X_tr_s = X_tr[:, sel_idx].astype(np.float64).copy()
    X_te_s = X_te[:, sel_idx].astype(np.float64).copy()

    # Apply asymmetric DRE mask: train unmasked, test masked
    _, X_te_deploy = apply_dre_mask(
        X_tr_s, X_te_s, selected_features, horizon, TAXONOMY_UCI
    )

    rf = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_tr_s, y_tr)  # Train on UNMASKED data

    # Predict on deployment-realistic (masked) test data
    y_pred = rf.predict(X_te_deploy)
    y_proba = (rf.predict_proba(X_te_deploy)[:, 1]
               if len(rf.classes_) > 1 else y_pred.astype(float))
    f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)

    return f1, y_proba


def run_one_seed(df_raw, dataset, seed, horizon, top_k):
    """Run paper-style + DRE eval for one seed at one horizon."""
    X, y, names = preprocess_uci(df_raw, horizon=horizon, verbose=False)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # IC-FS(full) — temporal filter on
    sel_full, alpha_full, f1_full_paper = select_best_ius(
        X_tr, y_tr, X_te, y_te, names, horizon, top_k, seed,
        apply_temporal_filter=True
    )

    # IC-FS(--temporal) — no filter (sanity check)
    sel_notemp, alpha_notemp, f1_notemp_paper = select_best_ius(
        X_tr, y_tr, X_te, y_te, names, horizon, top_k, seed,
        apply_temporal_filter=False
    )

    # DRE: mask + retrain + evaluate
    f1_full_deploy, y_proba_full = evaluate_under_dre(
        X_tr, y_tr, X_te, y_te, sel_full, horizon, names, seed
    )
    f1_notemp_deploy, y_proba_notemp = evaluate_under_dre(
        X_tr, y_tr, X_te, y_te, sel_notemp, horizon, names, seed
    )

    # Intervention metrics (Precision@20%, Recall@20%)
    prec20_full, rec20_full = precision_recall_at_top_k(y_te, y_proba_full, 0.20)
    prec20_notemp, rec20_notemp = precision_recall_at_top_k(y_te, y_proba_notemp, 0.20)

    # Actionability ratios
    ar_full = actionability_ratio(sel_full, TAXONOMY_UCI)
    ar_notemp = actionability_ratio(sel_notemp, TAXONOMY_UCI)
    ar_available_full = actionability_ratio_available(sel_full, horizon, TAXONOMY_UCI)
    ar_available_notemp = actionability_ratio_available(sel_notemp, horizon, TAXONOMY_UCI)

    # IUS metrics
    ius_full_paper = compute_ius_paper(f1_full_paper, sel_full, horizon, TAXONOMY_UCI)
    ius_notemp_paper = compute_ius_paper(f1_notemp_paper, sel_notemp, horizon, TAXONOMY_UCI)
    ius_full_deploy = compute_ius_deploy(f1_full_deploy, sel_full, horizon, TAXONOMY_UCI)
    ius_notemp_deploy = compute_ius_deploy(f1_notemp_deploy, sel_notemp, horizon, TAXONOMY_UCI)

    # Leakage diagnostics (expected τ ≈ 0 for UCI)
    tau_full = 1.0 - (ar_available_full / ar_full) if ar_full > 0 else 0.0
    tau_notemp = 1.0 - (ar_available_notemp / ar_notemp) if ar_notemp > 0 else 0.0

    # Temporal leakage flags (G1, G2, absences)
    temporal_features = ['G1', 'G2', 'absences']
    full_has_temporal = any(f in sel_full for f in temporal_features)
    notemp_has_temporal = any(f in sel_notemp for f in temporal_features)

    return {
        "seed": seed, "horizon": horizon, "dataset": dataset,
        "alpha_full": alpha_full, "alpha_notemp": alpha_notemp,
        # F1 values
        "f1_full_paper": f1_full_paper * 100,
        "f1_notemp_paper": f1_notemp_paper * 100,
        "f1_full_deploy": f1_full_deploy * 100,
        "f1_notemp_deploy": f1_notemp_deploy * 100,
        # Intervention metrics
        "precision20_full_deploy": prec20_full * 100,
        "recall20_full_deploy": rec20_full * 100,
        "precision20_notemp_deploy": prec20_notemp * 100,
        "recall20_notemp_deploy": rec20_notemp * 100,
        # AR metrics
        "AR_full": ar_full,
        "AR_notemp": ar_notemp,
        "AR_available_full": ar_available_full,
        "AR_available_notemp": ar_available_notemp,
        # IUS metrics
        "IUS_paper_full": ius_full_paper * 100,
        "IUS_paper_notemp": ius_notemp_paper * 100,
        "IUS_deploy_full": ius_full_deploy * 100,
        "IUS_deploy_notemp": ius_notemp_deploy * 100,
        # Leakage diagnostics
        "tau_full": tau_full,
        "tau_notemp": tau_notemp,
        "full_has_temporal": full_has_temporal,
        "notemp_has_temporal": notemp_has_temporal,
        "n_full": len(sel_full),
        "n_notemp": len(sel_notemp),
        "selected_full": "|".join(sel_full),
        "selected_notemp": "|".join(sel_notemp),
    }


def main():
    parser = argparse.ArgumentParser(
        description="UCI DRE Multi-seed Evaluation"
    )
    parser.add_argument("--dataset", type=str, default="math",
                        choices=["math", "portuguese"],
                        help="Dataset: 'math' or 'portuguese'")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Prediction horizon (0, 1, or 2). "
                             "If unspecified, runs all horizons.")
    parser.add_argument("--k", type=int, default=5,
                        help="Feature budget (default: 5)")
    parser.add_argument("--all-k", action="store_true",
                        help="Run all budgets k ∈ {5, 7, 10, 15}")
    args = parser.parse_args()

    dataset = args.dataset
    k_list = [5, 7, 10, 15] if args.all_k else [args.k]
    h_list = [0, 1, 2] if args.horizon is None else [args.horizon]

    out_dir = project_root / "results" / "uci" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"UCI DRE Multi-seed | dataset={dataset} | "
          f"horizons={h_list} | budgets={k_list} | n_seeds={len(RNG_SEEDS)}")
    print("=" * 80)

    # Load raw dataset once
    print(f"\n[1/3] Loading UCI {dataset} dataset...")
    df_raw = load_uci_dataset(dataset)
    print(f"  Loaded {len(df_raw)} students × {df_raw.shape[1]} raw columns")

    for k in k_list:
        print(f"\n{'='*80}")
        print(f"Feature budget k = {k}")
        print(f"{'='*80}")

        for h in h_list:
            print(f"\n[2/3] Running {len(RNG_SEEDS)} seeds for h={h}, k={k}...")

            out_csv = out_dir / f"k{k}" / f"dre_multi_uci_{dataset}_h{h}_k{k}.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)

            rows = []
            t0 = time.time()

            for s in RNG_SEEDS:
                t_seed = time.time()
                try:
                    r = run_one_seed(df_raw, dataset, s, h, k)
                    rows.append(r)
                    print(f"  seed={s:4d} ({time.time()-t_seed:5.0f}s) | "
                          f"full_dep={r['f1_full_deploy']:5.1f} "
                          f"notemp_dep={r['f1_notemp_deploy']:5.1f} "
                          f"tau_full={r['tau_full']:.3f} "
                          f"tau_notemp={r['tau_notemp']:.3f}")
                except Exception as e:
                    print(f"  seed={s} FAILED: {e}")

            df = pd.DataFrame(rows)
            df.to_csv(out_csv, index=False)
            print(f"\nSaved {out_csv}  (total {time.time()-t0:.0f}s)")

            if len(rows) < 2:
                print("\nNot enough seeds for stats")
                continue

            print(f"\n[3/3] Statistics ({len(rows)} seeds)")
            print("\n--- Bootstrap 95% CI ---")
            for col in ["f1_full_deploy", "f1_notemp_deploy",
                        "IUS_deploy_full", "IUS_deploy_notemp",
                        "tau_full", "tau_notemp"]:
                if col in df.columns:
                    v = df[col].values
                    print(f"  {col:<22}: mean={v.mean():6.2f}  "
                          f"std={v.std(ddof=1):5.2f}  "
                          f"95% CI=[{np.percentile(v, 2.5):5.2f}, "
                          f"{np.percentile(v, 97.5):5.2f}]")

            # Wilcoxon test
            a_ius = df["IUS_deploy_full"].values
            b_ius = df["IUS_deploy_notemp"].values
            if not np.allclose(a_ius, b_ius):
                try:
                    stat, p = wilcoxon(a_ius, b_ius, alternative="greater",
                                       zero_method="wilcox")
                    diff = a_ius - b_ius
                    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0
                    sig = ("***" if p < 0.001 else "**" if p < 0.01
                           else "*" if p < 0.0167 else "ns")
                    print(f"\n--- Wilcoxon: IC-FS(full) > IC-FS(--temporal) ---")
                    print(f"  IUS_deploy: W={stat:.0f}  p={p:.5f}  d={d:+.2f}  "
                          f"diff={diff.mean():+.2f}  [{sig}]")
                except ValueError as e:
                    print(f"\n  Wilcoxon failed ({e})")
            else:
                print("\n  IUS_deploy: identical — selections converge")

            # Leakage check
            print("\n--- Temporal leakage check (expected: none for UCI) ---")
            tau_full_mean = df["tau_full"].mean()
            tau_notemp_mean = df["tau_notemp"].mean()
            print(f"  Mean τ_full:   {tau_full_mean:.4f}  (expected ≈0)")
            print(f"  Mean τ_notemp: {tau_notemp_mean:.4f}  (expected ≈0)")
            if tau_full_mean > 0.05 or tau_notemp_mean > 0.05:
                print(f"  ⚠️  WARNING: τ > 0.05 detected! "
                      f"Check if preprocess_uci correctly drops G1/G2/absences.")
            else:
                print(f"  ✓ PASS: Temporal validity enforced correctly")


if __name__ == "__main__":
    main()

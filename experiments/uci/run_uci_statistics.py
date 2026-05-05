#./experiments/uci/
"""
================================================================================
Multi-seed Statistics on UCI — IC-FS variants Wilcoxon
================================================================================
Compares IC-FS(full) vs 3 ablation variants over 8 seeds at one horizon.

Variants:
  1. IC-FS (full): temporal filter + actionability weights + nested α
  2. IC-FS (--temporal): no temporal filter (like DE-FS)
  3. IC-FS (--actionability): all weights = 1.0 (pure prediction)
  4. HardFilter + DE-FS: drop Tier 0+3, then ensemble select

Output: results/uci/{dataset}/stat8_uci_{dataset}_h{horizon}.csv
Runtime: ~10-20 min per (dataset, horizon)

Usage:
    python experiments/uci/run_uci_statistics.py --dataset math --horizon 0
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
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.icfs.ic_fs import (
    Tier, _resolve_parent,
    feature_scores_for_selection, ic_fs_select,
    actionability_ratio, actionability_ratio_available,
    temporal_validity_score,
    compute_ius_paper, compute_ius_deploy,
    filter_by_horizon, apply_dre_mask,
)
from src.icfs.taxonomy_uci import TAXONOMY_UCI
from preprocess_uci import load_uci_dataset, preprocess_uci, split_uci

RNG_SEEDS = [42, 123, 456, 789, 1011, 2024, 3033, 4044]
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
TOP_K = 7
N_TREES = 100


def _eval(X_tr, y_tr, X_te, y_te, sel_idx, seed, cv_folds=3):
    """Train RF, return F1 + accuracy + CV F1."""
    rf = RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                  n_jobs=-1, class_weight='balanced')
    rf.fit(X_tr[:, sel_idx], y_tr)
    y_pred = rf.predict(X_te[:, sel_idx])
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv = cross_val_score(clone(rf), X_tr[:, sel_idx], y_tr, cv=skf,
                          scoring='f1_weighted', n_jobs=-1)
    return {"accuracy": acc, "f1": f1, "cv_mean": cv.mean(), "cv_std": cv.std()}


def _eval_dre(X_tr, y_tr, X_te, y_te, sel_names, sel_idx, horizon, seed, cv_folds=3):
    """DRE-honest evaluation: train on unmasked X_tr, predict on DRE-masked X_te."""
    X_tr_sel = X_tr[:, sel_idx]
    X_te_sel = X_te[:, sel_idx]
    _, X_te_deploy = apply_dre_mask(X_tr_sel, X_te_sel, sel_names, horizon, TAXONOMY_UCI)

    rf = RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                  n_jobs=-1, class_weight='balanced')
    rf.fit(X_tr_sel, y_tr)
    y_pred = rf.predict(X_te_deploy)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv = cross_val_score(clone(rf), X_tr_sel, y_tr, cv=skf,
                          scoring='f1_weighted', n_jobs=-1)
    return {"accuracy": acc, "f1": f1, "cv_mean": cv.mean(), "cv_std": cv.std()}


def run_full(X_tr, y_tr, X_te, y_te, names, horizon, seed):
    """IC-FS (full): temporal filter + α-sweep with nested validation."""
    available = filter_by_horizon(names, horizon, TAXONOMY_UCI)
    idx = [names.index(f) for f in available]
    X_tr_a, X_te_a = X_tr[:, idx], X_te[:, idx]

    # Nested validation for alpha selection
    val_seed = seed + 1000
    X_tr_inner, X_val_inner, y_tr_inner, y_val_inner = train_test_split(
        X_tr_a, y_tr, test_size=0.2, random_state=val_seed, stratify=y_tr
    )

    sdf = feature_scores_for_selection(X_tr_inner, y_tr_inner, available, TAXONOMY_UCI)

    best_alpha = None
    best_ius_val = -np.inf
    for alpha in ALPHA_GRID:
        sel = ic_fs_select(sdf, alpha, min(TOP_K, len(available)))
        sel_loc = [available.index(f) for f in sel]
        ev_val = _eval(X_tr_inner, y_tr_inner, X_val_inner, y_val_inner, sel_loc, val_seed)
        ius_val = compute_ius_deploy(ev_val["f1"], sel, horizon, TAXONOMY_UCI)
        if ius_val > best_ius_val:
            best_ius_val = ius_val
            best_alpha = alpha

    # Retrain with best alpha
    sdf_full = feature_scores_for_selection(X_tr_a, y_tr, available, TAXONOMY_UCI)
    sel_final = ic_fs_select(sdf_full, best_alpha, min(TOP_K, len(available)))
    sel_loc_final = [available.index(f) for f in sel_final]

    ev = _eval(X_tr_a, y_tr, X_te_a, y_te, sel_loc_final, seed)
    ar = actionability_ratio(sel_final, TAXONOMY_UCI)
    ar_avail = actionability_ratio_available(sel_final, horizon, TAXONOMY_UCI)
    tvs = temporal_validity_score(sel_final, horizon, TAXONOMY_UCI)
    ius_paper = compute_ius_paper(ev["f1"], sel_final, horizon, TAXONOMY_UCI)
    ius_deploy = compute_ius_deploy(ev["f1"], sel_final, horizon, TAXONOMY_UCI)

    return {"f1": ev["f1"] * 100, "accuracy": ev["accuracy"] * 100,
            "AR": ar, "AR_available": ar_avail, "TVS": tvs,
            "IUS_paper": ius_paper * 100, "IUS_deploy": ius_deploy * 100,
            "alpha": best_alpha, "n": len(sel_final), "sel": sel_final}


def run_no_temporal(X_tr, y_tr, X_te, y_te, names, horizon, seed):
    """IC-FS(--temporal): no temporal filter (like DE-FS)."""
    val_seed = seed + 1000
    X_tr_inner, X_val_inner, y_tr_inner, y_val_inner = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=val_seed, stratify=y_tr
    )

    sdf = feature_scores_for_selection(X_tr_inner, y_tr_inner, names, TAXONOMY_UCI)

    best_alpha = None
    best_ius_val = -np.inf
    for alpha in ALPHA_GRID:
        sel = ic_fs_select(sdf, alpha, TOP_K)
        sel_idx = [names.index(f) for f in sel]
        ev_val = _eval(X_tr_inner, y_tr_inner, X_val_inner, y_val_inner, sel_idx, val_seed)
        ius_val = compute_ius_deploy(ev_val["f1"], sel, horizon, TAXONOMY_UCI)
        if ius_val > best_ius_val:
            best_ius_val = ius_val
            best_alpha = alpha

    sdf_full = feature_scores_for_selection(X_tr, y_tr, names, TAXONOMY_UCI)
    sel_final = ic_fs_select(sdf_full, best_alpha, TOP_K)
    sel_idx_final = [names.index(f) for f in sel_final]

    # Use DRE evaluation (may have unavailable features)
    ev = _eval_dre(X_tr, y_tr, X_te, y_te, sel_final, sel_idx_final, horizon, seed)
    ar = actionability_ratio(sel_final, TAXONOMY_UCI)
    ar_avail = actionability_ratio_available(sel_final, horizon, TAXONOMY_UCI)
    tvs = temporal_validity_score(sel_final, horizon, TAXONOMY_UCI)
    ius_paper = compute_ius_paper(ev["f1"], sel_final, horizon, TAXONOMY_UCI)
    ius_deploy = compute_ius_deploy(ev["f1"], sel_final, horizon, TAXONOMY_UCI)

    return {"f1": ev["f1"] * 100, "accuracy": ev["accuracy"] * 100,
            "AR": ar, "AR_available": ar_avail, "TVS": tvs,
            "IUS_paper": ius_paper * 100, "IUS_deploy": ius_deploy * 100,
            "alpha": best_alpha, "n": len(sel_final), "sel": sel_final}


def run_no_action(X_tr, y_tr, X_te, y_te, names, horizon, seed):
    """IC-FS(--actionability): all weights = 1.0 → pure prediction."""
    available = filter_by_horizon(names, horizon, TAXONOMY_UCI)
    idx = [names.index(f) for f in available]
    X_tr_a, X_te_a = X_tr[:, idx], X_te[:, idx]
    sdf = feature_scores_for_selection(X_tr_a, y_tr, available, TAXONOMY_UCI).copy()
    sdf["actionability"] = 1.0

    sel = ic_fs_select(sdf, 1.0, min(TOP_K, len(available)))
    sel_loc = [available.index(f) for f in sel]
    ev = _eval(X_tr_a, y_tr, X_te_a, y_te, sel_loc, seed)
    ar = actionability_ratio(sel, TAXONOMY_UCI)
    ar_avail = actionability_ratio_available(sel, horizon, TAXONOMY_UCI)
    tvs = temporal_validity_score(sel, horizon, TAXONOMY_UCI)
    ius_paper = compute_ius_paper(ev["f1"], sel, horizon, TAXONOMY_UCI)
    ius_deploy = compute_ius_deploy(ev["f1"], sel, horizon, TAXONOMY_UCI)
    return {"f1": ev["f1"] * 100, "accuracy": ev["accuracy"] * 100,
             "AR": ar, "AR_available": ar_avail, "TVS": tvs,
             "IUS_paper": ius_paper * 100, "IUS_deploy": ius_deploy * 100,
             "alpha": np.nan, "n": len(sel), "sel": sel}


def run_hardfilter(X_tr, y_tr, X_te, y_te, names, horizon, seed):
    """HardFilter+DE-FS: drop Tier 0+3, then ensemble select."""
    available = filter_by_horizon(names, horizon, TAXONOMY_UCI)
    allowed = {Tier.PRE_SEMESTER, Tier.MID_SEMESTER}
    filtered = [f for f in available
                if _resolve_parent(f, TAXONOMY_UCI) is not None
                and _resolve_parent(f, TAXONOMY_UCI).tier in allowed]
    if not filtered:
        return None
    idx = [names.index(f) for f in filtered]
    X_tr_a, X_te_a = X_tr[:, idx], X_te[:, idx]
    top_k_eff = min(TOP_K, len(filtered))

    # DE-FS-style ensemble
    scaler = MinMaxScaler(); Xn = scaler.fit_transform(X_tr_a)
    c, _ = chi2(Xn, y_tr); c = np.nan_to_num(c, nan=0.0)
    mi = mutual_info_classif(X_tr_a, y_tr, random_state=seed)
    co = np.array([abs(np.corrcoef(X_tr_a[:, j], y_tr)[0, 1])
                    if np.std(X_tr_a[:, j]) > 1e-10 else 0.0
                    for j in range(X_tr_a.shape[1])])
    co = np.nan_to_num(co, nan=0.0)

    def nm(v): return (v - v.min()) / (v.max() - v.min() + 1e-10)
    ens = (nm(c) + nm(mi) + nm(co)) / 3
    top = np.argsort(ens)[::-1][:top_k_eff]
    sel = [filtered[i] for i in top]
    ev = _eval(X_tr_a, y_tr, X_te_a, y_te, list(top), seed)
    ar = actionability_ratio(sel, TAXONOMY_UCI)
    ar_avail = actionability_ratio_available(sel, horizon, TAXONOMY_UCI)
    tvs = temporal_validity_score(sel, horizon, TAXONOMY_UCI)
    ius_paper = compute_ius_paper(ev["f1"], sel, horizon, TAXONOMY_UCI)
    ius_deploy = compute_ius_deploy(ev["f1"], sel, horizon, TAXONOMY_UCI)
    return {"f1": ev["f1"] * 100, "accuracy": ev["accuracy"] * 100,
             "AR": ar, "AR_available": ar_avail, "TVS": tvs,
             "IUS_paper": ius_paper * 100, "IUS_deploy": ius_deploy * 100,
             "alpha": np.nan, "n": len(sel), "sel": sel}


def run_seed(df_raw, dataset, seed, horizon):
    """Run all 4 variants for one seed."""
    X, y, names = preprocess_uci(df_raw, horizon=horizon)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)

    r1 = run_full(X_tr, y_tr, X_te, y_te, names, horizon, seed)
    r2 = run_no_temporal(X_tr, y_tr, X_te, y_te, names, horizon, seed)
    r3 = run_no_action(X_tr, y_tr, X_te, y_te, names, horizon, seed)
    r4 = run_hardfilter(X_tr, y_tr, X_te, y_te, names, horizon, seed)

    return {
        "seed": seed, "dataset": dataset, "horizon": horizon,
        "IUS_paper_full": r1["IUS_paper"],
        "IUS_paper_noTemp": r2["IUS_paper"],
        "IUS_paper_noAction": r3["IUS_paper"],
        "IUS_paper_hardDEFS": r4["IUS_paper"] if r4 else np.nan,
        "IUS_deploy_full": r1["IUS_deploy"],
        "IUS_deploy_noTemp": r2["IUS_deploy"],
        "IUS_deploy_noAction": r3["IUS_deploy"],
        "IUS_deploy_hardDEFS": r4["IUS_deploy"] if r4 else np.nan,
        "AR_full": r1["AR"],
        "AR_available_full": r1["AR_available"],
        "AR_available_noTemp": r2["AR_available"],
        "F1_full": r1["f1"], "F1_noTemp": r2["f1"],
        "TVS_full": r1["TVS"], "TVS_noTemp": r2["TVS"],
        "alpha_full": r1["alpha"], "alpha_noTemp": r2["alpha"],
        "n_full": r1["n"], "n_noTemp": r2["n"],
    }


def cohens_d_paired(x, y):
    diff = np.asarray(x) - np.asarray(y)
    s = diff.std(ddof=1) if len(diff) > 1 else 0
    return float(diff.mean() / s) if s > 1e-10 else 0.0


def main():
    parser = argparse.ArgumentParser(description="UCI Statistics")
    parser.add_argument("--dataset", type=str, default="math",
                          help="'math' or 'portuguese'")
    parser.add_argument("--horizon", type=int, default=0,
                          help="Horizon (0, 1, or 2)")
    args = parser.parse_args()

    dataset = args.dataset.lower()
    if dataset in ["math", "mat"]:
        dataset = "math"
    elif dataset in ["portuguese", "por"]:
        dataset = "portuguese"

    horizon = args.horizon
    out_dir = project_root / "results" / "uci" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"stat8_uci_{dataset}_h{horizon}.csv"

    print("=" * 80)
    print(f"UCI Multi-seed Statistics | {dataset.upper()} | h={horizon} | n={len(RNG_SEEDS)}")
    print("=" * 80)

    df_raw = load_uci_dataset(dataset)
    print(f"[Load] {len(df_raw)} students")

    rows = []
    t0 = time.time()
    for s in RNG_SEEDS:
        t_s = time.time()
        try:
            r = run_seed(df_raw, dataset, s, horizon)
            rows.append(r)
            print(f"  seed={s:4d} ({time.time()-t_s:5.0f}s): "
                   f"full={r['IUS_deploy_full']:5.2f} noTemp={r['IUS_deploy_noTemp']:5.2f} "
                   f"noAct={r['IUS_deploy_noAction']:5.2f} hardDEFS={r['IUS_deploy_hardDEFS']:5.2f}")
        except Exception as e:
            print(f"  seed={s} FAILED: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}  (total {time.time()-t0:.0f}s)")

    if len(df) < 2:
        return

    print("\n--- Bootstrap 95% CI (Deployment-Honest Metrics) ---")
    for col in ["IUS_deploy_full", "IUS_deploy_noTemp", "IUS_deploy_noAction", "IUS_deploy_hardDEFS",
                 "AR_available_full", "AR_available_noTemp",
                 "F1_full", "F1_noTemp", "TVS_full", "TVS_noTemp", "AR_full"]:
        v = df[col].dropna().values
        if len(v) < 2: continue
        print(f"  {col:<24}: mean={v.mean():6.2f}  std={v.std(ddof=1):5.2f}  "
               f"95% CI=[{np.percentile(v, 2.5):5.2f}, {np.percentile(v, 97.5):5.2f}]")

    print("\n--- Wilcoxon signed-rank (one-sided 'greater'), Bonferroni α=0.0167 ---")
    print("Using IUS_deploy (deployment-honest metric):")
    a = df["IUS_deploy_full"].values
    for col in ["IUS_deploy_noTemp", "IUS_deploy_noAction", "IUS_deploy_hardDEFS"]:
        b = df[col].dropna().values
        if len(b) != len(a) or np.allclose(a, b):
            print(f"  full vs {col:<24}: identical or missing — skip")
            continue
        try:
            stat, p = wilcoxon(a, b, alternative="greater", zero_method="wilcox")
            d = cohens_d_paired(a, b)
            sig = ("***" if p < 0.001 else "**" if p < 0.01
                    else "*" if p < 0.0167 else "ns")
            print(f"  full vs {col:<24}: W={stat:5.0f} p={p:.5f} d={d:+.2f} "
                   f"diff={(a-b).mean():+.2f} [{sig}]")
        except ValueError as e:
            print(f"  full vs {col}: {e}")


if __name__ == "__main__":
    main()

#./experiments/oulad/
"""
================================================================================
Deployment-Realistic Evaluation (DRE) on OULAD — Multi-seed
================================================================================
Critical experiment: confirms that DE-FS-style (no temporal filter) suffers
F1 collapse when Tier-3 features (CMA/TMA scores) are masked at inference.

Protocol:
  For each seed, horizon ∈ {0, 1, 2}:
    1. Train IC-FS(full) with temporal filter
    2. Train IC-FS(-temporal) without filter
    3. Mask temporally-unavailable features in BOTH train and test
    4. Retrain on masked train, evaluate on masked test
    5. Compare paper-style F1 vs deployment F1

Output: dre_multi_oulad_h{0,1,2}.csv with 8 seeds each
Total runtime: ~2-3 hours on laptop with n_jobs=-1

Usage:
    python experiments/oulad/run_oulad_dre.py 0   # horizon 0
    python experiments/oulad/run_oulad_dre.py 1   # horizon 1
    python experiments/oulad/run_oulad_dre.py 2   # horizon 2
================================================================================
"""

from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.icfs.ic_fs import (
    feature_scores_for_selection, ic_fs_select,
    actionability_ratio, actionability_ratio_available,
    temporal_validity_score,
    compute_ius_deploy, compute_ius_paper,
    filter_by_horizon,
    apply_dre_mask,              # shared asymmetric DRE utility
)
from src.icfs.taxonomy_oulad import TAXONOMY_OULAD
from preprocess_oulad import preprocess_oulad, load_oulad_horizon

RNG_SEEDS = [42, 123, 456, 789, 1011, 2024, 3033, 4044]
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
TOP_K = 20
N_TREES = 100  # Issue 6 fix: match ICFSPipeline n_estimators=100 (paper §3.7)


def fit_predict_f1(X_tr, y_tr, X_te, y_te, sel_idx, random_state):
    """Train RF on selected features, return F1."""
    rf = RandomForestClassifier(n_estimators=N_TREES,
                                  random_state=random_state,
                                  n_jobs=-1, class_weight='balanced')
    rf.fit(X_tr[:, sel_idx], y_tr)
    y_pred = rf.predict(X_te[:, sel_idx])
    return f1_score(y_te, y_pred, average='weighted', zero_division=0)


def precision_recall_at_top_k(y_true, y_proba, top_k_pct=0.20):
    """
    Compute precision and recall at top k% of predictions.

    Args:
        y_true: True labels (1 = at-risk, 0 = not at-risk)
        y_proba: Predicted probabilities for at-risk class
        top_k_pct: Fraction of population to intervene on

    Returns:
        precision, recall at top k%
    """
    n = len(y_true)
    k = int(n * top_k_pct)
    if k == 0:
        return 0.0, 0.0

    # Issue 2 fix: sort ASCENDING by P(Pass) → lowest P(Pass) = highest at-risk.
    # Previously sorted descending, which ranked PASSING students first.
    top_k_idx = np.argsort(y_proba)[:k]

    # Issue 2 fix: count Fail/Withdrawn (y=0) in top-k.
    # Previously used y_true.sum() which counted y=1 (Pass) — the wrong class.
    tp = (y_true[top_k_idx] == 0).sum()
    precision = tp / k

    # Recall: of all at-risk students (y=0), what fraction did we capture?
    total_at_risk = (y_true == 0).sum()
    recall = tp / total_at_risk if total_at_risk > 0 else 0.0

    return precision, recall


def select_best_ius(X_tr, y_tr, X_te, y_te, names, horizon, random_state,
                      apply_temporal_filter: bool):
    """Run IC-FS α-sweep and return best-IUS selection.

    CRITICAL FIX (ESWA Reviewer 2 - WEAKNESS #2):
    Alpha selection now uses NESTED VALIDATION on training data only.
    Previously, alpha was selected by maximizing IUS on the test set,
    which constituted indirect test-set leakage.

    New protocol:
      1. Split training data into train_inner (80%) / val_inner (20%)
      2. Compute feature scores on train_inner
      3. For each alpha, select features and evaluate F1 on val_inner
      4. Select alpha* that maximizes IUS_val (computed on val_inner)
      5. Retrain on full training set with selected alpha*
      6. Return final selection (without using test set for alpha selection)

    apply_temporal_filter:
      True  → IC-FS (full): filter unavailable features before scoring
      False → IC-FS (-temporal): use all features (DE-FS-style)
    """
    if apply_temporal_filter:
        available = filter_by_horizon(names, horizon, TAXONOMY_OULAD)
        if not available:
            raise RuntimeError(f"No features available at horizon={horizon}")
        idx = [names.index(f) for f in available]
        X_tr_use = X_tr[:, idx]
        X_te_use = X_te[:, idx]
        feat_use = available
    else:
        X_tr_use = X_tr
        X_te_use = X_te
        feat_use = names

    # NESTED VALIDATION: split training data for alpha selection
    # Use a fixed validation seed (based on random_state) for reproducibility
    val_seed = random_state + 1000
    X_tr_inner, X_val_inner, y_tr_inner, y_val_inner = train_test_split(
        X_tr_use, y_tr, test_size=0.2, random_state=val_seed, stratify=y_tr
    )

    # Compute feature scores on INNER TRAINING SET only
    score_df = feature_scores_for_selection(X_tr_inner, y_tr_inner, feat_use,
                                              TAXONOMY_OULAD)

    # Alpha sweep on validation set (NOT test set)
    best_ius_val = -np.inf
    best_alpha = None
    for alpha in ALPHA_GRID:
        sel = ic_fs_select(score_df, alpha, min(TOP_K, len(feat_use)))
        sel_local = [feat_use.index(f) for f in sel]
        # Evaluate on VALIDATION set (inner), not test set
        f1_val = fit_predict_f1(X_tr_inner, y_tr_inner, X_val_inner, y_val_inner,
                                  sel_local, val_seed)
        # FIX: use deployment-honest IUS_deploy, not deprecated compute_ius
        ius_val = compute_ius_deploy(f1_val, sel, horizon, TAXONOMY_OULAD)
        if ius_val > best_ius_val:
            best_ius_val, best_alpha = ius_val, alpha

    # RETRAIN on full training set with selected alpha* (deployment-realistic)
    score_df_full = feature_scores_for_selection(X_tr_use, y_tr, feat_use,
                                                   TAXONOMY_OULAD)
    final_sel = ic_fs_select(score_df_full, best_alpha, min(TOP_K, len(feat_use)))

    # Evaluate F1 on test set (for reporting only, NOT for alpha selection)
    # FIX: return final_f1 directly — avoids fragile back-computation from IUS
    final_sel_local = [feat_use.index(f) for f in final_sel]
    final_f1 = fit_predict_f1(X_tr_use, y_tr, X_te_use, y_te,
                                final_sel_local, random_state)

    return final_sel, best_alpha, final_f1


def evaluate_under_dre(X_tr, y_tr, X_te, y_te, selected_features, horizon,
                         names, random_state):
    """DEPLOYMENT-REALISTIC EVALUATION (CORRECTED - ESWA Reviewer 2 CONCERN #6).

    Previous (INCORRECT) protocol:
      - Masked unavailable features in BOTH training and test sets
      - Trained model on mean-imputed training data
      - This is unrealistic: in deployment, you train on complete historical data

    New (CORRECT) protocol:
      1. Train on COMPLETE training data (all selected features available)
      2. At inference (test time), mask unavailable features with train-column means
      3. This matches real deployment: model trained on full history, predicts on partial data

    Returns:
        f1: weighted F1 score
        y_proba: predicted probabilities for at-risk class (for Precision@k)
    """
    sel_idx = [names.index(f) for f in selected_features]
    assert len(selected_features) == len(sel_idx), \
        "selected_features length must match derived sel_idx"
    X_tr_s = X_tr[:, sel_idx].astype(np.float64).copy()
    X_te_s = X_te[:, sel_idx].astype(np.float64).copy()

    # FIX: use shared apply_dre_mask — single authoritative asymmetric masking utility.
    # Trains on unmasked X_tr_s; masks only inference-time (X_te) columns.
    _, X_te_deploy = apply_dre_mask(X_tr_s, X_te_s, selected_features,
                                     horizon, TAXONOMY_OULAD)

    rf = RandomForestClassifier(n_estimators=N_TREES,
                                  random_state=random_state,
                                  n_jobs=-1, class_weight='balanced')
    rf.fit(X_tr_s, y_tr)  # ← Train on UNMASKED data

    # Predict on deployment-realistic (masked) test data
    y_pred = rf.predict(X_te_deploy)
    y_proba = rf.predict_proba(X_te_deploy)[:, 1] if len(rf.classes_) > 1 else y_pred.astype(float)
    f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    return f1, y_proba


def run_one_seed(df_raw, seed, horizon):
    """Run paper-style + DRE eval for one seed at one horizon."""
    X, y, names = preprocess_oulad(df_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                  random_state=seed, stratify=y)

    # IC-FS(full) — temporal filter on
    sel_full, alpha_full, f1_full_paper = select_best_ius(
        X_tr, y_tr, X_te, y_te, names, horizon, seed,
        apply_temporal_filter=True)

    # IC-FS(-temporal) — no filter (DE-FS analogue)
    sel_notemp, alpha_notemp, f1_notemp_paper = select_best_ius(
        X_tr, y_tr, X_te, y_te, names, horizon, seed,
        apply_temporal_filter=False)

    # DRE: mask + retrain + evaluate
    f1_full_deploy, y_proba_full = evaluate_under_dre(X_tr, y_tr, X_te, y_te,
                                                         sel_full, horizon, names, seed)
    f1_notemp_deploy, y_proba_notemp = evaluate_under_dre(X_tr, y_tr, X_te, y_te,
                                                             sel_notemp, horizon, names, seed)

    # Intervention metrics (Precision@20%, Recall@20%)
    prec20_full, rec20_full = precision_recall_at_top_k(y_te, y_proba_full, 0.20)
    prec20_notemp, rec20_notemp = precision_recall_at_top_k(y_te, y_proba_notemp, 0.20)

    # ─── EXISTING: paper-style AR (unchanged for comparison) ───
    ar_full = actionability_ratio(sel_full, TAXONOMY_OULAD)
    ar_notemp = actionability_ratio(sel_notemp, TAXONOMY_OULAD)

    # ─── NEW: deployment-honest AR_available ────────────────────
    ar_available_full = actionability_ratio_available(sel_full, horizon, TAXONOMY_OULAD)
    ar_available_notemp = actionability_ratio_available(sel_notemp, horizon, TAXONOMY_OULAD)

    # ─── EXISTING: old IUS (for comparison table) ───────────────
    ius_full_deploy_old = f1_full_deploy * ar_full  # TVS=1 for full, so same
    ius_notemp_deploy_old = f1_notemp_deploy * ar_notemp  # INFLATED — for demonstration

    # ─── NEW: correct IUS_deploy ────────────────────────────────
    ius_full_deploy_new = compute_ius_deploy(f1_full_deploy, sel_full, horizon, TAXONOMY_OULAD)
    ius_notemp_deploy_new = compute_ius_deploy(f1_notemp_deploy, sel_notemp, horizon, TAXONOMY_OULAD)

    # FIX: paper-style IUS computed from returned f1_paper, not back-derived from old IUS
    ius_full_paper = compute_ius_paper(f1_full_paper, sel_full, horizon, TAXONOMY_OULAD)
    ius_notemp_paper = compute_ius_paper(f1_notemp_paper, sel_notemp, horizon, TAXONOMY_OULAD)

    # Leakage diagnostics
    tau_full = 1.0 - (ar_available_full / ar_full) if ar_full > 0 else 0.0
    tau_notemp = 1.0 - (ar_available_notemp / ar_notemp) if ar_notemp > 0 else 0.0

    # Leakage flags: did the unfiltered version pick Tier-3 score features?
    tier3_features = ['score_CMA1', 'score_TMA1', 'score_CMA2', 'score_TMA2',
                       'weighted_assessment_score_to_date']
    full_has_t3 = any(any(f.startswith(t) or f == t for t in tier3_features)
                       for f in sel_full)
    notemp_has_t3 = any(any(f.startswith(t) or f == t for t in tier3_features)
                          for f in sel_notemp)

    return {
        "seed": seed, "horizon": horizon,
        "alpha_full": alpha_full, "alpha_notemp": alpha_notemp,
        # F1 values (paper-style and deploy)
        "f1_full_paper": f1_full_paper * 100,
        "f1_notemp_paper": f1_notemp_paper * 100,
        "f1_full_deploy": f1_full_deploy * 100,
        "f1_notemp_deploy": f1_notemp_deploy * 100,
        # Intervention metrics (NEW)
        "precision20_full_deploy": prec20_full * 100,
        "recall20_full_deploy": rec20_full * 100,
        "precision20_notemp_deploy": prec20_notemp * 100,
        "recall20_notemp_deploy": rec20_notemp * 100,
        # AR: old (for comparison) and new (for primary reporting)
        "AR_full": ar_full,
        "AR_notemp": ar_notemp,  # Old: inflated at h=0
        "AR_available_full": ar_available_full,  # New: correct for full
        "AR_available_notemp": ar_available_notemp,  # New: reveals leakage at h=0
        # IUS: old (paper-style, inflated) vs new (deployment-honest)
        "IUS_paper_full": ius_full_paper * 100,
        "IUS_paper_notemp": ius_notemp_paper * 100,
        "IUS_deploy_old_full": ius_full_deploy_old * 100,  # F1_deploy × AR (old formula)
        "IUS_deploy_old_notemp": ius_notemp_deploy_old * 100,  # F1_deploy × AR (old formula)
        "IUS_deploy_full": ius_full_deploy_new * 100,  # PRIMARY metric
        "IUS_deploy_notemp": ius_notemp_deploy_new * 100,  # PRIMARY metric
        # Leakage diagnostics
        "tau_full": tau_full,
        "tau_notemp": tau_notemp,
        "full_has_T3": full_has_t3,
        "notemp_has_T3": notemp_has_t3,
        "n_full": len(sel_full), "n_notemp": len(sel_notemp),
        "selected_full": "|".join(sel_full),
        "selected_notemp": "|".join(sel_notemp),
    }


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out_dir = project_root / "results" / "oulad"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"dre_multi_oulad_h{horizon}.csv"

    print("=" * 80)
    print(f"OULAD DRE Multi-seed | horizon={horizon} | n_seeds={len(RNG_SEEDS)}")
    print("=" * 80)

    print(f"\n[1/3] Loading parquet for h={horizon}...")
    df_raw = load_oulad_horizon(horizon)
    print(f"  Loaded {len(df_raw)} enrollments × {df_raw.shape[1]} columns")
    print(f"  Pass rate: {df_raw['y'].mean():.3f}")

    print(f"\n[2/3] Running {len(RNG_SEEDS)} seeds...")
    rows = []
    t0 = time.time()
    for s in RNG_SEEDS:
        t_seed = time.time()
        try:
            r = run_one_seed(df_raw, s, horizon)
            rows.append(r)
            print(f"  seed={s:4d} ({time.time()-t_seed:5.0f}s) | "
                   f"full_dep={r['f1_full_deploy']:5.1f} "
                   f"notemp_dep={r['f1_notemp_deploy']:5.1f} "
                   f"diff={r['f1_full_deploy']-r['f1_notemp_deploy']:+5.1f} "
                   f"notemp_T3_leak={r['notemp_has_T3']}")
        except Exception as e:
            print(f"  seed={s} FAILED: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}  (total {time.time()-t0:.0f}s)")

    if len(rows) < 2:
        print("\nNot enough seeds for stats")
        return

    print(f"\n[3/3] Statistics ({len(rows)} seeds)")
    print("\n--- Bootstrap 95% CI ---")
    for col in ["f1_full_deploy", "f1_notemp_deploy",
                 "IUS_deploy_full", "IUS_deploy_notemp",
                 "f1_full_paper", "f1_notemp_paper",
                 "IUS_paper_full", "IUS_paper_notemp"]:
        v = df[col].values
        print(f"  {col:<22}: mean={v.mean():6.2f}  std={v.std(ddof=1):5.2f}  "
               f"95% CI=[{np.percentile(v, 2.5):5.2f}, {np.percentile(v, 97.5):5.2f}]")

    print("\n--- Wilcoxon: IC-FS(full) > IC-FS(-temporal) under DEPLOYMENT ---")
    print(f"  Bonferroni α = 0.05/3 = 0.0167 (3 horizons)")
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
            print(f"  IUS_deploy: W={stat:.0f}  p={p:.5f}  d={d:+.2f}  "
                   f"diff={diff.mean():+.2f}  [{sig}]")
        except ValueError as e:
            print(f"  IUS_deploy: Wilcoxon failed ({e})")
    else:
        print("  IUS_deploy: identical values — selections converge at this horizon")

    a_f1 = df["f1_full_deploy"].values
    b_f1 = df["f1_notemp_deploy"].values
    if not np.allclose(a_f1, b_f1):
        try:
            stat, p = wilcoxon(a_f1, b_f1, alternative="greater",
                                zero_method="wilcox")
            diff = a_f1 - b_f1
            d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0
            sig = ("***" if p < 0.001 else "**" if p < 0.01
                    else "*" if p < 0.0167 else "ns")
            print(f"  F1_deploy:  W={stat:.0f}  p={p:.5f}  d={d:+.2f}  "
                   f"diff={diff.mean():+.2f}  [{sig}]")
        except ValueError as e:
            print(f"  F1_deploy: Wilcoxon failed ({e})")

    print("\n--- Leakage exposure ---")
    n_notemp_leaks = df["notemp_has_T3"].sum()
    print(f"  IC-FS(-temporal) selected Tier-3 in {n_notemp_leaks}/{len(df)} seeds")
    if n_notemp_leaks > 0:
        sub = df[df["notemp_has_T3"]]
        f1_drop = sub["f1_notemp_paper"] - sub["f1_notemp_deploy"]
        print(f"    Mean F1 drop in those seeds: {f1_drop.mean():+.2f} pts")


if __name__ == "__main__":
    main()
#./experiments/oulad/
"""
================================================================================
External Baselines on OULAD: NSGA-II-MOFS, Stability Selection, Boruta
================================================================================
Runs 3 baseline feature-selection methods on OULAD per horizon, computes IUS
under same protocol as IC-FS for fair comparison.

NSGA-II:           pymoo, multi-objective (F1, AR), no scalarization
Stability Sel.:    Meinshausen-Bühlmann (2010), bootstrap L1-LogReg
Boruta:            Kursa & Rudnicki (2010) with temporal filter

Output: results/oulad/baselines_oulad_h{0,1,2}.csv
Runtime: ~30-45 min per horizon

Usage:
    python experiments/oulad/run_oulad_baselines.py 0
================================================================================
"""

from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.icfs.ic_fs import (
    actionability_ratio, actionability_ratio_available,
    temporal_validity_score,
    compute_ius_deploy, compute_ius_paper,
    filter_by_horizon, get_temporal_availability,
    apply_dre_mask,
)
from src.icfs.taxonomy_oulad import TAXONOMY_OULAD
from preprocess_oulad import preprocess_oulad, load_oulad_horizon

RANDOM_STATE = 42
TOP_K = 15
N_TREES = 100  # Issue 6 fix: match ICFSPipeline n_estimators=100 (paper §3.7)


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


# ─── Baseline 1: NSGA-II MOFS ─────────────────────────────────────────────
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.optimize import minimize


class FSProblem(ElementwiseProblem):
    """Multi-objective: minimize (-F1, -AR). Constraint: k_min ≤ |sel| ≤ k_max."""

    def __init__(self, X_in, y_in, X_val, y_val, names, k_min=8, k_max=20):
        self.X_in = X_in; self.y_in = y_in
        self.X_val = X_val; self.y_val = y_val
        self.names = names
        self.k_min = k_min; self.k_max = k_max
        self.clf = RandomForestClassifier(n_estimators=40,
                                            random_state=RANDOM_STATE,
                                            n_jobs=-1, class_weight='balanced')
        super().__init__(n_var=len(names), n_obj=2, n_constr=2,
                          xl=0, xu=1, vtype=bool)

    def _evaluate(self, x, out, *a, **kw):
        mask = x.astype(bool)
        k = int(mask.sum())
        if k < 1:
            out["F"] = [0.0, 0.0]
            out["G"] = [self.k_min - k, k - self.k_max]
            return
        sel = [self.names[i] for i in range(len(mask)) if mask[i]]
        idx = np.where(mask)[0]
        try:
            clf = clone(self.clf)
            clf.fit(self.X_in[:, idx], self.y_in)
            f1 = f1_score(self.y_val, clf.predict(self.X_val[:, idx]),
                           average='weighted', zero_division=0)
        except Exception:
            f1 = 0.0
        ar = actionability_ratio(sel, TAXONOMY_OULAD)
        out["F"] = [-f1, -ar]
        out["G"] = [self.k_min - k, k - self.k_max]


def run_nsga2(X_tr, y_tr, X_te, y_te, names, horizon,
                pop_size=40, n_gen=25):
    available = filter_by_horizon(names, horizon, TAXONOMY_OULAD)
    if not available:
        return None
    idx_avail = [names.index(f) for f in available]
    X_tr_a = X_tr[:, idx_avail]
    X_te_a = X_te[:, idx_avail]

    X_in, X_val, y_in, y_val = train_test_split(
        X_tr_a, y_tr, test_size=0.25,
        random_state=RANDOM_STATE, stratify=y_tr)

    problem = FSProblem(X_in, y_in, X_val, y_val, available,
                          k_min=8, k_max=min(20, len(available)))
    algo = NSGA2(pop_size=pop_size,
                  sampling=BinaryRandomSampling(),
                  crossover=TwoPointCrossover(),
                  mutation=BitflipMutation(prob=0.1),
                  eliminate_duplicates=True)
    res = minimize(problem, algo, ("n_gen", n_gen),
                     verbose=False, seed=RANDOM_STATE)

    # Find best-IUS Pareto solution evaluated on test set
    final_clf = RandomForestClassifier(n_estimators=N_TREES,
                                          random_state=RANDOM_STATE,
                                          n_jobs=-1, class_weight='balanced')
    best = None
    for x in res.X:
        mask = x.astype(bool)
        if mask.sum() < 1:
            continue
        sel = [available[i] for i in range(len(mask)) if mask[i]]
        loc = np.where(mask)[0]
        clf = clone(final_clf)
        clf.fit(X_tr_a[:, loc], y_tr)
        y_pred = clf.predict(X_te_a[:, loc])
        f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
        acc = accuracy_score(y_te, y_pred)

        # DRE masking — asymmetric: train on unmasked, mask only X_te.
        # Loop is a no-op here (temporal filter upstream ensures all sel features
        # are available), but we call apply_dre_mask for code uniformity and
        # future safety if the filter is ever removed or relaxed.
        _, X_te_deploy = apply_dre_mask(
            X_tr_a[:, loc], X_te_a[:, loc], sel, horizon, TAXONOMY_OULAD)
        clf_deploy = clone(final_clf)
        clf_deploy.fit(X_tr_a[:, loc], y_tr)   # train on UNMASKED
        y_pred_deploy = clf_deploy.predict(X_te_deploy)
        y_proba_deploy = clf_deploy.predict_proba(X_te_deploy)[:, 1] if len(clf_deploy.classes_) > 1 else y_pred_deploy.astype(float)
        f1_deploy = f1_score(y_te, y_pred_deploy, average='weighted', zero_division=0)

        # Intervention metrics
        prec20, rec20 = precision_recall_at_top_k(y_te, y_proba_deploy, 0.20)

        ar = actionability_ratio(sel, TAXONOMY_OULAD)
        ar_avail = actionability_ratio_available(sel, horizon, TAXONOMY_OULAD)
        tvs = temporal_validity_score(sel, horizon, TAXONOMY_OULAD)
        ius_paper = compute_ius_paper(f1, sel, horizon, TAXONOMY_OULAD)
        ius_deploy = compute_ius_deploy(f1_deploy, sel, horizon, TAXONOMY_OULAD)

        cand = {"accuracy": acc * 100, "f1_paper": f1 * 100, "f1_deploy": f1_deploy * 100,
                 "precision20_deploy": prec20 * 100, "recall20_deploy": rec20 * 100,
                 "AR": ar, "AR_available": ar_avail, "TVS": tvs,
                 "IUS_paper": ius_paper * 100, "IUS_deploy": ius_deploy * 100,
                 "n_features": int(mask.sum()),
                 "selected": "|".join(sel)}
        if best is None or cand["IUS_deploy"] > best["IUS_deploy"]:
            best = cand

    if best is None:
        return None
    best["method"] = "NSGA-II-MOFS"
    best["horizon"] = horizon
    return best


# ─── Baseline 2: Stability Selection (Meinshausen-Bühlmann) ──────────────
def run_stability_selection(X_tr, y_tr, X_te, y_te, names, horizon,
                                n_subsamples=40, top_k=TOP_K,
                                subsample_frac=0.75,
                                C_grid=(0.01, 0.1, 1.0)):
    available = filter_by_horizon(names, horizon, TAXONOMY_OULAD)
    if not available:
        return None
    idx_avail = [names.index(f) for f in available]
    X_tr_a = X_tr[:, idx_avail]
    X_te_a = X_te[:, idx_avail]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tr_a)

    n, p = X_scaled.shape
    counts = np.zeros(p)
    rng = np.random.RandomState(RANDOM_STATE)

    for _ in range(n_subsamples):
        idx_sub = rng.choice(n, int(n * subsample_frac), replace=False)
        Xs, ys = X_scaled[idx_sub], y_tr[idx_sub]
        if len(np.unique(ys)) < 2:
            continue
        for C in C_grid:
            try:
                lr = LogisticRegression(penalty="l1", solver="liblinear",
                                          C=C, max_iter=300,
                                          random_state=RANDOM_STATE,
                                          class_weight='balanced')
                lr.fit(Xs, ys)
                active = np.abs(lr.coef_[0]) > 1e-6
                counts[active] += 1
            except Exception:
                pass

    freq = counts / (n_subsamples * len(C_grid))
    order = np.argsort(freq)[::-1][:min(top_k, len(available))]
    sel = [available[i] for i in order]

    rf = RandomForestClassifier(n_estimators=N_TREES,
                                  random_state=RANDOM_STATE,
                                  n_jobs=-1, class_weight='balanced')
    rf.fit(X_tr_a[:, order], y_tr)
    y_pred = rf.predict(X_te_a[:, order])
    f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    acc = accuracy_score(y_te, y_pred)

    # DRE masking — asymmetric (see apply_dre_mask docstring).
    _, X_te_deploy = apply_dre_mask(
        X_tr_a[:, order], X_te_a[:, order], sel, horizon, TAXONOMY_OULAD)
    rf_deploy = clone(rf)
    rf_deploy.fit(X_tr_a[:, order], y_tr)   # train on UNMASKED
    y_pred_deploy = rf_deploy.predict(X_te_deploy)
    y_proba_deploy = rf_deploy.predict_proba(X_te_deploy)[:, 1] if len(rf_deploy.classes_) > 1 else y_pred_deploy.astype(float)
    f1_deploy = f1_score(y_te, y_pred_deploy, average='weighted', zero_division=0)

    # Intervention metrics
    prec20, rec20 = precision_recall_at_top_k(y_te, y_proba_deploy, 0.20)

    ar = actionability_ratio(sel, TAXONOMY_OULAD)
    ar_avail = actionability_ratio_available(sel, horizon, TAXONOMY_OULAD)
    tvs = temporal_validity_score(sel, horizon, TAXONOMY_OULAD)
    ius_paper = compute_ius_paper(f1, sel, horizon, TAXONOMY_OULAD)
    ius_deploy = compute_ius_deploy(f1_deploy, sel, horizon, TAXONOMY_OULAD)

    return {"method": "StabilitySelection", "horizon": horizon,
             "accuracy": acc * 100, "f1_paper": f1 * 100, "f1_deploy": f1_deploy * 100,
             "precision20_deploy": prec20 * 100, "recall20_deploy": rec20 * 100,
             "AR": ar, "AR_available": ar_avail, "TVS": tvs,
             "IUS_paper": ius_paper * 100, "IUS_deploy": ius_deploy * 100,
             "n_features": len(sel),
             "selected": "|".join(sel)}


# ─── Baseline 3: Boruta + temporal filter ────────────────────────────────
from boruta import BorutaPy


def run_boruta(X_tr, y_tr, X_te, y_te, names, horizon, max_iter=40):
    available = filter_by_horizon(names, horizon, TAXONOMY_OULAD)
    if not available:
        return None
    idx_avail = [names.index(f) for f in available]
    X_tr_a = X_tr[:, idx_avail]
    X_te_a = X_te[:, idx_avail]

    rf_b = RandomForestClassifier(n_estimators=80, n_jobs=-1,
                                     random_state=RANDOM_STATE,
                                     class_weight="balanced", max_depth=6)
    selector = BorutaPy(rf_b, n_estimators="auto", max_iter=max_iter,
                          random_state=RANDOM_STATE, verbose=0)
    try:
        selector.fit(X_tr_a, y_tr)
    except Exception as e:
        return {"method": "Boruta", "horizon": horizon, "error": str(e),
                 "IUS_paper": np.nan, "IUS_deploy": np.nan,
                 "f1_paper": np.nan, "f1_deploy": np.nan,
                 "AR": np.nan, "AR_available": np.nan, "TVS": np.nan,
                 "n_features": 0, "selected": ""}

    confirmed = selector.support_
    tentative = (selector.support_weak_
                  if hasattr(selector, "support_weak_")
                  else np.zeros(len(available), dtype=bool))
    sel_mask = confirmed | tentative
    if sel_mask.sum() == 0:
        sel_mask = selector.ranking_ <= 2

    sel = [available[i] for i in range(len(available)) if sel_mask[i]]
    loc = np.where(sel_mask)[0]
    if len(sel) == 0:
        return {"method": "Boruta", "horizon": horizon,
                 "IUS_paper": 0.0, "IUS_deploy": 0.0,
                 "f1_paper": 0.0, "f1_deploy": 0.0,
                 "AR": 0.0, "AR_available": 0.0, "TVS": 0.0,
                 "n_features": 0, "selected": ""}

    rf = RandomForestClassifier(n_estimators=N_TREES,
                                  random_state=RANDOM_STATE,
                                  n_jobs=-1, class_weight='balanced')
    rf.fit(X_tr_a[:, loc], y_tr)
    y_pred = rf.predict(X_te_a[:, loc])
    f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    acc = accuracy_score(y_te, y_pred)

    # DRE masking — asymmetric (see apply_dre_mask docstring).
    _, X_te_deploy = apply_dre_mask(
        X_tr_a[:, loc], X_te_a[:, loc], sel, horizon, TAXONOMY_OULAD)
    rf_deploy = clone(rf)
    rf_deploy.fit(X_tr_a[:, loc], y_tr)   # train on UNMASKED
    y_pred_deploy = rf_deploy.predict(X_te_deploy)
    y_proba_deploy = rf_deploy.predict_proba(X_te_deploy)[:, 1] if len(rf_deploy.classes_) > 1 else y_pred_deploy.astype(float)
    f1_deploy = f1_score(y_te, y_pred_deploy, average='weighted', zero_division=0)

    # Intervention metrics
    prec20, rec20 = precision_recall_at_top_k(y_te, y_proba_deploy, 0.20)

    ar = actionability_ratio(sel, TAXONOMY_OULAD)
    ar_avail = actionability_ratio_available(sel, horizon, TAXONOMY_OULAD)
    tvs = temporal_validity_score(sel, horizon, TAXONOMY_OULAD)
    ius_paper = compute_ius_paper(f1, sel, horizon, TAXONOMY_OULAD)
    ius_deploy = compute_ius_deploy(f1_deploy, sel, horizon, TAXONOMY_OULAD)

    return {"method": "Boruta", "horizon": horizon,
             "accuracy": acc * 100, "f1_paper": f1 * 100, "f1_deploy": f1_deploy * 100,
             "precision20_deploy": prec20 * 100, "recall20_deploy": rec20 * 100,
             "AR": ar, "AR_available": ar_avail, "TVS": tvs,
             "IUS_paper": ius_paper * 100, "IUS_deploy": ius_deploy * 100,
             "n_features": len(sel),
             "selected": "|".join(sel)}


# ─── Orchestration ───────────────────────────────────────────────────────
def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out_dir = project_root / "results" / "oulad"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"baselines_oulad_h{horizon}.csv"

    print("=" * 80)
    print(f"OULAD Baselines | horizon={horizon}")
    print("=" * 80)

    print(f"\n[Load] Reading parquet for h={horizon}...")
    df_raw = load_oulad_horizon(horizon)
    X, y, names = preprocess_oulad(df_raw, verbose=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                  random_state=RANDOM_STATE,
                                                  stratify=y)
    print(f"  Train={len(y_tr)} Test={len(y_te)} Features={len(names)}")

    rows = []

    print(f"\n[1/3] NSGA-II-MOFS (pop=40, gen=25)...")
    t0 = time.time()
    r1 = run_nsga2(X_tr, y_tr, X_te, y_te, names, horizon)
    if r1:
        print(f"  {time.time()-t0:.0f}s | IUS_deploy={r1['IUS_deploy']:.2f} "
               f"F1_deploy={r1['f1_deploy']:.2f} AR_avail={r1['AR_available']:.3f} "
               f"k={r1['n_features']}")
        rows.append(r1)

    print(f"\n[2/3] Stability Selection (40 subsamples, 3 C values)...")
    t0 = time.time()
    r2 = run_stability_selection(X_tr, y_tr, X_te, y_te, names, horizon)
    if r2:
        print(f"  {time.time()-t0:.0f}s | IUS_deploy={r2['IUS_deploy']:.2f} "
               f"F1_deploy={r2['f1_deploy']:.2f} AR_avail={r2['AR_available']:.3f} "
               f"k={r2['n_features']}")
        rows.append(r2)

    print(f"\n[3/3] Boruta + temporal filter (max_iter=40)...")
    t0 = time.time()
    r3 = run_boruta(X_tr, y_tr, X_te, y_te, names, horizon)
    if r3:
        print(f"  {time.time()-t0:.0f}s | IUS_deploy={r3.get('IUS_deploy', np.nan):.2f} "
               f"F1_deploy={r3.get('f1_deploy', np.nan):.2f} k={r3.get('n_features', 0)}")
        rows.append(r3)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    cols = ["method", "horizon", "accuracy", "f1_deploy", "AR", "AR_available",
            "TVS", "IUS_deploy", "IUS_paper", "n_features"]
    cols_avail = [c for c in cols if c in df_out.columns]
    print("\n--- Summary (Deployment-Honest Metrics) ---")
    print(df_out[cols_avail].to_string(index=False,
                                          float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
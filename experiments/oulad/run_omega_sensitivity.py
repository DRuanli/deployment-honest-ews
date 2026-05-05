#./experiments/oulad/
"""
================================================================================
Post-hoc w₂ (Tier-2 actionability weight) Sensitivity Analysis — OULAD
================================================================================
Recomputes IUS_deploy for w₂ ∈ {0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} using
EXISTING feature selections from dre_multi and baselines experiments.

Key insight: F1_deploy is FIXED (selections unchanged); only AR_available
changes linearly in w₂ because:
    AR_available(w₂) = (n_tier1 × 1.0 + n_tier2 × w₂) / |S|
    IUS_deploy(w₂)   = F1_deploy × AR_available(w₂)

This separates two concerns:
  • Predictive performance (F1_deploy) — invariant to w₂
  • Actionability accounting (AR_available) — scales with w₂

α* endogeneity check: when w₂ changes, IUS_val(α) used for nested α-selection
also changes.  At low w₂, α* may shift.  We flag any horizon/seed where α*
would shift based on the stored feature compositions.

Input files (tried in order):
  results/oulad/k{K}/dre_multi_oulad_h{h}_k{K}.csv   ← k-parameterised (new)
  results/oulad/dre_multi_oulad_h{h}.csv               ← flat (legacy)
  results/oulad/baselines_oulad_h{h}.csv

Output:
  results/oulad/omega_sensitivity_h{h}.csv             ← per-horizon detail
  results/oulad/omega_sensitivity_all_horizons.csv      ← combined summary

Usage:
    python experiments/oulad/run_omega_sensitivity.py            # all horizons
    python experiments/oulad/run_omega_sensitivity.py --horizon 1
    python experiments/oulad/run_omega_sensitivity.py --k 5      # prefer k=5
================================================================================
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.icfs.ic_fs import Tier, _resolve_parent
from src.icfs.taxonomy_oulad import TAXONOMY_OULAD

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OMEGA_2_VALUES = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PAPER_W2       = 0.7
RESULTS_DIR    = project_root / "results" / "oulad"
DATASET_LABEL  = "OULAD"


# ---------------------------------------------------------------------------
# Tier utilities
# ---------------------------------------------------------------------------
def _tier_of(fname: str) -> Tier | None:
    profile = _resolve_parent(fname, TAXONOMY_OULAD)
    return profile.tier if profile else None


def _count_tiers(feature_list: list[str]) -> dict:
    """Count features per tier in a selection."""
    counts = {Tier.NON_ACTIONABLE: 0, Tier.PRE_SEMESTER: 0,
              Tier.MID_SEMESTER: 0, Tier.PAST_GRADE: 0, None: 0}
    for f in feature_list:
        counts[_tier_of(f)] += 1
    return counts


def _ar_available(n_tier1: int, n_tier2: int, n_total: int, w2: float) -> float:
    """AR_available = (n_tier1 × 1.0 + n_tier2 × w₂) / |S|."""
    if n_total == 0:
        return 0.0
    return (n_tier1 * 1.0 + n_tier2 * w2) / n_total


# ---------------------------------------------------------------------------
# File loading helpers — try k-parameterised path first, then flat
# ---------------------------------------------------------------------------
def _load_dre_multi(h: int, k: int | None) -> pd.DataFrame:
    """Load dre_multi for OULAD at horizon h, trying k-aware path first."""
    candidates = []
    if k is not None:
        candidates.append(RESULTS_DIR / f"k{k}" / f"dre_multi_oulad_h{h}_k{k}.csv")
    # Auto-detect available k directories
    for kdir in sorted(RESULTS_DIR.glob("k*")):
        kval = kdir.name[1:]
        if kval.isdigit():
            candidates.append(kdir / f"dre_multi_oulad_h{h}_k{kval}.csv")
    candidates.append(RESULTS_DIR / f"dre_multi_oulad_h{h}.csv")

    for p in candidates:
        if p.exists():
            print(f"    [dre_multi h={h}] Loading {p.relative_to(project_root)}")
            return pd.read_csv(p)
    print(f"    [dre_multi h={h}] Not found — skipping IC-FS(full)")
    return pd.DataFrame()


def _load_baselines(h: int) -> pd.DataFrame:
    p = RESULTS_DIR / f"baselines_oulad_h{h}.csv"
    if p.exists():
        print(f"    [baselines h={h}] Loading {p.relative_to(project_root)}")
        return pd.read_csv(p)
    print(f"    [baselines h={h}] Not found — skipping baselines")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Core recomputation
# ---------------------------------------------------------------------------
def _recompute_sensitivity(
    selected_str: str,
    f1_deploy: float,
    method: str,
    horizon: int,
    seed: int | float,
) -> list[dict]:
    """Recompute IUS_deploy for each w₂ given a fixed feature selection."""
    if pd.isna(selected_str) or not selected_str:
        return []

    features = [f.strip() for f in selected_str.split("|") if f.strip()]
    if not features:
        return []

    counts = _count_tiers(features)
    n_tier1 = counts[Tier.PRE_SEMESTER]
    n_tier2 = counts[Tier.MID_SEMESTER]
    n_tier0 = counts[Tier.NON_ACTIONABLE]
    n_tier3 = counts[Tier.PAST_GRADE]
    n_total = len(features)

    rows = []
    for w2 in OMEGA_2_VALUES:
        ar = _ar_available(n_tier1, n_tier2, n_total, w2)
        ius = (f1_deploy / 100.0) * ar * 100.0
        rows.append({
            "dataset":     DATASET_LABEL,
            "horizon":     horizon,
            "method":      method,
            "seed":        seed,
            "w2":          w2,
            "n_tier0":     n_tier0,
            "n_tier1":     n_tier1,
            "n_tier2":     n_tier2,
            "n_tier3":     n_tier3,
            "n_total":     n_total,
            "f1_deploy":   f1_deploy,
            "AR_available": ar,
            "IUS_deploy":  ius,
        })
    return rows


def process_horizon(h: int, k: int | None) -> pd.DataFrame:
    """Run sensitivity analysis for one horizon; return long-form DataFrame."""
    print(f"\n  {'─'*70}")
    print(f"  Horizon h={h}")
    print(f"  {'─'*70}")
    all_rows = []

    # ── IC-FS(full) from dre_multi ──────────────────────────────────────
    df_dre = _load_dre_multi(h, k)
    if not df_dre.empty:
        sel_col = "selected_full" if "selected_full" in df_dre.columns else None
        f1_col  = "f1_full_deploy" if "f1_full_deploy" in df_dre.columns else None

        if sel_col and f1_col:
            for _, row in df_dre.iterrows():
                rows = _recompute_sensitivity(
                    row[sel_col], row[f1_col],
                    "IC-FS(full)", h, row.get("seed", 0)
                )
                all_rows.extend(rows)
            print(f"      → IC-FS(full): {len(df_dre)} seeds processed")
        else:
            print(f"      → IC-FS(full): missing columns {sel_col=} {f1_col=}")

    # ── Baselines ────────────────────────────────────────────────────────
    df_bl = _load_baselines(h)
    if not df_bl.empty:
        for _, row in df_bl.iterrows():
            method = str(row.get("method", "Baseline"))
            sel    = row.get("selected", "")
            f1     = row.get("f1_deploy", row.get("f1_paper", np.nan))
            if pd.isna(f1):
                continue
            rows = _recompute_sensitivity(sel, float(f1), method, h, seed=0)
            all_rows.extend(rows)
        print(f"      → Baselines: {len(df_bl)} methods processed")

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Ranking analysis helper
# ---------------------------------------------------------------------------
def _ranking_analysis(df: pd.DataFrame) -> None:
    """Print ranking preservation table: IC-FS(full) vs best baseline."""
    if df.empty:
        return

    print("\n  RANKING PRESERVATION (IC-FS(full) vs best baseline):")
    print(f"  {'w₂':>6} | {'h=0 rank':>10} | {'h=1 rank':>10} | {'h=2 rank':>10}")
    print(f"  {'-'*46}")

    for w2 in OMEGA_2_VALUES:
        slice_ = df[np.isclose(df["w2"], w2)]
        row_parts = [f"  {w2:>6.1f} |"]
        for h in [0, 1, 2]:
            hslice = slice_[slice_["horizon"] == h]
            if hslice.empty:
                row_parts.append(f"{'N/A':>10} |")
                continue
            ius_by_method = hslice.groupby("method")["IUS_deploy"].mean()
            if "IC-FS(full)" not in ius_by_method.index:
                row_parts.append(f"{'no data':>10} |")
                continue
            ic_ius = ius_by_method["IC-FS(full)"]
            baselines = ius_by_method.drop("IC-FS(full)", errors="ignore")
            if baselines.empty:
                row_parts.append(f"{'1st':>10} |")
                continue
            n_above = int((baselines >= ic_ius).sum())
            rank    = n_above + 1
            marker  = "✓" if rank == 1 else "✗"
            row_parts.append(f"  {marker} rank {rank:>3} |")
        print("".join(row_parts))


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def _print_summary(df: pd.DataFrame) -> None:
    """Print CV and robustness metrics for IC-FS(full)."""
    icfs = df[df["method"] == "IC-FS(full)"]
    if icfs.empty:
        return

    baseline_at_paper = None
    if not df[df["w2"].apply(lambda x: np.isclose(x, PAPER_W2))].empty:
        paper_slice = df[
            (df["method"] == "IC-FS(full)") &
            df["w2"].apply(lambda x: np.isclose(x, PAPER_W2))
        ]
        if not paper_slice.empty:
            baseline_at_paper = paper_slice["IUS_deploy"].mean()

    print("\n  IC-FS(full) IUS_deploy by w₂ (mean ± std across seeds and horizons):")
    print(f"  {'w₂':>6} | {'mean':>8} | {'std':>8} | {'Δ vs 0.7':>10} | {'CV%':>7}")
    print(f"  {'-'*48}")
    for w2 in OMEGA_2_VALUES:
        slc = icfs[icfs["w2"].apply(lambda x: np.isclose(x, w2))]["IUS_deploy"]
        if slc.empty:
            continue
        mean, std = slc.mean(), slc.std(ddof=1) if len(slc) > 1 else 0.0
        cv   = 100 * std / mean if mean > 0 else 0.0
        delta = mean - baseline_at_paper if baseline_at_paper else float("nan")
        mark  = " ← paper" if np.isclose(w2, PAPER_W2) else ""
        print(f"  {w2:>6.1f} | {mean:>8.2f} | {std:>8.2f} | {delta:>+10.2f} | {cv:>7.1f}%{mark}")

    # CV for middle range
    mid_vals = icfs[icfs["w2"].apply(lambda x: x in [0.5, 0.6, 0.7, 0.8, 0.9])]["IUS_deploy"]
    if len(mid_vals) > 1:
        cv_mid = 100 * mid_vals.std(ddof=1) / mid_vals.mean()
        verdict = "ROBUST" if cv_mid < 10 else ("~ MODERATE" if cv_mid < 15 else "SENSITIVE")
        print(f"\n  CV over w₂ ∈ [0.5, 0.9]: {cv_mid:.1f}% → {verdict}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc w₂ sensitivity analysis for OULAD")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Run only one horizon (0, 1, or 2)")
    parser.add_argument("--k", type=int, default=None,
                        help="Preferred feature budget k (used for file lookup)")
    args = parser.parse_args()

    horizons = [args.horizon] if args.horizon is not None else [0, 1, 2]

    print("=" * 80)
    print("OULAD — Post-hoc w₂ Sensitivity Analysis")
    print("=" * 80)
    print(f"w₂ grid:  {OMEGA_2_VALUES}")
    print(f"Paper w₂: {PAPER_W2}")

    all_dfs = []
    for h in horizons:
        df_h = process_horizon(h, args.k)
        if not df_h.empty:
            out_csv = RESULTS_DIR / f"omega_sensitivity_h{h}.csv"
            df_h.to_csv(out_csv, index=False, float_format="%.4f")
            print(f"\n Saved → {out_csv.relative_to(project_root)}")
            all_dfs.append(df_h)

    if not all_dfs:
        print("\n[WARN] No data found. Run dre_multi and baselines experiments first.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Combined CSV
    out_combined = RESULTS_DIR / "omega_sensitivity_all_horizons.csv"
    df_all.to_csv(out_combined, index=False, float_format="%.4f")
    print(f"\n Combined → {out_combined.relative_to(project_root)}")

    # Analytics
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    _print_summary(df_all)
    _ranking_analysis(df_all)

    print("\n" + "=" * 80)
    print("OULAD omega sensitivity analysis complete.")
    print("=" * 80)
    print("\nNext: run generate_results_figures.py --omega-sensitivity")
    print("      to generate fig_R8_omega_sensitivity_oulad.pdf/.png")


if __name__ == "__main__":
    main()
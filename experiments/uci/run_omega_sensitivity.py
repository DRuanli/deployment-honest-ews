#./experiments/uci/
"""
================================================================================
Post-hoc w₂ (Tier-2 actionability weight) Sensitivity Analysis — UCI
================================================================================
Same recomputation logic as experiments/oulad/run_omega_sensitivity.py but
for UCI Math and UCI Portuguese datasets.

Input files:
  results/uci/{dataset}/k{k}/stat8_uci_{dataset}_h{h}_k{k}.csv  ← k-aware
  results/uci/{dataset}/uci_{dataset}_icfs_multi_h{h}.csv        ← legacy
  results/uci/{dataset}/baselines_uci_{dataset}_h{h}.csv

Note: for IC-FS(full) selections we use the multi-seed experiments CSV
      (which stores the `selected` column of best-IUS features per seed).
      If only a k-aware stat8 file exists and does NOT contain a `selected_full`
      column, that horizon is skipped with a warning.

Output:
  results/uci/{dataset}/omega_sensitivity_{dataset}_h{h}.csv
  results/uci/{dataset}/omega_sensitivity_{dataset}_all_horizons.csv

Usage:
    python experiments/uci/run_omega_sensitivity.py --dataset math
    python experiments/uci/run_omega_sensitivity.py --dataset portuguese
    python experiments/uci/run_omega_sensitivity.py --dataset math --horizon 1
    python experiments/uci/run_omega_sensitivity.py --dataset math --k 10
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
from src.icfs.taxonomy_uci import TAXONOMY_UCI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OMEGA_2_VALUES = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PAPER_W2       = 0.7


# ---------------------------------------------------------------------------
# Tier utilities
# ---------------------------------------------------------------------------
def _tier_of(fname: str) -> Tier | None:
    profile = _resolve_parent(fname, TAXONOMY_UCI)
    return profile.tier if profile else None


def _count_tiers(feature_list: list[str]) -> dict:
    counts = {Tier.NON_ACTIONABLE: 0, Tier.PRE_SEMESTER: 0,
              Tier.MID_SEMESTER: 0, Tier.PAST_GRADE: 0, None: 0}
    for f in feature_list:
        counts[_tier_of(f)] += 1
    return counts


def _ar_available(n_tier1: int, n_tier2: int, n_total: int, w2: float) -> float:
    if n_total == 0:
        return 0.0
    return (n_tier1 * 1.0 + n_tier2 * w2) / n_total


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------
def _results_dir(dataset: str) -> Path:
    return project_root / "results" / "uci" / dataset


def _load_icfs_multi(dataset: str, h: int, k: int | None) -> pd.DataFrame:
    """
    Load IC-FS(full) multi-seed selections.
    Priority order:
      1. k-aware stat8 (if it has selected_full)
      2. Flat multi-seed icfs_multi CSV (has `selected` column)
    """
    rdir = _results_dir(dataset)

    # Try k-aware stat8 first (if caller specified k)
    if k is not None:
        p = rdir / f"k{k}" / f"stat8_uci_{dataset}_h{h}_k{k}.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "selected_full" in df.columns:
                print(f"    [IC-FS h={h}] {p.relative_to(project_root)}")
                return df.rename(columns={"selected_full": "selected",
                                          "F1_full": "f1_deploy",
                                          "IUS_deploy_full": "IUS_deploy"})

    # Auto-scan k-directories for stat8 with selected_full
    for kdir in sorted(rdir.glob("k*")):
        kval = kdir.name[1:]
        if not kval.isdigit():
            continue
        p = kdir / f"stat8_uci_{dataset}_h{h}_k{kval}.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "selected_full" in df.columns:
                print(f"    [IC-FS h={h}] {p.relative_to(project_root)}")
                return df.rename(columns={"selected_full": "selected",
                                          "F1_full": "f1_deploy",
                                          "IUS_deploy_full": "IUS_deploy"})

    # Fallback: flat multi-seed CSV (from run_uci_experiments.py)
    p_flat = rdir / f"uci_{dataset}_icfs_multi_h{h}.csv"
    if p_flat.exists():
        print(f"    [IC-FS h={h}] {p_flat.relative_to(project_root)}")
        return pd.read_csv(p_flat)

    print(f"    [IC-FS h={h}] Not found — skipping IC-FS(full)")
    return pd.DataFrame()


def _load_baselines(dataset: str, h: int) -> pd.DataFrame:
    p = _results_dir(dataset) / f"baselines_uci_{dataset}_h{h}.csv"
    if p.exists():
        print(f"    [baselines h={h}] {p.relative_to(project_root)}")
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
    dataset_label: str,
) -> list[dict]:
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
        ar  = _ar_available(n_tier1, n_tier2, n_total, w2)
        ius = (f1_deploy / 100.0) * ar * 100.0
        rows.append({
            "dataset":     dataset_label,
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


def process_horizon(dataset: str, h: int, k: int | None) -> pd.DataFrame:
    print(f"\n  {'─'*70}")
    print(f"  Dataset: {dataset.upper()}  |  Horizon h={h}")
    print(f"  {'─'*70}")
    label = f"UCI-{dataset.capitalize()}"
    all_rows = []

    # ── IC-FS(full) ─────────────────────────────────────────────────────
    df_icfs = _load_icfs_multi(dataset, h, k)
    if not df_icfs.empty:
        # Determine column names (may differ between stat8 and icfs_multi)
        sel_col = next((c for c in ["selected", "selected_full"] if c in df_icfs.columns), None)
        f1_col  = next((c for c in ["f1_deploy", "F1_full", "f1_full_deploy"] if c in df_icfs.columns), None)

        if sel_col and f1_col:
            for _, row in df_icfs.iterrows():
                f1_val = row[f1_col]
                # multi-seed CSV stores F1 as percentage already; stat8 also
                rows = _recompute_sensitivity(
                    row[sel_col], float(f1_val), "IC-FS(full)",
                    h, row.get("seed", 0), label
                )
                all_rows.extend(rows)
            print(f"      → IC-FS(full): {len(df_icfs)} seeds processed")
        else:
            print(f"      → IC-FS(full): missing columns ({sel_col=}, {f1_col=})")

    # ── Baselines ────────────────────────────────────────────────────────
    df_bl = _load_baselines(dataset, h)
    if not df_bl.empty:
        for _, row in df_bl.iterrows():
            method = str(row.get("method", "Baseline"))
            sel    = row.get("selected", "")
            f1     = row.get("f1_deploy", row.get("f1_paper", np.nan))
            if pd.isna(f1):
                continue
            rows = _recompute_sensitivity(
                sel, float(f1), method, h, seed=0, dataset_label=label
            )
            all_rows.extend(rows)
        print(f"      → Baselines: {len(df_bl)} methods processed")

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Ranking preservation check
# ---------------------------------------------------------------------------
def _ranking_analysis(df: pd.DataFrame, dataset: str) -> None:
    if df.empty:
        return

    print(f"\n  RANKING PRESERVATION — {dataset.upper()}")
    print(f"  {'w₂':>6} | {'h=0':>10} | {'h=1':>10} | {'h=2':>10}")
    print(f"  {'-'*46}")

    for w2 in OMEGA_2_VALUES:
        sl = df[df["w2"].apply(lambda x: np.isclose(x, w2))]
        row_parts = [f"  {w2:>6.1f} |"]
        for h in [0, 1, 2]:
            hsl = sl[sl["horizon"] == h]
            if hsl.empty:
                row_parts.append(f"{'N/A':>10} |")
                continue
            ius_by_method = hsl.groupby("method")["IUS_deploy"].mean()
            if "IC-FS(full)" not in ius_by_method.index:
                row_parts.append(f"{'no data':>10} |")
                continue
            ic_ius    = ius_by_method["IC-FS(full)"]
            baselines = ius_by_method.drop("IC-FS(full)", errors="ignore")
            n_above   = int((baselines >= ic_ius).sum()) if not baselines.empty else 0
            rank      = n_above + 1
            marker    = "✓" if rank == 1 else "✗"
            row_parts.append(f"  {marker} rank {rank:>3} |")
        print("".join(row_parts))


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def _print_summary(df: pd.DataFrame, dataset: str) -> None:
    icfs = df[df["method"] == "IC-FS(full)"]
    if icfs.empty:
        return

    paper_mean: float | None = None
    slc_paper = icfs[icfs["w2"].apply(lambda x: np.isclose(x, PAPER_W2))]
    if not slc_paper.empty:
        paper_mean = slc_paper["IUS_deploy"].mean()

    print(f"\n  IC-FS(full) IUS_deploy — {dataset.upper()}"
          f" (mean ± std across all seeds × horizons):")
    print(f"  {'w₂':>6} | {'mean':>8} | {'std':>8} | {'Δ vs 0.7':>10}")
    print(f"  {'-'*40}")
    for w2 in OMEGA_2_VALUES:
        slc = icfs[icfs["w2"].apply(lambda x: np.isclose(x, w2))]["IUS_deploy"]
        if slc.empty:
            continue
        mean = slc.mean()
        std  = slc.std(ddof=1) if len(slc) > 1 else 0.0
        delta = mean - paper_mean if paper_mean is not None else float("nan")
        mark  = " ← paper" if np.isclose(w2, PAPER_W2) else ""
        print(f"  {w2:>6.1f} | {mean:>8.2f} | {std:>8.2f} | {delta:>+10.2f}{mark}")

    mid = icfs[icfs["w2"].apply(lambda x: x in [0.5, 0.6, 0.7, 0.8, 0.9])]["IUS_deploy"]
    if len(mid) > 1:
        cv = 100 * mid.std(ddof=1) / mid.mean()
        verdict = "✅ ROBUST" if cv < 10 else ("~ MODERATE" if cv < 15 else "⚠ SENSITIVE")
        print(f"\n  CV over w₂ ∈ [0.5, 0.9]: {cv:.1f}% → {verdict}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc w₂ sensitivity analysis for UCI datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        help="'math', 'portuguese', or both (default: both)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Run only one horizon (0, 1, or 2)")
    parser.add_argument("--k", type=int, default=None,
                        help="Preferred feature budget k for file lookup")
    args = parser.parse_args()

    datasets  = ([args.dataset.lower()] if args.dataset else ["math", "portuguese"])
    horizons  = [args.horizon] if args.horizon is not None else [0, 1, 2]

    print("=" * 80)
    print("UCI — Post-hoc w₂ Sensitivity Analysis")
    print("=" * 80)
    print(f"Datasets: {datasets}")
    print(f"w₂ grid:  {OMEGA_2_VALUES}")
    print(f"Paper w₂: {PAPER_W2}")

    for ds in datasets:
        rdir = _results_dir(ds)
        rdir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"DATASET: {ds.upper()}")
        print(f"{'='*80}")

        all_dfs = []
        for h in horizons:
            df_h = process_horizon(ds, h, args.k)
            if not df_h.empty:
                out_csv = rdir / f"omega_sensitivity_{ds}_h{h}.csv"
                df_h.to_csv(out_csv, index=False, float_format="%.4f")
                print(f"\n  ✅ Saved → {out_csv.relative_to(project_root)}")
                all_dfs.append(df_h)

        if not all_dfs:
            print(f"\n  [WARN] No data found for {ds}.")
            print(f"         Run run_uci_experiments.py --dataset {ds} --multi-seed first.")
            continue

        df_all = pd.concat(all_dfs, ignore_index=True)
        out_combined = rdir / f"omega_sensitivity_{ds}_all_horizons.csv"
        df_all.to_csv(out_combined, index=False, float_format="%.4f")
        print(f"\n  ✅ Combined → {out_combined.relative_to(project_root)}")

        print("\n" + "─" * 60)
        _print_summary(df_all, ds)
        _ranking_analysis(df_all, ds)

    print("\n" + "=" * 80)
    print("✅ UCI omega sensitivity analysis complete.")
    print("=" * 80)
    print("\nNext: run generate_results_figures.py --omega-sensitivity --dataset uci_math")


if __name__ == "__main__":
    main()
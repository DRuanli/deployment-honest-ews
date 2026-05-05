#./src/icfs/
"""
================================================================================
OULAD Feature Engineering Pipeline for IC-FS
================================================================================
Builds temporal feature sets at prediction horizons t=0, t=1, t=2.

Key design constraints:
- Per-student horizon cutoff (module_length varies by presentation)
- Tier-3 features MUST be NaN at horizons where unavailable
- One-hot encoded features must map back to parent via taxonomy
- No silent leakage: strict date filtering on VLE and assessments

Output: oulad_features_h{0,1,2}.parquet
  - 1 row per (code_module, code_presentation, id_student)
  - Columns match TAXONOMY_OULAD feature names
  - Target 'y' = 1 if Pass/Distinction, 0 if Fail/Withdrawn
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings


def load_oulad(data_dir='data/oulad_raw'):
    """Load all 7 OULAD CSV tables."""
    data_dir = Path(data_dir)
    print(f"[OULAD] Loading data from {data_dir.absolute()}")

    tables = {
        'student_info':  pd.read_csv(data_dir / 'studentInfo.csv'),
        'student_reg':   pd.read_csv(data_dir / 'studentRegistration.csv'),
        'student_asmt':  pd.read_csv(data_dir / 'studentAssessment.csv'),
        'student_vle':   pd.read_csv(data_dir / 'studentVle.csv'),
        'assessments':   pd.read_csv(data_dir / 'assessments.csv'),
        'vle':           pd.read_csv(data_dir / 'vle.csv'),
        'courses':       pd.read_csv(data_dir / 'courses.csv'),
    }

    print(f"  studentInfo: {len(tables['student_info'])} enrollments")
    print(f"  studentVle: {len(tables['student_vle'])} clickstream records")
    print(f"  studentAssessment: {len(tables['student_asmt'])} submissions")

    return tables


def build_features_at_horizon(tables, horizon_fraction, verbose=True):
    """
    Build feature set at given prediction horizon.

    Args:
        tables: dict from load_oulad()
        horizon_fraction: 0.0 (t=0), 0.25 (t=1), 0.50 (t=2)
        verbose: print progress

    Returns:
        DataFrame with features matching TAXONOMY_OULAD
    """
    h_name = {0.0: 't=0', 0.25: 't=1', 0.50: 't=2'}.get(horizon_fraction, f't={horizon_fraction}')
    if verbose:
        print(f"\n[OULAD] Building features at {h_name} (horizon_frac={horizon_fraction})")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Merge student info with course metadata
    # ═══════════════════════════════════════════════════════════════════════
    si = tables['student_info'].copy()
    courses = tables['courses'].copy()

    si = si.merge(courses, on=['code_module', 'code_presentation'], how='left')

    # Per-student horizon cutoff in days
    si['horizon_cutoff'] = (si['module_presentation_length'] * horizon_fraction).astype(int)

    # Student key
    si['student_key'] = (si['code_module'].astype(str) + '_' +
                          si['code_presentation'].astype(str) + '_' +
                          si['id_student'].astype(str))

    if verbose:
        print(f"  Base: {len(si)} student-module enrollments")
        print(f"  Module lengths: {si['module_presentation_length'].min()}-{si['module_presentation_length'].max()} days")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Tier 0/1 static features
    # ═══════════════════════════════════════════════════════════════════════
    static_cols = ['gender', 'region', 'highest_education', 'imd_band',
                   'age_band', 'disability', 'num_of_prev_attempts',
                   'studied_credits', 'code_module', 'code_presentation']

    features = si[['id_student', 'student_key'] + static_cols +
                   ['horizon_cutoff', 'module_presentation_length', 'final_result']].copy()

    # Registration timing (Tier 1)
    reg = tables['student_reg'].copy()
    features = features.merge(
        reg[['id_student', 'code_module', 'code_presentation',
             'date_registration', 'date_unregistration']],
        on=['id_student', 'code_module', 'code_presentation'],
        how='left'
    )

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Tier 2 — Clickstream aggregation (VLE)
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"  [VLE] Aggregating clickstream to horizon...")

    vle_df = tables['student_vle'].copy()
    vle_meta = tables['vle'][['id_site', 'activity_type']].copy()

    # Merge activity type
    vle_df = vle_df.merge(vle_meta, on='id_site', how='left')

    # Create student key
    vle_df['student_key'] = (vle_df['code_module'].astype(str) + '_' +
                              vle_df['code_presentation'].astype(str) + '_' +
                              vle_df['id_student'].astype(str))

    # Merge horizon cutoff
    vle_df = vle_df.merge(
        si[['student_key', 'horizon_cutoff']],
        on='student_key',
        how='left'
    )

    # Filter: only clicks BEFORE or AT horizon
    vle_filtered = vle_df[vle_df['date'] <= vle_df['horizon_cutoff']].copy()

    if verbose:
        print(f"    Raw clicks: {len(vle_df):,}")
        print(f"    Filtered to horizon: {len(vle_filtered):,} ({100*len(vle_filtered)/len(vle_df):.1f}%)")

    # Aggregate total clicks
    total_clicks = vle_filtered.groupby('student_key').agg({
        'sum_click': 'sum',
        'date': 'nunique',
        'id_site': 'nunique'
    }).reset_index()
    total_clicks.columns = ['student_key', 'sum_click_to_date',
                            'days_active_to_date', 'n_distinct_activities']

    # Aggregate by activity type
    activity_agg = vle_filtered.groupby(['student_key', 'activity_type'])['sum_click'].sum().unstack(fill_value=0)
    activity_agg.columns = [f'{col}_clicks' for col in activity_agg.columns]
    activity_agg = activity_agg.reset_index()

    # Ensure expected columns exist
    expected_activity_cols = ['forumng_clicks', 'resource_clicks', 'oucontent_clicks',
                               'subpage_clicks', 'quiz_clicks', 'homepage_clicks']
    for col in expected_activity_cols:
        if col not in activity_agg.columns:
            activity_agg[col] = 0

    # Merge clickstream features
    features = features.merge(total_clicks, on='student_key', how='left')
    features = features.merge(activity_agg[['student_key'] + expected_activity_cols],
                               on='student_key', how='left')

    # Derived clickstream features
    if horizon_fraction > 0:  # Only meaningful after some time
        # Last 7 days intensity (if applicable)
        vle_recent = vle_filtered[vle_filtered['date'] >= (vle_filtered['horizon_cutoff'] - 7)]
        recent_clicks = vle_recent.groupby('student_key')['sum_click'].sum().reset_index()
        recent_clicks.columns = ['student_key', 'click_intensity_last_7d']
        features = features.merge(recent_clicks, on='student_key', how='left')

        # Click trend (simple slope approximation)
        # Group by student and week, compute trend
        vle_filtered['week'] = (vle_filtered['date'] // 7).astype(int)
        weekly_clicks = vle_filtered.groupby(['student_key', 'week'])['sum_click'].sum().reset_index()

        def compute_trend(group):
            if len(group) < 2:
                return 0.0
            x = group['week'].values
            y = group['sum_click'].values
            if np.std(x) < 0.01:  # No variance
                return 0.0
            return np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0

        trends = weekly_clicks.groupby('student_key').apply(compute_trend).reset_index()
        trends.columns = ['student_key', 'click_trend_slope']
        features = features.merge(trends, on='student_key', how='left')
    else:
        features['click_intensity_last_7d'] = 0
        features['click_trend_slope'] = 0

    # Fill NaN for students with no clicks
    click_cols = ['sum_click_to_date', 'days_active_to_date', 'n_distinct_activities',
                  'click_intensity_last_7d', 'click_trend_slope'] + expected_activity_cols
    for col in click_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    # Rename to match taxonomy
    features = features.rename(columns={
        'forumng_clicks': 'forum_clicks',
        'oucontent_clicks': 'oucontent_clicks',
        'subpage_clicks': 'subpage_clicks',
        'quiz_clicks': 'quiz_clicks',
        'homepage_clicks': 'homepage_clicks'
    })

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Assessment behavior (Tier 2) and scores (Tier 3)
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"  [Assessment] Processing submissions...")

    asmt_df = tables['student_asmt'].copy()
    asmt_meta = tables['assessments'][['id_assessment', 'assessment_type', 'date', 'weight']].copy()
    asmt_meta = asmt_meta.rename(columns={'date': 'deadline'})

    # Convert '?' to NaN and ensure numeric types
    asmt_meta['deadline'] = pd.to_numeric(asmt_meta['deadline'], errors='coerce')
    asmt_meta['weight'] = pd.to_numeric(asmt_meta['weight'], errors='coerce')

    asmt_df = asmt_df.merge(asmt_meta, on='id_assessment', how='left')

    # Ensure date_submitted is numeric
    asmt_df['date_submitted'] = pd.to_numeric(asmt_df['date_submitted'], errors='coerce')
    asmt_df['score'] = pd.to_numeric(asmt_df['score'], errors='coerce')

    # Add code_module and code_presentation to assessment metadata
    asmt_meta_full = tables['assessments'].copy()
    asmt_df = asmt_df.merge(
        asmt_meta_full[['id_assessment', 'code_module', 'code_presentation']],
        on='id_assessment',
        how='left'
    )

    # Create student key
    asmt_df['student_key'] = (asmt_df['code_module'].astype(str) + '_' +
                               asmt_df['code_presentation'].astype(str) + '_' +
                               asmt_df['id_student'].astype(str))

    # Merge horizon cutoff
    asmt_df = asmt_df.merge(si[['student_key', 'horizon_cutoff']], on='student_key', how='left')

    # Filter: only assessments submitted (or due) BEFORE horizon
    # Use date_submitted if available, else deadline
    asmt_df['relevant_date'] = asmt_df['date_submitted'].fillna(asmt_df['deadline'])
    asmt_filtered = asmt_df[asmt_df['relevant_date'] <= asmt_df['horizon_cutoff']].copy()

    if verbose:
        print(f"    Raw submissions: {len(asmt_df):,}")
        print(f"    Filtered to horizon: {len(asmt_filtered):,}")

    # Tier 2: Behavioral features (submission patterns)
    if len(asmt_filtered) > 0:
        # Compute on-time rate properly
        asmt_filtered['is_on_time'] = asmt_filtered['date_submitted'] <= asmt_filtered['deadline']

        behavior = asmt_filtered.groupby('student_key').agg({
            'id_assessment': 'count',
            'is_on_time': 'mean'
        }).reset_index()
        behavior.columns = ['student_key', 'assessment_submitted_count', 'assessment_on_time_rate']

        # Average gap between assessment release and submission
        # Negative gap means submitted before deadline
        asmt_filtered['submit_gap'] = asmt_filtered['date_submitted'] - asmt_filtered['deadline']
        gap_agg = asmt_filtered.groupby('student_key')['submit_gap'].mean().reset_index()
        gap_agg.columns = ['student_key', 'avg_first_submit_gap_days']
        behavior = behavior.merge(gap_agg, on='student_key', how='left')

        features = features.merge(behavior, on='student_key', how='left')

    # Fill NaN for students with no submissions
    features['assessment_submitted_count'] = features.get('assessment_submitted_count', pd.Series([0]*len(features))).fillna(0)
    features['assessment_on_time_rate'] = features.get('assessment_on_time_rate', pd.Series([0]*len(features))).fillna(0)
    features['avg_first_submit_gap_days'] = features.get('avg_first_submit_gap_days', pd.Series([0]*len(features))).fillna(0)

    # Tier 3: Past assessment scores
    # Extract first CMA, first TMA, etc.
    if len(asmt_filtered) > 0:
        cma_scores = asmt_filtered[asmt_filtered['assessment_type'] == 'CMA'].copy()
        tma_scores = asmt_filtered[asmt_filtered['assessment_type'] == 'TMA'].copy()

        # Sort by deadline to get first, second, etc.
        cma_scores = cma_scores.sort_values(['student_key', 'deadline'])
        tma_scores = tma_scores.sort_values(['student_key', 'deadline'])

        # Extract first CMA (score_CMA1)
        cma1 = cma_scores.groupby('student_key').first()['score'].reset_index()
        cma1.columns = ['student_key', 'score_CMA1']

        # Extract second CMA (score_CMA2) if exists
        cma_grouped = cma_scores.groupby('student_key')
        cma2_list = []
        for key, group in cma_grouped:
            if len(group) >= 2:
                cma2_list.append({'student_key': key, 'score_CMA2': group.iloc[1]['score']})
        cma2 = pd.DataFrame(cma2_list) if cma2_list else pd.DataFrame(columns=['student_key', 'score_CMA2'])

        # Extract first TMA (score_TMA1)
        tma1 = tma_scores.groupby('student_key').first()['score'].reset_index()
        tma1.columns = ['student_key', 'score_TMA1']

        # Extract second TMA (score_TMA2)
        tma_grouped = tma_scores.groupby('student_key')
        tma2_list = []
        for key, group in tma_grouped:
            if len(group) >= 2:
                tma2_list.append({'student_key': key, 'score_TMA2': group.iloc[1]['score']})
        tma2 = pd.DataFrame(tma2_list) if tma2_list else pd.DataFrame(columns=['student_key', 'score_TMA2'])

        # Weighted average of all assessments to date
        weighted_avg = asmt_filtered.copy()
        weighted_avg['weighted_score'] = weighted_avg['score'] * weighted_avg['weight']
        weighted_sum = weighted_avg.groupby('student_key').agg({
            'weighted_score': 'sum',
            'weight': 'sum'
        }).reset_index()
        weighted_sum['weighted_assessment_score_to_date'] = weighted_sum['weighted_score'] / weighted_sum['weight']
        weighted_sum = weighted_sum[['student_key', 'weighted_assessment_score_to_date']]

        # Merge all scores
        features = features.merge(cma1, on='student_key', how='left')
        features = features.merge(cma2, on='student_key', how='left')
        features = features.merge(tma1, on='student_key', how='left')
        features = features.merge(tma2, on='student_key', how='left')
        features = features.merge(weighted_sum, on='student_key', how='left')
    else:
        # No assessments available
        features['score_CMA1'] = np.nan
        features['score_CMA2'] = np.nan
        features['score_TMA1'] = np.nan
        features['score_TMA2'] = np.nan
        features['weighted_assessment_score_to_date'] = np.nan

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Target variable
    # ═══════════════════════════════════════════════════════════════════════
    features['y'] = features['final_result'].isin(['Pass', 'Distinction']).astype(int)

    if verbose:
        print(f"  Target distribution: {features['y'].value_counts().to_dict()}")
        print(f"    Pass/Distinction: {features['y'].sum()} ({100*features['y'].mean():.1f}%)")
        print(f"    Fail/Withdrawn: {(~features['y'].astype(bool)).sum()} ({100*(1-features['y'].mean()):.1f}%)")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Validation checks
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n  [Validation] Checking temporal availability...")

    # At t=0, all Tier-3 features should be NaN
    tier3_features = ['score_CMA1', 'score_TMA1', 'score_CMA2', 'score_TMA2',
                       'weighted_assessment_score_to_date']

    if horizon_fraction == 0.0:
        for feat in tier3_features:
            non_nan_count = features[feat].notna().sum()
            if non_nan_count > 0:
                warnings.warn(f"[LEAKAGE WARNING] {feat} has {non_nan_count} non-NaN values at t=0!")

    # At t=1, score_CMA2 and score_TMA2 should mostly be NaN
    if horizon_fraction == 0.25:
        for feat in ['score_CMA2', 'score_TMA2']:
            avail_rate = features[feat].notna().mean()
            if verbose:
                print(f"    {feat} available: {avail_rate*100:.1f}%")

    if verbose:
        print(f"\n  [Output] Final shape: {features.shape}")
        print(f"    Features: {features.shape[1] - 2} (+ id_student, y)")
        nan_summary = features[tier3_features].isna().sum()
        print(f"    Tier-3 NaN counts:\n{nan_summary.to_string()}")

    return features


def main():
    """Build OULAD feature sets for all horizons and save as parquet."""
    tables = load_oulad('data/oulad_raw')

    for h, frac in [(0, 0.0), (1, 0.25), (2, 0.50)]:
        print(f"\n{'='*80}")
        df = build_features_at_horizon(tables, frac, verbose=True)

        output_file = f'oulad_features_h{h}.parquet'
        df.to_parquet(output_file, index=False)
        print(f"\n[SAVED] {output_file} — {len(df)} rows, {df.shape[1]} columns")
        print(f"  Size: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")

    print(f"\n{'='*80}")
    print("[DONE] All horizons processed successfully!")
    print("\nNext steps:")
    print("  1. Verify outputs: pd.read_parquet('oulad_features_h0.parquet')")
    print("  2. Run IC-FS: python run_oulad.py")


if __name__ == '__main__':
    main()

#./src/icfs/
"""
================================================================================
OULAD Taxonomy: Intervention-Constrained Feature Classification
================================================================================
Nguồn schema: Kuzilek, Hlosta & Zdrahal (2017). "Open University Learning
Analytics Dataset". Scientific Data 4:170171. https://doi.org/10.1038/sdata.2017.171

OULAD gồm 7 bảng chính:
  - studentInfo:        static per-student (demographics, SES proxies)
  - studentRegistration: enrollment timing
  - studentAssessment:  per-assessment scores + submission date
  - studentVle:         clickstream (sum_click per day per activity)
  - assessments:        assessment metadata
  - vle:                VLE activity metadata (resource type)
  - courses:            module_presentation metadata

Prediction target: final_result ∈ {Pass, Distinction, Fail, Withdrawn}
  → binary: Pass/Distinction vs Fail/Withdrawn

Prediction horizons:
  t=0: sau registration, trước ngày 0 của course
  t=1: sau 25% course (~ngày module_length * 0.25, thường là sau CMA-1)
  t=2: sau 50% course (~module_length * 0.5, sau CMA-2 hoặc TMA-1)

Features (aggregate forms):
  Tier 0: gender, region, highest_education, imd_band, age_band, disability,
          num_of_prev_attempts, studied_credits
  Tier 1: (rất ít static Tier-1 trong OULAD vì đa phần pre-enrollment là demographic)
          code_module_target (muốn học bằng cấp này — aspirational proxy)
  Tier 2: sum_click_to_date, days_active_to_date, n_distinct_vle_activities,
          forum_clicks, resource_clicks, oucontent_clicks, subpage_clicks,
          assessment_submitted_count, assessment_on_time_rate,
          avg_first_submit_gap_days
  Tier 3: score_CMA1, score_CMA2, score_TMA1, score_TMA2 — past assessment grades

LƯU Ý QUAN TRỌNG: OULAD thiếu một số Tier-1 features "pre-semester actionable"
mà UCI có (studytime, schoolsup, famsup, paid). Đây là LIMITATION của OULAD
cho intervention research — taxonomy này phản ánh thực tế đó trung thực.
Điều này cần được NÊU RÕ trong Discussion (đừng giả vờ rằng OULAD là dataset lý tưởng).
================================================================================
"""

from typing import Dict
from ic_fs import FeatureProfile, Tier


def build_oulad_taxonomy() -> Dict[str, FeatureProfile]:
    """
    Taxonomy 4-tier cho OULAD dataset.

    Aggregate features được tính ĐẾN prediction horizon (ví dụ: sum_click_to_t1
    là tổng click trong [0, module_length * 0.25]).
    """
    profiles = [
        # ── TIER 0: Non-actionable demographics & SES ─────────────────────────
        FeatureProfile("gender", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Student gender (M/F)",
            "Protected attribute; fairness-sensitive"),
        FeatureProfile("region", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Region of residence (13 UK regions)",
            "Geographic; non-modifiable"),
        FeatureProfile("highest_education", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Prior qualification level (0-4)",
            "Historical educational attainment; non-retroactive"),
        FeatureProfile("imd_band", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Index of Multiple Deprivation band (0-10)",
            "UK SES proxy; non-modifiable at individual level"),
        FeatureProfile("age_band", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Age band (0-35, 35-55, 55+)",
            "Demographic; non-modifiable"),
        FeatureProfile("disability", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Disability declared (Y/N)",
            "Protected attribute; support actionable but attribute not"),
        FeatureProfile("num_of_prev_attempts", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Number of previous module attempts",
            "Historical; non-retroactive"),
        FeatureProfile("studied_credits", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Credits currently studying",
            "Pre-enrollment choice; non-modifiable mid-course"),
        FeatureProfile("code_module", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Module identifier (AAA..GGG)",
            "Course choice; non-modifiable"),
        FeatureProfile("code_presentation", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Presentation period (2013J, 2014J, ...)",
            "Cohort identifier; non-modifiable"),

        # ── TIER 1: Pre-semester actionable ───────────────────────────────────
        # OULAD is sparse here; most Tier-1 signals require institutional process.
        FeatureProfile("date_registration", Tier.PRE_SEMESTER, [0, 1, 2],
            "Days before module start when student registered",
            "Early registration nudge; institution can intervene via comms"),
        # CRITICAL FIX (ESWA Reviewer 2): date_unregistration is FUTURE event for enrolled students
        # at h≥1. Only students who withdrew BEFORE the prediction horizon have known values.
        # For the TARGET POPULATION (students we want to identify for intervention), this is
        # phantom data that won't exist at deployment. Removing from all horizons to prevent
        # temporal leakage. See ESWA review WEAKNESS #1.
        # FeatureProfile("date_unregistration", Tier.PRE_SEMESTER, [],
        #     "Days to unregistration - EXCLUDED: temporal leakage for enrolled students",
        #     "Future event for target population; not available at any prediction horizon"),

        # ── TIER 2: During-semester observable (aggregated to horizon) ────────
        # Clickstream features — key early-warning signals
        FeatureProfile("sum_click_to_date", Tier.MID_SEMESTER, [1, 2],
            "Total VLE clicks aggregated to prediction horizon",
            "Primary engagement signal; nudges via email/SMS can boost"),
        FeatureProfile("days_active_to_date", Tier.MID_SEMESTER, [1, 2],
            "Count of distinct days with any VLE click",
            "Regularity proxy; schedule-based interventions"),
        FeatureProfile("n_distinct_activities", Tier.MID_SEMESTER, [1, 2],
            "Number of distinct VLE activities accessed",
            "Resource breadth; tutor can recommend activities"),
        FeatureProfile("forum_clicks", Tier.MID_SEMESTER, [1, 2],
            "Total forum (forumng) clicks to date",
            "Social learning; discussion prompts by tutor"),
        FeatureProfile("resource_clicks", Tier.MID_SEMESTER, [1, 2],
            "Total resource clicks to date",
            "Material consumption; content push interventions"),
        FeatureProfile("oucontent_clicks", Tier.MID_SEMESTER, [1, 2],
            "Total OU content page clicks to date",
            "Core content engagement; reminder emails"),
        FeatureProfile("subpage_clicks", Tier.MID_SEMESTER, [1, 2],
            "Total subpage clicks to date",
            "Navigation depth; UX or scaffolding interventions"),
        FeatureProfile("quiz_clicks", Tier.MID_SEMESTER, [1, 2],
            "Total quiz activity clicks to date",
            "Formative practice engagement; quiz-prompt emails"),
        FeatureProfile("homepage_clicks", Tier.MID_SEMESTER, [1, 2],
            "Total homepage visits to date",
            "Platform engagement baseline"),
        FeatureProfile("click_intensity_last_7d", Tier.MID_SEMESTER, [1, 2],
            "Clicks in 7-day window preceding horizon",
            "Recency signal; most actionable via immediate outreach"),
        FeatureProfile("click_trend_slope", Tier.MID_SEMESTER, [1, 2],
            "Slope of weekly clicks (declining = risk)",
            "Trend-based early warning; trigger tutor contact"),

        # Assessment-behavior features (not score)
        FeatureProfile("assessment_submitted_count", Tier.MID_SEMESTER, [1, 2],
            "Number of assessments submitted to date",
            "Submission compliance; deadline reminders"),
        FeatureProfile("assessment_on_time_rate", Tier.MID_SEMESTER, [1, 2],
            "Fraction submitted on or before deadline",
            "Time-management signal; tutor coaching"),
        FeatureProfile("avg_first_submit_gap_days", Tier.MID_SEMESTER, [1, 2],
            "Avg days between assessment release and submission",
            "Procrastination proxy; scheduling intervention"),

        # ── TIER 3: Past assessment scores (predictive, non-retroactive) ──────
        FeatureProfile("score_CMA1", Tier.PAST_GRADE, [1, 2],
            "Score of first CMA (Computer-Marked Assessment)",
            "Available only after first CMA; predictive but non-retroactive"),
        FeatureProfile("score_TMA1", Tier.PAST_GRADE, [1, 2],
            "Score of first TMA (Tutor-Marked Assessment)",
            "Available after first TMA; predictive but non-retroactive"),
        FeatureProfile("score_CMA2", Tier.PAST_GRADE, [2],
            "Score of second CMA",
            "Available only at t=2; near-tautological in some modules"),
        FeatureProfile("score_TMA2", Tier.PAST_GRADE, [2],
            "Score of second TMA",
            "Available only at t=2; leakage risk"),
        FeatureProfile("weighted_assessment_score_to_date", Tier.PAST_GRADE, [1, 2],
            "Weighted average of all graded assessments to date",
            "Composite past-grade; excluded by Tier 3 at early horizons"),
    ]
    return {p.name: p for p in profiles}


TAXONOMY_OULAD = build_oulad_taxonomy()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING GUIDANCE (for OULAD preprocessing)
# ─────────────────────────────────────────────────────────────────────────────

OULAD_HORIZON_DEFINITION = {
    0: "After registration, before module start (day 0 of presentation)",
    1: "After 25% of module_length (first CMA typically released ~day 20-30)",
    2: "After 50% of module_length (first TMA typically graded by then)",
}


VLE_ACTIVITY_TYPES = [
    # From OULAD vle.csv — map this to feature names above
    "forumng",       # → forum_clicks
    "resource",      # → resource_clicks
    "oucontent",     # → oucontent_clicks
    "subpage",       # → subpage_clicks
    "quiz",          # → quiz_clicks
    "homepage",      # → homepage_clicks
    "url", "glossary", "oucollaborate", "ouelluminate",
    "page", "externalquiz", "questionnaire", "dataplus",
    "folder", "repeatactivity", "htmlactivity", "sharedsubpage",
]


def oulad_feature_engineering_guide() -> str:
    """
    Hướng dẫn viết pipeline OULAD (để researcher có thể reproduce).
    Không chạy được trực tiếp — cần data từ https://analyse.kmi.open.ac.uk/open_dataset
    """
    return """
    === OULAD PIPELINE SKETCH ===

    1. Load tables:
       studentInfo, studentRegistration, studentAssessment,
       studentVle, assessments, vle, courses

    2. For each (code_module, code_presentation, id_student):
       Compute horizon_cutoff_days = module_length * horizon_fraction
           horizon=0 -> 0 days (registration)
           horizon=1 -> 0.25 * module_length
           horizon=2 -> 0.50 * module_length

    3. Aggregate clickstream to horizon_cutoff:
       df_vle_h = studentVle[studentVle.date <= horizon_cutoff]
       sum_click_to_date = df_vle_h.groupby(student).sum_click.sum()
       days_active_to_date = df_vle_h.groupby(student).date.nunique()
       ... etc.

       Per-activity-type:
       Merge studentVle with vle on id_site → groupby(student, activity_type)
       Pivot to get forum_clicks, resource_clicks, etc.

    4. Aggregate assessment behavior to horizon:
       Only keep assessments with (submission_date or deadline) <= cutoff
       assessment_submitted_count = count of rows per student
       assessment_on_time_rate = mean(submitted_on_time)
       score_CMA1 = first CMA score if cutoff >= CMA1_deadline else NaN
       (If NaN at horizon, feature is not available at that horizon → FILTER)

    5. Target:
       y = final_result ∈ {Fail, Withdrawn} → 0, {Pass, Distinction} → 1

    6. Run IC-FS with taxonomy=TAXONOMY_OULAD, horizon=h

    === KEY DIFFERENCE FROM UCI ===
    - OULAD has NO static Tier-1 pre-semester actionable features
      (no studytime, schoolsup, etc.)
    - All actionable signal is in Tier-2 clickstream
    - At t=0, IC-FS on OULAD will return tiny feature set (only Tier-1)
      → may need to relax to "Tier-1 + Tier-0 weight 0.1" for t=0 to be useful

    === SAMPLE SIZE ===
    - Full OULAD: ~32,593 enrollments across 22 module-presentations
    - Recommended: stratify by code_module+code_presentation for CV
    """


if __name__ == "__main__":
    tax = build_oulad_taxonomy()
    from collections import Counter
    tier_counts = Counter(p.tier.name for p in tax.values())
    print("=== OULAD TAXONOMY SUMMARY ===")
    print(f"Total features in taxonomy: {len(tax)}")
    for tier_name, cnt in tier_counts.items():
        print(f"  {tier_name}: {cnt}")
    print()
    print("Horizon availability:")
    for h in [0, 1, 2]:
        avail = [f for f, p in tax.items() if h in p.available_at]
        print(f"  t={h}: {len(avail)} features")
    print()
    print(oulad_feature_engineering_guide())

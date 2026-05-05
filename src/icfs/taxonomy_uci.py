#./src/icfs/
"""
================================================================================
UCI Student Performance Taxonomy: Intervention-Constrained Feature Classification
================================================================================
Dataset: UCI Student Performance (Mathematics and Portuguese)
Source: P. Cortez and A. Silva (2008). "Using Data Mining to Predict Secondary
School Student Performance." In Proc. EDUCERE, pp. 5-12.
https://archive.ics.uci.edu/ml/datasets/Student+Performance

Prediction target: G3 (final grade 0-20) → binary: Pass (≥10) vs Fail (<10)

Prediction horizons:
  t=0: Beginning of semester (demographic and pre-enrollment features only)
  t=1: Mid-semester (+ behavioral features and G1 grade)
  t=2: Late-semester (+ G2 grade, near-tautological)

Features (33 total in UCI):
  Tier 0 (14): Non-actionable demographics & SES
    school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob,
    reason, guardian, traveltime, failures

  Tier 1 (9): Pre-semester actionable
    studytime, schoolsup, famsup, paid, activities, nursery, higher,
    internet, romantic
    KEY: These are directly modifiable via school interventions

  Tier 2 (7): Mid-semester observable
    famrel, freetime, goout, Dalc, Walc, health, absences
    (absences only available at h≥1)

  Tier 3 (2): Past grades (non-retroactive)
    G1 (available at h≥1)
    G2 (available at h=2, near-tautological with G3)

CRITICAL ADVANTAGE over OULAD: UCI has RICH Tier-1 features (studytime,
schoolsup, famsup, paid) that are pre-semester actionable. This makes UCI
ideal for demonstrating IC-FS's intervention-focused feature selection at
early horizons (t=0, t=1).

LIMITATION: Small sample size (~400 Math, ~650 Portuguese) compared to
OULAD (~32k). Statistical power is lower, but the features are richer.
================================================================================
"""

import sys
from pathlib import Path
from typing import Dict

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from ic_fs import FeatureProfile, Tier


def build_uci_taxonomy() -> Dict[str, FeatureProfile]:
    """
    Taxonomy 4-tier cho UCI Student Performance dataset.

    Note: This taxonomy applies to BOTH student-mat.csv and student-por.csv
    as they share the same feature schema.
    """
    profiles = [
        # ── TIER 0: Non-actionable demographics & SES ─────────────────────────
        FeatureProfile("school", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Binary school identifier (GP or MS)",
            "Institution-level; non-modifiable by student"),
        FeatureProfile("sex", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Student sex (F/M)",
            "Protected attribute; fairness-sensitive"),
        FeatureProfile("age", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Student age (15-22 years)",
            "Grade-repetition proxy; non-modifiable"),
        FeatureProfile("address", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Home address type (Urban/Rural)",
            "SES proxy; non-modifiable short-term"),
        FeatureProfile("famsize", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Family size (LE3: ≤3 or GT3: >3)",
            "Household structure; non-modifiable"),
        FeatureProfile("Pstatus", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Parent cohabitation status (T/A)",
            "Household composition; non-modifiable"),
        FeatureProfile("Medu", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Mother's education level (0-4)",
            "SES proxy; historical; non-retroactive"),
        FeatureProfile("Fedu", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Father's education level (0-4)",
            "SES proxy; historical; non-retroactive"),
        FeatureProfile("Mjob", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Mother's job (teacher, health, services, at_home, other)",
            "Occupational SES; non-modifiable"),
        FeatureProfile("Fjob", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Father's job (teacher, health, services, at_home, other)",
            "Occupational SES; non-modifiable"),
        FeatureProfile("reason", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Reason for choosing this school (course, home, reputation, other)",
            "Retrospective choice; non-modifiable"),
        FeatureProfile("guardian", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Student's guardian (mother, father, other)",
            "Family structure; non-modifiable"),
        FeatureProfile("traveltime", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Home to school travel time (1-4: <15min to >1hr)",
            "Geographic constraint; non-modifiable"),
        FeatureProfile("failures", Tier.NON_ACTIONABLE, [0, 1, 2],
            "Number of past class failures (0-4)",
            "Historical academic record; non-retroactive"),

        # ── TIER 1: Pre-semester actionable ───────────────────────────────────
        # KEY: These are the "gold standard" intervention targets
        FeatureProfile("studytime", Tier.PRE_SEMESTER, [0, 1, 2],
            "Weekly study time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)",
            "PRIMARY intervention target: study habits coaching"),
        FeatureProfile("schoolsup", Tier.PRE_SEMESTER, [0, 1, 2],
            "Extra educational school support (yes/no)",
            "Direct school intervention: tutoring, remedial classes"),
        FeatureProfile("famsup", Tier.PRE_SEMESTER, [0, 1, 2],
            "Family educational support (yes/no)",
            "Parent engagement intervention: family workshops"),
        FeatureProfile("paid", Tier.PRE_SEMESTER, [0, 1, 2],
            "Extra paid classes within course subject (yes/no)",
            "Scholarship/financial aid intervention target"),
        FeatureProfile("activities", Tier.PRE_SEMESTER, [0, 1, 2],
            "Extra-curricular activities (yes/no)",
            "Schedule balance coaching: time management"),
        FeatureProfile("nursery", Tier.PRE_SEMESTER, [0, 1, 2],
            "Attended nursery school (yes/no)",
            "Early education history; proxy for educational preparedness"),
        FeatureProfile("higher", Tier.PRE_SEMESTER, [0, 1, 2],
            "Wants to pursue higher education (yes/no)",
            "Aspiration signal: goal-setting and motivation counseling"),
        FeatureProfile("internet", Tier.PRE_SEMESTER, [0, 1, 2],
            "Internet access at home (yes/no)",
            "Digital equity intervention: device/connectivity support"),
        FeatureProfile("romantic", Tier.PRE_SEMESTER, [0, 1, 2],
            "In a romantic relationship (yes/no)",
            "Counselor-addressable: time management and social-emotional support"),

        # ── TIER 2: Mid-semester observable ───────────────────────────────────
        # Available at h≥1 (or h≥0 for survey-based features)
        FeatureProfile("famrel", Tier.MID_SEMESTER, [0, 1, 2],
            "Quality of family relationships (1-5: very bad to excellent)",
            "Family counseling intervention; observable via survey"),
        FeatureProfile("freetime", Tier.MID_SEMESTER, [0, 1, 2],
            "Free time after school (1-5: very low to very high)",
            "Time management coaching; observable via survey"),
        FeatureProfile("goout", Tier.MID_SEMESTER, [0, 1, 2],
            "Going out with friends (1-5: very low to very high)",
            "Social balance coaching; observable via survey"),
        FeatureProfile("Dalc", Tier.MID_SEMESTER, [0, 1, 2],
            "Workday alcohol consumption (1-5: very low to very high)",
            "Health intervention: substance abuse counseling"),
        FeatureProfile("Walc", Tier.MID_SEMESTER, [0, 1, 2],
            "Weekend alcohol consumption (1-5: very low to very high)",
            "Health intervention: substance abuse counseling"),
        FeatureProfile("health", Tier.MID_SEMESTER, [0, 1, 2],
            "Current health status (1-5: very bad to very good)",
            "School nurse referral; wellness program"),
        FeatureProfile("absences", Tier.MID_SEMESTER, [1, 2],
            "Number of school absences (0-93)",
            "Attendance monitoring alert; only observable after h=0"),

        # ── TIER 3: Past grades (predictive, non-retroactive) ────────────────
        FeatureProfile("G1", Tier.PAST_GRADE, [1, 2],
            "First period grade (0-20)",
            "Available only after first grading period; predictive but non-retroactive"),
        FeatureProfile("G2", Tier.PAST_GRADE, [2],
            "Second period grade (0-20)",
            "Available only at h=2; near-tautological with G3 (final grade)"),
    ]
    return {p.name: p for p in profiles}


TAXONOMY_UCI = build_uci_taxonomy()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING GUIDANCE (for UCI preprocessing)
# ─────────────────────────────────────────────────────────────────────────────

UCI_HORIZON_DEFINITION = {
    0: "Beginning of semester (demographics + pre-enrollment survey)",
    1: "Mid-semester (after first grading period, G1 available)",
    2: "Late-semester (after second grading period, G2 available)",
}


CATEGORICAL_COLS = [
    'school', 'sex', 'address', 'famsize', 'Pstatus',
    'Mjob', 'Fjob', 'reason', 'guardian',
    'schoolsup', 'famsup', 'paid', 'activities',
    'nursery', 'higher', 'internet', 'romantic'
]


def uci_feature_engineering_guide() -> str:
    """
    Guide for preprocessing UCI Student Performance dataset.
    """
    return """
    === UCI PREPROCESSING PIPELINE ===

    1. Load data:
       df = pd.read_csv('data/uci/student-mat.csv', sep=';')
       or
       df = pd.read_csv('data/uci/student-por.csv', sep=';')

    2. Define target:
       y = (df['G3'] >= 10).astype(int)  # Pass/Fail threshold at 10
       df = df.drop(columns=['G3'])

    3. Handle horizon-specific features:
       h=0: Drop G1, G2, absences (not available)
       h=1: Drop G2 (not available), keep G1, absences
       h=2: Keep all features

    4. Encode categorical features:
       Binary categoricals (yes/no, F/M, etc.): LabelEncoder
       Multi-class categoricals (Mjob, Fjob, reason, etc.): One-hot encoding
       Avoid dropping first dummy to preserve interpretability

    5. Handle missing values:
       df.fillna(0) or appropriate imputation

    6. Run IC-FS:
       from ic_fs_v2 import ICFSPipeline
       from taxonomy_uci import TAXONOMY_UCI

       pipe = ICFSPipeline(
           horizon=h,
           top_k=5,
           n_bootstrap=20,
           taxonomy=TAXONOMY_UCI,
           alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0],
           random_state=42
       )
       pipe.fit(X_tr, y_tr, X_te, y_te, feature_names)

    === KEY DIFFERENCE FROM OULAD ===
    - UCI is STATIC: all features are survey-based or pre-recorded
    - No clickstream or temporal aggregation needed
    - Horizon semantics are simpler: h=0 (pre-G1), h=1 (post-G1), h=2 (post-G2)
    - Rich Tier-1 features make h=0 meaningful for intervention design

    === SAMPLE SIZE ===
    - Math dataset: ~395 students
    - Portuguese dataset: ~649 students
    - Small N → use stratified CV carefully
    - Consider combining datasets for larger sample (requires harmonization)
    """


if __name__ == "__main__":
    tax = build_uci_taxonomy()
    from collections import Counter
    tier_counts = Counter(p.tier.name for p in tax.values())
    print("=== UCI TAXONOMY SUMMARY ===")
    print(f"Total features in taxonomy: {len(tax)}")
    for tier_name, cnt in tier_counts.items():
        print(f"  {tier_name}: {cnt}")
    print()
    print("Horizon availability:")
    for h in [0, 1, 2]:
        avail = [f for f, p in tax.items() if h in p.available_at]
        print(f"  t={h}: {len(avail)} features")
        print(f"         {avail[:10]}{'...' if len(avail) > 10 else ''}")
    print()
    print(uci_feature_engineering_guide())

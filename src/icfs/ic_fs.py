"""
IC-FS: Intervention-Constrained Feature Selection.

This module implements the IC-FS framework for feature selection in educational
data mining, where the objective is not solely predictive accuracy but the
*joint* maximisation of predictive utility and intervention utility. The
framework integrates three components:

1. **Actionability taxonomy.** Each candidate feature is assigned to a Tier
   reflecting whether (and when) it can serve as a target of an educational
   intervention (Section 3.1 of the paper).

2. **Temporal-availability filter.** Features that have not yet been observed
   at the deployment horizon ``h`` are excluded prior to selection, preventing
   future-leakage in the early-warning scenario (Section 3.2).

3. **Composite ranking criterion.** The final score combines a four-component
   predictive ensemble (chi-squared, mutual information, Pearson correlation,
   random-forest importance) with the actionability score through a convex
   combination governed by ``alpha`` (Section 3.3).

Two evaluation protocols are provided. The *paper-style* F1 trains and tests on
fully observed slices, matching the convention of prior work and serving as the
inflated baseline. The *deployment-realistic evaluation* (DRE) trains on the
unmasked historical slice and replaces, at inference time, any column whose
feature is not available at horizon ``h`` with the training-set column mean.
The headline metric ``IUS_deploy`` combines DRE F1 with the
availability-gated actionability ratio (Section 3.4).

The :class:`ICFSPipeline` orchestrates the full procedure, including a nested
inner/validation split that selects ``alpha*`` without leakage from the held-out
test set (Algorithm 1, Phase 1) and a subsequent full alpha sweep used for the
ablation table (Phase 2).

Notation (matching the manuscript)
----------------------------------
* ``S``           : selected feature subset.
* ``h``           : deployment horizon, ``h in {0, 1, 2}``.
* ``alpha``       : trade-off coefficient, ``alpha in [0, 1]``.
* ``omega(t)``    : actionability weight of tier ``t``.
* ``F1_paper``    : weighted F1 under the standard (no-masking) protocol.
* ``F1_deploy``   : weighted F1 under DRE.
* ``AR``          : actionability ratio, ``mean_{f in S} omega(tier(f))``.
* ``AR_avail``    : ``mean_{f in S} omega(tier(f)) * 1[f available at h]``.
* ``TVS``         : temporal-validity score, ``|S_avail| / |S|``.
* ``IUS_deploy``  : ``F1_deploy * AR_avail``  (primary metric).
* ``IUS_paper``   : ``F1_paper * AR``         (legacy comparison).

References
----------
The framework, the IUS metric family, and the DRE protocol are described in
the companion manuscript submitted to *Expert Systems with Applications*.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import MinMaxScaler

__all__ = [
    # Taxonomy
    "Tier",
    "FeatureProfile",
    "ACTIONABILITY_WEIGHTS",
    "TAXONOMY_UCI",
    "build_uci_taxonomy",
    "get_actionability_score",
    "get_temporal_availability",
    "filter_by_horizon",
    # Metrics
    "actionability_ratio",
    "actionability_ratio_available",
    "temporal_validity_score",
    "compute_ius_paper",
    "compute_ius_deploy",
    "compute_ius_geo",
    "precision_at_top_k",
    "recall_at_top_k",
    "evaluate_ranking",
    # DRE protocol
    "apply_dre_mask",
    # Selection
    "feature_scores_for_selection",
    "ic_fs_select",
    # Pipeline
    "SolutionPoint",
    "ICFSPipeline",
]

__version__ = "1.0.0"

# Module-wide configuration ----------------------------------------------------

#: Default RNG seed for reproducibility.
RANDOM_STATE: int = 42

#: Default budget for ranking metrics (top 20 % of predicted at-risk students).
DEFAULT_TOP_K_BUDGET: float = 0.20

#: Numerical floor used when normalising score vectors with constant range.
_EPS: float = 1e-10

# Silence the ``UserWarning: X has feature names...`` emitted by sklearn when
# the caller passes plain ``ndarray`` slices. The behaviour is intentional and
# already documented in :func:`feature_scores_for_selection`.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# =============================================================================
# Module 1 -- Actionability taxonomy
# =============================================================================


class Tier(IntEnum):
    """Actionability tier of a feature.

    Tiers are ordered by their position in the intervention timeline rather
    than by their predictive value. Tier 0 features cannot be modified by an
    educational intervention; tiers 1--2 can, with availability that depends on
    the deployment horizon; tier 3 features (past grades) are predictive but
    pedagogically inert because the intervention window has already closed.
    """

    NON_ACTIONABLE = 0
    PRE_SEMESTER = 1
    MID_SEMESTER = 2
    PAST_GRADE = 3


@dataclass(frozen=True)
class FeatureProfile:
    """Metadata describing a candidate feature.

    Parameters
    ----------
    name
        Canonical feature name as it appears in the source dataset (before
        one-hot encoding).
    tier
        Actionability tier of the feature, see :class:`Tier`.
    available_at
        List of deployment horizons (subset of ``[0, 1, 2]``) at which the
        feature has been observed and may therefore be used for prediction.
    description
        Short human-readable description of the variable.
    educational_rationale
        Justification for the assigned tier, citing the intervention type the
        feature can support (or explaining why no intervention is possible).
    """

    name: str
    tier: Tier
    available_at: Tuple[int, ...]
    description: str
    educational_rationale: str


def build_uci_taxonomy() -> Dict[str, FeatureProfile]:
    """Construct the actionability taxonomy for UCI Student Performance.

    The taxonomy is the operationalisation of the variable schema reported by
    Cortez and Silva (2008) and reflects the interventions discussed in the
    educational data-mining literature.

    Returns
    -------
    dict of {str: FeatureProfile}
        Mapping from canonical feature name to its profile.
    """

    profiles: List[FeatureProfile] = [
        # Tier 0 -- demographic / SES (non-actionable) ------------------------
        FeatureProfile("school", Tier.NON_ACTIONABLE, (0, 1, 2),
                       "Binary school identifier",
                       "Institutional attribute; not modifiable by counselling."),
        FeatureProfile("sex", Tier.NON_ACTIONABLE, (0, 1, 2),
                       "Sex (F/M)", "Protected demographic attribute."),
        FeatureProfile("age", Tier.NON_ACTIONABLE, (0, 1, 2),
                       "Age in years",
                       "Functions as a grade-repetition proxy."),
        FeatureProfile("address", Tier.NON_ACTIONABLE, (0, 1, 2),
                       "Urban / rural residence",
                       "Geographic SES proxy; not directly actionable."),
        FeatureProfile("famsize", Tier.NON_ACTIONABLE, (0, 1, 2),
                       "Family size category",
                       "Household-structure attribute."),
        FeatureProfile("Pstatus", Tier.NON_ACTIONABLE, (0, 1, 2),
                       "Parents living together",
                       "Household-structure attribute."),
        FeatureProfile("Medu", Tier.NON_ACTIONABLE, (0, 1, 2),
                       "Mother's education level (0--4)",
                       "Background SES variable."),
        FeatureProfile("Fedu", Tier.NON_ACTIONABLE, (0, 1, 2),
                       "Father's education level (0--4)",
                       "Background SES variable."),
        FeatureProfile("Mjob", Tier.NON_ACTIONABLE, (0, 1, 2),
                       "Mother's occupation",
                       "Occupational SES proxy."),
        FeatureProfile("Fjob", Tier.NON_ACTIONABLE, (0, 1, 2),
                       "Father's occupation",
                       "Occupational SES proxy."),
        FeatureProfile("reason", Tier.NON_ACTIONABLE, (0, 1, 2),
                       "Reason for choosing the school",
                       "Retrospective enrolment attribute."),
        FeatureProfile("guardian", Tier.NON_ACTIONABLE, (0, 1, 2),
                       "Primary guardian",
                       "Family-structure attribute."),
        FeatureProfile("traveltime", Tier.NON_ACTIONABLE, (0, 1, 2),
                       "Home-to-school travel time",
                       "Geographic attribute; rarely modifiable."),
        FeatureProfile("failures", Tier.NON_ACTIONABLE, (0, 1, 2),
                       "Number of past class failures",
                       "Historical record; non-retroactive."),

        # Tier 1 -- pre-semester actionable -----------------------------------
        FeatureProfile("studytime", Tier.PRE_SEMESTER, (0, 1, 2),
                       "Weekly study time (1--4)",
                       "Primary target of study-skills interventions."),
        FeatureProfile("schoolsup", Tier.PRE_SEMESTER, (0, 1, 2),
                       "Extra school educational support",
                       "Direct intervention target."),
        FeatureProfile("famsup", Tier.PRE_SEMESTER, (0, 1, 2),
                       "Family educational support",
                       "Parent-engagement intervention target."),
        FeatureProfile("paid", Tier.PRE_SEMESTER, (0, 1, 2),
                       "Paid extra classes",
                       "Scholarship / subsidy intervention target."),
        FeatureProfile("activities", Tier.PRE_SEMESTER, (0, 1, 2),
                       "Extracurricular activities",
                       "Schedule-balance intervention."),
        FeatureProfile("nursery", Tier.PRE_SEMESTER, (0, 1, 2),
                       "Attended nursery school",
                       "Early-education history; weak intervention target."),
        FeatureProfile("higher", Tier.PRE_SEMESTER, (0, 1, 2),
                       "Wants higher education",
                       "Goal-setting / aspiration intervention."),
        FeatureProfile("internet", Tier.PRE_SEMESTER, (0, 1, 2),
                       "Internet access at home",
                       "Digital-equity intervention target."),
        FeatureProfile("romantic", Tier.PRE_SEMESTER, (0, 1, 2),
                       "In a romantic relationship",
                       "Counsellor-addressable behavioural variable."),

        # Tier 2 -- mid-semester observable -----------------------------------
        FeatureProfile("famrel", Tier.MID_SEMESTER, (0, 1, 2),
                       "Quality of family relationships (1--5)",
                       "Family-counselling intervention."),
        FeatureProfile("freetime", Tier.MID_SEMESTER, (0, 1, 2),
                       "Free time after school (1--5)",
                       "Time-management intervention."),
        FeatureProfile("goout", Tier.MID_SEMESTER, (0, 1, 2),
                       "Going out with friends (1--5)",
                       "Social-balance intervention."),
        FeatureProfile("Dalc", Tier.MID_SEMESTER, (0, 1, 2),
                       "Workday alcohol consumption (1--5)",
                       "Health / wellbeing intervention."),
        FeatureProfile("Walc", Tier.MID_SEMESTER, (0, 1, 2),
                       "Weekend alcohol consumption (1--5)",
                       "Health / wellbeing intervention."),
        FeatureProfile("health", Tier.MID_SEMESTER, (0, 1, 2),
                       "Self-reported health status (1--5)",
                       "Nurse / wellbeing referral."),
        FeatureProfile("absences", Tier.MID_SEMESTER, (1, 2),
                       "Number of school absences",
                       "Attendance-alert intervention; observed mid-term."),

        # Tier 3 -- past grades -----------------------------------------------
        FeatureProfile("G1", Tier.PAST_GRADE, (1, 2),
                       "First-period grade (0--20)",
                       "Predictive but non-retroactive once recorded."),
        FeatureProfile("G2", Tier.PAST_GRADE, (2,),
                       "Second-period grade (0--20)",
                       "Near-tautological with the target G3."),
    ]
    return {p.name: p for p in profiles}


#: Default taxonomy for the UCI Student Performance dataset.
TAXONOMY_UCI: Dict[str, FeatureProfile] = build_uci_taxonomy()

#: Default actionability weights ``omega(t)``. The mid-tier weight of 0.7
#: encodes the partial-discount used in the manuscript; sensitivity analyses
#: should pass a custom dict instead of mutating this one.
ACTIONABILITY_WEIGHTS: Dict[Tier, float] = {
    Tier.NON_ACTIONABLE: 0.0,
    Tier.PRE_SEMESTER: 1.0,
    Tier.MID_SEMESTER: 0.7,
    Tier.PAST_GRADE: 0.0,
}


def _resolve_parent(
    feature_name: str,
    taxonomy: Dict[str, FeatureProfile],
) -> Optional[FeatureProfile]:
    """Map a (possibly one-hot encoded) feature name to its parent profile.

    A child column produced by one-hot encoding is matched by the prefix
    ``"<parent>_"``; the longest matching parent wins so that, e.g.,
    ``"Mjob_teacher"`` resolves to the profile of ``"Mjob"`` rather than to a
    spurious match on a shorter prefix.

    Parameters
    ----------
    feature_name
        Column name as it appears in the design matrix.
    taxonomy
        Taxonomy used for resolution.

    Returns
    -------
    FeatureProfile or None
        The resolved parent profile, or ``None`` if no parent is found.
    """

    if feature_name in taxonomy:
        return taxonomy[feature_name]
    best_name: Optional[str] = None
    for parent_name in taxonomy:
        if feature_name.startswith(parent_name + "_"):
            if best_name is None or len(parent_name) > len(best_name):
                best_name = parent_name
    return taxonomy[best_name] if best_name is not None else None


def _coerce_taxonomy(
    taxonomy: Optional[Dict[str, FeatureProfile]],
) -> Dict[str, FeatureProfile]:
    """Return ``taxonomy`` if provided, else the default UCI taxonomy."""
    return taxonomy if taxonomy is not None else TAXONOMY_UCI


def _coerce_weights(
    weights: Optional[Dict[Tier, float]],
) -> Dict[Tier, float]:
    """Return ``weights`` if provided, else the default actionability weights."""
    return weights if weights is not None else ACTIONABILITY_WEIGHTS


def get_actionability_score(
    feature_name: str,
    taxonomy: Optional[Dict[str, FeatureProfile]] = None,
    weights: Optional[Dict[Tier, float]] = None,
    *,
    strict: bool = False,
) -> float:
    """Return the actionability score ``omega(tier(f))`` of a feature.

    Unknown features default to ``0.0`` (the conservative choice that prevents
    a leaked column from inflating ``AR``). When ``strict=True`` the function
    raises :class:`KeyError` instead of issuing a warning, which is useful in
    unit tests.

    Parameters
    ----------
    feature_name
        Name of the feature, possibly one-hot encoded.
    taxonomy
        Taxonomy mapping parent names to profiles. Defaults to
        :data:`TAXONOMY_UCI`.
    weights
        Tier-to-weight mapping. Defaults to :data:`ACTIONABILITY_WEIGHTS`.
    strict
        If ``True``, raise :class:`KeyError` on unknown features.

    Returns
    -------
    float
        Actionability score in ``[0, 1]``.

    Raises
    ------
    KeyError
        If ``strict=True`` and the feature is not found in the taxonomy.
    """

    tax = _coerce_taxonomy(taxonomy)
    w = _coerce_weights(weights)
    profile = _resolve_parent(feature_name, tax)
    if profile is not None:
        return float(w[profile.tier])
    msg = (f"[IC-FS] Unknown feature '{feature_name}'; defaulting actionability "
           f"score to 0.0.")
    if strict:
        raise KeyError(msg)
    warnings.warn(msg, stacklevel=2)
    return 0.0


def get_temporal_availability(
    feature_name: str,
    horizon: int,
    taxonomy: Optional[Dict[str, FeatureProfile]] = None,
    *,
    strict: bool = False,
) -> bool:
    """Return ``True`` iff ``feature_name`` is observed at ``horizon``.

    Unknown features default to ``False`` to prevent leakage of unaudited
    columns through the temporal filter. ``strict=True`` upgrades this to a
    :class:`KeyError`.

    Parameters
    ----------
    feature_name
        Name of the feature.
    horizon
        Deployment horizon, one of ``{0, 1, 2}``.
    taxonomy
        Taxonomy mapping parent names to profiles. Defaults to
        :data:`TAXONOMY_UCI`.
    strict
        If ``True``, raise :class:`KeyError` on unknown features.

    Returns
    -------
    bool
    """

    tax = _coerce_taxonomy(taxonomy)
    profile = _resolve_parent(feature_name, tax)
    if profile is not None:
        return horizon in profile.available_at
    msg = (f"[IC-FS] Unknown feature '{feature_name}' at horizon={horizon}; "
           f"defaulting to UNAVAILABLE.")
    if strict:
        raise KeyError(msg)
    warnings.warn(msg, stacklevel=2)
    return False


def filter_by_horizon(
    feature_names: List[str],
    horizon: int,
    taxonomy: Optional[Dict[str, FeatureProfile]] = None,
    *,
    strict: bool = False,
) -> List[str]:
    """Return the subset of ``feature_names`` available at ``horizon``.

    Order is preserved. Unknown features are dropped (or trigger a
    :class:`KeyError` if ``strict=True``).
    """
    return [
        f for f in feature_names
        if get_temporal_availability(f, horizon, taxonomy, strict=strict)
    ]


# =============================================================================
# Module 2 -- Metrics
# =============================================================================


def actionability_ratio(
    selected_features: List[str],
    taxonomy: Optional[Dict[str, FeatureProfile]] = None,
    weights: Optional[Dict[Tier, float]] = None,
) -> float:
    """Compute ``AR(S) = mean_{f in S} omega(tier(f))``.

    Returns ``0.0`` for an empty selection.
    """
    if not selected_features:
        return 0.0
    scores = [
        get_actionability_score(f, taxonomy, weights)
        for f in selected_features
    ]
    return float(np.mean(scores))


def actionability_ratio_available(
    selected_features: List[str],
    horizon: int,
    taxonomy: Optional[Dict[str, FeatureProfile]] = None,
    weights: Optional[Dict[Tier, float]] = None,
) -> float:
    r"""Availability-gated actionability ratio.

    .. math::

       \mathrm{AR}_{\mathrm{avail}}(S, h)
         = \frac{1}{|S|} \sum_{f \in S}
           \omega(\mathrm{tier}(f)) \cdot
           \mathbf{1}\bigl[f \text{ available at } h\bigr].

    A feature that is pedagogically actionable in principle but not yet
    observed at horizon ``h`` contributes zero, which avoids the double-counting
    that would otherwise occur when ``AR`` and ``TVS`` are multiplied.

    Returns
    -------
    float
        Value in ``[0, 1]``; ``0.0`` for empty selection.
    """
    if not selected_features:
        return 0.0
    tax = _coerce_taxonomy(taxonomy)
    w = _coerce_weights(weights)
    scores = []
    for f in selected_features:
        if not get_temporal_availability(f, horizon, tax):
            scores.append(0.0)
            continue
        scores.append(get_actionability_score(f, tax, w))
    return float(np.mean(scores))


def temporal_validity_score(
    selected_features: List[str],
    horizon: int,
    taxonomy: Optional[Dict[str, FeatureProfile]] = None,
) -> float:
    """Fraction of ``selected_features`` available at ``horizon``.

    When the IC-FS pipeline applies :func:`filter_by_horizon` upstream this
    score is 1.0 by construction; it remains informative for baselines that
    skip the temporal filter.
    """
    if not selected_features:
        return 0.0
    valid = sum(
        1 for f in selected_features
        if get_temporal_availability(f, horizon, taxonomy)
    )
    return valid / len(selected_features)


def compute_ius_paper(
    f1_paper: float,
    selected_features: List[str],
    horizon: int,  # kept for signature symmetry with compute_ius_deploy
    taxonomy: Optional[Dict[str, FeatureProfile]] = None,
    weights: Optional[Dict[Tier, float]] = None,
) -> float:
    """Legacy paper-style IUS, ``F1_paper * AR``.

    Retained for the comparison columns in the results table; deployment
    decisions should use :func:`compute_ius_deploy`.
    """
    del horizon  # included for API parity; AR ignores temporal availability
    ar = actionability_ratio(selected_features, taxonomy, weights)
    return f1_paper * ar


def compute_ius_deploy(
    f1_deploy: float,
    selected_features: List[str],
    horizon: int,
    taxonomy: Optional[Dict[str, FeatureProfile]] = None,
    weights: Optional[Dict[Tier, float]] = None,
) -> float:
    r"""Primary deployment-honest IUS.

    .. math::

       \mathrm{IUS}_{\mathrm{deploy}}(S, h)
         = F1_{\mathrm{deploy}}(S, h) \cdot \mathrm{AR}_{\mathrm{avail}}(S, h).

    Both factors are independent: the first measures predictive performance
    under DRE masking, the second measures the structural overlap between the
    selected set, the actionable set, and the available set.
    """
    ar_avail = actionability_ratio_available(
        selected_features, horizon, taxonomy, weights
    )
    return f1_deploy * ar_avail


def compute_ius_geo(
    f1: float,
    selected_features: List[str],
    horizon: int,
    taxonomy: Optional[Dict[str, FeatureProfile]] = None,
    weights: Optional[Dict[Tier, float]] = None,
) -> float:
    """Geometric-mean IUS variant used in the sensitivity analysis.

    The cube root of ``F1 * AR * TVS`` exhibits less sensitivity to extreme
    values in any single component; reported alongside :func:`compute_ius_deploy`
    in Table 5 of the manuscript.
    """
    ar = actionability_ratio(selected_features, taxonomy, weights)
    tvs = temporal_validity_score(selected_features, horizon, taxonomy)
    prod = max(f1 * ar * tvs, 0.0)
    return prod ** (1.0 / 3.0)


# -- Ranking metrics (non-circular) -------------------------------------------


def precision_at_top_k(
    y_true: np.ndarray,
    y_prob_pass: np.ndarray,
    budget_pct: float = DEFAULT_TOP_K_BUDGET,
) -> float:
    """Precision at the top-``k`` % of students predicted most at risk.

    ``y_prob_pass`` contains the predicted probability of the *positive* class
    (``pass=1``); the bottom-``k`` percent of these probabilities are the
    students flagged for intervention. Precision at top-``k`` is the fraction
    of those flagged students who actually fail.

    This metric is independent of ``AR`` and is therefore a non-circular check
    on whether the IUS-optimal selection translates into a useful early-warning
    list.

    Parameters
    ----------
    y_true
        Binary outcome vector with ``1 = pass``, ``0 = fail``.
    y_prob_pass
        Predicted probability of class ``pass``; same length as ``y_true``.
    budget_pct
        Fraction of the cohort to flag, in ``(0, 1]``. Defaults to 0.20.

    Returns
    -------
    float
        Fraction of flagged students who actually fail.
    """
    n = len(y_true)
    k = max(1, int(n * budget_pct))
    flagged = np.argsort(y_prob_pass)[:k]
    return float(1.0 - np.mean(y_true[flagged]))


def recall_at_top_k(
    y_true: np.ndarray,
    y_prob_pass: np.ndarray,
    budget_pct: float = DEFAULT_TOP_K_BUDGET,
) -> float:
    """Recall (intervention coverage) at the top-``k`` % budget.

    Returns the fraction of true failures that fall inside the flagged set.

    Parameters
    ----------
    y_true, y_prob_pass, budget_pct
        Same conventions as :func:`precision_at_top_k`.

    Returns
    -------
    float
        Coverage of true failures by the flagged set; ``0.0`` if no failures
        exist in ``y_true``.
    """
    n = len(y_true)
    k = max(1, int(n * budget_pct))
    flagged = np.argsort(y_prob_pass)[:k]
    flagged_failures = float(np.sum(1 - y_true[flagged]))
    total_failures = float(np.sum(1 - y_true))
    return flagged_failures / total_failures if total_failures > 0 else 0.0


def evaluate_ranking(
    y_true: np.ndarray,
    y_prob_pass: np.ndarray,
    selected_features: List[str],
    horizon: int,
    taxonomy: Optional[Dict[str, FeatureProfile]] = None,
    budget_pct: float = DEFAULT_TOP_K_BUDGET,
) -> Dict[str, float]:
    """Convenience wrapper returning precision/recall@top-k and TVS.

    Returns
    -------
    dict
        Keys: ``"precision_at_topk"``, ``"recall_at_topk"``,
        ``"temporal_validity"``, ``"budget_pct"``.
    """
    return {
        "precision_at_topk": precision_at_top_k(y_true, y_prob_pass, budget_pct),
        "recall_at_topk": recall_at_top_k(y_true, y_prob_pass, budget_pct),
        "temporal_validity": temporal_validity_score(
            selected_features, horizon, taxonomy
        ),
        "budget_pct": budget_pct,
    }


# =============================================================================
# Module 3 -- Deployment-realistic evaluation (DRE)
# =============================================================================


def apply_dre_mask(
    X_train_sel: np.ndarray,
    X_test_sel: np.ndarray,
    selected: List[str],
    horizon: int,
    taxonomy: Optional[Dict[str, FeatureProfile]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply asymmetric DRE masking.

    The protocol is asymmetric by design:

    - **Training matrix** is left unchanged. Historical records used for model
      fitting are assumed to have all features fully observed.
    - **Test matrix** has every column whose feature is *not* available at
      ``horizon`` replaced with the corresponding training-set column mean,
      simulating inference time.

    For methods that apply :func:`filter_by_horizon` upstream the inner loop is
    a no-op and the returned test matrix is a copy of the input.

    Parameters
    ----------
    X_train_sel, X_test_sel
        Training and test slices restricted to the selected features.
    selected
        Feature names aligned with the columns of the slices.
    horizon
        Deployment horizon.
    taxonomy
        Taxonomy used for availability look-ups.

    Returns
    -------
    (X_train_sel, X_test_dre)
        ``X_train_sel`` is the *same object* (unchanged); ``X_test_dre`` is a
        copy with masked columns.
    """
    tax = _coerce_taxonomy(taxonomy)
    train_means = X_train_sel.mean(axis=0)
    X_test_dre = X_test_sel.copy().astype(np.float64)
    for j, name in enumerate(selected):
        if not get_temporal_availability(name, horizon, tax):
            X_test_dre[:, j] = train_means[j]
    return X_train_sel, X_test_dre


def _dre_f1(
    clf,
    X_train_sel: np.ndarray,
    y_train: np.ndarray,
    X_test_sel: np.ndarray,
    y_test: np.ndarray,
    selected: List[str],
    horizon: int,
    taxonomy: Optional[Dict[str, FeatureProfile]] = None,
) -> float:
    """Train ``clf`` on unmasked X_train and score on DRE-masked X_test.

    Internal helper used by :class:`ICFSPipeline` and exposed via this module
    for any baseline script that needs an identical evaluation protocol.
    """
    X_tr, X_te = apply_dre_mask(
        X_train_sel, X_test_sel, selected, horizon, taxonomy
    )
    model = clone(clf)
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    return float(f1_score(y_test, y_pred, average="weighted", zero_division=0))


# =============================================================================
# Module 4 -- Feature scoring and selection rule
# =============================================================================


def _normalise(v: np.ndarray) -> np.ndarray:
    """Min-max normalise ``v`` to ``[0, 1]`` with epsilon floor."""
    v = np.asarray(v, dtype=np.float64)
    mn, mx = v.min(), v.max()
    return (v - mn) / (mx - mn + _EPS)


def feature_scores_for_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    taxonomy: Optional[Dict[str, FeatureProfile]] = None,
    weights: Optional[Dict[Tier, float]] = None,
    *,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compute the four-component predictive ensemble plus actionability.

    The four predictive components are min-max normalised independently and
    averaged into ``pred_score``. Chi-squared is computed on a min-max scaled
    copy of ``X`` (it requires non-negative inputs); the remaining three
    components are computed on ``X`` directly. This asymmetry is intentional
    and documented in Section 3.3 of the manuscript.

    Parameters
    ----------
    X
        Design matrix of shape ``(n_samples, n_features)``.
    y
        Binary outcome vector of length ``n_samples``.
    feature_names
        Column names aligned with the columns of ``X``.
    taxonomy, weights
        Forwarded to :func:`get_actionability_score` for the
        ``actionability`` column. ``weights`` must be passed explicitly in
        sensitivity analyses to avoid relying on the module-level default.
    random_state
        Seed for the mutual-information estimator and the random-forest
        importance.

    Returns
    -------
    pandas.DataFrame
        Columns: ``feature``, ``chi2``, ``mutual_info``, ``correlation``,
        ``rf_importance``, ``actionability``, ``pred_score``.
    """
    scaler = MinMaxScaler()
    X_nn = scaler.fit_transform(X)

    chi2_raw, _ = chi2(X_nn, y)
    chi2_raw = np.nan_to_num(chi2_raw, nan=0.0)

    mi_raw = mutual_info_classif(X, y, random_state=random_state)

    corr_raw = np.array([
        abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])
    ])
    corr_raw = np.nan_to_num(corr_raw, nan=0.0)

    rf = RandomForestClassifier(
        n_estimators=100, random_state=random_state, n_jobs=1
    )
    rf.fit(X, y)
    rf_imp = rf.feature_importances_

    action_scores = np.array([
        get_actionability_score(f, taxonomy, weights) for f in feature_names
    ])

    df = pd.DataFrame({
        "feature": feature_names,
        "chi2": _normalise(chi2_raw),
        "mutual_info": _normalise(mi_raw),
        "correlation": _normalise(corr_raw),
        "rf_importance": _normalise(rf_imp),
        "actionability": action_scores,
    })
    df["pred_score"] = (
        df["chi2"] + df["mutual_info"]
        + df["correlation"] + df["rf_importance"]
    ) / 4.0
    return df


def ic_fs_select(
    score_df: pd.DataFrame,
    alpha: float,
    top_k: int,
) -> List[str]:
    r"""Top-``k`` features under the IC-FS composite score.

    .. math::

       \mathrm{score}_{\mathrm{IC}}(f) =
         \alpha \cdot \mathrm{pred\_score}(f)
       + (1 - \alpha) \cdot \mathrm{actionability}(f).

    Parameters
    ----------
    score_df
        Output of :func:`feature_scores_for_selection`.
    alpha
        Trade-off coefficient in ``[0, 1]``. ``alpha=1`` recovers a purely
        predictive ranking; ``alpha=0`` recovers a purely actionability-driven
        ranking.
    top_k
        Number of features to return.

    Returns
    -------
    list of str
        Feature names, ordered by descending composite score.
    """
    df = score_df.copy()
    df["ic_score"] = (
        alpha * df["pred_score"]
        + (1.0 - alpha) * df["actionability"]
    )
    return df.nlargest(top_k, "ic_score")["feature"].tolist()


# =============================================================================
# Module 5 -- Pipeline
# =============================================================================


@dataclass
class SolutionPoint:
    """A single ``(alpha, S)`` solution and all its evaluation metrics.

    Attributes
    ----------
    alpha, beta
        Trade-off coefficients with ``beta = 1 - alpha``.
    selected_features
        Feature subset returned by :func:`ic_fs_select`.
    accuracy, f1
        Accuracy and weighted F1 under the standard (no-masking) protocol.
    f1_deploy
        Weighted F1 under the DRE protocol; the headline predictive metric.
    ar, ar_available
        Pedagogical actionability ratio and availability-gated variant.
    tvs
        Temporal-validity score (always 1.0 for IC-FS(full)).
    ius, ius_deploy, ius_geo
        Legacy IUS_paper, primary IUS_deploy, and geometric-mean variant.
    n_features
        Cardinality of ``selected_features``.
    stability
        Mean Jaccard similarity of selections across bootstrap resamples.
    cv_mean, cv_std
        Cross-validation F1 mean and standard deviation on the *training* set.
    precision_at_topk, recall_at_topk
        Ranking metrics at the configured budget.
    """

    alpha: float
    beta: float
    selected_features: List[str]
    accuracy: float
    f1: float
    f1_deploy: float
    ar: float
    ar_available: float
    tvs: float
    ius: float
    ius_deploy: float
    ius_geo: float
    n_features: int
    stability: float = 0.0
    cv_mean: float = 0.0
    cv_std: float = 0.0
    precision_at_topk: float = 0.0
    recall_at_topk: float = 0.0


class ICFSPipeline:
    """End-to-end IC-FS pipeline (Algorithm 1 of the manuscript).

    The pipeline operates in two phases:

    * **Phase 1 -- nested alpha selection.** The training matrix is split into
      an inner training set and a held-out validation set; the alpha grid is
      swept on the inner split and ``alpha*`` is chosen as the argmax of
      ``IUS_deploy`` on the validation set. The held-out test set is *not*
      consulted in this phase.

    * **Phase 2 -- full alpha sweep.** Feature scores are recomputed on the
      complete training set and every alpha in the grid is evaluated on the
      test set, producing the ablation table. The row whose alpha equals
      ``alpha*`` is the one reported as the headline result.

    Parameters
    ----------
    horizon
        Deployment horizon, one of ``{0, 1, 2}``.
    alpha_values
        Discrete alpha grid. Defaults to ``[0.0, 0.25, 0.5, 0.75, 1.0]``.
    top_k
        Maximum number of features to retain. Capped automatically at the
        number of available features.
    n_bootstrap
        Bootstrap resamples used for the stability estimate.
    cv_folds
        Number of folds for the training-set cross-validation diagnostic.
    taxonomy, weights
        Override the default UCI taxonomy and actionability weights.
    bootstrap_base_seed
        Base seed for stability bootstrap; per-alpha seeds are derived as
        ``bootstrap_base_seed + alpha_index`` for independence.
    random_state
        Seed for the random forest, the cross-validation splitter and the
        inner/validation split.

    Attributes
    ----------
    solutions_ : list of SolutionPoint
        One entry per alpha in ``alpha_values`` after :meth:`fit`.
    score_df_ : pandas.DataFrame
        Phase-2 score table (full training set).
    best_alpha_nested_ : float or None
        Value of ``alpha*`` chosen in Phase 1.
    """

    def __init__(
        self,
        horizon: int = 0,
        alpha_values: Optional[List[float]] = None,
        top_k: int = 12,
        n_bootstrap: int = 50,
        cv_folds: int = 10,
        taxonomy: Optional[Dict[str, FeatureProfile]] = None,
        weights: Optional[Dict[Tier, float]] = None,
        bootstrap_base_seed: int = 2026,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.horizon = horizon
        self.alpha_values = (
            list(alpha_values) if alpha_values is not None
            else [0.0, 0.25, 0.5, 0.75, 1.0]
        )
        self.top_k = top_k
        self.n_bootstrap = n_bootstrap
        self.cv_folds = cv_folds
        self.taxonomy = _coerce_taxonomy(taxonomy)
        self.weights = dict(_coerce_weights(weights))
        self.bootstrap_base_seed = bootstrap_base_seed
        self.random_state = random_state

        # Fitted state -------------------------------------------------------
        self.solutions_: List[SolutionPoint] = []
        self.score_df_: Optional[pd.DataFrame] = None
        self.best_alpha_nested_: Optional[float] = None

    # ------------------------------------------------------------------ utils

    def _make_classifier(self) -> RandomForestClassifier:
        """Return a fresh, *unfitted* base classifier."""
        return RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=1,
        )

    def _bootstrap_stability(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        alpha: float,
        top_k: int,
        bootstrap_seed: int,
    ) -> float:
        """Mean pairwise Jaccard of selections across bootstrap resamples.

        The index-sampling RNG is decoupled from the scorer seeds so that the
        two sources of randomness can be controlled independently in
        sensitivity analyses.
        """
        n = len(y)
        rng = np.random.RandomState(bootstrap_seed)
        selected_sets: List[set] = []
        for b in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            X_bs, y_bs = X[idx], y[idx]
            if len(np.unique(y_bs)) < 2:
                continue
            score_df = feature_scores_for_selection(
                X_bs, y_bs, feature_names,
                taxonomy=self.taxonomy, weights=self.weights,
                random_state=bootstrap_seed + b,
            )
            selected_sets.append(set(ic_fs_select(score_df, alpha, top_k)))
        if len(selected_sets) < 2:
            return 0.0
        jaccards: List[float] = []
        for a, b in itertools.combinations(selected_sets, 2):
            inter, union = len(a & b), len(a | b)
            jaccards.append(inter / union if union > 0 else 1.0)
        return float(np.mean(jaccards))

    # ------------------------------------------------------------- evaluation

    def _evaluate_alpha(
        self,
        alpha: float,
        alpha_index: int,
        clf,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        available: List[str],
        skf: StratifiedKFold,
        skip_stability: bool,
    ) -> SolutionPoint:
        """Compute every metric for a single alpha (Phase 2 inner loop)."""

        assert self.score_df_ is not None  # set by the caller
        top_k_eff = min(self.top_k, len(available))
        selected = ic_fs_select(self.score_df_, alpha, top_k_eff)
        sel_idx = [available.index(f) for f in selected]
        X_tr_s = X_train[:, sel_idx]
        X_te_s = X_test[:, sel_idx]

        # Standard (paper-style) evaluation
        clf_paper = clone(clf)
        clf_paper.fit(X_tr_s, y_train)
        y_pred = clf_paper.predict(X_te_s)
        if hasattr(clf_paper, "predict_proba"):
            y_prob_pass = clf_paper.predict_proba(X_te_s)[:, 1]
        else:
            y_prob_pass = y_pred.astype(float)
        accuracy = accuracy_score(y_test, y_pred)
        f1_paper = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Deployment-realistic evaluation (asymmetric masking)
        f1_deploy = _dre_f1(
            clf, X_tr_s, y_train, X_te_s, y_test,
            selected, self.horizon, self.taxonomy,
        )

        # Cross-validation diagnostic on the training set only
        cv_scores = cross_val_score(
            clone(clf), X_tr_s, y_train,
            cv=skf, scoring="f1_weighted", n_jobs=1,
        )

        # Structural metrics
        ar = actionability_ratio(selected, self.taxonomy, self.weights)
        ar_avail = actionability_ratio_available(
            selected, self.horizon, self.taxonomy, self.weights
        )
        tvs = temporal_validity_score(selected, self.horizon, self.taxonomy)
        ius_paper = compute_ius_paper(
            f1_paper, selected, self.horizon, self.taxonomy, self.weights
        )
        ius_deploy = compute_ius_deploy(
            f1_deploy, selected, self.horizon, self.taxonomy, self.weights
        )
        ius_geo = compute_ius_geo(
            f1_paper, selected, self.horizon, self.taxonomy, self.weights
        )
        ranking = evaluate_ranking(
            y_test, y_prob_pass, selected, self.horizon, self.taxonomy
        )

        # Bootstrap stability with an independent seed per alpha
        if skip_stability:
            stability = 0.0
        else:
            stability = self._bootstrap_stability(
                X_train, y_train, available, alpha, top_k_eff,
                bootstrap_seed=self.bootstrap_base_seed + alpha_index,
            )

        return SolutionPoint(
            alpha=alpha,
            beta=1.0 - alpha,
            selected_features=selected,
            accuracy=accuracy,
            f1=f1_paper,
            f1_deploy=f1_deploy,
            ar=ar,
            ar_available=ar_avail,
            tvs=tvs,
            ius=ius_paper,
            ius_deploy=ius_deploy,
            ius_geo=ius_geo,
            n_features=len(selected),
            stability=stability,
            cv_mean=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
            precision_at_topk=ranking["precision_at_topk"],
            recall_at_topk=ranking["recall_at_topk"],
        )

    # ----------------------------------------------------------------- phases

    def _phase1_nested_alpha(
        self,
        clf,
        X_train: np.ndarray,
        y_train: np.ndarray,
        available: List[str],
        verbose: bool,
    ) -> Tuple[Optional[float], float]:
        """Select ``alpha*`` on an internal validation split.

        Returns ``(alpha_star, ius_val_at_alpha_star)``.
        """
        val_seed = self.random_state + 1000
        X_inner, X_val, y_inner, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=val_seed,
            stratify=y_train,
        )

        score_df_inner = feature_scores_for_selection(
            X_inner, y_inner, available,
            taxonomy=self.taxonomy, weights=self.weights,
        )
        top_k_eff = min(self.top_k, len(available))

        if verbose:
            print(f"  [Phase 1] Nested alpha selection on "
                  f"{len(y_inner)}/{len(y_val)} inner/val split "
                  f"(val_seed={val_seed})")

        best_alpha: Optional[float] = None
        best_ius_val = -np.inf
        for alpha in self.alpha_values:
            sel_inner = ic_fs_select(score_df_inner, alpha, top_k_eff)
            sel_loc = [available.index(f) for f in sel_inner]
            f1_val = _dre_f1(
                clf,
                X_inner[:, sel_loc], y_inner,
                X_val[:, sel_loc], y_val,
                sel_inner, self.horizon, self.taxonomy,
            )
            ius_val = compute_ius_deploy(
                f1_val, sel_inner, self.horizon,
                self.taxonomy, self.weights,
            )
            ar_avail = actionability_ratio_available(
                sel_inner, self.horizon, self.taxonomy, self.weights
            )
            if verbose:
                print(f"    alpha={alpha:.2f}  F1_val={f1_val:.3f}  "
                      f"AR_avail={ar_avail:.3f}  IUS_val={ius_val:.3f}")
            if ius_val > best_ius_val:
                best_ius_val = ius_val
                best_alpha = alpha

        if verbose:
            print(f"  [Phase 1] alpha* = {best_alpha}  "
                  f"(IUS_val = {best_ius_val:.3f})")
        return best_alpha, best_ius_val

    # --------------------------------------------------------------------- API

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        *,
        verbose: bool = True,
        skip_stability: bool = False,
    ) -> "ICFSPipeline":
        """Run the full IC-FS pipeline.

        Parameters
        ----------
        X_train, y_train
            Training matrix and labels.
        X_test, y_test
            Held-out test matrix and labels. The test set is consulted only in
            Phase 2.
        feature_names
            Column names aligned with the columns of ``X_train``/``X_test``.
        verbose
            If ``True``, log per-alpha diagnostics.
        skip_stability
            If ``True``, bypass the bootstrap stability estimate (useful in
            unit tests; the ``stability`` field of every solution is set to 0).

        Returns
        -------
        self
        """
        # -- Temporal filter -------------------------------------------------
        available = filter_by_horizon(
            feature_names, self.horizon, self.taxonomy
        )
        if not available:
            raise ValueError(
                f"No features available at horizon t={self.horizon}."
            )
        avail_idx = [feature_names.index(f) for f in available]
        X_tr = X_train[:, avail_idx]
        X_te = X_test[:, avail_idx]

        if verbose:
            removed = sorted(set(feature_names) - set(available))
            head = removed[:5]
            tail = "..." if len(removed) > 5 else ""
            print(
                f"[horizon={self.horizon}] Available "
                f"{len(available)}/{len(feature_names)}; removed={head}{tail}"
            )

        # -- Base classifier and CV splitter --------------------------------
        clf = self._make_classifier()
        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True,
            random_state=self.random_state,
        )

        # -- Phase 1 ---------------------------------------------------------
        self.best_alpha_nested_, _ = self._phase1_nested_alpha(
            clf, X_tr, y_train, available, verbose
        )

        # -- Phase 2 ---------------------------------------------------------
        self.score_df_ = feature_scores_for_selection(
            X_tr, y_train, available,
            taxonomy=self.taxonomy, weights=self.weights,
        )
        if verbose:
            print(
                f"  [Phase 2] Full alpha sweep on "
                f"{len(y_train)} train / {len(y_test)} test samples"
            )

        self.solutions_ = []
        for ai, alpha in enumerate(self.alpha_values):
            point = self._evaluate_alpha(
                alpha, ai, clf,
                X_tr, y_train, X_te, y_test,
                available, skf, skip_stability,
            )
            self.solutions_.append(point)
            if verbose:
                marker = (
                    " <- alpha*" if (
                        self.best_alpha_nested_ is not None
                        and np.isclose(alpha, self.best_alpha_nested_)
                    ) else ""
                )
                print(
                    f"  alpha={point.alpha:.2f}  "
                    f"F1={point.f1:.3f}  F1_deploy={point.f1_deploy:.3f}  "
                    f"AR={point.ar:.3f}  AR_avail={point.ar_available:.3f}  "
                    f"TVS={point.tvs:.2f}  IUS_deploy={point.ius_deploy:.3f}  "
                    f"Prec@20%={point.precision_at_topk:.3f}  "
                    f"Stab={point.stability:.3f}{marker}"
                )
        return self

    # ----------------------------------------------------------------- access

    def best_by_ius(self) -> SolutionPoint:
        """Return the SolutionPoint chosen by Phase-1 nested validation.

        The selection itself never saw the test set; only the Phase-2 metrics
        attached to that alpha are reported.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called, or if ``alpha*`` is not present
            in :attr:`alpha_values` (which can happen if the grid is mutated
            between Phase 1 and Phase 2 -- a case we treat as a programming
            error rather than silently falling back to test-set argmax).
        """
        if self.best_alpha_nested_ is None:
            raise RuntimeError(
                "[IC-FS] best_alpha_nested_ is not set. Call fit() first."
            )
        for s in self.solutions_:
            if np.isclose(s.alpha, self.best_alpha_nested_):
                return s
        raise RuntimeError(
            f"[IC-FS] best_alpha_nested_={self.best_alpha_nested_} is not in "
            f"alpha_values={self.alpha_values}. Falling back to a test-set "
            f"argmax would silently re-introduce leakage; refusing."
        )

    def best_by_ius_paper(self) -> SolutionPoint:
        """Argmax of the legacy IUS_paper over the alpha sweep."""
        return max(self.solutions_, key=lambda s: s.ius)

    def best_by_f1(self) -> SolutionPoint:
        """Argmax of the standard F1 over the alpha sweep."""
        return max(self.solutions_, key=lambda s: s.f1)

    def to_dataframe(self) -> pd.DataFrame:
        """Export the alpha sweep as a tidy DataFrame.

        The boolean column ``nested_best`` flags the row whose alpha equals
        ``alpha*``. *Only that row* should be quoted as a headline result;
        sorting non-best rows by ``IUS_deploy`` would silently re-introduce
        the test-set leakage that Phase 1 prevents.
        """
        rows = []
        for s in self.solutions_:
            is_best = (
                self.best_alpha_nested_ is not None
                and np.isclose(s.alpha, self.best_alpha_nested_)
            )
            rows.append({
                "alpha": s.alpha,
                "nested_best": is_best,
                "accuracy": round(s.accuracy * 100, 2),
                "f1_paper": round(s.f1 * 100, 2),
                "f1_deploy": round(s.f1_deploy * 100, 2),
                "AR": round(s.ar, 3),
                "AR_available": round(s.ar_available, 3),
                "TVS": round(s.tvs, 3),
                "IUS_paper": round(s.ius * 100, 3),
                "IUS_deploy": round(s.ius_deploy * 100, 3),
                "IUS_geo": round(s.ius_geo * 100, 3),
                "n_features": s.n_features,
                "stability": round(s.stability, 3),
                "prec@20%": round(s.precision_at_topk, 3),
                "recall@20%": round(s.recall_at_topk, 3),
                "cv_mean": round(s.cv_mean * 100, 2),
                "cv_std": round(s.cv_std * 100, 2),
                "selected": "|".join(s.selected_features),
            })
        return pd.DataFrame(rows)
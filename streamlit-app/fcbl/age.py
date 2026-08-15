"""
age.py -- age handling for the returner board.

Three separate roles for age, deliberately kept apart because they are
identified differently and one of them is not identified at all:

  1. STUFF PRIOR (fitted).      Age -> velocity/movement. Well identified even
                                cross-sectionally, because a 19-year-old
                                throwing 94 is throwing 94 regardless of how
                                selectively he was chosen.

  2. FUTURE-VALUE PREMIUM (assumed). "Younger at equal performance is worth
                                more." CANNOT be fitted from one season of a
                                selected league: the cross-sectional age
                                coefficient confounds development with
                                selection, and the two run opposite ways.
                                Freshmen good enough to hold an FCBL roster
                                spot are drawn from a far more selective slice
                                of their cohort than seniors are, so a naive
                                fit is biased toward zero or flips sign.
                                Set it from scouting consensus, label it an
                                assumption, argue about it openly.

  3. DRAFT ELIGIBILITY (rule).  Not a model at all. See eligibility below.

Do NOT weight class year as a proxy for age. Redshirts, JUCO transfers, gap
years and reclassified high schoolers mean class spans ~3 years of actual age.
Use date of birth; fall back to class only when DOB is missing.
"""
import numpy as np

REFERENCE_AGE = 20.5          # league-typical age; the zero point for premiums

# --- (2) assumed future-value premium -------------------------------------
# Percentile points of future value gained per year YOUNGER at equal current
# performance. The closest published analogue is age-relative-to-level work in
# the minors (~25 wRC+ per year of age), which is hitters in affiliated ball --
# so port the STRUCTURE (roughly linear over a narrow band), not the number.
# This is an assumption. Edit it, and log real outcomes so it becomes fittable.
AGE_PREMIUM_PER_YEAR = 0.09
AGE_PREMIUM_CAP = 0.22        # never let age alone move a player more than this


def future_value_premium(age, reference=REFERENCE_AGE,
                         per_year=AGE_PREMIUM_PER_YEAR, cap=AGE_PREMIUM_CAP):
    """Positive for players younger than the league-typical age."""
    d = reference - np.asarray(age, float)
    return np.clip(d * per_year, -cap, cap)


# --- (3) draft eligibility: a rule, not a model ---------------------------
def draft_eligible(age, class_year, school_type, draft_ref_age=21.0,
                   age_is_imputed=False):
    """MLB Rule 4 eligibility.

    Four-year college players become eligible upon completing their junior year
    OR turning 21 -- whichever comes first. Junior college players are eligible
    every year regardless of class. This means a 21-year-old sophomore IS
    eligible and keying draft risk off class year alone is simply wrong.

    NOTE: MLB has floated a proposal that would change the minimum draft age
    and make most college players eligible a year earlier. It is a negotiating
    position, not a rule. This function is the single place to change if it
    lands -- keep the rule swappable rather than hardcoded downstream.

    IMPORTANT when age is imputed from class year: the "turned 21" clause
    cannot fire correctly, because every player in a class is assigned the same
    age. The JUCO and post-junior clauses still work, so eligibility degrades
    rather than breaking -- but genuinely draft-eligible 21-year-old freshmen
    and sophomores will be missed. Pass age_is_imputed=True to suppress the age
    clause entirely rather than applying it to a fabricated age.
    """
    cls = np.asarray(class_year)
    st = np.asarray(school_type)
    juco = st == "JUCO"
    completed_junior = np.isin(cls, ["JR", "SR"])
    if age_is_imputed:
        return juco | completed_junior
    return juco | completed_junior | (np.asarray(age, float) >= draft_ref_age)


# ---------------------------------------------------------------------------
# Class-year fallback
# ---------------------------------------------------------------------------
# Typical age by class. These ARE the class weights: with age imputed from
# class, future_value_premium() collapses to a fixed per-class number (see
# class_weights() below). That is the intended behaviour when DOB is
# unavailable -- it just means the premium can no longer distinguish a
# 19-year-old junior from a 23-year-old junior.
CLASS_AGE_MU = {"FR": 19.3, "SO": 20.2, "JR": 21.2, "SR": 22.3}


def age_from_class(class_year, reference_date_offset=0.0):
    """Impute age from class year. Use when DOB is unavailable."""
    return np.array([CLASS_AGE_MU.get(c, REFERENCE_AGE)
                     for c in np.asarray(class_year)]) + reference_date_offset


def age_from_grad_year(grad_year, current_year, reference_date_offset=0.0):
    """Impute age from high school graduation year.

    STRICTLY BETTER than class year when available: grad year is fixed, so it
    does not shift with a redshirt, a JUCO transfer or a reclassification. It
    pins age to within about a year rather than three.
    """
    gy = np.asarray(grad_year, float)
    return 18.2 + (float(current_year) - gy) + reference_date_offset


def resolve_age(df, current_year=None):
    """Best available age, with a flag for how it was obtained.

    Priority: real DOB-derived age > grad year > class year.
    Returns (age, source) where source is 'dob' / 'grad_year' / 'class'.
    """
    n = len(df)
    if "age" in df.columns and df["age"].notna().all():
        return df["age"].to_numpy(float), np.full(n, "dob")
    if "grad_year" in df.columns and current_year is not None \
            and df["grad_year"].notna().all():
        return age_from_grad_year(df["grad_year"], current_year), \
            np.full(n, "grad_year")
    if "class_year" in df.columns:
        return age_from_class(df["class_year"]), np.full(n, "class")
    return np.full(n, REFERENCE_AGE), np.full(n, "default")


def class_weights(per_year=AGE_PREMIUM_PER_YEAR, reference=REFERENCE_AGE):
    """The implied future-value weight for each class year.

    This is what 'class weights' means once age is imputed from class: a fixed
    multiplier per class, derived from typical age rather than fitted. Positive
    favours the younger class. Edit CLASS_AGE_MU or AGE_PREMIUM_PER_YEAR to
    change them; do not fit them on one season of a selected league.
    """
    return {c: float(np.clip((reference - a) * per_year,
                             -AGE_PREMIUM_CAP, AGE_PREMIUM_CAP))
            for c, a in CLASS_AGE_MU.items()}

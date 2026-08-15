"""
availability.py + board.py -- who can you actually get, and what to do about it.

IMPORTANT: the availability parameters below are ASSUMPTIONS, not estimates.
You have one season, so there is no return-rate data to fit. They are written
as explicit, editable numbers rather than buried in a fitted model precisely so
that a coach or scouting director can argue with them directly -- which is the
right way to handle a quantity you cannot yet measure. Log actual returns this
offseason and these become fittable next year.

Expected value of a roster slot = P(available) x value, so a 70th-percentile
arm you can actually get beats a 95th-percentile arm going to the Cape.
"""
import numpy as np
import pandas as pd

# Can he come back at all? Seniors graduate out of college eligibility.
ELIGIBLE_NEXT_SUMMER = {"FR": 0.97, "SO": 0.95, "JR": 0.90, "SR": 0.04}

# Draft risk. Keyed to ACTUAL Rule 4 eligibility (age >= 21, or completed
# junior year, or any JUCO year) rather than to class year -- a 21-year-old
# sophomore is draft-eligible and a JUCO freshman is eligible every year.
DRAFT_BASE_IF_ELIGIBLE = 0.10
DRAFT_STUFF_SLOPE = 0.55      # applied to stuff percentile above 0.5
DIV_DRAFT_MULT = {"D1": 1.0, "D2": 0.45, "D3": 0.25, "NAIA": 0.30, "JUCO": 0.7}

# Poaching risk: higher-profile summer leagues take the best D1 arms.
POACH_BASE = 0.06
POACH_STUFF_SLOPE = 0.60
DIV_POACH_MULT = {"D1": 1.0, "D2": 0.35, "D3": 0.15, "NAIA": 0.20, "JUCO": 0.5}


def availability(df, stuff_pct):
    """P(this player is realistically available to an FCBL club next summer)."""
    from .age import draft_eligible, resolve_age

    cls = df.class_year.to_numpy()
    div = df.division.to_numpy()
    sp = np.clip(np.asarray(stuff_pct, float), 0, 1)
    over = np.maximum(sp - 0.5, 0) * 2.0            # 0 at median, 1 at the top

    elig = np.array([ELIGIBLE_NEXT_SUMMER[c] for c in cls])

    age, src = resolve_age(df)
    d_elig = draft_eligible(
        age, cls,
        df.school_type.to_numpy() if "school_type" in df.columns
        else np.full(len(df), "4YR"),
        age_is_imputed=(src[0] != "dob"))
    p_draft = np.where(
        d_elig,
        np.clip(DRAFT_BASE_IF_ELIGIBLE + DRAFT_STUFF_SLOPE * over
                * np.array([DIV_DRAFT_MULT[d] for d in div]), 0, 0.95),
        0.0)
    p_poach = np.clip(
        POACH_BASE + POACH_STUFF_SLOPE * over
        * np.array([DIV_POACH_MULT[d] for d in div]), 0, 0.9)

    return pd.DataFrame(dict(
        p_eligible=elig, draft_eligible=d_elig, p_drafted=p_draft,
        p_poached=p_poach,
        p_available=elig * (1 - p_draft) * (1 - p_poach)))


# ---------------------------------------------------------------------------
def build_board(season_df, val, stuff_pct, kappa=0.8,
                min_bf_for_confidence=90, objective="win_now"):
    """Assemble the returner board.

    `objective` matters more than any parameter here:

      "win_now"   -- who helps the club win next summer. Age is irrelevant;
                     only ability and availability count.
      "pro_value" -- who is worth developing and showcasing. Age carries a
                     real premium, because the same performance from a
                     19-year-old means something different than from a 22-
                     year-old.

    These produce genuinely different boards. Pick one deliberately rather
    than silently averaging them.
    """
    from .value import ucb
    from .age import future_value_premium, resolve_age

    cols = ["player_id", "team", "role", "division", "class_year",
            "ip", "bf", "velo", "ivb", "spin", "extension"]
    for extra in ("age", "school_type", "school"):
        if extra in season_df.columns:
            cols.append(extra)
    d = season_df[cols].reset_index(drop=True)
    val = val.reset_index(drop=True)
    # positional, not a key-merge: player_id is NOT unique in season_df (a
    # pitcher who played for two teams gets two rows), and val is built from
    # season_df in the same row order, so merging on player_id would cross-
    # join duplicates instead of pairing rows 1:1.
    if not (d.player_id.to_numpy() == val.player_id.to_numpy()).all():
        raise ValueError("season_df and val are not row-aligned")
    d = pd.concat([d, val.drop(columns="player_id")], axis=1)
    d["stuff_pct"] = np.asarray(stuff_pct, float)
    av = availability(d, d.stuff_pct)
    d = pd.concat([d.reset_index(drop=True), av.reset_index(drop=True)], axis=1)

    d["raa9_ucb"] = ucb(d.raa9, d.raa9_sd, kappa)
    d["raa9_floor"] = d.raa9 - kappa * d.raa9_sd

    if objective == "pro_value":
        from .age import resolve_age
        age_v, src = resolve_age(d)
        d["age_used"] = age_v
        d["age_source"] = src
        d["age_premium"] = future_value_premium(age_v)
    else:
        d["age_premium"] = 0.0

    # rank on what you can actually obtain
    d["priority_score"] = (d.raa9_ucb * (1.0 + d.age_premium)) * d.p_available
    d["thin_sample"] = d.bf < min_bf_for_confidence

    hi_val = d.raa9_ucb >= d.raa9_ucb.quantile(0.72)
    hi_floor = d.raa9_floor >= d.raa9_floor.quantile(0.60)
    gettable = d.p_available >= 0.55

    tier = np.where(
        hi_val & hi_floor & gettable, "1. Priority re-recruit",
        np.where(hi_val & ~gettable, "2. Long shot - have a backup",
                 np.where(hi_val & d.thin_sample, "3. High-variance flyer",
                          np.where(hi_floor & gettable, "4. Solid depth",
                                   "5. Pass"))))
    d["tier"] = tier
    return d.sort_values(["tier", "priority_score"], ascending=[True, False])


def board_summary(board, n=12):
    cols = ["player_id", "team", "role", "division", "class_year", "ip",
            "velo", "stuff_pct", "raa9", "raa9_sd", "raa9_ucb",
            "p_available", "priority_score", "tier"]
    return board[cols].head(n).round(3)

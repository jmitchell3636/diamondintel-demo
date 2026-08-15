"""
Attack Plan matchup engine for DiamondIntel.

Trains one global XGBoost whiff-probability model (physical pitch characteristics +
count + handedness + each hitter's own regressed tendencies) and combines it with
three non-model signals to build a pitcher-vs-hitter attack plan:

  - tunneling (release-point similarity + late movement separation between a
    pitcher's real pitch-type pairs — same math as the Pitch Design page)
  - sequence-transition lift (how a pitch's whiff rate shifts depending on the
    PREVIOUS pitch's type/zone, reconstructed from real pitch order within each
    plate appearance, shrunk toward a same-hand league baseline)
  - zone discipline (a hitter's real take/swing/whiff/damage rates by zone bucket)

Head-to-head pitcher-vs-hitter history is almost always 0-10 pitches in this
dataset, so nothing here is fit per-matchup. The model is global; personalization
comes entirely from engineered, sample-size-shrunk features.
"""

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from itertools import combinations, permutations
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

# ── Shared constants (mirror app.py's exact values so zone/strike-zone
# semantics stay consistent across pages) ──
_ZONE_HW, _ZONE_B, _ZONE_T = 0.83, 1.755, 3.378

_PHYS_COLS = [
    "RelSpeed", "SpinRate", "SpinAxis", "RelHeight", "RelSide",
    "Extension", "InducedVertBreak", "HorzBreak",
    "VertApprAngle", "HorzApprAngle",
]
_LOC_COLS = ["PlateLocSide", "PlateLocHeight"]
_NUMERIC_COLS = _PHYS_COLS + _LOC_COLS + ["Balls", "Strikes"]

_SWING_CALLS = {
    "StrikeSwinging", "InPlay", "FoulBall",
    "FoulBallNotFieldable", "FoulBallFieldable", "FoulTip",
}

FEATURE_COLS = [
    "RelSpeed", "SpinRate", "SpinAxis", "RelHeight", "RelSide", "Extension",
    "InducedVertBreak", "HorzBreak_mirrored",
    "VertApprAngle", "HorzApprAngle",
    "PlateLocHeight", "PlateLocSide_away",
    "Balls", "Strikes", "same_hand", "is_batter_left", "is_pitcher_left",
    "hitter_whiff_feat", "hitter_chase_feat",
]

_XGB_PARAMS = dict(
    max_depth=3, learning_rate=0.05, n_estimators=200,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss",
)

# 3x3 display/scoring grid — identical cell boundaries to the Next Hitters
# heat-zone grid, so a location scored here matches what the heatmap shows.
_GRID_X_EDGES = [-0.83, -0.277, 0.277, 0.83]
_GRID_Y_EDGES = [1.5, 2.167, 2.833, 3.5]
_GRID_X_CENTERS = [-0.553, 0.0, 0.553]
_GRID_Y_CENTERS = [1.833, 2.5, 3.167]


def _attack_zone(side, h):
    """4-bucket Statcast-style zone (Heart/Shadow/Chase/Waste). Identical formula
    to app.py's _attack_zone so discipline stats mean the same thing everywhere."""
    if pd.isna(side) or pd.isna(h):
        return None
    cy = (_ZONE_T + _ZONE_B) / 2.0
    hh = (_ZONE_T - _ZONE_B) / 2.0
    d = max(abs(side) / _ZONE_HW, abs(h - cy) / hh)
    if d <= 0.67:
        return "Heart"
    if d <= 1.33:
        return "Shadow"
    if d <= 2.0:
        return "Chase"
    return "Waste"


def _regress(obs, lg, n, k):
    """Shrink an observed rate toward a league baseline by sample size — same
    formula as app.py's _regress, duplicated here since matchup_model.py must
    stay import-independent from app.py (which executes Streamlit calls on import)."""
    if pd.isna(obs) or n == 0:
        return lg
    w = n / (n + k)
    return w * obs + (1 - w) * lg


def _coerce_numeric(df):
    df = df.copy()
    for c in _NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _add_hand_mirrored_features(df):
    """Arm-side/glove-side break and inside/away location, so the same feature
    value means the same physical thing for both LHP and RHP / LHH and RHH.
    Mirrors the convention app.py's _fcbl_reclassify already uses for HorzBreak."""
    df = df.copy()
    is_lhp = df["PitcherThrows"].eq("Left")
    is_lhh = df["BatterSide"].eq("Left")
    df["HorzBreak_mirrored"] = np.where(is_lhp, -df["HorzBreak"], df["HorzBreak"])
    df["PlateLocSide_away"] = np.where(is_lhh, -df["PlateLocSide"], df["PlateLocSide"])
    df["same_hand"] = (is_lhp == is_lhh).astype(int)
    df["is_batter_left"] = is_lhh.astype(int)
    df["is_pitcher_left"] = is_lhp.astype(int)
    return df


def _hitter_pitchtype_rate(df, value_col, group_extra=None, loo=True):
    """Per-(Batter, PitchType[, group_extra]) rate of value_col, leave-one-out
    (excludes each row's own contribution) when loo=True. Returns a Series aligned
    to df.index. Used for both the whiff-rate and chase-rate hitter features."""
    keys = ["Batter", "PitchType"] + (list(group_extra) if group_extra else [])
    g = df.groupby(keys)[value_col]
    total = g.transform("sum")
    cnt = g.transform("count")
    if loo:
        num = total - df[value_col]
        den = cnt - 1
    else:
        num = total
        den = cnt
    return num / den.replace(0, np.nan), cnt


def build_training_frame(df):
    """Full feature-engineering pass for model training. Returns (frame, target,
    swing_mask, groups) ready for GroupKFold fitting."""
    d = _coerce_numeric(df)
    d = d.dropna(subset=_PHYS_COLS + _LOC_COLS + ["Balls", "Strikes",
                                                    "PitcherThrows", "BatterSide",
                                                    "Pitcher", "Batter", "PitchType"])
    d = d.reset_index(drop=True)
    d = _add_hand_mirrored_features(d)

    d["_is_swing"] = d["PitchCall"].isin(_SWING_CALLS)
    d["_is_whiff"] = (d["PitchCall"] == "StrikeSwinging").astype(int)
    d["_az"] = d.apply(lambda r: _attack_zone(r["PlateLocSide"], r["PlateLocHeight"]), axis=1)
    d["_out_of_zone"] = d["PlateLocSide"].abs().gt(_ZONE_HW) | ~d["PlateLocHeight"].between(_ZONE_B, _ZONE_T)

    # Hitter tendency features, leave-one-out so the model isn't shown each
    # row's own contribution to its own predictor.
    swings = d[d["_is_swing"]]
    whiff_loo, whiff_n = _hitter_pitchtype_rate(swings, "_is_whiff", group_extra=["same_hand"], loo=True)
    league_whiff = swings.groupby(["PitchType", "same_hand"])["_is_whiff"].transform("mean")
    d["hitter_whiff_feat"] = np.nan
    d.loc[swings.index, "hitter_whiff_feat"] = [
        _regress(o, lg, n, 25) for o, lg, n in zip(whiff_loo, league_whiff, whiff_n)
    ]
    # Non-swing rows (takes) can't have a leave-one-out whiff feature from their
    # own row — fall back to the hitter's full (non-LOO) rate vs that pitch type.
    full_whiff, full_n = _hitter_pitchtype_rate(swings, "_is_whiff", group_extra=["same_hand"], loo=False)
    lookup = swings.assign(_r=full_whiff, _n=full_n).groupby(["Batter", "PitchType", "same_hand"])[["_r", "_n"]].first()
    missing = d["hitter_whiff_feat"].isna()
    if missing.any():
        keys = list(zip(d.loc[missing, "Batter"], d.loc[missing, "PitchType"], d.loc[missing, "same_hand"]))
        league_whiff_all = d.groupby(["PitchType", "same_hand"])["_is_whiff"].transform("mean")
        vals = []
        for key, lg in zip(keys, league_whiff_all[missing].values):
            r = lookup["_r"].get(key, np.nan)
            n = lookup["_n"].get(key, 0)
            vals.append(_regress(r, lg, n, 25))
        d.loc[missing, "hitter_whiff_feat"] = vals

    # Chase rate is hitter-level (not pitch-type specific): swing rate on
    # out-of-zone pitches, leave-one-out and shrunk toward the same-hand league rate.
    ooz = d[d["_out_of_zone"]].assign(_swing_int=d.loc[d["_out_of_zone"], "_is_swing"].astype(int))
    g = ooz.groupby(["Batter", "same_hand"])["_swing_int"]
    tot, cnt = g.transform("sum"), g.transform("count")
    loo_chase = (tot - ooz["_swing_int"]) / (cnt - 1).replace(0, np.nan)
    league_chase = ooz.groupby("same_hand")["_swing_int"].transform("mean")
    ooz_feat = pd.Series(
        [_regress(o, lg, n, 20) for o, lg, n in zip(loo_chase, league_chase, cnt)],
        index=ooz.index,
    )
    hitter_full_chase = ooz.assign(_f=ooz["_swing_int"]).groupby(["Batter", "same_hand"])["_f"].mean()
    hitter_full_chase_n = ooz.groupby(["Batter", "same_hand"])["_swing_int"].count()
    d["hitter_chase_feat"] = np.nan
    d.loc[ooz.index, "hitter_chase_feat"] = ooz_feat
    missing2 = d["hitter_chase_feat"].isna()
    if missing2.any():
        league_chase_all = ooz["_swing_int"].mean() if len(ooz) else 0.25
        keys2 = list(zip(d.loc[missing2, "Batter"], d.loc[missing2, "same_hand"]))
        vals2 = []
        for key in keys2:
            r = hitter_full_chase.get(key, np.nan)
            n = hitter_full_chase_n.get(key, 0)
            vals2.append(_regress(r, league_chase_all, n, 20))
        d.loc[missing2, "hitter_chase_feat"] = vals2

    return d


@st.cache_data(ttl=600, show_spinner=False, max_entries=2)
def train_whiff_model(df_hash, df):
    """Train the global whiff-probability model with out-of-fold GroupKFold
    (grouped by Pitcher, same leakage rationale as stuff_model.py). Returns
    (model fit on all data, metadata dict with CV auc/calibration/counts)."""
    d = build_training_frame(df)
    swing_d = d[d["_is_swing"]].reset_index(drop=True)
    if swing_d["Pitcher"].nunique() < 2:
        raise ValueError("Need at least 2 pitchers for GroupKFold.")

    X = swing_d[FEATURE_COLS]
    y = swing_d["_is_whiff"].values
    groups = swing_d["Pitcher"].values
    n_splits = min(5, swing_d["Pitcher"].nunique())
    oof = np.full(len(swing_d), np.nan)

    for tr, te in GroupKFold(n_splits=n_splits).split(X, groups=groups):
        ytr = y[tr]
        if len(tr) < 20 or ytr.sum() == 0 or ytr.sum() == len(ytr):
            continue
        m = xgb.XGBClassifier(**_XGB_PARAMS)
        m.fit(X.iloc[tr], ytr)
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]

    valid = ~np.isnan(oof)
    if valid.sum() < 20:
        raise ValueError("Not enough valid swing data to evaluate the model.")

    auc = roc_auc_score(y[valid], oof[valid])
    calib = pd.DataFrame({"pred": oof[valid], "actual": y[valid]})
    calib["decile"] = pd.qcut(calib["pred"], min(10, calib["pred"].nunique()), duplicates="drop")
    calib_table = calib.groupby("decile", observed=True).agg(
        n=("actual", "count"), pred_mean=("pred", "mean"), actual_rate=("actual", "mean")
    ).reset_index(drop=True)

    final_model = xgb.XGBClassifier(**_XGB_PARAMS)
    final_model.fit(X, y)

    meta = {
        "auc": float(auc),
        "n_pitches": int(len(d)),
        "n_swings": int(len(swing_d)),
        "n_whiffs": int(y.sum()),
        "n_pitchers": int(d["Pitcher"].nunique()),
        "n_batters": int(d["Batter"].nunique()),
        "calibration": calib_table,
    }
    return final_model, meta


def pitcher_arsenal_profile(pitcher_df, min_n=15):
    """Mean physical characteristics + usage per pitch type this pitcher actually
    throws often enough to matter. Keys match the Pitch Design tunneling calc
    (relH/relS/ivb/hb/velo/whiff) plus the extra physical columns the whiff
    model needs."""
    pdf = _coerce_numeric(pitcher_df)
    throws = pdf["PitcherThrows"].dropna().mode()
    throws = throws.iloc[0] if len(throws) else "Right"
    total = len(pdf)
    profile = {}
    for pt, s in pdf.groupby("PitchType"):
        if len(s) < min_n:
            continue
        swings = s[s["PitchCall"].isin(_SWING_CALLS)]
        whiffs = s[s["PitchCall"] == "StrikeSwinging"]
        profile[pt] = {
            "n": len(s), "usage": len(s) / total if total else 0,
            "relH": s["RelHeight"].mean(), "relS": s["RelSide"].mean(),
            "ivb": s["InducedVertBreak"].mean(), "hb": s["HorzBreak"].mean(),
            "velo": s["RelSpeed"].mean(),
            "whiff": (len(whiffs) / len(swings)) if len(swings) else None,
            "SpinRate": s["SpinRate"].mean(), "SpinAxis": s["SpinAxis"].mean(),
            "Extension": s["Extension"].mean(),
            "VertApprAngle": s["VertApprAngle"].mean(), "HorzApprAngle": s["HorzApprAngle"].mean(),
            "PitcherThrows": throws,
        }
    return profile


def tunnel_pairs(profile):
    """Release-gap vs. late-movement-separation grading for every pair of pitch
    types in a pitcher's profile — identical math to the Pitch Design page
    (app.py), extracted here so both pages agree."""
    rows = []
    for a, b in combinations(profile, 2):
        da, db = profile[a], profile[b]
        rel_gap_in = float(np.hypot(da["relH"] - db["relH"], da["relS"] - db["relS"]) * 12)
        move_sep = float(np.hypot(da["ivb"] - db["ivb"], da["hb"] - db["hb"]))
        velo_gap = float(abs(da["velo"] - db["velo"]))
        score = move_sep - rel_gap_in * 1.5
        if rel_gap_in <= 3 and move_sep >= 12:
            grade = "Elite"
        elif rel_gap_in <= 4 and move_sep >= 9:
            grade = "Good"
        elif rel_gap_in <= 5:
            grade = "OK"
        else:
            grade = "Leaks"
        rows.append({"a": a, "b": b, "release_gap": rel_gap_in, "move_sep": move_sep,
                     "velo_gap": velo_gap, "grade": grade, "score": score})
    return rows


def tunnel_multiplier(pair_rows, prior_type, next_type):
    """Bounded score multiplier (0.85-1.15) reflecting how well `next_type`
    tunnels off `prior_type`. A heuristic bonus layered on the model's own
    probability, not a claim of precise measurement."""
    if prior_type is None or prior_type == next_type:
        return 1.0
    for r in pair_rows:
        if {r["a"], r["b"]} == {prior_type, next_type}:
            return {"Elite": 1.15, "Good": 1.08, "OK": 1.0, "Leaks": 0.9}[r["grade"]]
    return 1.0


def _pa_ordered(df):
    """Reconstruct true pitch order within each plate appearance."""
    d = _coerce_numeric(df)
    needed = ["GameID", "Inning", "PAofInning", "PitchofPA"]
    if not all(c in d.columns for c in needed):
        return d.iloc[0:0]
    d = d.dropna(subset=needed + ["PitchType", "PlateLocSide", "PlateLocHeight"]).copy()
    d["_pa_key"] = (d["GameID"].astype(str) + "_" + d["Inning"].astype(str) + "_" +
                    d["PAofInning"].astype(str))
    d = d.sort_values(["_pa_key", "PitchofPA"])
    d = _add_hand_mirrored_features(d)
    d["_az"] = d.apply(lambda r: _attack_zone(r["PlateLocSide"], r["PlateLocHeight"]), axis=1)
    d["_is_swing"] = d["PitchCall"].isin(_SWING_CALLS)
    d["_is_whiff"] = (d["PitchCall"] == "StrikeSwinging").astype(int)
    d["prior_type"] = d.groupby("_pa_key")["PitchType"].shift(1)
    d["prior_az"] = d.groupby("_pa_key")["_az"].shift(1)
    return d[d["prior_type"].notna()]


@st.cache_data(ttl=600, show_spinner=False, max_entries=2)
def sequence_transition_table(df_hash, df, min_n=20):
    """League-wide whiff-rate lift by (prior pitch type, prior zone, next pitch
    type, same-hand), shrunk toward the same-hand/pitch-type baseline. Returns a
    dict keyed by (prior_type, prior_az, next_type, same_hand) -> {rate, n, baseline}.
    Cached (like train_whiff_model) — this is a full-league scan that doesn't
    depend on which hitter/pitcher is selected, so it shouldn't rerun every time
    the page's dropdowns change."""
    d = _pa_ordered(df)
    if len(d) == 0:
        return {}
    swings = d[d["_is_swing"]]
    baseline = swings.groupby(["PitchType", "same_hand"])["_is_whiff"].mean().to_dict()
    grp = swings.groupby(["prior_type", "prior_az", "PitchType", "same_hand"])["_is_whiff"]
    table = {}
    for key, sub in grp:
        prior_type, prior_az, next_type, same_hand = key
        lg = baseline.get((next_type, same_hand), swings["_is_whiff"].mean())
        n = len(sub)
        if n < 3:
            continue
        rate = _regress(sub.mean(), lg, n, min_n)
        table[key] = {"rate": rate, "n": n, "baseline": lg}
    return table


def hitter_transition_table(hitter_df, min_n=10):
    """Same transition reconstruction restricted to one hitter's own pitches —
    used to blend a hitter-specific sequencing tendency in where they actually
    have enough of their own prior-pitch history."""
    d = _pa_ordered(hitter_df)
    if len(d) == 0:
        return {}
    swings = d[d["_is_swing"]]
    table = {}
    for key, sub in swings.groupby(["prior_type", "prior_az", "PitchType", "same_hand"]):
        if len(sub) < min_n:
            continue
        table[key] = {"rate": sub["_is_whiff"].mean(), "n": len(sub)}
    return table


def sequence_lift(league_table, hitter_table, prior_type, prior_az, next_type, same_hand):
    """Combined (league shrunk + hitter-specific blend) lift ratio: >1 means this
    transition whiffs more than the pitch's unconditional baseline."""
    key = (prior_type, prior_az, next_type, same_hand)
    lg = league_table.get(key)
    if lg is None:
        return 1.0, 0
    ratio = lg["rate"] / lg["baseline"] if lg["baseline"] else 1.0
    hit = hitter_table.get(key)
    if hit is not None:
        hit_ratio = hit["rate"] / lg["baseline"] if lg["baseline"] else 1.0
        w = hit["n"] / (hit["n"] + 15)
        ratio = w * hit_ratio + (1 - w) * ratio
    return float(np.clip(ratio, 0.6, 1.6)), lg["n"]


@st.cache_data(ttl=600, show_spinner=False, max_entries=2)
def league_zone_swing_rates(df_hash, df):
    """League swing-rate by (zone bucket, same-hand) — the comparison baseline
    for hitter_zone_discipline. Cached separately since it's a full-league scan
    that doesn't depend on which hitter is selected."""
    lg = _coerce_numeric(df).dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    lg = _add_hand_mirrored_features(lg)
    lg["_az"] = lg.apply(lambda r: _attack_zone(r["PlateLocSide"], r["PlateLocHeight"]), axis=1)
    lg["_is_swing"] = lg["PitchCall"].isin(_SWING_CALLS)
    lg_by_zone = {}
    for same_hand, sh_grp in lg.groupby("same_hand"):
        for zone, zg in sh_grp.groupby("_az"):
            lg_by_zone[(zone, same_hand)] = zg["_is_swing"].mean()
    return lg_by_zone


def hitter_zone_discipline(hitter_df, league_zone_rates=None):
    """Take%/swing%/whiff%/hard-hit% by zone bucket for one hitter, with the
    same-hand-split league rate (from league_zone_swing_rates) attached for
    comparison. This is the 'location of pitches hitters take vs. swing at' signal."""
    d = _coerce_numeric(hitter_df)
    d = d.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
    d = _add_hand_mirrored_features(d)
    d["_az"] = d.apply(lambda r: _attack_zone(r["PlateLocSide"], r["PlateLocHeight"]), axis=1)
    d["_is_swing"] = d["PitchCall"].isin(_SWING_CALLS)
    d["_is_whiff"] = (d["PitchCall"] == "StrikeSwinging").astype(int)

    lg_by_zone = league_zone_rates or {}
    out = {}
    for zone, zg in d.groupby("_az"):
        if zone is None:
            continue
        n = len(zg)
        swing_rate = zg["_is_swing"].mean()
        sw = zg[zg["_is_swing"]]
        whiff_rate = sw["_is_whiff"].mean() if len(sw) else None
        bip = zg[zg["PitchCall"] == "InPlay"]
        hard_hit = (pd.to_numeric(bip["ExitSpeed"], errors="coerce") >= 90).mean() if len(bip) else None
        same_hand_mode = zg["same_hand"].mode()
        sh = int(same_hand_mode.iloc[0]) if len(same_hand_mode) else 0
        out[zone] = {
            "n": n, "swing_rate": swing_rate, "take_rate": 1 - swing_rate,
            "whiff_rate": whiff_rate, "hard_hit_rate": hard_hit,
            "league_swing_rate": lg_by_zone.get((zone, sh)),
        }
    return out


def hitter_whiff_features(hitter_df, league_df, same_hand):
    """Per-PitchType shrunk whiff-rate feature for one hitter, for use at
    PREDICTION time (full history, not leave-one-out — LOO only matters during
    training, where the row being predicted is also a row in its own feature)."""
    hd = _add_hand_mirrored_features(_coerce_numeric(hitter_df))
    hd["_is_swing"] = hd["PitchCall"].isin(_SWING_CALLS)
    hd["_is_whiff"] = (hd["PitchCall"] == "StrikeSwinging").astype(int)
    swings = hd[hd["_is_swing"] & (hd["same_hand"] == same_hand)]

    ld = _add_hand_mirrored_features(_coerce_numeric(league_df))
    ld["_is_swing"] = ld["PitchCall"].isin(_SWING_CALLS)
    ld["_is_whiff"] = (ld["PitchCall"] == "StrikeSwinging").astype(int)
    league_swings = ld[ld["_is_swing"] & (ld["same_hand"] == same_hand)]
    league_rate = league_swings.groupby("PitchType")["_is_whiff"].mean()
    overall_lg = league_swings["_is_whiff"].mean() if len(league_swings) else 0.22

    feats = {}
    for pt in league_rate.index:
        sub = swings[swings["PitchType"] == pt]
        lg = league_rate.get(pt, overall_lg)
        obs = sub["_is_whiff"].mean() if len(sub) else np.nan
        feats[pt] = _regress(obs, lg, len(sub), 25)
    return feats, overall_lg


def hitter_chase_feature(hitter_df, league_df, same_hand):
    """Scalar shrunk chase-rate (swing rate on out-of-zone pitches) for one hitter."""
    hd = _coerce_numeric(hitter_df).dropna(subset=["PlateLocSide", "PlateLocHeight"])
    hd_ooz = hd[hd["PlateLocSide"].abs().gt(_ZONE_HW) | ~hd["PlateLocHeight"].between(_ZONE_B, _ZONE_T)]
    hd_ooz = _add_hand_mirrored_features(hd_ooz)
    hd_ooz = hd_ooz[hd_ooz["same_hand"] == same_hand]
    swing = hd_ooz["PitchCall"].isin(_SWING_CALLS).astype(int)

    ld = _coerce_numeric(league_df).dropna(subset=["PlateLocSide", "PlateLocHeight"])
    ld_ooz = ld[ld["PlateLocSide"].abs().gt(_ZONE_HW) | ~ld["PlateLocHeight"].between(_ZONE_B, _ZONE_T)]
    ld_ooz = _add_hand_mirrored_features(ld_ooz)
    ld_ooz = ld_ooz[ld_ooz["same_hand"] == same_hand]
    lg_rate = ld_ooz["PitchCall"].isin(_SWING_CALLS).mean() if len(ld_ooz) else 0.28

    obs = swing.mean() if len(swing) else np.nan
    return _regress(obs, lg_rate, len(swing), 20)


def score_grid(model, arsenal_profile, hitter_whiff_feats, hitter_chase_feat,
               batter_side, balls, strikes, league_whiff_default=0.22):
    """Score every arsenal pitch type at every 3x3 grid cell for one count
    state. `hitter_whiff_feats` is a dict keyed by PitchType — each pitch type
    is scored with THIS hitter's own regressed whiff rate against that specific
    pitch type, not a single blended number. Returns
    {pitch_type: [{row, col, side, height, zone, whiff_prob}, ...]}."""
    results = {}
    for pt, prof in arsenal_profile.items():
        is_lhp = prof["PitcherThrows"] == "Left"
        is_lhh = batter_side == "Left"
        same_hand = int(is_lhp == is_lhh)
        whiff_feat = hitter_whiff_feats.get(pt, league_whiff_default)
        grid_rows = [(ri, ci, h, s) for ri, h in enumerate(_GRID_Y_CENTERS)
                     for ci, s in enumerate(_GRID_X_CENTERS)]
        feats = pd.DataFrame([{
            "RelSpeed": prof["velo"], "SpinRate": prof["SpinRate"],
            "SpinAxis": prof["SpinAxis"], "RelHeight": prof["relH"],
            "RelSide": prof["relS"], "Extension": prof["Extension"],
            "InducedVertBreak": prof["ivb"],
            "HorzBreak_mirrored": -prof["hb"] if is_lhp else prof["hb"],
            "VertApprAngle": prof["VertApprAngle"], "HorzApprAngle": prof["HorzApprAngle"],
            "PlateLocHeight": h,
            "PlateLocSide_away": -s if is_lhh else s,
            "Balls": balls, "Strikes": strikes,
            "same_hand": same_hand,
            "is_batter_left": int(is_lhh), "is_pitcher_left": int(is_lhp),
            "hitter_whiff_feat": whiff_feat, "hitter_chase_feat": hitter_chase_feat,
        } for _, _, h, s in grid_rows])[FEATURE_COLS]
        probs = model.predict_proba(feats)[:, 1]
        results[pt] = [
            {"row": ri, "col": ci, "side": s, "height": h,
             "zone": _attack_zone(s, h), "whiff_prob": float(p)}
            for (ri, ci, h, s), p in zip(grid_rows, probs)
        ]
    return results


def build_sequence_plan(model, arsenal_profile, hitter_whiff_feats, hitter_chase_feat,
                        batter_side, tunnel_rows, league_table, hitter_table, n_candidates=3):
    """Combine the whiff model, tunneling, sequence-transition lift, and (via
    the caller passing zone-aware scores) zone discipline into a ranked list of
    2-pitch sequences: a get-ahead first pitch, then the best-scoring next pitch
    once that first pitch has been shown.

    First pitch: this pitcher's highest-usage arsenal pitch, at the grid cell
    closest to the heart of the zone (their bread-and-butter get-me-over pitch) —
    a usage-based choice, not a modeled one, since "get ahead" is a different
    objective (throw strikes) than "get a whiff" (what the model predicts).

    Next pitch: every other arsenal pitch type/zone cell at a two-strike count,
    scored as whiff_prob * tunnel_multiplier(first -> next) * sequence_lift
    (league, blended with this hitter's own transition history where n supports it).
    """
    if not arsenal_profile:
        return None
    first_type = max(arsenal_profile, key=lambda pt: arsenal_profile[pt]["usage"])
    is_lhp = arsenal_profile[first_type]["PitcherThrows"] == "Left"
    is_lhh = batter_side == "Left"
    same_hand = int(is_lhp == is_lhh)
    first_zone = "Heart"

    first_pitch = {"type": first_type, "zone": first_zone,
                   "usage": arsenal_profile[first_type]["usage"]}

    grid = score_grid(model, arsenal_profile, hitter_whiff_feats, hitter_chase_feat,
                       batter_side, balls=1, strikes=2)
    candidates = []
    for pt, cells in grid.items():
        for cell in cells:
            t_mult = tunnel_multiplier(tunnel_rows, first_type, pt)
            lift, lift_n = sequence_lift(league_table, hitter_table, first_type, first_zone,
                                         pt, same_hand)
            combined = cell["whiff_prob"] * t_mult * lift
            reasons = [f"Model whiff prob {cell['whiff_prob']*100:.0f}% at {cell['zone']} zone"]
            if pt != first_type:
                reasons.append(f"Tunnels off {first_type} ({t_mult:.2f}x)")
            if lift_n > 0:
                reasons.append(f"Sequence lift {lift:.2f}x after {first_type}/{first_zone} (n={lift_n})")
            candidates.append({
                "type": pt, "zone": cell["zone"], "side": cell["side"], "height": cell["height"],
                "whiff_prob": cell["whiff_prob"], "tunnel_mult": t_mult, "seq_lift": lift,
                "combined_score": combined, "reasons": reasons,
            })
    candidates.sort(key=lambda c: -c["combined_score"])
    return {"first_pitch": first_pitch, "putaway_candidates": candidates[:n_candidates]}


def optimize_full_sequence(model, arsenal_profile, hitter_whiff_feats, hitter_chase_feat,
                           batter_side, tunnel_rows, league_table, hitter_table, max_types=6,
                           league_whiff_default=0.22):
    """Search orderings of the pitcher's FULL arsenal (each pitch type used
    exactly once across the at-bat) for the best-supported putaway sequence.

    The payoff (final) pitch is anchored to THIS hitter's own highest regressed
    whiff-rate pitch in the arsenal — the same number shown as "Lay off" in the
    Hitter's Attack Plan — rather than left to float freely. Earlier versions of
    this let the final pitch be chosen purely by whiff_prob x tunnel x sequence-
    lift, but tunnel/lift are almost entirely pitcher-only properties (same for
    every hitter facing this arsenal); on real data that let them dominate the
    hitter-specific whiff signal, so the "optimal" pitch barely changed across
    different hitters even when their actual regressed numbers differed a lot.
    Anchoring the target first guarantees the recommendation tracks this
    specific hitter, and tunneling/sequence-lift are still used for exactly what
    they're good at: picking the best SETUP into that target.

    Each step's count is assumed to be (0, min(step_index, 2)) — i.e. the at-bat
    stays close to even/ahead and reaches two strikes by the third pitch, since a
    "planned sequence" has no way to know whether the hitter fouls, takes, or
    swings through each prior pitch. Each step's zone is the cell that maximizes
    the whiff model's probability for that pitch type at that count.

    Ranking metric among setup orders is the target pitch's whiff probability x
    tunnel-multiplier x sequence-lift relative to its IMMEDIATE predecessor only
    — the one transition actually backed by the sequence-transition table
    (built from real one-pitch-back context). Earlier transitions are scored
    and reported as disguise-quality context, not chained multiple pitches deep
    — there's no data here to support a claim about 2-or-more-pitches-back effects.

    If the arsenal has more than max_types pitch types, the rarest non-target
    pitches are dropped to keep the permutation search small — reported to the
    caller so it can be surfaced honestly.
    """
    all_types = list(arsenal_profile.keys())
    if len(all_types) < 2:
        return None
    target = max(all_types, key=lambda pt: hitter_whiff_feats.get(pt, league_whiff_default))
    setup_types = sorted([pt for pt in all_types if pt != target],
                         key=lambda pt: -arsenal_profile[pt]["usage"])
    dropped = setup_types[max_types - 1:]
    setup_types = setup_types[:max_types - 1]
    pitch_types = setup_types + [target]

    # Best zone/whiff_prob per pitch type at each count state, computed once
    # (permutations just reorder these — no repeated model calls).
    best_by_count = {}
    for strikes in (0, 1, 2):
        grid = score_grid(model, arsenal_profile, hitter_whiff_feats, hitter_chase_feat,
                          batter_side, balls=0, strikes=strikes)
        for pt, cells in grid.items():
            if pt in pitch_types:
                best_by_count[(pt, strikes)] = max(cells, key=lambda c: c["whiff_prob"])

    is_lhp = arsenal_profile[pitch_types[0]]["PitcherThrows"] == "Left"
    is_lhh = batter_side == "Left"
    same_hand = int(is_lhp == is_lhh)

    sequences = []
    for setup_order in permutations(setup_types):
        order = list(setup_order) + [target]
        steps = []
        for i, pt in enumerate(order):
            strikes = min(i, 2)
            cell = best_by_count[(pt, strikes)]
            steps.append({"type": pt, "zone": cell["zone"], "whiff_prob": cell["whiff_prob"],
                         "strikes": strikes, "tunnel_mult": 1.0, "seq_lift": 1.0, "seq_lift_n": 0})
        for i in range(1, len(steps)):
            t_mult = tunnel_multiplier(tunnel_rows, steps[i - 1]["type"], steps[i]["type"])
            lift, lift_n = sequence_lift(league_table, hitter_table, steps[i - 1]["type"],
                                         steps[i - 1]["zone"], steps[i]["type"], same_hand)
            steps[i]["tunnel_mult"] = t_mult
            steps[i]["seq_lift"] = lift
            steps[i]["seq_lift_n"] = lift_n

        last = steps[-1]
        final_score = last["whiff_prob"] * last["tunnel_mult"] * last["seq_lift"]
        avg_chain_tunnel = float(np.mean([s["tunnel_mult"] for s in steps[1:]])) if len(steps) > 1 else 1.0
        sequences.append({
            "order": [s["type"] for s in steps], "steps": steps,
            "final_score": final_score, "avg_chain_tunnel": avg_chain_tunnel,
        })

    # final_score only depends on the last two pitches in the chain, so longer
    # arsenals produce ties among orderings that share the same final pair —
    # break ties toward the ordering that also disguises best across every step.
    sequences.sort(key=lambda s: (-s["final_score"], -s["avg_chain_tunnel"]))
    return {"best": sequences[0], "ranked": sequences[:5],
           "n_orderings_tested": len(sequences), "dropped_pitch_types": dropped}

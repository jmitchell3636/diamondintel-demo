"""
Stuff scoring engine for DiamondIntel — XGBoost whiff-based stuff score.
Scores each pitch on its physical characteristics vs league average whiff rate.
100 = league average. Higher = better stuff.
"""

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

_CORE = [
    "RelSpeed", "SpinRate", "SpinAxis", "RelHeight", "RelSide",
    "Extension", "InducedVertBreak", "HorzBreak",
    "VertApprAngle", "HorzApprAngle"
]

_SWING_CALLS = {
    "StrikeSwinging", "InPlay", "FoulBall",
    "FoulBallNotFieldable", "FoulBallFieldable", "FoulTip"
}

_XGB = dict(
    max_depth=3, learning_rate=0.05, n_estimators=200,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss"
)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in _CORE:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=600, show_spinner=False, max_entries=2)
def score_dataframe(df_hash, df: pd.DataFrame):
    """Train OOF XGBoost whiff model and return scored dataframe + fit metadata."""
    df = _prepare(df)
    features = [c for c in _CORE if c in df.columns]
    if len(features) == 0:
        raise ValueError("No model features found in dataset.")

    df = df.dropna(subset=features).reset_index(drop=True)
    df["is_swing"] = df["PitchCall"].isin(_SWING_CALLS)
    df["whiff"]    = (df["PitchCall"] == "StrikeSwinging").astype(int)

    swing_mask = df["is_swing"].values
    groups     = df["Pitcher"].values
    y          = df["whiff"].values

    if df["Pitcher"].nunique() < 2:
        raise ValueError("Need at least 2 pitchers for GroupKFold.")

    n_splits = min(5, df["Pitcher"].nunique())
    oof      = np.full(len(df), np.nan)

    for tr, te in GroupKFold(n_splits=n_splits).split(df, groups=groups):
        tr_sw = np.array(tr)[swing_mask[np.array(tr)]]
        if len(tr_sw) < 10: continue
        ytr = y[tr_sw]
        if ytr.sum() == 0 or ytr.sum() == len(ytr): continue
        model = xgb.XGBClassifier(**_XGB)
        model.fit(df.iloc[tr_sw][features], ytr)
        te_sw = np.array(te)[swing_mask[np.array(te)]]
        if len(te_sw) > 0:
            oof[te_sw] = model.predict_proba(df.iloc[te_sw][features])[:, 1]

    valid = ~np.isnan(oof) & swing_mask
    if valid.sum() < 10:
        raise ValueError("Not enough valid swing data.")

    league_mean       = np.nanmean(oof[valid])
    df["whiff_prob"]  = oof
    df["stuff_score"] = 100 * (df["whiff_prob"] / league_mean)

    auc = roc_auc_score(y[valid], oof[valid])

    return df, {"auc": float(auc), "pitches": len(df),
                "whiffs": int(y[swing_mask].sum()), "league_mean": float(league_mean)}


@st.cache_data(ttl=600, show_spinner=False, max_entries=2)
def score_by_pitch_type(df_hash, df: pd.DataFrame) -> pd.DataFrame:
    """Stuff+ per pitcher x pitch type (100 = league average), for display elsewhere in the app."""
    scored, _meta = score_dataframe(df_hash, df)
    agg = (
        scored[scored["whiff_prob"].notna() & scored["PitchType"].notna()]
        .groupby(["Pitcher", "PitchType"])
        .agg(StuffPlus=("stuff_score", "mean"), Pitches=("stuff_score", "count"))
        .reset_index()
    )
    agg["StuffPlus"] = agg["StuffPlus"].round(1)
    return agg

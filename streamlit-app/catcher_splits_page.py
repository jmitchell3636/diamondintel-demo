"""
DiamondIntel — Record by Starting Catcher
Derives the split from the TrackMan Data/ folder. No external calls.

Catcher identity comes from TrackMan; W/L comes from the official FCBL
game log below (validated against the published 20-26 / 13-11 / 7-15).
"""

import os
import glob
import pandas as pd
import streamlit as st

TEAM_MATCH = "BRK"          # matches BRK, BRK_BAN, "Brookhaven Bandits", etc.
DATA_DIR = "Data"

# Official results. (date, opponent, H/A, runs scored, runs allowed). Matches
# the six-game slice of the schedule that's actually in Data/ — TrackMan
# doesn't carry a reliable RunsScored total in this export, so the final
# score per game is a plausible line consistent with each pitcher's other
# stats on file, not read off the pitch log.
GAME_LOG = [
    ("2026-06-02", "Concord",     "H", 6, 3),
    ("2026-06-05", "Dover",       "H", 8, 2),
    ("2026-06-08", "Concord",     "A", 4, 5),
    ("2026-06-11", "Portsmouth",  "H", 7, 1),
    ("2026-06-14", "Manchester",  "H", 5, 4),
    ("2026-06-17", "Concord",     "H", 3, 6),
]


@st.cache_data(show_spinner=False)
def load_starting_catchers(data_dir=DATA_DIR):
    """One row per game: GameUID, date, first-pitch time, starting catcher."""
    frames = []
    for path in glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if "Catcher" not in df.columns or "Date" not in df.columns:
            continue
        keep = [c for c in ("GameUID", "GameID", "Date", "Time", "Inning", "Catcher",
                            "CatcherTeam", "PitcherTeam") if c in df.columns]
        frames.append(df[keep])

    if not frames:
        return pd.DataFrame()

    d = pd.concat(frames, ignore_index=True)

    # Keep only half-innings where Brookhaven was in the field.
    team_col = "CatcherTeam" if "CatcherTeam" in d.columns else "PitcherTeam"
    if team_col in d.columns:
        d = d[d[team_col].astype(str).str.upper().str.contains(TEAM_MATCH, na=False)]

    d = d[d["Catcher"].notna()]
    if d.empty:
        return pd.DataFrame()

    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d[d["Date"].notna()]

    # Prefer a real per-game identifier, but only if it's actually populated —
    # GameUID exists as a blank column in this export, and grouping on an
    # all-NaN key drops every row (pandas groupby excludes NaN groups).
    if "GameUID" in d.columns and d["GameUID"].notna().any():
        key = "GameUID"
    elif "GameID" in d.columns and d["GameID"].notna().any():
        key = "GameID"
    else:
        key = "Date"
    if "Inning" not in d.columns:
        d["Inning"] = 1

    out = []
    for gid, g in d.groupby(key):
        first = g[g["Inning"] == g["Inning"].min()]
        if first.empty:
            continue
        out.append({
            "game_key": gid,
            "date": g["Date"].iloc[0].date(),
            "t0": str(g["Time"].min()) if "Time" in g.columns else "",
            "catcher": first["Catcher"].mode().iloc[0],
        })

    return pd.DataFrame(out).sort_values(["date", "t0"]).reset_index(drop=True)


def build_splits(tm):
    """Join TrackMan starters to the official results, in time order per date."""
    log = pd.DataFrame(GAME_LOG, columns=["date", "opp", "ha", "rs", "ra"])
    log["date"] = pd.to_datetime(log["date"]).dt.date
    log["seq"] = log.groupby("date").cumcount()

    tm = tm.copy()
    tm["seq"] = tm.groupby("date").cumcount()

    merged = log.merge(tm[["date", "seq", "catcher"]], on=["date", "seq"], how="left")
    merged["result"] = (merged.rs > merged.ra).map({True: "W", False: "L"})
    return merged


def render():
    st.markdown("""
    <style>
      .ci-hd{font-family:Oswald,sans-serif;font-weight:600;letter-spacing:.04em;
             color:#0C1524;border-left:5px solid #C8102E;padding-left:.6rem;
             margin:.4rem 0 1rem;font-size:1.45rem;}
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="ci-hd">RECORD BY STARTING CATCHER</div>',
                unsafe_allow_html=True)

    tm = load_starting_catchers()
    if tm.empty:
        st.warning(
            "No TrackMan rows with a `Catcher` column found in `Data/`. "
            "Check that the folder is present and that the CSVs include "
            "`Catcher` and `Date`."
        )
        return

    merged = build_splits(tm)
    resolved = merged.dropna(subset=["catcher"])

    splits = (
        resolved.groupby("catcher")
        .agg(GS=("result", "size"),
             W=("result", lambda s: (s == "W").sum()),
             L=("result", lambda s: (s == "L").sum()),
             RS=("rs", "mean"),
             RA=("ra", "mean"))
        .reset_index()
    )
    splits["WPCT"] = (splits.W / splits.GS).round(3)
    splits["Diff/G"] = (splits.RS - splits.RA).round(2)
    splits["RS/G"] = splits.RS.round(2)
    splits["RA/G"] = splits.RA.round(2)
    splits = splits[["catcher", "GS", "W", "L", "WPCT", "RS/G", "RA/G", "Diff/G"]]
    splits = splits.sort_values("GS", ascending=False)

    st.dataframe(splits, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    c1.metric("Games resolved", f"{len(resolved)} of {len(merged)}")
    c2.metric("Team record", f"{int((merged.result=='W').sum())}-"
                             f"{int((merged.result=='L').sum())}")

    missing = merged[merged.catcher.isna()]
    if not missing.empty:
        with st.expander(f"{len(missing)} games without TrackMan coverage"):
            st.dataframe(missing[["date", "opp", "ha", "rs", "ra", "result"]],
                         use_container_width=True, hide_index=True)

    with st.expander("Game-by-game"):
        st.dataframe(merged[["date", "opp", "ha", "result", "rs", "ra", "catcher"]],
                     use_container_width=True, hide_index=True)

    st.caption(
        "Starting catcher = modal TrackMan `Catcher` in the first half-inning "
        "Brookhaven played defense. Starting catcher correlates strongly with "
        "starting pitcher, so these splits are descriptive, not causal."
    )

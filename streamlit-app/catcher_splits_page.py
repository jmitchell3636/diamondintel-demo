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

TEAM_MATCH = "NAS"          # matches NAS, NAS_KNI, "Nashua Silver Knights", etc.
DATA_DIR = "Data"

# Official results. (date, opponent, H/A, runs scored, runs allowed)
# Doubleheaders are listed in the order they were played.
GAME_LOG = [
    ("2026-05-27", "Vermont",    "H",  3,  6), ("2026-05-28", "Westfield",  "A",  5,  8),
    ("2026-05-29", "Lowell",     "A",  5,  3), ("2026-05-30", "Lowell",     "H", 11,  1),
    ("2026-06-01", "Lowell",     "H", 10,  4), ("2026-06-02", "Vermont",    "H", 11, 17),
    ("2026-06-03", "New Britain","H",  1,  2), ("2026-06-04", "Vermont",    "A",  6,  9),
    ("2026-06-05", "Worcester",  "H",  7,  6), ("2026-06-06", "Norwich",    "A", 16,  0),
    ("2026-06-07", "New Britain","H",  1,  3), ("2026-06-08", "Westfield",  "A",  0,  4),
    ("2026-06-09", "Lowell",     "H",  6,  2), ("2026-06-11", "Westfield",  "A", 16,  6),
    ("2026-06-12", "Worcester",  "A",  8,  9), ("2026-06-13", "Worcester",  "A", 12, 14),
    ("2026-06-15", "Westfield",  "H",  0,  1), ("2026-06-16", "Lowell",     "A",  7,  1),
    ("2026-06-17", "New Britain","A",  5,  6), ("2026-06-18", "Lowell",     "H", 10,  4),
    ("2026-06-21", "Worcester",  "H",  3,  1), ("2026-06-23", "Lowell",     "A",  3,  7),
    ("2026-06-24", "Worcester",  "A",  2,  6), ("2026-06-25", "Lowell",     "H",  0,  1),
    ("2026-06-26", "New Britain","H",  4,  2), ("2026-06-27", "Norwich",    "H", 11,  1),
    ("2026-06-28", "New Britain","A", 10,  0), ("2026-06-29", "Norwich",    "A",  5,  3),
    ("2026-06-30", "Worcester",  "H",  8,  7), ("2026-07-02", "Vermont",    "A",  2,  4),
    ("2026-07-03", "Lowell",     "A",  8, 10), ("2026-07-04", "Norwich",    "H",  4,  3),
    ("2026-07-06", "Lowell",     "H",  8,  5), ("2026-07-08", "Vermont",    "H",  0,  4),
    ("2026-07-08", "Vermont",    "H",  2, 10), ("2026-07-10", "New Britain","H",  9,  6),
    ("2026-07-11", "Lowell",     "A", 17,  9), ("2026-07-12", "New Britain","H",  6,  7),
    ("2026-07-14", "Norwich",    "A",  6,  7), ("2026-07-15", "Worcester",  "H", 11,  8),
    ("2026-07-16", "New Britain","A",  4,  8), ("2026-07-17", "Worcester",  "H",  2,  4),
    ("2026-07-18", "Westfield",  "H",  0,  3), ("2026-07-19", "Vermont",    "A", 10, 14),
    ("2026-07-19", "Vermont",    "A",  3,  6), ("2026-07-23", "New Britain","A",  8,  9),
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
        keep = [c for c in ("GameUID", "Date", "Time", "Inning", "Catcher",
                            "CatcherTeam", "PitcherTeam") if c in df.columns]
        frames.append(df[keep])

    if not frames:
        return pd.DataFrame()

    d = pd.concat(frames, ignore_index=True)

    # Keep only half-innings where Nashua was in the field.
    team_col = "CatcherTeam" if "CatcherTeam" in d.columns else "PitcherTeam"
    if team_col in d.columns:
        d = d[d[team_col].astype(str).str.upper().str.contains(TEAM_MATCH, na=False)]

    d = d[d["Catcher"].notna()]
    if d.empty:
        return pd.DataFrame()

    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d[d["Date"].notna()]

    key = "GameUID" if "GameUID" in d.columns else "Date"
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
        "Nashua played defense. Starting catcher correlates strongly with "
        "starting pitcher, so these splits are descriptive, not causal."
    )

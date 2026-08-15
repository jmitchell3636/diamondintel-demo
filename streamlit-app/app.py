import catcher_splits_page
import returner_board_page
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from collections import defaultdict
from scipy.stats import gaussian_kde

# 
#  CONFIG
# 
MY_TEAM = "BRK_BAN"

def _find_data_dir():
    candidates = [
        Path(__file__).parent / "Data",
        Path(__file__).parent / "data",
        Path.cwd() / "Data",
        Path.cwd() / "data",
        Path("/mount/src/baseball-analytics/Data"),
        Path("/mount/src/baseball-analytics/data"),
    ]
    for p in candidates:
        if p.exists() and list(p.glob("*.csv")):
            return p
    return Path(__file__).parent / "Data"

DATA_DIR = _find_data_dir()

TEAM_LABELS = {
    "BRK_BAN": "Brookhaven Bandits",
    "CON_RIV": "Concord River Cats",
    "DOV_ANC": "Dover Anchors",
    "POR_PRI": "Portsmouth Privateers",
    "MAN_MIL": "Manchester Millers",
}

# Just the city/region, no mascot -- used on the Returner Board (Front Office)
# where a short location reads better than the full team name.
TEAM_LOCATIONS = {
    "BRK_BAN": "Brookhaven",
    "CON_RIV": "Concord",
    "DOV_ANC": "Dover",
    "POR_PRI": "Portsmouth",
    "MAN_MIL": "Manchester",
}

# Teams hidden from every team/opponent picker in the app (they never show up
# as a selectable team or opponent). The underlying pitch data is NOT dropped —
# a game against one of these teams still counts fully in MY_TEAM's own stats
# (pitching against their batters, hitting against their pitchers); only the
# opposing team's own identity is hidden from browsing.
EXCLUDED_TEAMS = set()

def _team_options(series):
    """Unique team codes from a BatterTeam/PitcherTeam column, sorted, with
    EXCLUDED_TEAMS hidden. Use this everywhere a team picker is built instead
    of a raw sorted(series.dropna().unique()) so hidden teams never appear."""
    return sorted(t for t in series.dropna().unique() if t not in EXCLUDED_TEAMS)

# Full roster — "Last, First": "BASE_POS"
# Base positions: C, 1B, IF, OF, RHP, LHP
# IF players will be assigned 2B/3B/SS in the lineup builder
# OF players will be assigned LF/CF/RF
# Players removed from the active roster / lineup selection lists.
# Their historical data is kept (still counts in league/team stats); they are
# just hidden from roster pickers and lineup builders. Match on last name,
# case-insensitive, so spelling variants are caught.
REMOVED_FROM_ROSTER = {"jones", "gopal", "collins", "martorano", "mortorano", "piwnicki", "unknown"}

def _is_removed(name):
    if not isinstance(name, str):
        return False
    return name.split(",")[0].strip().lower() in REMOVED_FROM_ROSTER

def _player_options(series):
    """Unique player names from a Batter/Pitcher column, sorted, with
    REMOVED_FROM_ROSTER names hidden. Use this everywhere a player picker is
    built instead of a raw sorted(series.dropna().unique()) so a removed
    player never appears, no matter which page builds the list."""
    return sorted(n for n in series.dropna().unique() if not _is_removed(n))

# Players who left the 2026 roster (per official Futures League site) but whose
# historical data should still count everywhere EXCEPT the stat/leaderboard
# reports below — unlike REMOVED_FROM_ROSTER, they stay selectable in the
# Lineup Builder, Batter/Pitcher Analysis, Pitcher Scouting, Game Plan, etc.
REPORT_HIDDEN = {"dantoni", "kolb", "stead", "bump", "wilkes", "durham",
                 "skourides", "rodriguez", "shaffer", "gettinger", "norris",
                 "cross", "maiorano"}

def _is_report_hidden(name):
    if not isinstance(name, str):
        return False
    return name.split(",")[0].strip().lower() in REPORT_HIDDEN

def _player_options_reports(series):
    """Like _player_options, but also hides REPORT_HIDDEN names. Use this in
    the Matchup Tool, xBA Report, League Rankings, Hot/Cold Tracker, Hitter/
    Pitcher Stat Lines, Barrel Report, and Catcher Report pickers only."""
    return sorted(n for n in series.dropna().unique()
                  if not _is_removed(n) and not _is_report_hidden(n))

ROSTER = {
    "Callahan, Derek":  "OF",
    "Whitfield, Owen":  "OF",
    "Reyes, Julian":    "1B",
    "Pike, Jordan":     "OF",
    "Boyd, Marcus":     "C",
    "Alvarez, Sam":     "IF",
    "Odom, Casey":      "IF",
    "Nakashima, Kevin": "IF",
    "Lang, Trevor":     "IF",
    "Corbin, Miles":    "IF",
    "Trager, Will":     "C",
    "Brooks, Tyler":    "RHP",
    "Bennett, Cole":    "LHP",
    "Frost, Adam":      "RHP",
    "Delacruz, Ray":    "RHP",
    "Ito, Mason":       "LHP",
    "Sharpe, Devon":    "RHP",
}


PITCH_COLORS = {
    "Four-Seam": "#3b82f6", "Sinker":    "#22c55e", "Cutter":   "#8b5cf6",
    "Slider":    "#ef4444", "Curveball": "#f59e0b", "Changeup": "#06b6d4",
    "Splitter":  "#ec4899", "Knuckleball":"#f97316","Other":    "#64748b",
}

# Count situations pitch-mix usage is broken out by — shared by the Pitcher
# Scouting "Overall Pitch Mix" section and the Pitcher vs Team hitter report.
COUNT_GROUPS = {
    "First Pitch (0-0)":      [(0,0)],
    "Hitter Counts":          [(2,0),(3,0),(2,1),(3,1)],
    "Even Counts":            [(1,1),(2,2)],
    "Pitcher Counts":         [(0,1),(0,2),(1,2)],
    "Two-Strike":             [(0,2),(1,2),(2,2),(3,2)],
    "Full Count (3-2)":       [(3,2)],
}


def _render_count_group_mix(pitches, warn_missing=True):
    """Pitch-type usage broken out by count situation (COUNT_GROUPS), as 2
    rows of 3 cards with per-pitch usage bars."""
    _balls_max = pd.to_numeric(pitches["Balls"], errors="coerce").max()
    _strikes_max = pd.to_numeric(pitches["Strikes"], errors="coerce").max()
    _counts_ok = (pd.notna(_balls_max) and _balls_max >= 1) or \
                 (pd.notna(_strikes_max) and _strikes_max >= 1)
    if not _counts_ok and warn_missing:
        st.warning("⚠ Count data (balls/strikes) appears missing here — likely an incomplete "
                   "TrackMan file. Count-based splits below will be blank.")

    grp_list = list(COUNT_GROUPS.items())
    for row_start in range(0, len(grp_list), 3):
        row_groups = grp_list[row_start:row_start + 3]
        grp_cols = st.columns(3)
        for ci, (grp_name, count_list) in enumerate(row_groups):
            grp_pitches = pitches[pitches.apply(lambda r: (r["Balls"], r["Strikes"]) in count_list, axis=1)]
            n = len(grp_pitches)
            with grp_cols[ci]:
                st.markdown(f"**{grp_name}** <span style='color:#64748b;font-size:0.8rem;'>({n} pitches)</span>",
                    unsafe_allow_html=True)
                if n == 0:
                    st.caption("No data")
                    continue
                mix = grp_pitches["PitchType"].value_counts()
                for ptype, cnt in mix.items():
                    pct = cnt / n
                    color = PITCH_COLORS.get(ptype, "#64748b")
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:3px;'>"
                        f"<span style='width:80px;font-size:0.78rem;color:{color};font-weight:600;"
                        f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{ptype}</span>"
                        f"<div style='flex:1;background:#E2E8F0;border-radius:3px;height:8px;overflow:hidden;'>"
                        f"<div style='width:{pct:.0%};height:100%;background:{color};border-radius:3px;'></div></div>"
                        f"<span style='width:36px;text-align:right;font-size:0.78rem;color:#64748b;'>{pct:.0%}</span>"
                        f"</div>", unsafe_allow_html=True)
        st.divider()

# Normalize inconsistent Trackman pitch type tags to standard names
PITCH_NORMALIZE = {
    # Four-Seam variants (case-sensitive Trackman tags)
    "FourSeamFastball": "Four-Seam", "FourSeamFastBall": "Four-Seam",
    "Four-Seam": "Four-Seam", "4-Seam": "Four-Seam",
    "Fastball": "Four-Seam", "FA": "Four-Seam",
    # Two-Seam / Sinker variants
    "TwoSeamFastball": "Sinker", "TwoSeamFastBall": "Sinker",
    "Two-Seam": "Sinker", "Sinker": "Sinker", "SI": "Sinker",
    # Cutter
    "Cutter": "Cutter", "Cut": "Cutter", "FC": "Cutter",
    # Slider / Sweeper
    "Slider": "Slider", "SL": "Slider", "Sweeper": "Slider",
    # Curveball
    "Curveball": "Curveball", "CurveBall": "Curveball",
    "12-6 Curveball": "Curveball", "CB": "Curveball",
    # Changeup
    "Changeup": "Changeup", "ChangeUp": "Changeup",
    "Change-up": "Changeup", "CH": "Changeup",
    # Splitter
    "Splitter": "Splitter", "Split": "Splitter", "FS": "Splitter",
    # Other
    "Knuckleball": "Knuckleball", "Other": "Other", "Undefined": None,
}

def normalize_pitch(pt):
    if pd.isna(pt) or pt == "Undefined":
        return None
    return PITCH_NORMALIZE.get(pt, pt)

# ─────────────────────────────────────────
#  HANDEDNESS OVERRIDES
#  Corrects known Trackman data entry errors
#  verified against official Futures League roster
# ─────────────────────────────────────────
PITCHER_THROWS_OVERRIDES = {
    # ── CONFIRMED MISMATCHES (CSV wrong, official source correct) ──────────
    # Sullivan, Owen (WOR_BRA): CSV=Left, official roster=RHP → Right
    "Sullivan, Owen":    "Right",
    # Smith, Caden (LOW_SPI1): CSV=Right, Trackman portal + lowellspinners.com=LHP → Left
    "Smith, Caden":      "Left",
}

BATTER_SIDE_OVERRIDES = {
    # Fischer shows both L and R — official roster: Bats Left
    "Fischer, Peter":    "Left",
    # Hewett shows both L and R — needs coach confirmation, defaulting to official
    "Hewett, Bennett":   "Left",
    # Name variants — normalize to single canonical side
    "Cervoni, Mike":     "Right",
    "Cervoni, Michael":  "Right",
    # Camilleri bats Left — Trackman has him as Right (lowellspinners.com: B/T L/R)
    "Camilleri, Lorenzo": "Left",
    # Chance bats Left — Trackman has him as Right (lowellspinners.com: B/T L/R)
    "Chance, Cal":        "Left",
}

# Name spelling normalizations — maps Trackman typos to canonical names
NAME_OVERRIDES = {
    "Janks, Trent":      "Jenks, Trent",
    "Chernovetz , Brady":"Chernovetz, Brady",  # stray space before comma
    "Cervoni, Mike":     "Cervoni, Michael",
    "Mortorano, Tommy":  "Martorano, Tommy",  # Trackman misspelling
    "Martorano, Thomas": "Martorano, Tommy",  # formal name variant
    "Keblinsky, Peter":  "Keblinsky, Pete",   # name variant — merge to one canonical spelling
    "Marsh Jr. , Shaun": "Marsh Jr., Shaun",  # stray space before comma
    "Hennessey , Tommy": "Hennessey, Tommy",  # stray space before comma
    "ortiz, jayden":     "Ortiz, Jayden",     # lowercase entry variant
}

# Per-GAME pitcher corrections — for when an opposing TrackMan operator entered
# the WRONG pitcher for a specific game. Unlike NAME_OVERRIDES (which remaps a
# name everywhere), this only remaps within ONE GameID, so the same name stays
# correct in every other game.
#   key:   (GameID, "Wrong, Name")   value: "Correct, Name"
# Example:
#   ("20260529-LeLacheurPark-1", "Smith, John"): "Doe, Jane",
GAME_PITCHER_OVERRIDES = {
}

BATTING_BASE_POSITIONS = {"C", "1B", "IF", "OF"}

# All position players can play any position — full flexibility
ALL_FIELD_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
POS_OPTIONS = {bp: ALL_FIELD_POSITIONS for bp in BATTING_BASE_POSITIONS}

# Positions that must be unique in a lineup
UNIQUE_POSITIONS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"}

POS_COLORS = {
    "C":"#f59e0b", "1B":"#22c55e", "2B":"#a78bfa",
    "3B":"#ef4444", "SS":"#3b82f6", "LF":"#06b6d4",
    "CF":"#06b6d4", "RF":"#06b6d4", "DH":"#f97316",
}

st.set_page_config(
    page_title="DiamondIntel · Brookhaven Bandits",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  VISUAL THEME
#  One place. Colors match the Plotly figures (white field, slate rules) so the
#  page and the charts read as a single surface. Oswald sets headings in condensed
#  uppercase — scoreboard vernacular, not a neutral sans. Numbers everywhere use
#  tabular figures so stat columns align vertically like a box score.
# ─────────────────────────────────────────
_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
  --midnight:#FFFFFF; --panel:#F1F5F9; --rule:#E2E8F0; --knights:#C8102E;
  /* Text scale, darkest-first. Every value measured against the #FFFFFF field. */
  --chalk:#1E293B;         /* body, table cells, metric values */
  --dust-bright:#475569;   /* captions, widget + metric labels, table headers */
  --steel:#334155;         /* headings — size carries hierarchy, not brightness */
  --dust:#64748B;          /* incidental metadata only */
}

html, body, [class*="css"]{
  font-family:'Inter',-apple-system,'Segoe UI',sans-serif;
  font-feature-settings:'tnum' 1,'cv05' 1;   /* tabular figures */
}

/* Headings: condensed, uppercase, tight. The title gets the one red rule. */
h1,h2,h3{
  font-family:'Oswald','Inter',sans-serif !important;
  text-transform:uppercase; letter-spacing:.045em; color:var(--steel) !important;
}
h1{ font-size:1.95rem !important; font-weight:600 !important;
    padding-bottom:.4rem; margin-bottom:.9rem;
    border-bottom:2px solid var(--knights); }
h2{ font-size:1.28rem !important; font-weight:600 !important; margin-top:1.5rem !important; }
h3{ font-size:1.02rem !important; font-weight:500 !important; color:var(--steel) !important;
    letter-spacing:.08em; }
h4{ font-family:'Oswald','Inter',sans-serif !important; text-transform:uppercase;
    letter-spacing:.07em; font-size:.86rem !important; color:var(--dust-bright) !important; }

/* Captions: quiet, never competing with the data. */
[data-testid="stCaptionContainer"]{ color:var(--dust-bright) !important; font-size:.8rem; line-height:1.5; }
[data-testid="stCaptionContainer"] p{ color:var(--dust-bright) !important; }

/* Metrics as panels rather than floating text. */
[data-testid="stMetric"]{
  background:var(--panel); border:1px solid var(--rule); border-left:3px solid var(--knights);
  border-radius:6px; padding:.7rem .85rem;
}
[data-testid="stMetricLabel"]{
  font-family:'Oswald','Inter',sans-serif; text-transform:uppercase;
  letter-spacing:.07em; font-size:.72rem !important; color:var(--dust-bright) !important;
}
[data-testid="stMetricValue"]{
  font-family:'IBM Plex Mono',monospace !important; font-weight:500;
  font-size:1.5rem !important; color:var(--chalk) !important;
}
[data-testid="stMetricDelta"]{ font-size:.76rem !important; }

/* Tables: numbers monospaced so columns line up. */
[data-testid="stDataFrame"]{ border:1px solid var(--rule); border-radius:6px; }
[data-testid="stDataFrame"] div[role="gridcell"]{
  font-family:'IBM Plex Mono',monospace; font-size:.82rem;
}
[data-testid="stDataFrame"] div[role="columnheader"]{
  font-family:'Oswald','Inter',sans-serif; text-transform:uppercase;
  letter-spacing:.05em; font-size:.74rem; background:var(--panel); color:var(--dust-bright);
}

/* Sidebar: a rail, not a second page. */
section[data-testid="stSidebar"]{ background:var(--panel); border-right:1px solid var(--rule); }
section[data-testid="stSidebar"] h1{ border-bottom:none; font-size:1.3rem !important;
  color:var(--steel) !important; }

/* Nav: labels are read constantly, so they sit at full text contrast, not muted.
   Streamlit wraps radio label text in a markdown <p>, so both need the colour. */
div[role="radiogroup"] label,
div[role="radiogroup"] label p,
div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p{
  color:var(--chalk) !important; font-size:.92rem; font-weight:500;
}
div[role="radiogroup"] label:hover,
div[role="radiogroup"] label:hover p{ color:#0F172A !important; }

/* Widget labels (selectbox, slider, number input) */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p{
  color:var(--dust-bright) !important; font-weight:500;
}

/* Sidebar text sits a touch brighter still — it is the primary navigation. */
section[data-testid="stSidebar"] label p{ color:var(--chalk) !important; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{ color:var(--dust-bright) !important; }

/* Expander + tab headers */
[data-testid="stExpander"] summary p{ color:var(--chalk) !important; }
button[data-baseweb="tab"]{ color:var(--dust-bright) !important; }
button[data-baseweb="tab"][aria-selected="true"]{ color:#0F172A !important; }

/* Dividers: hairlines, not gutters. */
hr{ border-color:var(--rule) !important; margin:1.15rem 0 !important; }

/* Buttons + downloads: quiet until wanted. */
.stButton>button, .stDownloadButton>button{
  font-family:'Oswald','Inter',sans-serif; text-transform:uppercase; letter-spacing:.06em;
  font-size:.8rem; border:1px solid var(--rule); background:var(--panel); color:var(--chalk);
  border-radius:5px;
}
.stButton>button:hover, .stDownloadButton>button:hover{
  border-color:var(--knights); color:#0F172A;
}

/* Keep the top chrome out of the way. */
#MainMenu, footer{ visibility:hidden; }
.block-container{ padding-top:2.1rem; padding-bottom:2.5rem; }

/* Quality floor: visible keyboard focus, honour reduced motion. */
:focus-visible{ outline:2px solid var(--knights); outline-offset:2px; }
@media (prefers-reduced-motion: reduce){ *{ animation:none !important; transition:none !important; } }
</style>
"""
st.markdown(_THEME_CSS, unsafe_allow_html=True)

#
#  DATA LOADING
# 


def _render_kde_heatmap(pitches, weight_col=None, key_suffix="", title=""):
    """Render a KDE heatmap of pitch locations using plotly."""
    import plotly.graph_objects as _go_kde
    from scipy.stats import gaussian_kde as _gkde
    loc = pitches[pitches["PlateLocSide"].notna() & pitches["PlateLocHeight"].notna()]
    if len(loc) < 5:
        st.info("Not enough location data (need 5+).")
        return
    if weight_col and weight_col in loc.columns and loc[weight_col].notna().any():
        w = loc[weight_col].fillna(loc[weight_col].mean()).values
        w = np.clip(w, 0.1, None); w = w / w.sum()
    else:
        w = None
    xi = np.linspace(-2.0, 2.0, 60)
    yi = np.linspace(-0.1, 4.5, 60)
    xx, yy = np.meshgrid(xi, yi)
    # Use ALL pitches, but make the density estimate robust to far-flung tracking
    # outliers: a few pitches recorded far off the plate would otherwise inflate
    # gaussian_kde's covariance and smear the heat blob thin inside the view.
    # We feed every pitch, but set the bandwidth from the ROBUST spread (the bulk
    # of pitches via the interquartile range), so outliers can't wash it out.
    _sx = loc["PlateLocSide"].values
    _sy = loc["PlateLocHeight"].values
    def _robust_bw(arr):
        iqr = np.subtract(*np.percentile(arr, [75, 25]))
        spread = (iqr / 1.349) if iqr > 0 else (np.std(arr) or 1.0)
        n = max(len(arr), 2)
        return max(0.15, 1.06 * spread * n ** (-1/5)) / (np.std(arr) or 1.0)
    try:
        bw = float(np.clip(np.mean([_robust_bw(_sx), _robust_bw(_sy)]), 0.05, 0.5))
        kde = _gkde(np.vstack([_sx, _sy]), weights=w, bw_method=bw)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    except Exception:
        try:
            kde = _gkde(np.vstack([_sx, _sy]), bw_method=0.33)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        except Exception:
            st.info("Could not compute heatmap.")
            return
    fig = _go_kde.Figure()
    fig.add_trace(_go_kde.Heatmap(
        x=xi, y=yi, z=zz,
        colorscale=[[0,"#ffffff"],[0.3,"#bfdbfe"],[0.6,"#f59e0b"],[0.8,"#ef4444"],[1.0,"#7f1d1d"]],
        showscale=False, zsmooth="best",
    ))
    # Strike zone
    fig.add_trace(_go_kde.Scatter(
        x=[-0.83, 0.83, 0.83, -0.83, -0.83],
        y=[1.5, 1.5, 3.5, 3.5, 1.5],
        mode="lines", line=dict(color="rgba(15,23,42,0.55)", width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))
    # Home plate — pitcher view (point facing up toward strike zone)
    # x-axis matches PlateLocSide: negative = pitcher's left, positive = pitcher's right
    _pw = 0.708
    plate_x = [_pw, -_pw, -_pw,  0.0,  _pw,  _pw]
    plate_y = [0.0,   0.0,  0.20, 0.40, 0.20, 0.0]
    fig.add_trace(_go_kde.Scatter(
        x=plate_x, y=plate_y,
        mode="lines", fill="toself",
        fillcolor="rgba(15,23,42,0.12)",
        line=dict(color="rgba(15,23,42,0.65)", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        xaxis=dict(range=[-2, 2], visible=False, scaleanchor="y", scaleratio=0.87),
        yaxis=dict(range=[-0.1, 4.5], visible=False),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"kde_{key_suffix}")
    if title:
        st.caption(title)


def _kde_heatmap_png(pitches, weight_col=None, width_in=2.1, height_in=2.35):
    """Same KDE math as _render_kde_heatmap, rendered with matplotlib to PNG
    bytes instead of an interactive Plotly figure — for embedding in a PDF
    (there's no kaleido in this environment to rasterize Plotly directly).
    Returns None if there isn't enough location data."""
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.stats import gaussian_kde as _gkde

    loc = pitches[pitches["PlateLocSide"].notna() & pitches["PlateLocHeight"].notna()]
    if len(loc) < 5:
        return None
    if weight_col and weight_col in loc.columns and loc[weight_col].notna().any():
        w = loc[weight_col].fillna(loc[weight_col].mean()).values
        w = np.clip(w, 0.1, None); w = w / w.sum()
    else:
        w = None
    xi = np.linspace(-2.0, 2.0, 60)
    yi = np.linspace(-0.1, 4.5, 60)
    xx, yy = np.meshgrid(xi, yi)
    _sx = loc["PlateLocSide"].values
    _sy = loc["PlateLocHeight"].values

    def _robust_bw(arr):
        iqr = np.subtract(*np.percentile(arr, [75, 25]))
        spread = (iqr / 1.349) if iqr > 0 else (np.std(arr) or 1.0)
        n = max(len(arr), 2)
        return max(0.15, 1.06 * spread * n ** (-1 / 5)) / (np.std(arr) or 1.0)

    try:
        bw = float(np.clip(np.mean([_robust_bw(_sx), _robust_bw(_sy)]), 0.05, 0.5))
        kde = _gkde(np.vstack([_sx, _sy]), weights=w, bw_method=bw)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    except Exception:
        try:
            kde = _gkde(np.vstack([_sx, _sy]), bw_method=0.33)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        except Exception:
            return None

    fig, ax = _plt.subplots(figsize=(width_in, height_in), dpi=150)
    cmap = LinearSegmentedColormap.from_list(
        "heat", ["#ffffff", "#bfdbfe", "#f59e0b", "#ef4444", "#7f1d1d"])
    ax.pcolormesh(xx, yy, zz, cmap=cmap, shading="gouraud")
    ax.plot([-0.83, 0.83, 0.83, -0.83, -0.83], [1.5, 1.5, 3.5, 3.5, 1.5],
            color="#0f172a", linewidth=1.2, linestyle=":", alpha=0.6)
    pw = 0.708
    plate_x = [pw, -pw, -pw, 0.0, pw, pw]
    plate_y = [0.0, 0.0, 0.20, 0.40, 0.20, 0.0]
    ax.fill(plate_x, plate_y, color="#0f172a", alpha=0.12)
    ax.plot(plate_x, plate_y, color="#0f172a", linewidth=1.2, alpha=0.65)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.1, 4.5)
    ax.set_aspect(0.87)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    _plt.close(fig)
    return buf.getvalue()


def _ab_outcome(row):
    """Classify the terminal outcome of a plate appearance from its last
    pitch. Shared by the At-Bat Log section and the hitter scouting PDF."""
    pr = row.get("PlayResult", "")
    kb = row.get("KorBB", "")
    pc = row.get("PitchCall", "")
    if pr in ("Single", "Double", "Triple", "HomeRun"):
        return pr
    if kb == "Walk":
        return "Walk"
    if pc == "HitByPitch":
        return "HBP"
    if kb == "Strikeout":
        return "Strikeout"
    if pr in ("Out", "FieldersChoice", "Error", "Sacrifice"):
        return pr
    return str(pr) if pd.notna(pr) and pr not in ("", "Undefined") else "—"


def _build_ab_log(hp):
    """Every plate appearance in hp (already filtered to one batter vs one
    pitcher), broken into pitch-by-pitch detail. Returns a list of dicts:
    {date, inning, outcome, exit_speed, angle, distance, direction, pitches:[...]}.
    Empty list if there's no usable at-bat structure (missing GameID etc.)."""
    gc_ab = [c for c in ["GameID", "Inning", "Top/Bottom", "PAofInning"] if c in hp.columns]
    if not gc_ab:
        return []
    sort_cols = gc_ab + (["PitchofPA"] if "PitchofPA" in hp.columns else [])
    hp = hp.sort_values(sort_cols)

    results = []
    for _, ab in hp.groupby(gc_ab, sort=False):
        if "PitchofPA" in ab.columns:
            ab = ab.sort_values("PitchofPA")
        last = ab.iloc[-1]
        pitches = []
        for pnum, (_, prow) in enumerate(ab.iterrows(), 1):
            side_v, height_v = prow.get("PlateLocSide"), prow.get("PlateLocHeight")
            has_loc = pd.notna(side_v) and pd.notna(height_v)
            pitches.append({
                "num": pnum,
                "count": (f"{int(prow['Balls'])}-{int(prow['Strikes'])}"
                          if pd.notna(prow.get("Balls")) and pd.notna(prow.get("Strikes")) else "—"),
                "type": prow.get("PitchType") if pd.notna(prow.get("PitchType")) else "—",
                "velo": prow.get("RelSpeed"),
                "ivb": prow.get("InducedVertBreak"),
                "hb": prow.get("HorzBreak"),
                "side": side_v,
                "height": height_v,
                "zone": (attack_zone(side_v, height_v) or "—") if has_loc else "—",
                "call": prow.get("PitchCall") if pd.notna(prow.get("PitchCall")) else "—",
            })
        results.append({
            "date": pd.to_datetime(last.get("Date"), errors="coerce"),
            "inning": last.get("Inning"),
            "outcome": _ab_outcome(last),
            "exit_speed": last.get("ExitSpeed"),
            "angle": last.get("Angle"),
            "distance": last.get("Distance"),
            "direction": last.get("Direction"),
            "pitches": pitches,
        })
    return results


# Fixed-order categorical groups for pitch-call markers on the at-bat pitch
# plot — order/hues match the app's palette slots so a call's color stays
# stable no matter which at-bat or how many groups actually appear.
_PITCH_CALL_GROUPS = [
    ("Ball", {"BallCalled", "BallinDirt"}, "#2a78d6"),
    ("Called Strike", {"StrikeCalled"}, "#eb6834"),
    ("Whiff", {"StrikeSwinging"}, "#e34948"),
    ("Foul", {"FoulBallNotFieldable", "FoulBallFieldable", "FoulBall", "FoulTip"}, "#eda100"),
    ("In Play", {"InPlay"}, "#1baf7a"),
    ("Hit By Pitch", {"HitByPitch"}, "#e87ba4"),
]


def _pitch_call_group(call):
    for name, calls, color in _PITCH_CALL_GROUPS:
        if call in calls:
            return name, color
    return "Other", "#64748b"


def _render_ab_pitch_plot(pitches, key):
    """Scatter every pitch of a plate appearance as dots in the strike zone —
    same zone box, home plate, and borderless white chrome as the KDE
    heatmap (_render_kde_heatmap), the app's most-used pitch-location style."""
    import plotly.graph_objects as _go_pl
    loc_pitches = [p for p in pitches if pd.notna(p.get("side")) and pd.notna(p.get("height"))]
    if not loc_pitches:
        st.caption("No pitch location data for this at-bat.")
        return

    fig = _go_pl.Figure()
    fig.add_trace(_go_pl.Scatter(
        x=[-0.83, 0.83, 0.83, -0.83, -0.83], y=[1.5, 1.5, 3.5, 3.5, 1.5],
        mode="lines", line=dict(color="rgba(15,23,42,0.55)", width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip"))
    pw = 0.708
    fig.add_trace(_go_pl.Scatter(
        x=[pw, -pw, -pw, 0.0, pw, pw], y=[0.0, 0.0, 0.20, 0.40, 0.20, 0.0],
        mode="lines", fill="toself", fillcolor="rgba(15,23,42,0.12)",
        line=dict(color="rgba(15,23,42,0.65)", width=1.5),
        showlegend=False, hoverinfo="skip"))

    seen = set()
    for p in loc_pitches:
        group, color = _pitch_call_group(p.get("call"))
        velo_txt = f"{p['velo']:.1f} mph" if pd.notna(p.get("velo")) else "velo —"
        fig.add_trace(_go_pl.Scatter(
            x=[p["side"]], y=[p["height"]],
            mode="markers",
            marker=dict(size=14, color=color, symbol="circle", opacity=0.85,
                        line=dict(color="rgba(15,23,42,0.55)", width=1)),
            name=group, legendgroup=group, showlegend=group not in seen,
            hovertemplate=(f"Pitch {p['num']} · {p['count']}<br>{p['type']} · {velo_txt}"
                            f"<br>{p.get('call')}<extra></extra>"),
        ))
        seen.add(group)

    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=30),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        xaxis=dict(range=[-2, 2], visible=False, scaleanchor="y", scaleratio=0.87),
        yaxis=dict(range=[-0.1, 4.5], visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, x=0,
                    font=dict(size=9.5), itemsizing="constant"),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"abplot_{key}")


def _ab_pitch_plot_png(pitches, width_in=2.3, height_in=2.6):
    """Matplotlib PNG twin of _render_ab_pitch_plot — same zone box, home
    plate, and dot markers — for embedding a pitch-by-pitch zone plot per
    at-bat in the PDF report (no kaleido available to rasterize Plotly)."""
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    loc_pitches = [p for p in pitches if pd.notna(p.get("side")) and pd.notna(p.get("height"))]
    if not loc_pitches:
        return None

    fig, ax = _plt.subplots(figsize=(width_in, height_in), dpi=150)
    ax.plot([-0.83, 0.83, 0.83, -0.83, -0.83], [1.5, 1.5, 3.5, 3.5, 1.5],
            color="#0f172a", linewidth=1.0, linestyle=":", alpha=0.6)
    pw = 0.708
    plate_x = [pw, -pw, -pw, 0.0, pw, pw]
    plate_y = [0.0, 0.0, 0.20, 0.40, 0.20, 0.0]
    ax.fill(plate_x, plate_y, color="#0f172a", alpha=0.12)
    ax.plot(plate_x, plate_y, color="#0f172a", linewidth=1.0, alpha=0.65)

    for p in loc_pitches:
        _, color = _pitch_call_group(p.get("call"))
        ax.scatter([p["side"]], [p["height"]], s=90, color=color, alpha=0.85,
                   edgecolors="#0f172a", linewidths=0.6, zorder=3)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.1, 4.5)
    ax.set_aspect(0.87)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    _plt.close(fig)
    return buf.getvalue()


def _fcbl_reclassify(combined):
    """
    Vectorized pitch reclassification — fast boolean masks, no row iteration.
    """
    df = combined.copy()
    feat = ["RelSpeed","InducedVertBreak","HorzBreak","SpinRate"]
    if not all(f in df.columns for f in feat):
        return df

    # Per-pitcher fastball baseline (p85 velocity) — use map not join to avoid fragmentation
    fb_map = df.groupby("Pitcher")["RelSpeed"].quantile(0.85)
    fb_velo = df["Pitcher"].map(fb_map).fillna(88.0)

    drop    = (fb_velo - df["RelSpeed"]).fillna(99)
    hb_arm  = np.where(df["PitcherThrows"] == "Left",
                       -df["HorzBreak"].fillna(0),
                        df["HorzBreak"].fillna(0))
    pt       = df["PitchType"]
    sp_known = df["SpinRate"].notna()
    sp       = df["SpinRate"].fillna(0)
    ivb      = df["InducedVertBreak"].fillna(0)
    hba      = pd.Series(hb_arm, index=df.index)
    vlo      = df["RelSpeed"].fillna(0)

    # sp_known guards every "low spin" comparison below: a pitch with no spin
    # reading gets fillna(0)'d for the ">=" checks (harmlessly fails a "high
    # spin" bar), but a bare "< threshold" check would treat that same missing
    # reading as "confirmed low spin" and misclassify it — so those two rules
    # require an actual measurement before they're allowed to fire.
    new_pt = pt.copy()
    new_pt = new_pt.mask(sp_known & (sp < 1350) & ~pt.isin(["Splitter","Curveball"]), "Splitter")
    new_pt = new_pt.mask((pt == "Changeup") & (drop <= 6) & (sp >= 2000) & (ivb >= 8), "Sinker")
    new_pt = new_pt.mask((pt == "Changeup") & (sp >= 2150) & (hba >= 4), "Sinker")
    new_pt = new_pt.mask((pt == "Sinker")   & sp_known & (sp < 1650)  & (drop >= 5), "Changeup")
    new_pt = new_pt.mask((pt == "Slider")   & (hba < -13)  & (vlo < 87), "Sweeper")
    # Slider/Cutter mixup: TrackMan's auto-classifier inconsistently splits
    # some pitchers' cutters into "Slider" tags. A real cutter sits close to
    # the pitcher's own fastball velo (small drop) with real carry (high
    # IVB); a real slider gives up more velo and has little-to-no carry.
    # Thresholds calibrated against manually-tagged Slider/Cutter pitches
    # league-wide (~1% false-positive rate on verified sliders).
    new_pt = new_pt.mask((pt == "Slider")   & (drop < 6)    & (ivb >= 7), "Cutter")
    new_pt = new_pt.mask((pt == "Changeup") & (vlo < 72), "Curveball")

    # Preserve manually-tagged pitches: only keep reclassification where the
    # pitch was NOT manually tagged. Where it WAS tagged, restore the original.
    if "_was_tagged" in df.columns:
        new_pt = new_pt.where(~df["_was_tagged"], pt)

    df["PitchType"] = new_pt
    return df

@st.cache_data(ttl=600, show_spinner=False, max_entries=2)  # v15 — per-game pitcher overrides
def load_data():
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()
    dfs = []
    for f in csv_files:
        # Skip positioning files — they share PitchUIDs with game files
        if "playerpositioning" in f.name.lower() or "positioning" in f.name.lower():
            continue
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.copy()

    # Merge duplicate/misspelled pitcher names so a player's data isn't split.
    # Keys are stripped/lowercased for matching; values are the canonical name.
    _PITCHER_NAME_FIXES = {
        "mortorano, tommy": "Martorano, Thomas",
        "martorano, tommy": "Martorano, Thomas",
        "martorano, thomas": "Martorano, Thomas",
    }
    if "Pitcher" in combined.columns:
        def _fix_pitcher(n):
            if pd.isna(n):
                return n
            key = str(n).strip().lower()
            return _PITCHER_NAME_FIXES.get(key, str(n).strip())
        combined["Pitcher"] = combined["Pitcher"].apply(_fix_pitcher)
    # Force the numeric TrackMan columns to real numbers. On the full dataset some
    # CSVs carry stray text/blank values that make pandas read a whole column as
    # strings (object dtype), which breaks .quantile() and other math. Coerce any
    # non-numeric entries to NaN so every downstream calc has clean numeric types.
    _numeric_cols = [
        "RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "SpinAxis",
        "RelHeight", "RelSide", "Extension", "VertApprAngle", "HorzApprAngle",
        "PlateLocHeight", "PlateLocSide", "ZoneSpeed", "ExitSpeed", "Angle",
        "Direction", "Distance", "HangTime", "Balls", "Strikes",
        "pfxx", "pfxz", "Bearing",
    ]
    for _c in _numeric_cols:
        if _c in combined.columns:
            combined[_c] = pd.to_numeric(combined[_c], errors="coerce")

    # Repair missing/blank GameID. Several pages list games via
    # df["GameID"].dropna().unique(), so any row with an empty GameID silently
    # vanishes from those pages. Rebuild it from Date + Stadium (the same
    # YYYYMMDD-Stadium-1 shape TrackMan uses), falling back to GameUID.
    if "GameID" in combined.columns:
        gid = combined["GameID"].astype("string").str.strip()
        blank = gid.isna() | (gid == "") | (gid.str.lower() == "nan")
        if blank.any():
            def _mk_gid(row):
                d = str(row.get("Date", "")).replace("-", "").strip()[:8]
                stad = str(row.get("Stadium", "")).strip().replace(" ", "")
                if d and stad:
                    return f"{d}-{stad}-1"
                guid = str(row.get("GameUID", "")).strip()
                return guid if guid and guid.lower() != "nan" else None
            combined.loc[blank, "GameID"] = combined.loc[blank].apply(_mk_gid, axis=1)
    tagged = combined.get("TaggedPitchType", pd.Series(dtype=str))
    auto   = combined.get("AutoPitchType",   pd.Series(dtype=str))
    # Use tagged type when available, fall back to auto
    raw_type = np.where(tagged.isna() | (tagged == "Undefined"), auto, tagged)
    # Track which pitches were MANUALLY tagged (real TaggedPitchType present).
    # The FCBL reclassifier must NOT overwrite these — only auto-classified ones.
    combined["_was_tagged"] = ~(tagged.isna() | (tagged == "Undefined"))
    # Normalize to standard names (fixes FourSeamFastball vs Four-Seam etc.)
    def _norm(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        x = str(x).strip()
        if x in ("nan", "None", "Undefined", ""):
            return None
        return PITCH_NORMALIZE.get(x, x)
    combined["PitchType"] = pd.Series(raw_type).apply(_norm)
    if "PitchUID" in combined.columns:
        # Only dedup rows that actually have a PitchUID — don't drop rows with null UIDs
        has_uid  = combined["PitchUID"].notna()
        no_uid   = combined[~has_uid]
        with_uid = combined[has_uid].drop_duplicates(subset="PitchUID")
        combined = pd.concat([with_uid, no_uid], ignore_index=True)

    # Apply handedness and name overrides
    if "Pitcher" in combined.columns:
        # Fix pitcher throwing hands (vectorized: map override by name, keep original where absent)
        _p = combined["Pitcher"].astype(str).str.strip()
        _override = _p.map(PITCHER_THROWS_OVERRIDES)
        combined["PitcherThrows"] = _override.where(_override.notna(), combined["PitcherThrows"])
    if "Batter" in combined.columns:
        # Fix batter sides (vectorized)
        _b = combined["Batter"].astype(str).str.strip()
        _override_b = _b.map(BATTER_SIDE_OVERRIDES)
        combined["BatterSide"] = _override_b.where(_override_b.notna(), combined["BatterSide"])
        # Fix name spellings
        combined["Batter"] = combined["Batter"].map(
            lambda n: NAME_OVERRIDES.get(str(n).strip(), str(n).strip()) if pd.notna(n) else n
        )
    if "Pitcher" in combined.columns:
        combined["Pitcher"] = combined["Pitcher"].map(
            lambda n: NAME_OVERRIDES.get(str(n).strip(), str(n).strip()) if pd.notna(n) else n
        )

    # Per-game pitcher corrections (wrong pitcher entered by an operator for one game)
    if GAME_PITCHER_OVERRIDES and "Pitcher" in combined.columns and "GameID" in combined.columns:
        # Vectorized: build (GameID, Pitcher) tuple key and map only where an override exists
        _keys = list(zip(combined["GameID"].astype(str).str.strip(),
                         combined["Pitcher"].astype(str).str.strip()))
        _mapped = pd.Series([GAME_PITCHER_OVERRIDES.get(k) for k in _keys], index=combined.index)
        combined["Pitcher"] = _mapped.where(_mapped.notna(), combined["Pitcher"])

    combined = _fcbl_reclassify(combined)

    # Drop pitch types that only occur once for a given pitcher (likely a
    # classification blip, not a real part of their arsenal).
    if "Pitcher" in combined.columns and "PitchType" in combined.columns:
        counts = combined.groupby(["Pitcher", "PitchType"])["PitchType"].transform("size")
        combined = combined[combined["PitchType"].isna() | (counts > 1)]

    return combined

df_all = load_data()

# Apply manual Pitch Editor corrections (on top of automatic reclassification)
def _apply_pitch_corrections(df):
    import json
    # Load from session state first, then from file
    corr = st.session_state.get("_pitch_corr", {})
    if not corr:
        try:
            corr_file = DATA_DIR / "pitch_corrections.json"
            if corr_file.exists():
                with open(corr_file) as _f:
                    corr = json.load(_f)
                st.session_state["_pitch_corr"] = corr
        except Exception:
            pass
    if not corr or "PitchUID" not in df.columns:
        return df
    uid_map = {k[4:]: v for k, v in corr.items() if k.startswith("uid:")}
    if not uid_map:
        return df
    mask = df["PitchUID"].isin(uid_map)
    if not mask.any():
        return df   # no rows actually affected — skip the expensive copy entirely
    df = df.copy()
    df.loc[mask, "PitchType"] = df.loc[mask, "PitchUID"].map(uid_map)
    return df

df_all = _apply_pitch_corrections(df_all)

# Coerce numeric columns ONCE — eliminates 7 full-dataset copies.
for _nc in ["Balls","Strikes","PlateLocSide","PlateLocHeight","ExitSpeed","Angle",
            "RelSpeed","SpinRate","HorzBreak","InducedVertBreak","Extension",
            "RelHeight","RelSide","VertApprAngle","HorzApprAngle","ZoneSpeed",
            "Direction","Distance","HangTime","pfxx","pfxz","Bearing",
            "x0","y0","z0","vx0","vy0","vz0","ax0","ay0","az0"]:
    if _nc in df_all.columns:
        df_all[_nc] = pd.to_numeric(df_all[_nc], errors="coerce")

# ── Stuff+ scoring — runs once per data load, before any page renders ────────
# score_by_pitch_type is @st.cache_data keyed on (row count, columns), so it only
# retrains when the dataset actually changes (new games + "Reload data"), not on
# every click. NOTE: an earlier version of this trained on every boot unconditionally
# and spiked memory on the 1GB hosting instance (health check "connection reset") —
# the shape-based cache key avoids repeating that.
try:
    import stuff_model
    _stuff_hash = (len(df_all), tuple(df_all.columns))
    stuff_plus_df = stuff_model.score_by_pitch_type(_stuff_hash, df_all)
except Exception:
    stuff_plus_df = pd.DataFrame(columns=["Pitcher", "PitchType", "StuffPlus", "Pitches"])

# ── Park factors — shared across League Rankings, OPS+ Leaderboard, and
# Player WAR, all of which used to flag "not park-adjusted" as a known gap.
# Single-season, ~25-30 games per park, so an unregressed factor would be
# wildly noisy — shrunk toward 1.0 by sample size and clipped to a sane
# range. Uses combined runs/game (both teams) at each stadium as the signal;
# FIP-style metrics are mostly driven by HR rate, but a stadium's own HR
# sample here is too thin (a few dozen fly balls) to trust on its own, so
# the general runs environment is used as the best available proxy instead
# of a separate, noisier HR-only park factor. ──
@st.cache_data(ttl=600, max_entries=2)
def _compute_park_factors(_hash):
    """Returns (game_pf, pf_table): game_pf is {GameID: park_factor} for
    looking up any game's park factor by ID; pf_table is a small per-stadium
    summary (games, avg runs/game, factor) for display/debugging."""
    if "GameID" not in df_all.columns or "Stadium" not in df_all.columns:
        return {}, pd.DataFrame()

    game_stadium = {}
    game_runs = {}
    for gid, grp in df_all.groupby("GameID"):
        stadium = grp["Stadium"].dropna().iloc[0] if grp["Stadium"].notna().any() else None
        if stadium is None:
            continue
        game_stadium[gid] = stadium
        game_runs[gid] = grp["RunsScored"].fillna(0).sum() if "RunsScored" in grp.columns else 0.0

    if not game_stadium:
        return {}, pd.DataFrame()

    league_rpg = sum(game_runs.values()) / len(game_runs)

    stadium_runs = defaultdict(list)
    for gid, stadium in game_stadium.items():
        stadium_runs[stadium].append(game_runs[gid])

    K_PF = 20            # shrinkage constant — at 20 games, 50/50 blend with league average
    PF_MIN, PF_MAX = 0.85, 1.15
    stadium_pf = {}
    pf_rows = []
    for stadium, runs_list in stadium_runs.items():
        n = len(runs_list)
        avg = sum(runs_list) / n
        raw_pf = avg / league_rpg if league_rpg else 1.0
        w = n / (n + K_PF)
        shrunk = max(PF_MIN, min(PF_MAX, w * raw_pf + (1 - w) * 1.0))
        stadium_pf[stadium] = shrunk
        pf_rows.append({"Stadium": stadium, "Games": n, "Avg Runs/G": round(avg, 2),
                        "Park Factor": round(shrunk, 3)})

    game_pf = {gid: stadium_pf.get(stadium, 1.0) for gid, stadium in game_stadium.items()}
    pf_table = pd.DataFrame(pf_rows).sort_values("Park Factor", ascending=False).reset_index(drop=True)
    return game_pf, pf_table

# Demo fallback for Data/official_stats.csv, which hasn't been exported for
# this club yet. Covers the Brookhaven roster only (Season Report only ever
# looks up players on MY_TEAM) with a plausible full-season line — a bigger
# sample than the handful of TrackMan games in Data/, which is the point:
# this is what the "official season line" section looks like once the
# league stats PDF is on file. Delete this block once official_stats.csv
# is added and this falls back to reading the real file automatically.
_DEMO_OFFICIAL_STATS = [
    {"player": "Callahan, Derek", "g": 30, "ab": 118, "h": 39, "doubles": 9, "triples": 1, "hr": 4,
     "rbi": 22, "bb": 14, "so": 19, "sb": 8, "ba": 0.331, "obp": 0.402, "slg": 0.525, "ops": 0.927},
    {"player": "Whitfield, Owen", "g": 30, "ab": 115, "h": 35, "doubles": 7, "triples": 0, "hr": 6,
     "rbi": 27, "bb": 16, "so": 24, "sb": 3, "ba": 0.304, "obp": 0.389, "slg": 0.522, "ops": 0.911},
    {"player": "Reyes, Julian", "g": 28, "ab": 104, "h": 30, "doubles": 6, "triples": 1, "hr": 2,
     "rbi": 15, "bb": 10, "so": 17, "sb": 5, "ba": 0.288, "obp": 0.351, "slg": 0.423, "ops": 0.774},
    {"player": "Pike, Jordan", "g": 29, "ab": 109, "h": 33, "doubles": 8, "triples": 0, "hr": 5,
     "rbi": 24, "bb": 9, "so": 22, "sb": 2, "ba": 0.303, "obp": 0.356, "slg": 0.514, "ops": 0.870},
    {"player": "Boyd, Marcus", "g": 27, "ab": 97, "h": 25, "doubles": 4, "triples": 0, "hr": 3,
     "rbi": 18, "bb": 8, "so": 20, "sb": 0, "ba": 0.258, "obp": 0.314, "slg": 0.392, "ops": 0.706},
    {"player": "Alvarez, Sam", "g": 26, "ab": 92, "h": 24, "doubles": 5, "triples": 1, "hr": 1,
     "rbi": 11, "bb": 7, "so": 18, "sb": 6, "ba": 0.261, "obp": 0.313, "slg": 0.370, "ops": 0.683},
    {"player": "Odom, Casey", "g": 28, "ab": 101, "h": 27, "doubles": 6, "triples": 0, "hr": 2,
     "rbi": 14, "bb": 11, "so": 21, "sb": 4, "ba": 0.267, "obp": 0.339, "slg": 0.386, "ops": 0.725},
    {"player": "Nakashima, Kevin", "g": 25, "ab": 88, "h": 21, "doubles": 3, "triples": 0, "hr": 1,
     "rbi": 9, "bb": 6, "so": 16, "sb": 3, "ba": 0.239, "obp": 0.287, "slg": 0.307, "ops": 0.594},
    {"player": "Lang, Trevor", "g": 24, "ab": 80, "h": 19, "doubles": 4, "triples": 0, "hr": 2,
     "rbi": 10, "bb": 5, "so": 15, "sb": 1, "ba": 0.237, "obp": 0.282, "slg": 0.362, "ops": 0.645},
    {"player": "Brooks, Tyler", "app": 11, "gs": 11, "w": 6, "l": 2, "sv": 0, "ip": 58.0,
     "h": 48, "er": 22, "bb": 16, "so": 64, "era": 3.41, "whip": 1.10},
    {"player": "Frost, Adam", "app": 9, "gs": 5, "w": 3, "l": 2, "sv": 0, "ip": 32.1,
     "h": 30, "er": 15, "bb": 13, "so": 29, "era": 4.18, "whip": 1.33},
    {"player": "Sharpe, Devon", "app": 14, "gs": 0, "w": 2, "l": 1, "sv": 3, "ip": 19.2,
     "h": 15, "er": 7, "bb": 9, "so": 22, "era": 3.20, "whip": 1.22},
    {"player": "Ito, Mason", "app": 15, "gs": 0, "w": 1, "l": 0, "sv": 1, "ip": 17.0,
     "h": 14, "er": 6, "bb": 7, "so": 19, "era": 3.18, "whip": 1.24},
    {"player": "Delacruz, Ray", "app": 13, "gs": 0, "w": 1, "l": 2, "sv": 0, "ip": 15.1,
     "h": 16, "er": 9, "bb": 8, "so": 13, "era": 5.28, "whip": 1.57},
    {"player": "Bennett, Cole", "app": 12, "gs": 0, "w": 0, "l": 1, "sv": 2, "ip": 13.2,
     "h": 11, "er": 5, "bb": 6, "so": 15, "era": 3.29, "whip": 1.24},
]


@st.cache_data(ttl=300, max_entries=3)
def load_official_stats():
    """Load official stats from Presto Sports CSV export if present.
    Accepts flexible column names — maps common variants automatically."""
    stats_file = DATA_DIR / "official_stats.csv"
    if not stats_file.exists():
        df = pd.DataFrame(_DEMO_OFFICIAL_STATS)
    else:
        try:
            df = pd.read_csv(stats_file)
        except Exception:
            df = pd.DataFrame(_DEMO_OFFICIAL_STATS)

    # Normalize column names — lowercase, strip spaces
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Column name mapping — handles common Presto Sports export variants
    COL_MAP = {
        # Player name
        "player":       "player", "name": "player", "athlete": "player",
        # Batting average
        "avg":          "ba", "ba": "ba", "batting_avg": "ba", "batting_average": "ba",
        # OBP
        "obp":          "obp", "on_base_pct": "obp", "on_base_percentage": "obp",
        # SLG
        "slg":          "slg", "slg_pct": "slg", "slugging": "slg",
        # OPS
        "ops":          "ops",
        # Games
        "g":            "g", "gp":  "g", "games": "g",
        # At bats
        "ab":           "ab", "at_bats": "ab",
        # Hits
        "h":            "h", "hits": "h",
        # Runs
        "r":            "r", "runs": "r",
        # HR
        "hr":           "hr", "home_runs": "hr",
        # RBI
        "rbi":          "rbi", "rbis": "rbi",
        # BB
        "bb":           "bb", "walks": "bb",
        # K
        "so":           "so", "k":  "so", "strikeouts": "so", "ks": "so",
        # HBP
        "hbp":          "hbp", "hit_by_pitch": "hbp",
        # SAC
        "sf":           "sf", "sac_fly": "sf", "sacrifice_fly": "sf",
        "sh":           "sh", "sac_bunt": "sh", "sacrifice_hit": "sh",
        # Plate appearances — used straight when the league export carries it,
        # otherwise rebuilt from AB+BB+HBP+SF+SH (see _official_k_bb).
        "pa":           "pa", "plate_appearances": "pa",
        # SB
        "sb":           "sb", "stolen_bases": "sb",
        # 2B 3B
        "2b":           "doubles", "doubles": "doubles",
        "3b":           "triples", "triples": "triples",
        # ── Pitching (for the Season Report's official season line) ──
        "era":          "era", "w":   "w", "wins": "w",
        "l":            "l", "losses": "l",
        "sv":           "sv", "saves": "sv",
        "ip":           "ip", "innings_pitched": "ip",
        "whip":         "whip",
        "er":           "er", "earned_runs": "er",
        "bf":           "bf", "batters_faced": "bf",
        "app":          "app", "appearances": "app", "g_p": "app",
        "gs":           "gs", "games_started": "gs",
    }

    renamed = {}
    for col in df.columns:
        if col in COL_MAP:
            renamed[col] = COL_MAP[col]
    df = df.rename(columns=renamed)

    if "player" not in df.columns:
        return pd.DataFrame()

    # Normalize player name — try to match Trackman "Last, First" format
    # Presto Sports exports as "First Last" — convert to "Last, First"
    def normalize_name(n):
        if pd.isna(n):
            return n
        n = str(n).strip()
        # Already "Last, First" format
        if "," in n:
            return n
        # "First Last" → "Last, First"
        parts = n.split()
        if len(parts) >= 2:
            return f"{parts[-1]}, {' '.join(parts[:-1])}"
        return n

    df["player_norm"] = df["player"].map(normalize_name)

    # Convert numeric columns
    for col in ["ba","obp","slg","ops","g","ab","h","r","hr","rbi","bb","so","hbp","sf","sh","pa","sb",
                "doubles","triples",
                "era","w","l","sv","ip","whip","er","bf","app","gs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

official_stats = load_official_stats()

def get_official_stat(player_name, stat, default=None):
    """Get a stat from official_stats for a player, matching by name."""
    if official_stats.empty:
        return default
    # Try exact Trackman name match
    match = official_stats[official_stats["player_norm"] == player_name]
    if match.empty:
        # Try last name only
        last = player_name.split(",")[0].strip().lower()
        match = official_stats[official_stats["player_norm"].str.split(",").str[0].str.strip().str.lower() == last]
    if match.empty or stat not in official_stats.columns:
        return default
    val = match[stat].iloc[0]
    return val if pd.notna(val) else default


def to_ip(outs):
    """Convert raw outs to baseball IP notation (e.g. 10 outs = 3.1)"""
    if outs is None: return "—"
    outs = int(outs)
    return f"{outs // 3}.{outs % 3}"

def team_label(code):
    return TEAM_LABELS.get(code, code)

def team_location(code):
    return TEAM_LOCATIONS.get(code, code)

def player_last(name):
    if pd.isna(name):
        return name
    return name.split(",")[0].strip()

def base_pos(name):
    return ROSTER.get(name, "?")

def _safe_whiff(s):
    """Whiff rate (swings-and-misses / swings) for a slice of pitches, or None."""
    swings = s["PitchCall"].isin(["StrikeSwinging","FoulBallNotFieldable",
                                   "FoulBallFieldable","InPlay"]).sum()
    if swings == 0:
        return None
    whiffs = (s["PitchCall"] == "StrikeSwinging").sum()
    return whiffs / swings

def _ab_mask(df):
    """At-bat-ending pitches: a ball in play, a strikeout, or a foul ball that
    was actually caught for an out. The tracking data leaves PlayResult as
    'Undefined' on a caught foul (fieldable or not) instead of tagging it
    InPlay/Out, so without this it silently drops out of every AB count and
    never counts against the hitter's average even though OutsOnPlay shows
    the out really happened."""
    m = df["PitchCall"].eq("InPlay") | df["KorBB"].eq("Strikeout")
    if "OutsOnPlay" in df.columns:
        m = m | (df["PitchCall"].isin(["FoulBallFieldable", "FoulBallNotFieldable"]) &
                 (df["OutsOnPlay"].fillna(0) > 0))
    return m

def _goto_pitcher_scouting(player_id, team):
    """Deep-link a Returner Board pitcher into Pitcher Scouting.

    Must be used as a widget on_click callback, never called from inline
    script code after a plain `if st.button(...):`. By the time the main
    script body re-executes after a click, the sidebar's nav_cat/nav_page
    radios have already been instantiated for that run, and Streamlit
    forbids mutating an already-instantiated widget's session_state key
    later in the same run. Callbacks run before the script body, so the
    mutation is legal there.
    """
    all_teams = sorted(_team_options(df_all["PitcherTeam"]))
    sorted_teams = ([MY_TEAM] if MY_TEAM in all_teams else []) + \
        sorted(t for t in all_teams if t != MY_TEAM)
    st.session_state["nav_cat"] = "Pitchers"
    st.session_state["nav_page"] = "Pitcher Scouting"
    if team in sorted_teams:
        st.session_state["ps_team"] = sorted_teams.index(team)
    st.session_state["ps_pitcher"] = player_id
    st.session_state["rb_return_flag"] = True
    st.session_state["rb_last_player"] = player_id
    st.session_state["rb_last_kind"] = "pitcher"


def _goto_batter_analysis(player_id, team):
    """Deep-link a Returner Board hitter into Batter Analysis. See
    _goto_pitcher_scouting for why this must run as an on_click callback."""
    all_spray_teams = ([MY_TEAM] if MY_TEAM in _team_options(df_all["BatterTeam"]) else []) + \
        sorted(t for t in _team_options(df_all["BatterTeam"]) if t != MY_TEAM)
    st.session_state["nav_cat"] = "Hitters"
    st.session_state["nav_page"] = "Batter Analysis"
    if team in all_spray_teams:
        st.session_state["ba_team"] = all_spray_teams.index(team)
    st.session_state["ba_batter"] = player_id
    st.session_state["rb_return_flag"] = True
    st.session_state["rb_last_player"] = player_id
    st.session_state["rb_last_kind"] = "hitter"


#
#  SIDEBAR
#
with st.sidebar:
    st.markdown("## DiamondIntel")
    st.markdown(f"**{team_label(MY_TEAM)}**")
    st.divider()
    NAV_GROUPS = {
        "Hitters": ["Batter Analysis", "Player Report", "Hitter Stat Lines",
                    "Hot / Cold", "Barrel Report", "xBA Report"],
        "Pitchers": ["Pitcher Scouting", "Pitcher Stat Lines",
                     "Pitch Design", "Movement Plots", "Pitch Run Values", "3D Trajectories",
                     "Catcher Report", "Trends", "Bullpen Script", "Starters vs Bullpen"],
        "Matchups": ["Matchup Tool", "Game Plan", "Lineup Builder",
                     "Next Hitters", "Attack Plan (Beta)", "Bullpen", "Reliever Matchups","Catcher Splits",
                     "Pitcher vs Team"],
        "League": ["Report Generator", "Season Report", "League Rankings",
                   "OPS+ Leaderboard", "Player WAR", "Defensive Positioning",
                   "Team Totals"],
        "Front Office": ["Returner Board"],
    }
    NAV_DISPLAY_NAMES = {
        "Report Generator": "Printable Scouting Sheet",
        "Season Report": "End-of-Season Player Report",
        "League Rankings": "League Pitching Rankings",
        "Team Totals": "Season Totals by Opponent",
        "Attack Plan (Beta)": "Attack Plan",
        "Bullpen": "Bullpen Availability",
    }
    nav_cat = st.radio("Section", list(NAV_GROUPS.keys()),
                       horizontal=True, label_visibility="collapsed", key="nav_cat")
    page = st.radio("Page", NAV_GROUPS[nav_cat],
                    format_func=lambda p: NAV_DISPLAY_NAMES.get(p, p),
                    label_visibility="collapsed", key="nav_page")
    st.divider()
    if df_all.empty:
        st.error("No data found in Data/ folder.")
    else:
        games   = df_all["GameID"].nunique() if "GameID" in df_all.columns else "?"
        pitches = len(df_all)
        st.caption(f"**{pitches:,}** pitches · **{games}** games")
        if "Date" in df_all.columns:
            latest = pd.to_datetime(df_all["Date"], errors="coerce").max()
            if pd.notna(latest):
                st.caption(f"Last game: **{latest.strftime('%b %d, %Y')}**")
        if st.button("🔄 Reload data", key="reload_data", use_container_width=True,
                     help="Clears the cache and re-reads every CSV in the Data/ folder. "
                          "Use this after adding a new game so it shows up without waiting."):
            st.cache_data.clear()
            st.rerun()
    st.divider()
    # Official stats status
    if not official_stats.empty:
        st.caption(f"Official stats: **{len(official_stats)} players** loaded")
    else:
        st.caption("No official_stats.csv — using Trackman calculations")

    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

if df_all.empty:
    st.error("No CSV files found. Add Trackman CSVs to the Data/ folder in GitHub and refresh.")
    with st.expander("Debug info"):
        st.code(f"""
app.py location : {Path(__file__)}
DATA_DIR        : {DATA_DIR}
DATA_DIR exists : {DATA_DIR.exists()}
cwd             : {Path.cwd()}
cwd contents    : {list(Path.cwd().iterdir())[:10]}
mount           : {list(Path('/mount/src').iterdir()) if Path('/mount/src').exists() else 'not found'}
        """)
    st.stop()

# 
#  HELPERS — FIELD
# 
def direction_to_xy(direction, distance):
    # Bird's eye view: left field = left (negative x), right field = right (positive x)
    # Trackman direction: negative = left field, positive = right field
    rad  = np.radians(direction)
    dist = np.clip(distance, 0, 450)
    return np.sin(rad) * dist, np.cos(rad) * dist

def draw_field(fig):
    for sign in [-1, 1]:
        rad = np.radians(sign * 45)
        fig.add_shape(type="line", x0=0, y0=0,
            x1=np.sin(rad)*400, y1=np.cos(rad)*400,
            line=dict(color="rgba(255,255,255,0.25)", width=1.5))
    angles = np.linspace(-45, 45, 100)
    fig.add_trace(go.Scatter(
        x=[np.sin(np.radians(a))*400 for a in angles],
        y=[np.cos(np.radians(-a))*400 for a in angles],
        mode="lines", line=dict(color="rgba(255,255,255,0.2)", width=1.5),
        hoverinfo="skip", showlegend=False))
    for dist, label in [(150,"150'"),(250,"250'"),(330,"330'"),(400,"400'")]:
        angs = np.linspace(-45, 45, 80)
        fig.add_trace(go.Scatter(
            x=[np.sin(np.radians(-a))*dist for a in angs],
            y=[np.cos(np.radians(-a))*dist for a in angs],
            mode="lines", line=dict(color="rgba(255,255,255,0.08)", width=1, dash="dot"),
            hoverinfo="skip", showlegend=False))
        fig.add_annotation(x=0, y=dist, text=label, showarrow=False,
            font=dict(size=9, color="rgba(255,255,255,0.2)"), yshift=4)
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=95*np.cos(theta), y=95*np.sin(theta)+60,
        fill="toself", fillcolor="rgba(120,80,40,0.15)",
        line=dict(color="rgba(120,80,40,0.3)", width=1),
        hoverinfo="skip", showlegend=False))
    for bx, by in [(0,0),(63.6,63.6),(0,127.3),(-63.6,63.6)]:
        fig.add_trace(go.Scatter(x=[bx], y=[by], mode="markers",
            marker=dict(symbol="square", size=7, color="white",
                        line=dict(color="white", width=1)),
            hoverinfo="skip", showlegend=False))
    return fig

# Shift recommendation removed — coaches determine positioning

# 
#  EXPECTED STATS MODEL (xBA, xSLG, xwOBA)
#  Lookup table from MLB Statcast research, calibrated to FCBL
# 
# Calibration factors — recalibrated from 932 ABs across 15 FCBL games
# Formula: per-AB (includes strikeouts in denominator, excludes BB/HBP)
XBA_CAL_FACTOR   = 1.2894
XSLG_CAL_FACTOR  = 1.0500
XWOBA_CAL_FACTOR = 1.1500
WOBA_WEIGHTS = {"1b": 0.888, "2b": 1.271, "3b": 1.616, "hr": 2.101}

STATCAST_LOOKUP = {
    ("<70","<-10"):(0.03,0.00,0.00,0.00),("<70","-10-0"):(0.08,0.01,0.00,0.00),
    ("<70","0-10"):(0.12,0.02,0.00,0.00),("<70","10-20"):(0.10,0.03,0.01,0.00),
    ("<70","20-25"):(0.06,0.02,0.00,0.00),("<70","25-30"):(0.04,0.01,0.00,0.00),
    ("<70","30-40"):(0.02,0.01,0.00,0.00),("<70","40-50"):(0.01,0.00,0.00,0.00),("<70","50+"):(0.01,0.00,0.00,0.00),
    ("70-80","<-10"):(0.05,0.00,0.00,0.00),("70-80","-10-0"):(0.18,0.02,0.00,0.00),
    ("70-80","0-10"):(0.28,0.04,0.01,0.00),("70-80","10-20"):(0.22,0.08,0.02,0.00),
    ("70-80","20-25"):(0.12,0.06,0.01,0.00),("70-80","25-30"):(0.07,0.04,0.01,0.00),
    ("70-80","30-40"):(0.03,0.02,0.00,0.00),("70-80","40-50"):(0.02,0.01,0.00,0.00),("70-80","50+"):(0.01,0.00,0.00,0.00),
    ("80-85","<-10"):(0.06,0.01,0.00,0.00),("80-85","-10-0"):(0.24,0.03,0.00,0.00),
    ("80-85","0-10"):(0.36,0.07,0.01,0.00),("80-85","10-20"):(0.32,0.14,0.03,0.00),
    ("80-85","20-25"):(0.18,0.12,0.02,0.01),("80-85","25-30"):(0.08,0.08,0.01,0.02),
    ("80-85","30-40"):(0.04,0.04,0.01,0.03),("80-85","40-50"):(0.02,0.02,0.00,0.01),("80-85","50+"):(0.01,0.00,0.00,0.00),
    ("85-90","<-10"):(0.07,0.01,0.00,0.00),("85-90","-10-0"):(0.28,0.04,0.01,0.00),
    ("85-90","0-10"):(0.40,0.10,0.02,0.00),("85-90","10-20"):(0.38,0.20,0.04,0.01),
    ("85-90","20-25"):(0.22,0.18,0.03,0.04),("85-90","25-30"):(0.10,0.14,0.02,0.08),
    ("85-90","30-40"):(0.05,0.08,0.01,0.07),("85-90","40-50"):(0.02,0.03,0.00,0.03),("85-90","50+"):(0.01,0.01,0.00,0.00),
    ("90-95","<-10"):(0.08,0.02,0.00,0.00),("90-95","-10-0"):(0.30,0.06,0.01,0.00),
    ("90-95","0-10"):(0.42,0.14,0.03,0.00),("90-95","10-20"):(0.40,0.26,0.05,0.02),
    ("90-95","20-25"):(0.24,0.24,0.04,0.10),("90-95","25-30"):(0.10,0.18,0.02,0.18),
    ("90-95","30-40"):(0.05,0.10,0.01,0.16),("90-95","40-50"):(0.02,0.04,0.00,0.07),("90-95","50+"):(0.01,0.01,0.00,0.01),
    ("95-100","<-10"):(0.10,0.02,0.00,0.00),("95-100","-10-0"):(0.32,0.08,0.01,0.00),
    ("95-100","0-10"):(0.44,0.18,0.03,0.00),("95-100","10-20"):(0.40,0.30,0.05,0.05),
    ("95-100","20-25"):(0.22,0.28,0.04,0.18),("95-100","25-30"):(0.08,0.20,0.02,0.30),
    ("95-100","30-40"):(0.04,0.12,0.01,0.28),("95-100","40-50"):(0.02,0.05,0.00,0.14),("95-100","50+"):(0.01,0.01,0.00,0.02),
    ("100-105","<-10"):(0.12,0.03,0.00,0.00),("100-105","-10-0"):(0.33,0.10,0.01,0.00),
    ("100-105","0-10"):(0.44,0.22,0.04,0.00),("100-105","10-20"):(0.38,0.32,0.05,0.10),
    ("100-105","20-25"):(0.18,0.28,0.04,0.28),("100-105","25-30"):(0.06,0.18,0.02,0.42),
    ("100-105","30-40"):(0.03,0.10,0.01,0.40),("100-105","40-50"):(0.02,0.05,0.00,0.22),("100-105","50+"):(0.01,0.02,0.00,0.04),
    ("105+","<-10"):(0.14,0.04,0.00,0.00),("105+","-10-0"):(0.34,0.12,0.02,0.00),
    ("105+","0-10"):(0.43,0.24,0.04,0.01),("105+","10-20"):(0.34,0.32,0.05,0.16),
    ("105+","20-25"):(0.14,0.26,0.03,0.38),("105+","25-30"):(0.04,0.14,0.02,0.54),
    ("105+","30-40"):(0.02,0.08,0.01,0.52),("105+","40-50"):(0.01,0.04,0.00,0.32),("105+","50+"):(0.01,0.02,0.00,0.06),
}

def _ev_bin(ev):
    if ev<70: return "<70"
    elif ev<80: return "70-80"
    elif ev<85: return "80-85"
    elif ev<90: return "85-90"
    elif ev<95: return "90-95"
    elif ev<100: return "95-100"
    elif ev<105: return "100-105"
    else: return "105+"

def _la_bin(la):
    if la<-10: return "<-10"
    elif la<0: return "-10-0"
    elif la<10: return "0-10"
    elif la<20: return "10-20"
    elif la<25: return "20-25"
    elif la<30: return "25-30"
    elif la<40: return "30-40"
    elif la<50: return "40-50"
    else: return "50+"

def _raw_probs(ev, la):
    return STATCAST_LOOKUP.get((_ev_bin(ev), _la_bin(la)), (0.0, 0.0, 0.0, 0.0))

# ── Statcast-style attack zones (Heart / Shadow / Chase / Waste) ──
# Based on distance from strike-zone center. Calibrated to match MLB proportions
# (Heart ~14%, Shadow ~40%, Chase ~26%, Waste ~20% of all pitches).
_AZ_HW = 0.83                      # zone half-width (ft)
_AZ_TOP, _AZ_BOT = 3.378, 1.755    # zone top/bottom (ft)
_AZ_CH = (_AZ_TOP + _AZ_BOT) / 2   # vertical center
_AZ_HH = (_AZ_TOP - _AZ_BOT) / 2   # zone half-height
_AZ_ORDER = ["Heart", "Shadow", "Chase", "Waste"]

def attack_zone(side, height):
    """Classify a pitch location into Heart / Shadow / Chase / Waste."""
    if pd.isna(side) or pd.isna(height):
        return None
    dx = abs(side) / _AZ_HW
    dy = abs(height - _AZ_CH) / _AZ_HH
    d = max(dx, dy)          # 1.0 = exactly at the zone edge
    if d <= 0.55:
        return "Heart"
    if d <= 1.33:
        return "Shadow"
    if d <= 2.0:
        return "Chase"
    return "Waste"

@st.cache_data(ttl=600, max_entries=2)
def _league_attack_zone_rates(df, mode):
    """League distribution across attack zones. mode='pitch' = all pitches (for
    pitchers); mode='swing' = swings only (for hitters)."""
    d = df.copy()
    d["PlateLocSide"] = pd.to_numeric(d["PlateLocSide"], errors="coerce")
    d["PlateLocHeight"] = pd.to_numeric(d["PlateLocHeight"], errors="coerce")
    if mode == "swing":
        d = d[d["PitchCall"].isin(["StrikeSwinging", "FoulBallNotFieldable",
                                   "FoulBallFieldable", "InPlay"])]
    d = d.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    z = d.apply(lambda r: attack_zone(r["PlateLocSide"], r["PlateLocHeight"]), axis=1)
    z = z.dropna()
    if len(z) == 0:
        return {k: 0.0 for k in _AZ_ORDER}
    vc = z.value_counts(normalize=True)
    return {k: float(vc.get(k, 0.0)) * 100 for k in _AZ_ORDER}


def _attack_zone(side, h):
    """Statcast-style attack zone from plate location (feet).
    Heart (middle), Shadow (edges), Chase (just out), Waste (far out).
    Uses the app's actual strike zone (half-width 0.83 ft, height 1.755-3.378 ft)
    so chase/zone rates stay consistent with the rest of the app."""
    if side is None or h is None or (isinstance(side, float) and np.isnan(side)) or \
       (isinstance(h, float) and np.isnan(h)):
        return None
    # zone center 2.5665 ft (midpoint of 1.755-3.378); half-height 0.8115 ft
    cy = (3.378 + 1.755) / 2.0
    hh = (3.378 - 1.755) / 2.0
    d = max(abs(side) / 0.83, abs(h - cy) / hh)
    if d <= 0.67:
        return "Heart"
    if d <= 1.33:
        return "Shadow"
    if d <= 2.0:
        return "Chase"
    return "Waste"

_ATTACK_SWING = {"StrikeSwinging", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"}
# Descriptive swing-decision run values (count-based linear-weight proxy; NOT a
# fitted model — labeled honestly in the UI).
_ATTACK_TAKE_RV = {"StrikeCalled": -0.05, "BallCalled": 0.06, "BallinDirt": 0.06,
                   "HitByPitch": 0.10}
_ATTACK_OUT_RV = {"Single": 0.47, "Double": 0.78, "Triple": 1.05, "HomeRun": 1.40,
                  "Out": -0.27, "Error": 0.30, "FieldersChoice": -0.27, "Sacrifice": -0.10}
_ATTACK_SWING_RV = {"StrikeSwinging": -0.12, "FoulBallNotFieldable": -0.04,
                    "FoulBallFieldable": -0.04}

def _attack_zone_frame(df):
    """Return a copy with _az zone label, excluding games with broken tracking
    (PitchCall mostly 'Undefined'), which would corrupt swing/take rates."""
    bad = set()
    for g, grp in df.groupby("GameID"):
        if (grp["PitchCall"] == "Undefined").mean() > 0.5:
            bad.add(g)
    d = df[~df["GameID"].isin(bad)].copy()
    d["PlateLocSide"] = pd.to_numeric(d["PlateLocSide"], errors="coerce")
    d["PlateLocHeight"] = pd.to_numeric(d["PlateLocHeight"], errors="coerce")
    d = d[d["PlateLocSide"].notna() & d["PlateLocHeight"].notna()].copy()
    d["_az"] = d.apply(lambda r: _attack_zone(r["PlateLocSide"], r["PlateLocHeight"]), axis=1)
    d["_swing"] = d["PitchCall"].isin(_ATTACK_SWING)
    return d

def _attack_decision_rv(row):
    """Run value of the pitch outcome (descriptive proxy)."""
    if row["PitchCall"] == "InPlay":
        return _ATTACK_OUT_RV.get(row.get("PlayResult"), 0.0)
    if row["PitchCall"] in _ATTACK_SWING_RV:
        return _ATTACK_SWING_RV[row["PitchCall"]]
    return _ATTACK_TAKE_RV.get(row["PitchCall"], 0.0)

def _attack_zone_svg(zone_values, title="", is_rate=True):
    """Attack-zone grid drawn TO SCALE against the true TrackMan strike zone.

    True zone: half-width 9.96 in (8.5 in half-plate + 1.46 in ball radius),
    height 21.06-40.53 in (half-height 9.74 in about the 30.79 in center).
    Bands are scaled distance d = max(|x|/9.96, |y-cy|/9.74):
      Heart d<=0.67 -> +/-6.7 in x, +/-6.5 in y
      Shadow d<=1.33 -> +/-13.2 in x, +/-13.0 in y
      Chase  d<=2.00 -> +/-19.9 in x, +/-19.5 in y
      Waste  beyond
    """
    HW_IN, HH_IN = 9.96, 9.74
    S = 6.9            # px per inch
    CX = CY = 200.0
    def rect(bx_in, by_in):
        w, h = 2 * bx_in * S, 2 * by_in * S
        return CX - bx_in * S, CY - by_in * S, w, h
    def col(v, vmax):
        if vmax <= 0:
            return "#1e293b"
        t = max(0.0, min(1.0, v / vmax))
        r = int(30 + t * (239 - 30)); g = int(58 + t * (68 - 58)); b = int(138 - t * (138 - 68))
        return "rgb(" + str(r) + "," + str(g) + "," + str(b) + ")"
    vals = [abs(x) for x in zone_values.values()] or [1]
    vmax = max(vals)
    def fmt(v):
        return (str(round(v)) + "%") if is_rate else ("+" if v >= 0 else "") + format(v, ".2f")

    bands = [("Waste", 23.0, 23.0), ("Chase", 19.92, 19.48),
             ("Shadow", 13.25, 12.95), ("Heart", 6.67, 6.53)]
    parts = []
    for name, bx, by in bands:
        x, y, w, h = rect(bx, by)
        c = col(zone_values.get(name, 0), vmax)
        parts.append('<rect x="' + format(x, ".1f") + '" y="' + format(y, ".1f") +
                     '" width="' + format(w, ".1f") + '" height="' + format(h, ".1f") +
                     '" fill="' + c + '" stroke="#475569" stroke-width="1"/>')
    # true strike zone outline (dashed green), drawn on top
    zx, zy, zw, zh = rect(HW_IN, HH_IN)
    parts.append('<rect x="' + format(zx, ".1f") + '" y="' + format(zy, ".1f") +
                 '" width="' + format(zw, ".1f") + '" height="' + format(zh, ".1f") +
                 '" fill="none" stroke="#4ade80" stroke-width="2.5" stroke-dasharray="5 4"/>')
    # band value labels, stacked at the top inside each band
    label_y = {"Waste": CY - 23.0 * S + 15, "Chase": CY - 19.48 * S + 15,
               "Shadow": CY - 12.95 * S + 15, "Heart": CY - 6.53 * S + 14}
    meas = {"Waste": "beyond 19.9 in", "Chase": "13.3-19.9 in",
            "Shadow": "6.7-13.3 in", "Heart": "within 6.7 in"}
    for name, _, _ in bands:
        v = fmt(zone_values.get(name, 0))
        parts.append('<text x="200" y="' + format(label_y[name], ".0f") +
                     '" fill="#f8fafc" font-size="11.5" font-weight="600" text-anchor="middle">' +
                     name + " " + v + "</text>")
    for name, _, _ in bands:
        parts.append('<text x="200" y="' + format(label_y[name] + 12, ".0f") +
                     '" fill="#cbd5e1" font-size="8.5" text-anchor="middle">' + meas[name] + "</text>")
    # strike-zone dimension callout
    parts.append('<text x="200" y="392" fill="#4ade80" font-size="9.5" text-anchor="middle">'
                 'dashed = true strike zone: 19.9 in wide, 21.1-40.5 in tall</text>')
    body = "".join(parts)
    return ('<div style="text-align:center;font-family:sans-serif;">'
            '<div style="color:#1e293b;font-size:13px;margin-bottom:2px;font-weight:600;">' + title + "</div>"
            '<svg viewBox="0 0 400 400" width="330" height="330">' + body + "</svg></div>")


# ── True-strike-zone swing rates (Z-Swing / O-Swing) ──
# NOTE: the attack-zone BANDS are not the strike zone. The Shadow band straddles
# the border (~52% of Shadow pitches are actually balls). Z/O-Swing therefore use
# the real TrackMan zone, not Heart+Shadow.
_ZONE_HW, _ZONE_B, _ZONE_T = 0.83, 1.755, 3.378

def _true_zone_swing(d):
    """Return (z_swing_pct, o_swing_pct, n_in, n_out) from an _attack_zone_frame."""
    if len(d) == 0:
        return 0.0, 0.0, 0, 0
    inz = (d["PlateLocSide"].abs() <= _ZONE_HW) & d["PlateLocHeight"].between(_ZONE_B, _ZONE_T)
    a, b = d[inz], d[~inz]
    z = a["_swing"].mean() * 100 if len(a) else 0.0
    o = b["_swing"].mean() * 100 if len(b) else 0.0
    return z, o, len(a), len(b)


# ── Barrel / hard-hit (same frozen thresholds as the Barrel Report page) ──
_BARREL_SCORE_THRESH = 76.82
_HARDHIT_EV = 90.0

def _barrel_score(ev, la):
    if pd.isna(ev) or pd.isna(la):
        return np.nan
    la_fit = max(0.0, 1.0 - abs(la - 18) / 22.0)
    return ev * la_fit

def _quality_rates(bip):
    """Barrel% and HardHit% over batted balls with EV+LA."""
    b = bip[bip["ExitSpeed"].notna() & bip["Angle"].notna()]
    if len(b) == 0:
        return np.nan, np.nan, 0
    sc = b.apply(lambda r: _barrel_score(r["ExitSpeed"], r["Angle"]), axis=1)
    barrel = (sc >= _BARREL_SCORE_THRESH).mean() * 100
    hard = (b["ExitSpeed"] >= _HARDHIT_EV).mean() * 100
    return barrel, hard, len(b)


def _heat3x3_svg(grid, title="", fmt_fn=None, subtitle=""):
    """3x3 heat grid over the strike zone (catcher view). grid = 3x3 list of
    (value, n) with row 0 = up. Cells sized to the true zone: 6.64 in wide each,
    6.49 in tall each (19.9 in / 3 by 19.5 in / 3)."""
    if fmt_fn is None:
        fmt_fn = lambda v: format(v, ".3f")
    vals = [c[0] for row in grid for c in row if c[0] == c[0]]
    lo, hi = (min(vals), max(vals)) if vals else (0, 1)
    rng = (hi - lo) or 1
    S = 12.0
    cw, chh = 6.64 * S, 6.49 * S
    x0, y0 = 200 - 1.5 * cw, 190 - 1.5 * chh
    parts = []
    for i, row in enumerate(grid):
        for j, (v, n) in enumerate(row):
            x, y = x0 + j * cw, y0 + i * chh
            if v != v:
                fill = "#111827"; txt = "-"
            else:
                t = (v - lo) / rng
                r = int(30 + t * (239 - 30)); g = int(58 + t * (68 - 58)); b = int(138 - t * (138 - 68))
                fill = "rgb(" + str(r) + "," + str(g) + "," + str(b) + ")"
                txt = fmt_fn(v)
            parts.append('<rect x="' + format(x, ".1f") + '" y="' + format(y, ".1f") +
                         '" width="' + format(cw, ".1f") + '" height="' + format(chh, ".1f") +
                         '" fill="' + fill + '" stroke="#475569"/>')
            parts.append('<text x="' + format(x + cw / 2, ".1f") + '" y="' + format(y + chh / 2, ".1f") +
                         '" fill="#f8fafc" font-size="13" font-weight="600" text-anchor="middle">' + txt + "</text>")
            parts.append('<text x="' + format(x + cw / 2, ".1f") + '" y="' + format(y + chh / 2 + 14, ".1f") +
                         '" fill="#cbd5e1" font-size="9" text-anchor="middle">n=' + str(n) + "</text>")
    parts.append('<text x="200" y="' + format(y0 + 3 * chh + 18, ".0f") +
                 '" fill="#475569" font-size="9.5" text-anchor="middle">'
                 'catcher view - zone 19.9 in wide, 21.1-40.5 in tall - each cell 6.6 x 6.5 in</text>')
    sub = ('<div style="color:#475569;font-size:10.5px;">' + subtitle + "</div>") if subtitle else ""
    return ('<div style="text-align:center;font-family:sans-serif;">'
            '<div style="color:#1e293b;font-size:13px;font-weight:600;">' + title + "</div>" + sub +
            '<svg viewBox="0 0 400 300" width="330" height="248">' + "".join(parts) + "</svg></div>")


# ── Optimal shift — recommended fielder positions from a hitter's spray ──
# Standard (no-shift) alignment, feet from home plate — x: - = left field/3B
# side, + = right field/1B side (matches direction_to_xy's sign convention);
# y = depth. Modeled on typical starting depths: corner infielders (1B/3B)
# play shallower and closer to their foul lines; middle infielders (2B/SS)
# play deeper and much closer to the 2B bag/centerline than the corners do;
# outfielders spread evenly with center field deepest.
_STANDARD_FIELD_POS = {
    "P": (0, 59), "C": (0, 5),
    "3B": (-70, 96), "SS": (-46, 148), "2B": (38, 133), "1B": (57, 89),
    "LF": (-135, 295), "CF": (0, 320), "RF": (135, 295),
}
_SHIFTABLE = ["3B", "SS", "2B", "1B", "LF", "CF", "RF"]

# How far each fielder is allowed to shade off their standard spot (feet),
# clamped so nobody ever drifts into a neighboring position's territory or
# crosses the middle of the infield/outfield they don't cover.
_POSITION_BOUNDS = {
    "3B": (-95, -45), "SS": (-75, -20), "2B": (15, 75), "1B": (30, 90),
    "LF": (-185, -95), "CF": (-50, 50), "RF": (95, 185),
}

def _shift_positions(bp, side):
    """Recommended fielder (x,y) ft-from-plate for this hitter, shaded toward
    his real pull tendency (groundballs move the infield, air balls move the
    outfield — with OF depth nudged by his average air-ball distance).
    Returns (positions dict, n_bip) — n_bip lets callers show a low-sample flag."""
    bip = bp[bp["PitchCall"].eq("InPlay") & bp["Direction"].notna() &
             bp["Distance"].notna() & bp["Angle"].notna()]
    pos = dict(_STANDARD_FIELD_POS)
    n = len(bip)
    if n < 8:
        return pos, n

    gb  = bip[bip["Angle"] < 10]
    air = bip[bip["Angle"] >= 10]
    # Pull side is the side OPPOSITE the batter's stance: RHH pulls to left
    # field (negative direction/x), LHH pulls to right field (positive).
    pull_sign = -1 if side != "Left" else 1

    def _pull_pct(sub):
        if len(sub) < 3:
            return 0.5
        is_pull = (sub["Direction"] < 0) if pull_sign < 0 else (sub["Direction"] > 0)
        return float(is_pull.mean())

    gb_pull  = _pull_pct(gb)
    air_pull = _pull_pct(air)
    avg_air_dist = float(air["Distance"].mean()) if len(air) >= 3 else 300.0
    depth_scale = float(np.clip(avg_air_dist / 300.0, 0.85, 1.2))

    MAX_INF_SHIFT, MAX_OF_SHIFT = 35.0, 55.0
    inf_shift = pull_sign * (gb_pull - 0.5) * 2 * MAX_INF_SHIFT
    of_shift  = pull_sign * (air_pull - 0.5) * 2 * MAX_OF_SHIFT

    def _clamped(label, x):
        lo, hi = _POSITION_BOUNDS[label]
        return float(np.clip(x, lo, hi))

    bx, by = _STANDARD_FIELD_POS["3B"];  pos["3B"] = (_clamped("3B", bx + inf_shift * 0.6), by)
    bx, by = _STANDARD_FIELD_POS["SS"];  pos["SS"] = (_clamped("SS", bx + inf_shift), by)
    bx, by = _STANDARD_FIELD_POS["2B"];  pos["2B"] = (_clamped("2B", bx + inf_shift), by)
    bx, by = _STANDARD_FIELD_POS["1B"];  pos["1B"] = (_clamped("1B", bx + inf_shift * 0.6), by)
    bx, by = _STANDARD_FIELD_POS["LF"];  pos["LF"] = (_clamped("LF", bx + of_shift), by * depth_scale)
    bx, by = _STANDARD_FIELD_POS["CF"];  pos["CF"] = (_clamped("CF", bx + of_shift * 0.5), by * depth_scale)
    bx, by = _STANDARD_FIELD_POS["RF"];  pos["RF"] = (_clamped("RF", bx + of_shift), by * depth_scale)
    return pos, n

def _field_svg(positions, title="", width=240):
    """Top-down field diagram (foul lines + outfield arc) with a labeled dot
    per fielder. positions: {label: (x_ft, y_ft)}, same sign convention as
    direction_to_xy (- = left field, + = right field)."""
    VB_W, VB_H = 340, 330
    S = 0.62
    cx, cy = VB_W / 2, VB_H - 14
    def px(x, y):
        return cx + x * S, cy - y * S
    parts = []
    for sign in (-1, 1):
        fx, fy = px(sign * 300 * 0.7071, 300 * 0.7071)
        parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{fx:.1f}" y2="{fy:.1f}" '
                     f'stroke="#94a3b8" stroke-width="1"/>')
    arc = []
    for deg in range(-45, 46, 5):
        rad = np.radians(deg)
        x, y = 380 * np.sin(rad), 380 * np.cos(rad)
        arc.append(px(x, y))
    path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in arc)
    parts.append(f'<path d="{path}" fill="none" stroke="#94a3b8" stroke-width="1"/>')
    hx, hy = px(0, 0)
    parts.append(f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="3" fill="#1e293b"/>')
    for label, (x, y) in positions.items():
        fx, fy = px(x, y)
        color = "#64748b" if label in ("P", "C") else "#C8102E"
        parts.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="10" fill="{color}" '
                     f'stroke="#fff" stroke-width="1.5"/>')
        parts.append(f'<text x="{fx:.1f}" y="{fy+3:.1f}" font-size="8.5" fill="#fff" '
                     f'text-anchor="middle" font-weight="700">{label}</text>')
    height = int(VB_H * width / VB_W)
    body = "".join(parts)
    ttl = f'<div style="font-size:0.78rem;font-weight:700;color:#1e293b;margin-bottom:2px;">{title}</div>' if title else ""
    return (f'<div style="text-align:center;font-family:sans-serif;">{ttl}'
            f'<svg viewBox="0 0 {VB_W} {VB_H}" width="{width}" height="{height}">{body}</svg></div>')


def calc_xba(ev, la):
    p1,p2,p3,ph = _raw_probs(ev, la)
    return min(0.990, (p1+p2+p3+ph) * XBA_CAL_FACTOR)

def calc_xslg(ev, la):
    p1,p2,p3,ph = _raw_probs(ev, la)
    return min(3.990, (p1*1 + p2*2 + p3*3 + ph*4) * XSLG_CAL_FACTOR)

def calc_xwoba_bip(ev, la):
    p1,p2,p3,ph = _raw_probs(ev, la)
    raw = p1*WOBA_WEIGHTS["1b"]+p2*WOBA_WEIGHTS["2b"]+p3*WOBA_WEIGHTS["3b"]+ph*WOBA_WEIGHTS["hr"]
    return min(3.990, raw * XWOBA_CAL_FACTOR)

def batter_expected_stats(bp_df):
    """Compute xBA, xSLG, xwOBA per plate appearance (not per BIP).
    Denominator = AB (PA - BB - HBP - SAC), numerator = sum of hit probs on fair BIP.
    Strikeouts contribute 0 to numerator but count in denominator — same as real BA.
    """
    # Get AB count from last pitch of each PA
    group_cols = [c for c in ["GameID","Inning","Top/Bottom","PAofInning"] if c in bp_df.columns]
    if group_cols:
        last = bp_df.groupby(group_cols).last().reset_index()
    else:
        last = bp_df.copy()

    is_bb  = last["KorBB"].eq("Walk") if "KorBB" in last.columns else pd.Series(False, index=last.index)
    is_hbp = last["PitchCall"].eq("HitByPitch") if "PitchCall" in last.columns else pd.Series(False, index=last.index)
    is_sac = last["PlayResult"].eq("Sacrifice") if "PlayResult" in last.columns else pd.Series(False, index=last.index)
    ab = (~is_bb & ~is_hbp & ~is_sac).sum()

    if ab == 0:
        return None, None, None, 0

    # Fair balls in play
    fair = bp_df[
        bp_df["ExitSpeed"].notna() & bp_df["Angle"].notna() &
        bp_df["Direction"].notna() &
        (bp_df["Distance"].fillna(0) >= 10) &
        (bp_df["Direction"].abs() <= 45)
    ].copy()

    if len(fair) == 0:
        return None, None, None, 0

    fair["_xba"]   = fair.apply(lambda r: calc_xba(r["ExitSpeed"], r["Angle"]), axis=1)
    fair["_xslg"]  = fair.apply(lambda r: calc_xslg(r["ExitSpeed"], r["Angle"]), axis=1)
    fair["_xwoba"] = fair.apply(lambda r: calc_xwoba_bip(r["ExitSpeed"], r["Angle"]), axis=1)

    n_bip = len(fair)
    xba   = round(fair["_xba"].sum()   / ab, 3)
    xslg  = round(fair["_xslg"].sum()  / ab, 3)
    xwoba = round(fair["_xwoba"].sum() / ab, 3)

    return xba, xslg, xwoba, n_bip

def batter_xba(bp_df):
    xba, _, _, n = batter_expected_stats(bp_df)
    return xba, n

# 
#  BATTER STATS HELPER
# 
def compute_batter_stats(batter, my_pitches, opp_hand):
    bp = my_pitches[my_pitches["Batter"] == batter]
    bb = bp[bp["ExitSpeed"].notna()]
    side = bp["BatterSide"].iloc[0] if "BatterSide" in bp.columns and len(bp) > 0 else "?"
    pa   = bp[bp["PitchofPA"] == 1].shape[0] if "PitchofPA" in bp.columns else len(bp)
    hits = bp["PlayResult"].isin(["Single","Double","Triple","HomeRun"]).sum() if "PlayResult" in bp.columns else 0
    bb_c = bp["KorBB"].eq("Walk").sum() if "KorBB" in bp.columns else 0
    hbp  = bp["PitchCall"].eq("HitByPitch").sum() if "PitchCall" in bp.columns else 0
    sac  = bp["PlayResult"].eq("Sacrifice").sum() if "PlayResult" in bp.columns else 0
    obp_calc = (hits + bb_c + hbp) / max(pa - sac, 1)

    # Use official stats when available — override Trackman calculations
    off_ba  = get_official_stat(batter, "ba")
    off_obp = get_official_stat(batter, "obp")
    off_slg = get_official_stat(batter, "slg")
    off_pa  = get_official_stat(batter, "ab")  # use AB as PA proxy if available
    off_g   = get_official_stat(batter, "g")
    off_bb  = get_official_stat(batter, "bb")
    off_so  = get_official_stat(batter, "so")
    has_official = off_ba is not None

    obp = off_obp if off_obp is not None else obp_calc
    if off_pa is not None: pa = int(off_pa)
    if off_bb is not None: bb_c = int(off_bb)
    k_pct_official = (int(off_so) / max(int(off_pa), 1)) if (off_so is not None and off_pa is not None) else None
    bb_pct_official = (int(off_bb) / max(int(off_pa), 1)) if (off_bb is not None and off_pa is not None) else None
    avg_ev   = bb["ExitSpeed"].mean() if len(bb) > 0 else None
    max_ev   = bb["ExitSpeed"].max()  if len(bb) > 0 else None
    hard_pct = (bb["ExitSpeed"] >= 90).mean() if len(bb) > 0 else None
    avg_la   = bb["Angle"].mean() if len(bb) > 0 and "Angle" in bb.columns else None
    ev_rhp   = bb[bb["PitcherThrows"]=="Right"]["ExitSpeed"].mean() if len(bb) > 0 else None
    ev_lhp   = bb[bb["PitcherThrows"]=="Left"]["ExitSpeed"].mean()  if len(bb) > 0 else None
    # OPS = OBP + SLG
    singles  = bp["PlayResult"].eq("Single").sum()   if "PlayResult" in bp.columns else 0
    doubles  = bp["PlayResult"].eq("Double").sum()   if "PlayResult" in bp.columns else 0
    triples  = bp["PlayResult"].eq("Triple").sum()   if "PlayResult" in bp.columns else 0
    hr       = bp["PlayResult"].eq("HomeRun").sum()  if "PlayResult" in bp.columns else 0
    slg_num  = singles + 2*doubles + 3*triples + 4*hr
    slg_den  = max(pa - bb_c - hbp - sac, 1)
    slg_calc = slg_num / slg_den
    off_slg  = get_official_stat(batter, "slg")
    slg      = off_slg if off_slg is not None else slg_calc
    ops      = obp + slg

    # ── Lineup ranking score: wOBA from actual outcomes, shrunk toward league
    # by PA so small-sample flukes don't outrank real producers. ──
    _WOBA_W = {"BB": 0.69, "HBP": 0.72, "1B": 0.89, "2B": 1.27, "3B": 1.62, "HR": 2.10}
    _woba_num = (_WOBA_W["BB"] * bb_c + _WOBA_W["HBP"] * hbp +
                 _WOBA_W["1B"] * singles + _WOBA_W["2B"] * doubles +
                 _WOBA_W["3B"] * triples + _WOBA_W["HR"] * hr)
    _woba_raw = _woba_num / max(pa, 1)
    _LEAGUE_WOBA = 0.320     # FCBL-ish league average to shrink toward
    _K_PA = 12               # at 12 PA, 50/50 blend of player vs league
    _w = pa / (pa + _K_PA)
    woba_shrunk = _w * _woba_raw + (1 - _w) * _LEAGUE_WOBA

    # Platoon nudge kept as a small adjustment on top of wOBA.
    obp_score    = obp * 40
    ev_score     = ((avg_ev or 70) - 70) / 30 * 30
    platoon_score = 0
    platoon_adv   = False
    if opp_hand == "Right" and ev_rhp is not None:
        platoon_score = ((ev_rhp - 75) / 20) * 20
        platoon_adv   = side == "Left"
    elif opp_hand == "Left" and ev_lhp is not None:
        platoon_score = ((ev_lhp - 75) / 20) * 20
        platoon_adv   = side == "Right"
    # Score is now wOBA-driven (scaled to a familiar range) with a small platoon tilt.
    raw_score = woba_shrunk * 100 + (platoon_score * 0.15)

    # Sample-size handling is already baked into woba_shrunk above (it blends
    # the player's wOBA toward league by PA), so use the score directly here —
    # no second shrinkage, which would over-flatten everyone.
    weight = pa / (pa + 10)   # kept for display/other uses
    total = raw_score
    k_pct  = k_pct_official if k_pct_official is not None else bp["KorBB"].eq("Strikeout").sum() / max(pa, 1)
    bb_pct = bb_pct_official if bb_pct_official is not None else bb_c / max(pa, 1)
    xba, xslg, xwoba, xba_n = batter_expected_stats(bp)
    xba, xslg, xwoba, xba_n = batter_expected_stats(bp)
    return dict(Batter=batter, BasPos=base_pos(batter), Side=side, PA=pa,
        OBP=obp, OPS=ops, KPct=k_pct, BBPct=bb_pct, HasOfficial=has_official,
        AvgEV=avg_ev, MaxEV=max_ev, HardPct=hard_pct, AvgLA=avg_la,
        EV_RHP=ev_rhp, EV_LHP=ev_lhp, PlatoonAdv=platoon_adv, Score=total,
        xBA=xba, xSLG=xslg, xwOBA=xwoba, xBA_n=xba_n)


# ─────────────────────────────────────────
#  SEASON REPORT — shared calc helpers
#  (exit-velo percentiles, FIP/xFIP twin for use outside the Pitcher
#  Scouting tab, and a from-scratch xERA estimate built on tracked
#  batted-ball data since no formula for it exists anywhere else here)
# ─────────────────────────────────────────
def _ev_stats(bip):
    """(avg_ev, max_ev, ev90) from a balls-in-play slice, or (None, None, None)."""
    ev = pd.to_numeric(bip["ExitSpeed"], errors="coerce").dropna()
    if len(ev) == 0:
        return None, None, None
    return float(ev.mean()), float(ev.max()), float(ev.quantile(0.90))


_SEASON_FIP_CONSTANT = 3.10   # same anchor used in the Pitcher Scouting Metrics tab
_SEASON_LEAGUE_HR_FB = 0.08

def _season_pitcher_fip_metrics(pitcher_df):
    """FIP/xFIP + supporting counts for one pitcher's pitches. A standalone
    copy of the Pitcher Scouting Metrics tab's calc_pitcher_metrics (that one
    is a nested function local to that tab), kept in sync by formula/constants
    so the Season Report's numbers always match what's shown there."""
    d = pitcher_df
    k = d["KorBB"].eq("Strikeout").sum()
    if k == 0:
        k = d[d["PitchCall"].isin(["StrikeSwinging", "StrikeCalled"]) & (d["Strikes"] == 2)].shape[0]
    bb  = d["KorBB"].eq("Walk").sum()
    hbp = d["PitchCall"].eq("HitByPitch").sum()
    hr  = d["PlayResult"].eq("HomeRun").sum()
    outs = d["OutsOnPlay"].fillna(0).sum() + k
    ip = outs / 3
    fb_tagged = d["TaggedHitType"].eq("FlyBall").sum()
    fb_auto = d["AutoHitType"].eq("FlyBall").sum() if "AutoHitType" in d.columns else 0
    fb = max(fb_tagged, fb_auto)
    fip = (13*hr + 3*(bb+hbp) - 2*k) / ip + _SEASON_FIP_CONSTANT if ip >= 2.0 else None
    xhr = fb * _SEASON_LEAGUE_HR_FB
    xfip = (13*xhr + 3*(bb+hbp) - 2*k) / ip + _SEASON_FIP_CONSTANT if ip >= 2.0 else None
    return dict(IP=to_ip(outs), ip_dec=ip, K=int(k), BB=int(bb), HBP=int(hbp), HR=int(hr),
                FIP=round(fip, 2) if fip is not None else None,
                xFIP=round(xfip, 2) if xfip is not None else None)


_XERA_WOBA_SCALE = 1.15    # approx MLB "runs per 1.0 wOBA point" scale
_XERA_PA_PER_9   = 38.5    # approx batters faced per 9 innings
_XERA_BB_W, _XERA_HBP_W = 0.69, 0.72

def _pa_xwoba_against(pitcher_df):
    """Full-PA expected-wOBA-against: actual BB/HBP get their standard wOBA
    weight, strikeouts contribute 0, and balls in play get their expected
    wOBA-on-contact from tracked exit velo/launch angle (calc_xwoba_bip).
    Returns (xwoba_per_pa, pa) or (None, 0) if there's no PA data."""
    d = pitcher_df
    pa = d[d["PitchofPA"] == 1].shape[0] if "PitchofPA" in d.columns else 0
    if pa == 0:
        return None, 0
    bb = d["KorBB"].eq("Walk").sum()
    hbp = d["PitchCall"].eq("HitByPitch").sum()
    bip = d[d["PitchCall"].eq("InPlay") & d["ExitSpeed"].notna() & d["Angle"].notna()]
    bip_sum = bip.apply(lambda r: calc_xwoba_bip(r["ExitSpeed"], r["Angle"]), axis=1).sum() if len(bip) else 0.0
    numer = bb*_XERA_BB_W + hbp*_XERA_HBP_W + bip_sum
    return numer / pa, pa

def calc_xera_estimate(pitcher_df, league_df, min_pa=20):
    """DiamondIntel's own xERA estimate — NOT Statcast's proprietary metric,
    which is a black-box regression fit to years of MLB data we don't have.
    This version rescales expected-wOBA-against (built from the same tracked
    exit-velo/launch-angle data used everywhere else in the app) onto a
    runs-per-9 scale, anchored to the same league-average run environment as
    the FIP constant. Treat it as a real, data-driven estimate — not a
    industry-standard number. Returns None below min_pa."""
    xwoba_pa, pa = _pa_xwoba_against(pitcher_df)
    if xwoba_pa is None or pa < min_pa:
        return None
    lg_xwoba_pa, lg_pa = _pa_xwoba_against(league_df)
    if lg_xwoba_pa is None or lg_pa == 0:
        return None
    runs_above_avg_per_pa = (xwoba_pa - lg_xwoba_pa) / _XERA_WOBA_SCALE
    xera = _SEASON_FIP_CONSTANT + runs_above_avg_per_pa * _XERA_PA_PER_9
    return round(max(0.0, xera), 2)


#
#  PAGE: LINEUP BUILDER
#

# 
#  PAGE: SPRAY CHART
# 
def _xwoba_on_contact(ev, la):
    """Rough xwOBA-on-contact from exit velo + launch angle."""
    if pd.isna(ev) or pd.isna(la):
        return np.nan
    if ev >= 95 and 8 <= la <= 32:
        return 1.20
    if ev >= 90 and 5 <= la <= 35:
        return 0.75
    if 0 <= la <= 40 and ev >= 80:
        return 0.45
    return 0.18

@st.cache_data(ttl=600, max_entries=3)
def _build_gameplan_pdf(pitcher, throws_lbl, opp, ars_rows, tend_rows, lineup_rows,
                        attack_notes, hand_lbl, logo_path="assets/nashua_logo.png"):
    """Build a branded PDF game plan; returns bytes. Logo used if the file exists."""
    import io, os
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    NAVY = colors.HexColor("#0a2240")
    SILVER = colors.HexColor("#8a9199")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.55*inch, bottomMargin=0.55*inch)
    styles = getSampleStyleSheet()
    story = []
    if logo_path and os.path.exists(logo_path):
        try:
            story.append(Image(logo_path, width=0.95*inch, height=0.95*inch))
        except Exception:
            pass
    title = ParagraphStyle("gp_t", parent=styles["Title"], textColor=NAVY,
                           fontSize=18, spaceAfter=2, alignment=0)
    sub = ParagraphStyle("gp_s", parent=styles["Normal"], textColor=SILVER,
                         fontSize=11, spaceAfter=10)
    story.append(Paragraph("Brookhaven Bandits Baseball Scouting", title))
    story.append(Paragraph("Pre-Game Game Plan &mdash; vs " + pitcher + " (" + throws_lbl +
                           ") &middot; " + opp, sub))
    h = ParagraphStyle("gp_h", parent=styles["Heading2"], textColor=NAVY,
                       fontSize=13, spaceBefore=10, spaceAfter=4)

    def _tbl(data):
        t = Table(data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f2f5")]),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))
        return t

    story.append(Paragraph("Arsenal &amp; Usage", h))
    story.append(_tbl([["Pitch", "Usage", "Velo", "Spin"]] +
                      [[r["Pitch"], r["Usage"], r["Velo"], r["Spin"]] for r in ars_rows]))
    if tend_rows:
        story.append(Paragraph("Tendencies by Count", h))
        story.append(_tbl([["Count", "n", "Likely pitches"]] +
                          [[r["Count"], str(r["n"]), r["Likely"]] for r in tend_rows]))
    story.append(Paragraph("Ranked Projected Matchups vs " + hand_lbl, h))
    story.append(_tbl([["#", "Hitter", "Side", "BA", "Proj xwOBA"]] +
                      [[str(i+1), r["Hitter"], r["Side"], r["BA"], r["Proj"]]
                       for i, r in enumerate(lineup_rows)]))
    story.append(Paragraph("Attack Plans", h))
    body = ParagraphStyle("gp_b", parent=styles["Normal"], fontSize=9, spaceAfter=2)
    for note in attack_notes:
        story.append(Paragraph(note, body))
    story.append(Spacer(1, 10))
    foot = ParagraphStyle("gp_f", parent=styles["Normal"], textColor=SILVER, fontSize=7.5)
    story.append(Paragraph("Projections are shrinkage-based (lean on league averages when data "
                           "is thin) &mdash; directional game-planning tools, not guarantees.", foot))
    doc.build(story)
    return buf.getvalue()


@st.cache_data(ttl=600, max_entries=3)
def _build_hitter_scouting_pdf(hitter_name, hand_lbl, pitcher_name, throws_lbl,
                               vs_hand, mix_rows, ab_log,
                               logo_path="assets/nashua_logo.png"):
    """Printable scouting report for one Bandits hitter facing one
    specific opposing pitcher: this pitcher's location/contact/whiff heat
    maps and pitch mix — both scoped to the hitter's own batting side, since
    a pitcher's mix and locations shift by who's up — plus every recorded
    head-to-head at-bat between the two of them, pitch by pitch, each with
    its own catcher's-eye zone plot, on its own page."""
    import io, os
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image, PageBreak, KeepTogether)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    RED = colors.HexColor("#C8102E")     # Knights red — matches the app's --knights accent
    INK = colors.HexColor("#1E293B")
    STRIPE = colors.HexColor("#FBEAEA")  # faint red row stripe
    RULE = colors.HexColor("#E2E8F0")
    AB_OUTCOME_HEX = {
        "Single": "#22c55e", "Double": "#3b82f6", "Triple": "#8b5cf6",
        "HomeRun": "#f59e0b", "Walk": "#06b6d4", "HBP": "#06b6d4",
        "Out": "#C8102E", "FieldersChoice": "#f97316", "Error": "#ec4899",
        "Strikeout": "#C8102E", "Sacrifice": "#64748b", "—": "#475569",
    }

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.55*inch, bottomMargin=0.55*inch)
    styles = getSampleStyleSheet()
    story = []

    def _rule(height=2.0):
        t = Table([[""]], colWidths=[7.4*inch], rowHeights=[height])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), RED),
                               ("TOPPADDING", (0, 0), (-1, -1), 0),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        return t

    title = ParagraphStyle("hr_t", parent=styles["Title"], textColor=RED,
                           fontSize=19, spaceAfter=0, alignment=0)
    sub = ParagraphStyle("hr_s", parent=styles["Normal"], textColor=INK,
                         fontSize=11, spaceAfter=0)
    title_block = [Paragraph("Brookhaven Bandits Baseball Scouting", title),
                   Paragraph(f"{hitter_name} ({hand_lbl}) vs {pitcher_name} ({throws_lbl})", sub)]
    if logo_path and os.path.exists(logo_path):
        try:
            head = Table([[Image(logo_path, width=0.7*inch, height=0.7*inch), title_block]],
                        colWidths=[0.85*inch, 6.55*inch])
        except Exception:
            head = Table([[title_block]], colWidths=[7.4*inch])
    else:
        head = Table([[title_block]], colWidths=[7.4*inch])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("LEFTPADDING", (0, 0), (0, 0), 0)]))
    story.append(head)
    story.append(Spacer(1, 6))
    story.append(_rule(2.4))
    story.append(Spacer(1, 10))

    h = ParagraphStyle("hr_h", parent=styles["Heading2"], textColor=RED,
                       fontSize=13, spaceBefore=10, spaceAfter=3)
    body = ParagraphStyle("hr_b", parent=styles["Normal"], textColor=INK, fontSize=9, spaceAfter=2)

    def _section(title_txt):
        story.append(Paragraph(title_txt, h))
        story.append(_rule(1.1))
        story.append(Spacer(1, 5))

    def _tbl(data, col_widths=None):
        t = Table(data, hAlign="LEFT", colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), RED),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("TEXTCOLOR", (0, 1), (-1, -1), INK),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, RULE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, STRIPE]),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
        return t

    # ── Location tendencies (heat maps), scoped to this hitter's side ──
    _section(f"{pitcher_name} vs {hand_lbl} — Location Tendencies")
    bip = vs_hand[vs_hand["ExitSpeed"].notna() & vs_hand["PlateLocSide"].notna() &
                  vs_hand["PlateLocHeight"].notna()]
    SWING_C_PDF = {"StrikeSwinging","InPlay","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}
    sw = vs_hand[vs_hand["PitchCall"].isin(SWING_C_PDF) & vs_hand["PlateLocSide"].notna() &
                 vs_hand["PlateLocHeight"].notna()].copy()
    if len(sw):
        sw["_whiff_w"] = sw["PitchCall"].eq("StrikeSwinging").astype(float)

    panels = [(vs_hand, None, "Pitch Location"), (bip, "ExitSpeed", "Hard Contact"),
              (sw, "_whiff_w", "Whiffs")]
    imgs = [_kde_heatmap_png(data, weight_col=wcol) if len(data) >= 5 else None
            for data, wcol, _lbl in panels]

    if any(imgs):
        cap_row, img_row = [], []
        for (_, _, lbl), png in zip(panels, imgs):
            cap_row.append(Paragraph(f"<b>{lbl}</b>", body))
            img_row.append(Image(io.BytesIO(png), width=2.1*inch, height=2.35*inch)
                           if png else Paragraph("Not enough data", body))
        heat_table = Table([cap_row, img_row], colWidths=[2.2*inch]*3)
        heat_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(heat_table)
    else:
        story.append(Paragraph(f"Not enough location data vs {hand_lbl} to build heat maps.", body))

    # ── Pitch mix vs this hand ──
    _section(f"Pitch Mix vs {hand_lbl}")
    if mix_rows:
        story.append(_tbl([["Pitch","Usage","Pitches","Velo","IVB","HB","Spin"]] +
                          [[r["Pitch"], r["Usage"], str(r["Pitches"]), r["Velo"],
                            r["IVB"], r["HB"], r["Spin"]] for r in mix_rows]))
    else:
        story.append(Paragraph(f"No recorded pitches vs {hand_lbl} for this pitcher.", body))

    # ── Pitch usage by count situation — 0-0, pitcher's counts, hitter's
    # counts, 0-2, and full count (not the whole 12-count matrix; those are
    # the situations a coach actually calls a pitch by) ──
    _section(f"Pitch Usage by Count vs {hand_lbl}")
    PDF_COUNT_GROUPS = [
        ("0-0",              [(0, 0)]),
        ("Pitcher's Counts", [(0, 1), (0, 2), (1, 2)]),
        ("Hitter's Counts",  [(2, 0), (3, 0), (2, 1), (3, 1)]),
        ("0-2",              [(0, 2)]),
        ("Full Count (3-2)", [(3, 2)]),
    ]
    vs_hand_clean = vs_hand[vs_hand["PitchType"].notna() & (vs_hand["PitchType"] != "None")]
    pdf_pitch_types = [r["Pitch"] for r in mix_rows] if mix_rows else \
        sorted(vs_hand_clean["PitchType"].dropna().unique())
    group_n = {}
    group_pct = {}
    for gname, counts in PDF_COUNT_GROUPS:
        grp = vs_hand_clean[vs_hand_clean.apply(lambda r: (r["Balls"], r["Strikes"]) in counts, axis=1)]
        group_n[gname] = len(grp)
        group_pct[gname] = grp["PitchType"].value_counts(normalize=True)

    if pdf_pitch_types and any(n > 0 for n in group_n.values()):
        header = ["Pitch"] + [f"{g} (n={group_n[g]})" for g, _ in PDF_COUNT_GROUPS]
        rows = []
        for pt in pdf_pitch_types:
            row = [pt]
            for gname, _ in PDF_COUNT_GROUPS:
                pct = group_pct[gname].get(pt) if group_n[gname] > 0 else None
                row.append(f"{pct:.0%}" if pct is not None else "—")
            rows.append(row)
        story.append(_tbl([header] + rows, col_widths=[1.3*inch] + [1.22*inch]*5))
    else:
        story.append(Paragraph(f"Not enough count data vs {hand_lbl} to break out by count.", body))

    # ── Head-to-head at-bat history — its own page, one zone plot per AB ──
    story.append(PageBreak())
    _section(f"At-Bat History — {hitter_name} vs {pitcher_name}")
    if not ab_log:
        story.append(Paragraph("No recorded plate appearances between these two players.", body))
    else:
        ab_title_style = ParagraphStyle("hr_ab", parent=body, fontSize=10, spaceAfter=3)
        for i, ab in enumerate(ab_log, 1):
            date_str = ab["date"].strftime("%b %d, %Y") if pd.notna(ab["date"]) else "—"
            inning = int(ab["inning"]) if pd.notna(ab["inning"]) else "?"
            oc_hex = AB_OUTCOME_HEX.get(ab["outcome"], "#64748b")
            bb_txt = ""
            if pd.notna(ab.get("exit_speed")):
                bb_parts = [f"EV {ab['exit_speed']:.1f} mph"]
                if pd.notna(ab.get("angle")):
                    bb_parts.append(f"LA {ab['angle']:.0f}&deg;")
                if pd.notna(ab.get("distance")):
                    bb_parts.append(f"{ab['distance']:.0f} ft")
                bb_txt = ("  &mdash;  <font color='#64748b' size=8>"
                          + " &middot; ".join(bb_parts) + "</font>")
            ab_block = [Paragraph(
                f"<b>AB {i}</b> — {date_str}, Inning {inning} &mdash; "
                f"<font color='{oc_hex}'><b>{ab['outcome']}</b></font>{bb_txt}", ab_title_style)]
            prows = [["Pitch #","Count","Type","Velo","IVB","HB","Zone","Call"]]
            for p in ab["pitches"]:
                prows.append([
                    str(p["num"]), p["count"], p["type"],
                    f"{p['velo']:.1f}" if pd.notna(p["velo"]) else "—",
                    f"{p['ivb']:.1f}\"" if pd.notna(p["ivb"]) else "—",
                    f"{p['hb']:.1f}\"" if pd.notna(p["hb"]) else "—",
                    p["zone"], p["call"],
                ])
            pitch_tbl = _tbl(prows, col_widths=[0.4*inch, 0.45*inch, 0.65*inch, 0.45*inch,
                                                0.45*inch, 0.45*inch, 0.7*inch, 0.75*inch])
            zone_png = _ab_pitch_plot_png(ab["pitches"])
            zone_cell = (Image(io.BytesIO(zone_png), width=1.9*inch, height=2.15*inch)
                        if zone_png else Paragraph("No location data", body))
            combo = Table([[pitch_tbl, zone_cell]], colWidths=[4.55*inch, 2.2*inch])
            combo.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                       ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
            ab_block.append(combo)
            ab_block.append(Spacer(1, 8))
            story.append(KeepTogether(ab_block))

    doc.build(story)
    return buf.getvalue()


# ─────────────────────────────────────────
#  SCOUTING SHEET (printable one-page report)
# ─────────────────────────────────────────
_SS_HITS  = ["Single", "Double", "Triple", "HomeRun"]
_SS_TAKE  = {"BallCalled", "StrikeCalled", "BallinDirt", "HitByPitch"}
_SS_STRK  = {"StrikeCalled", "StrikeSwinging", "FoulBallNotFieldable",
             "FoulBallFieldable", "InPlay"}

def _ss_clean_counts(df):
    """Drop games whose count columns never advance (corrupt tracking)."""
    bad = set()
    for g, grp in df.groupby("GameID"):
        b = pd.to_numeric(grp["Balls"], errors="coerce").max()
        s = pd.to_numeric(grp["Strikes"], errors="coerce").max()
        if b == 0 and s == 0:
            bad.add(g)
    return df[~df["GameID"].isin(bad)]

def _ss_ba(sub):
    ab = int((_ab_mask(sub)).sum())
    h = int(sub["PlayResult"].isin(_SS_HITS).sum())
    return h, ab

def _ss_count_take(sub, balls, strikes):
    """Return (take%, in-zone take%) for one count.
    T%    = takes / all pitches seen in this count.
    IZ T% = takes on pitches inside the strike zone / pitches inside the zone.
            i.e. how often he lets a strike go by in this count."""
    c = sub[(sub["Balls"] == balls) & (sub["Strikes"] == strikes)]
    if len(c) == 0:
        return None, None
    took = c["PitchCall"].isin(_SS_TAKE)
    t_pct = took.mean() * 100
    loc = c[c["PlateLocSide"].notna() & c["PlateLocHeight"].notna()]
    inz = loc[(loc["PlateLocSide"].abs() <= 0.83) &
              loc["PlateLocHeight"].between(1.755, 3.378)]
    if len(inz) == 0:
        return t_pct, None
    return t_pct, inz["PitchCall"].isin(_SS_TAKE).mean() * 100

def _ss_pct(v):
    return "—" if v is None or v != v else f"{v:.0f}%"

def _ss_avg(h, ab):
    return "—" if ab == 0 else f"{h/ab:.3f}"

def _ss_jersey_map():
    """Optional jersey numbers from Data/jersey_numbers.json.

    Two accepted shapes:
      flat   {"Smith, Connor": "12", "Jones, Ray": "7"}
      byteam {"WOR_BRA": {"Smith, Connor": "12"}, "NAS_SIL": {...}}
    Names must match TrackMan's "Last, First". Missing players stay blank so the
    column can still be written in by hand.
    """
    import json
    try:
        p = DATA_DIR / "jersey_numbers.json"
        if not p.exists():
            return {}
        with open(p) as f:
            raw = json.load(f)
    except Exception:
        return {}
    flat = {}
    for k, v in raw.items():
        if isinstance(v, dict):          # by-team shape: flatten it
            flat.update(v)
        else:
            flat[k] = v
    return {str(k): str(val) for k, val in flat.items()}


def _scout_tables(df_all, team, min_pitches=10):
    """Build the pitcher and hitter scouting tables for one team."""
    jersey = _ss_jersey_map()
    d = df_all
    dc = _ss_clean_counts(d)

    prows = []
    pd_team = d[d["PitcherTeam"] == team]
    for p in sorted(pd_team["Pitcher"].dropna().unique()):
        if _is_removed(p):
            continue
        pp = pd_team[pd_team["Pitcher"] == p]
        if len(pp) < min_pitches:
            continue
        hl, al = _ss_ba(pp[pp["BatterSide"] == "Left"])
        hr, ar = _ss_ba(pp[pp["BatterSide"] == "Right"])
        ppc = dc[(dc["PitcherTeam"] == team) & (dc["Pitcher"] == p)]
        fp = ppc[ppc["PitchofPA"] == 1]
        def fps(s):
            return None if len(s) == 0 else s["PitchCall"].isin(_SS_STRK).mean() * 100
        prows.append({
            "Pitcher": player_last(p), "#": jersey.get(p, ""),
            "v LHH": _ss_avg(hl, al), "v RHH": _ss_avg(hr, ar),
            "FPS%": _ss_pct(fps(fp)),
            "FPS% L": _ss_pct(fps(fp[fp["BatterSide"] == "Left"])),
            "FPS% R": _ss_pct(fps(fp[fp["BatterSide"] == "Right"])),
        })

    hrows = []
    hd_team = d[d["BatterTeam"] == team]
    for b in sorted(hd_team["Batter"].dropna().unique()):
        if _is_removed(b):
            continue
        bp = hd_team[hd_team["Batter"] == b]
        if len(bp) < min_pitches:
            continue
        hl, al = _ss_ba(bp[bp["PitcherThrows"] == "Left"])
        hr, ar = _ss_ba(bp[bp["PitcherThrows"] == "Right"])
        bpc = dc[(dc["BatterTeam"] == team) & (dc["Batter"] == b)]
        t00, i00 = _ss_count_take(bpc, 0, 0)
        t10, i10 = _ss_count_take(bpc, 1, 0)
        t01, i01 = _ss_count_take(bpc, 0, 1)
        hrows.append({
            "Hitter": player_last(b), "#": jersey.get(b, ""),
            "v LHP": _ss_avg(hl, al), "v RHP": _ss_avg(hr, ar),
            "0-0 T%": _ss_pct(t00), "0-0 IZ T%": _ss_pct(i00),
            "1-0 T%": _ss_pct(t10), "1-0 IZ T%": _ss_pct(i10),
            "0-1 T%": _ss_pct(t01), "0-1 IZ T%": _ss_pct(i01),
            "L-AB": f"{hl}-{al}", "R-AB": f"{hr}-{ar}",
        })
    return pd.DataFrame(prows), pd.DataFrame(hrows)


def _build_scout_pdf(team_name, pit_df, hit_df):
    """One-page landscape PDF that fills the sheet, at the largest readable type.

    Three type sizes are measured separately rather than forcing one size on
    everything: the player name, the body values, and the column headers. Headers
    like "0-0 IZ T%" are long and would otherwise drag every cell down with them.
    A single scale factor then shrinks all three together, and only as far as it
    must, until the sheet lands on one page. Row padding finally expands to take
    up whatever vertical space is left.
    """
    import io
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics

    NAVY = colors.HexColor("#111827")
    PAGE = landscape(letter)
    LR, TOP, BOT = 0.35 * inch, 0.3 * inch, 0.3 * inch
    AVAIL_W = PAGE[0] - 2 * LR
    PAD_X = 5
    NAME_W, HASH_W = 132.0, 44.0

    def _widths(df):
        cols = list(df.columns)
        others = [c for c in cols if c not in (cols[0], "#")]
        rest = AVAIL_W - NAME_W - (HASH_W if "#" in cols else 0)
        each = rest / max(len(others), 1)
        return [NAME_W if c == cols[0] else (HASH_W if c == "#" else each) for c in cols]

    def _fit(texts, width, font, cap):
        """Largest size at which every string fits `width`."""
        avail = width - 2 * PAD_X
        fs = cap
        for txt in texts:
            txt = str(txt)
            if not txt:
                continue
            w = pdfmetrics.stringWidth(txt, font, 10.0)
            if w > 0:
                fs = min(fs, 10.0 * avail / w)
        return max(4.0, fs)

    def _sizes(df):
        cols, w = list(df.columns), _widths(df)
        name = _fit(df[cols[0]], w[0], "Helvetica-Bold", 15.0)
        head = _fit(cols, min(x for c, x in zip(cols, w) if c != cols[0]),
                    "Helvetica-Bold", 12.0)
        body_cols = [c for c in cols[1:]]
        body = min([_fit(df[c], w[cols.index(c)], "Helvetica", 14.0) for c in body_cols] or [12.0])
        return name, body, head

    sz = [_sizes(d) for d in (pit_df, hit_df) if len(d)]
    name_fs = min(x[0] for x in sz) if sz else 12.0
    body_fs = min(x[1] for x in sz) if sz else 11.0
    head_fs = min(x[2] for x in sz) if sz else 9.0

    def _table(df, k, pad):
        nf, bf, hf = name_fs * k, body_fs * k, head_fs * k
        data = [list(df.columns)] + df.astype(str).values.tolist()
        tb = Table(data, colWidths=_widths(df), hAlign="LEFT", repeatRows=1)
        tb.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), hf),
            ("LEADING", (0, 0), (-1, 0), hf + 1.5),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),   # player names
            ("FONTSIZE", (0, 1), (0, -1), nf),
            ("FONTSIZE", (1, 1), (-1, -1), bf),
            ("LEADING", (0, 1), (-1, -1), max(nf, bf) + 1.6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9ca3af")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f2f4")]),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), pad),
            ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
            ("LEFTPADDING", (0, 0), (-1, -1), PAD_X),
            ("RIGHTPADDING", (0, 0), (-1, -1), PAD_X),
        ]))
        return tb

    def _story(k, pad):
        big = body_fs * k >= 8.0
        styles = getSampleStyleSheet()
        t = ParagraphStyle("ss_t", parent=styles["Title"], fontSize=18 if big else 13,
                           spaceAfter=1, alignment=0, textColor=NAVY)
        cap = ParagraphStyle("ss_s", parent=styles["Normal"], fontSize=8.5 if big else 6.8,
                             leading=10 if big else 8, spaceAfter=6 if big else 3,
                             textColor=colors.HexColor("#6b7280"))
        hh = ParagraphStyle("ss_h", parent=styles["Heading2"], fontSize=12.5 if big else 9,
                            spaceBefore=4 if big else 1, spaceAfter=3 if big else 1,
                            textColor=NAVY)
        st_ = [Paragraph(team_name + " &mdash; Scouting Sheet", t),
               Paragraph("v LHH/RHH and v LHP/RHP are batting average. FPS% = first-pitch strike "
                         "rate. T% = take rate in that count. IZ T% = take rate on pitches inside "
                         "the strike zone. L-AB / R-AB = hits-at bats by opposing hand.", cap)]
        if len(pit_df):
            st_ += [Paragraph("Pitchers", hh), _table(pit_df, k, pad)]
        if len(hit_df):
            st_ += [Spacer(1, 8 if big else 3), Paragraph("Hitters", hh), _table(hit_df, k, pad)]
        return st_

    def _render(k, pad):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=PAGE, topMargin=TOP, bottomMargin=BOT,
                                leftMargin=LR, rightMargin=LR)
        doc.build(_story(k, pad))
        return buf.getvalue(), doc.page

    def _pad_for(k):
        b = body_fs * k
        return 2.6 if b >= 8.0 else (1.6 if b >= 6.0 else 0.9)

    out, k, fitted = None, 1.0, False
    while k >= 0.34:
        out, pages = _render(k, _pad_for(k))
        if pages <= 1:
            fitted = True
            break
        k -= 0.04
    if not fitted:
        out, _ = _render(0.34, 0.8)
        k = 0.34

    pad = _pad_for(k)
    while pad < 10.0:
        cand, pages = _render(k, pad + 0.5)
        if pages > 1:
            break
        out, pad = cand, pad + 0.5
    return out

def _pdf_unavailable(exc):
    """Explain, precisely, why a PDF could not be built."""
    import traceback
    if isinstance(exc, ModuleNotFoundError) and "reportlab" in str(exc):
        st.error("PDF export needs the **reportlab** package, which is not installed on this "
                 "deployment. Add a line reading `reportlab` to `requirements.txt` in the repo "
                 "and let the app rebuild.")
    else:
        st.error("The PDF could not be built: " + type(exc).__name__ + " — " + str(exc))
        with st.expander("Show details"):
            st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def _league_pitch_baselines(df):
    """Per-pitch-type league baselines: whiff%, xwOBA-on-contact, hard-hit%.
    These are the regression targets for sparse hitter samples."""
    bip = df[df["PitchCall"] == "InPlay"].copy()
    bip["_xw"] = bip.apply(lambda r: _xwoba_on_contact(r["ExitSpeed"], r["Angle"]), axis=1)
    base = {}
    for pt in df["PitchType"].dropna().unique():
        a = df[df["PitchType"] == pt]
        b = bip[bip["PitchType"] == pt]
        sw = a["PitchCall"].isin(["StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable","InPlay"]).sum()
        wh = (a["PitchCall"] == "StrikeSwinging").sum()
        base[pt] = {
            "whiff": (wh / sw) if sw else 0.22,
            "xw": b["_xw"].mean() if len(b) else 0.37,
            "hh": (b["ExitSpeed"] >= 90).mean() if len(b) else 0.27,
        }
    return base

def _regress(obs, lg, n, k):
    """Shrink an observed rate toward the league baseline by sample size."""
    if pd.isna(obs) or n == 0:
        return lg
    w = n / (n + k)
    return w * obs + (1 - w) * lg

def project_matchup(batter, pitcher, df, baselines):
    """Project a hitter vs a pitcher: expected xwOBA, whiff%, hard-hit%.
    Combines the hitter's per-pitch-type tendencies (regressed for sample size)
    weighted by the pitcher's actual pitch usage. Returns dict or None.
    NOTE: small-sample projection — leans on league baselines when data is thin."""
    pp = df[df["Pitcher"] == pitcher]
    if len(pp) == 0:
        return None
    mix = (pp["PitchType"].value_counts() / len(pp)).to_dict()
    bp = df[df["Batter"] == batter]
    proj_wh = proj_xw = proj_hh = 0.0
    wsum = 0.0
    total_n = 0
    for pt, usage in mix.items():
        if usage < 0.05 or pt not in baselines:
            continue
        hp = bp[bp["PitchType"] == pt]
        n = len(hp)
        nbip = hp[hp["PitchCall"] == "InPlay"]
        total_n += n
        sw = hp["PitchCall"].isin(["StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable","InPlay"]).sum()
        o_wh = ((hp["PitchCall"] == "StrikeSwinging").sum() / sw) if sw else np.nan
        o_xw = nbip.apply(lambda r: _xwoba_on_contact(r["ExitSpeed"], r["Angle"]), axis=1).mean() if len(nbip) else np.nan
        o_hh = (nbip["ExitSpeed"] >= 90).mean() if len(nbip) else np.nan
        proj_wh += usage * _regress(o_wh, baselines[pt]["whiff"], int(sw), 40)
        proj_xw += usage * _regress(o_xw, baselines[pt]["xw"], len(nbip), 8)
        proj_hh += usage * _regress(o_hh, baselines[pt]["hh"], len(nbip), 8)
        wsum += usage
    if wsum == 0:
        return None
    return {
        "whiff": proj_wh / wsum,
        "xwoba": proj_xw / wsum,
        "hardhit": proj_hh / wsum,
        "n_seen": total_n,
    }

def _lineup_pitcher_profile(pitcher_name, all_pitches):
    """Arsenal profile for one pitcher — computed once per pitcher selection
    and reused across every hitter scored against him. Pitch shape (velo,
    movement) is pooled overall since a given pitch's physical shape doesn't
    change with who's at the plate, but USAGE is tracked separately per
    batter side, since pitchers routinely change their mix by handedness
    (e.g. more changeups to opposite-hand hitters, more sliders same-hand)."""
    if not pitcher_name:
        return None
    pp = all_pitches[all_pitches["Pitcher"] == pitcher_name]
    if len(pp) == 0:
        return None
    hand_mode = pp["PitcherThrows"].mode()
    hand = hand_mode.iloc[0] if len(hand_mode) else None
    grp_n = len(pp)

    usage_by_side = {}
    for side_val, side_grp in pp.groupby("BatterSide"):
        if side_val not in ("Right", "Left"):
            continue
        usage_by_side[side_val] = (side_grp["PitchType"].value_counts() / len(side_grp)).to_dict()

    arsenal = {}
    for pt, sub in pp.groupby("PitchType"):
        if pd.isna(pt) or len(sub) < 15:
            continue
        hb_arm = (-sub["HorzBreak"] if hand == "Left" else sub["HorzBreak"])
        arsenal[pt] = {
            "usage": len(sub) / grp_n,   # overall fallback when a side split is missing
            "velo": sub["RelSpeed"].mean(),
            "ivb": sub["InducedVertBreak"].mean(),
            "hb_arm": hb_arm.mean(),
        }
    return {"hand": hand, "arsenal": arsenal, "usage_by_side": usage_by_side}


def score_matchup(batter, pitcher_profile, all_pitches, rm_baselines):
    """Score a hitter vs a specific pitcher: projected xwOBA (the main stat)
    plus whiff%, both read from the hitter's own history against pitches that
    match THIS pitcher's actual velocity and movement — not just the same
    PitchType label — weighted by how often the pitcher actually throws each
    pitch to THIS hitter's side (not his overall pooled mix), with a platoon
    read from his throwing hand layered on top. Same approach as the Reliever
    Matchup Planner's stuff-similarity scoring, but with a tighter similarity
    window: this is one specific known opponent, not a whole bullpen of arms
    to screen at once.

    Returns (score, details, low_sample, proj_xwoba) — proj_xwoba is the raw
    projected number (None if no matchup could be computed) for display."""
    if pitcher_profile is None or not pitcher_profile["arsenal"]:
        return 0, [], False, None

    bp = all_pitches[all_pitches["Batter"] == batter]
    if len(bp) == 0:
        return 0, [], True, None
    bp = bp.copy()
    bp["_HB_arm"] = np.where(bp["PitcherThrows"].eq("Left"), -bp["HorzBreak"], bp["HorzBreak"])
    side_mode = bp["BatterSide"].mode()
    side = side_mode.iloc[0] if len(side_mode) else None

    SWING_C_LU = {"StrikeSwinging","InPlay","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}
    # Loosened from the original 1.5/2.5/2.5 — that was flagging almost every
    # hitter/pitch combo as low-sample against this league's per-batter pitch
    # counts. Still tighter than the Reliever Matchup Planner's 2.5/4.0/4.0 —
    # one known opponent's exact arsenal, not screening a whole bullpen at once.
    VELO_TOL, IVB_TOL, HB_TOL = 2.0, 3.5, 3.5
    LG_WHIFF_DEFAULT, LG_XWOBA_DEFAULT = 0.22, 0.44
    side_usage = pitcher_profile["usage_by_side"].get(side, {})

    proj_whiff, proj_xwoba, wsum = 0.0, 0.0, 0.0
    details, low_sample = [], False
    for pt, prof in pitcher_profile["arsenal"].items():
        usage = side_usage.get(pt, prof["usage"])   # what he throws to THIS side
        if usage < 0.05 or pd.isna(prof["velo"]):
            continue
        similar = bp[
            (bp["RelSpeed"] - prof["velo"]).abs().le(VELO_TOL) &
            (bp["InducedVertBreak"] - prof["ivb"]).abs().le(IVB_TOL) &
            (bp["_HB_arm"] - prof["hb_arm"]).abs().le(HB_TOL)
        ]
        sw = similar["PitchCall"].isin(SWING_C_LU).sum()
        wh = similar["PitchCall"].eq("StrikeSwinging").sum()
        obs_whiff = wh / sw if sw > 0 else np.nan

        fair = similar[similar["ExitSpeed"].notna() & similar["Angle"].notna() &
                       (similar["Distance"].fillna(0) >= 10) &
                       (similar["Direction"].fillna(999).abs() <= 45)]
        obs_xwoba = (fair.apply(lambda r: calc_xwoba_bip(r["ExitSpeed"], r["Angle"]), axis=1).mean()
                     if len(fair) >= 2 else np.nan)

        n_similar = len(similar)
        if n_similar < 3:
            low_sample = True

        lg = rm_baselines.get(pt, {"whiff": LG_WHIFF_DEFAULT, "xw": LG_XWOBA_DEFAULT})
        pw = _regress(obs_whiff, lg["whiff"], int(sw), 15)
        px = _regress(obs_xwoba, lg["xw"], len(fair), 6)
        proj_whiff += usage * pw
        proj_xwoba += usage * px
        wsum += usage
        details.append({"pitch": pt, "usage": usage, "xwoba": round(px, 3),
                        "whiff": round(pw, 3), "n": n_similar, "low": n_similar < 3})

    if wsum == 0:
        return 0, [], True, None

    proj_whiff /= wsum
    proj_xwoba /= wsum

    score = 50.0
    score += (proj_xwoba - LG_XWOBA_DEFAULT) * 60   # xwOBA — the main stat, dominant weight
    score -= (proj_whiff - LG_WHIFF_DEFAULT) * 40    # whiff — secondary, still bad for the batter

    hand = pitcher_profile["hand"]
    if hand in ("Right", "Left") and side in ("Right", "Left"):
        score += 4 if hand != side else -3   # platoon: opposite-hand favors the batter

    details.sort(key=lambda d: d["usage"], reverse=True)
    return round(min(max(score, 0), 100), 1), details, low_sample, round(proj_xwoba, 3)


if page == "Batter Analysis":
    if st.session_state.get("rb_return_flag"):
        def _back_to_big_board():
            st.session_state["nav_cat"] = "Front Office"
            st.session_state["nav_page"] = "Returner Board"
            st.session_state.pop("rb_return_flag", None)
        st.button("← Back to Big Board", key="rb_back_btn_hit", on_click=_back_to_big_board)

    st.title("Batter Analysis")

    all_spray_teams = ([MY_TEAM] if MY_TEAM in _team_options(df_all["BatterTeam"]) else []) +                       sorted([t for t in _team_options(df_all["BatterTeam"]) if t != MY_TEAM])
    spray_labels    = [team_label(t) for t in all_spray_teams]

    # A Returner Board deep-link can hand us a team/batter this run's data
    # doesn't have -- drop it rather than let the selectboxes below crash on
    # a value outside their own options.
    if st.session_state.get("ba_team") not in range(len(all_spray_teams)):
        st.session_state.pop("ba_team", None)

    col1, col2, col3, col4 = st.columns([1.4, 1.8, 1.2, 1.2])
    with col1:
        team_idx = st.selectbox("Team",
            options=range(len(all_spray_teams)),
            format_func=lambda i: spray_labels[i], key="ba_team")
        selected_team = all_spray_teams[team_idx] if all_spray_teams else None
    all_team_batters = (sorted(df_all[df_all["BatterTeam"]==selected_team]["Batter"].dropna().unique())
                        if selected_team else [])
    opp_batters = [b for b in all_team_batters if not _is_removed(b)]
    # Same override as Pitcher Scouting: a deep-linked hitter who's since
    # left the active roster is still a legitimate returner-board target,
    # so let them back into the picker even though normal browsing hides
    # REMOVED_FROM_ROSTER names everywhere else.
    pending_b = st.session_state.get("ba_batter")
    if pending_b in all_team_batters and pending_b not in opp_batters:
        opp_batters = [pending_b] + opp_batters
    if st.session_state.get("ba_batter") not in ([""] + opp_batters):
        st.session_state.pop("ba_batter", None)
    with col2:
        batter = st.selectbox("Batter", options=[""]+opp_batters,
            format_func=lambda x: x if x else "Select batter…", key="ba_batter")
    with col3:
        hand_filter = st.selectbox("vs. Pitcher Hand",
            options=["All","Right","Left"],
            format_func=lambda x: {"All":"All","Right":"vs RHP","Left":"vs LHP"}[x])
    with col4:
        color_by = st.selectbox("Color By",
            options=["ev","la","result"],
            format_func=lambda x: {"ev":"Exit Velocity","la":"Launch Angle","result":"Hit Result"}[x])

    st.divider()

    if not batter:
        st.info(" Select an opponent team and batter to view their spray chart.")
        st.stop()

    mask = (df_all["Batter"]==batter) & df_all["ExitSpeed"].notna() &            df_all["Direction"].notna() & df_all["Angle"].notna()
    if hand_filter != "All":
        mask &= df_all["PitcherThrows"] == hand_filter
    balls = df_all[mask].copy()

    # Filter to fair balls only:
    # 1. Distance >= 10 ft (removes foul tips / bad tracking)
    # 2. |Direction| <= 45° (removes foul balls outside the lines)
    dist_col = balls["Distance"].fillna(
        balls.get("LastTrackedDistance", pd.Series(dtype=float))).fillna(0)
    balls = balls[(dist_col >= 10) & (balls["Direction"].abs() <= 45)].copy()

    if balls.empty:
        st.warning(f"No tracked batted balls found for **{batter}** with current filters.")
        st.stop()

    dist_col = balls["Distance"].fillna(
        balls.get("LastTrackedDistance", pd.Series(dtype=float))).fillna(250)
    balls = balls.copy()
    balls["fx"], balls["fy"] = direction_to_xy(balls["Direction"].values, dist_col.values)
    batter_side = balls["BatterSide"].iloc[0] if "BatterSide" in balls.columns else "Right"

    hand_color = {"Left":"#3b82f6","Right":"#9ca3af","Switch":"#c084fc"}.get(batter_side,"#64748b")
    st.markdown(
        f"### {batter} "
        f"<span style='background:#1a2235;color:{hand_color};font-size:0.85rem;"
        f"padding:3px 10px;border-radius:99px;font-weight:700;vertical-align:middle;'>"
        f"Bats {batter_side}</span> — {len(balls)} batted balls",
        unsafe_allow_html=True)
    if hand_filter != "All":
        st.caption(f"Filter: vs {'RHP' if hand_filter=='Right' else 'LHP'}")

    fig = go.Figure()
    fig = draw_field(fig)

    if len(balls) >= 6:  # Only show heatmap with enough fair balls
        try:
            kde = gaussian_kde(np.vstack([balls["fx"], balls["fy"]]), bw_method=0.35)
            gx, gy = np.meshgrid(np.linspace(-400,400,120), np.linspace(-20,500,120))
            z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            fig.add_trace(go.Contour(
                x=np.linspace(-400,400,120), y=np.linspace(-20,500,120), z=z,
                colorscale=[[0,"rgba(0,0,0,0)"],[0.3,"rgba(239,68,68,0.15)"],
                            [0.7,"rgba(239,68,68,0.35)"],[1.0,"rgba(239,68,68,0.55)"]],
                showscale=False, showlegend=False,
                contours=dict(showlines=False, coloring="fill"),
                hoverinfo="skip", ncontours=20))
        except Exception:
            pass

    if color_by == "ev":
        marker = dict(color=balls["ExitSpeed"], colorscale="RdYlGn",
            cmin=balls["ExitSpeed"].min(), cmax=balls["ExitSpeed"].max(),
            colorbar=dict(title="EV (mph)", thickness=12, len=0.6),
            size=10, line=dict(color="rgba(0,0,0,0.6)",width=1), opacity=0.88)
        hover = balls.apply(lambda r:
            f"<b>{player_last(r['Batter'])}</b><br>EV: {r['ExitSpeed']:.1f} mph<br>"
            f"LA: {r['Angle']:.1f}°<br>Dist: {r.get('Distance',0):.0f} ft<br>"
            f"Result: {r.get('PlayResult','—')}<br>vs {r.get('PitcherThrows','?')}HP", axis=1)
    elif color_by == "la":
        marker = dict(color=balls["Angle"], colorscale="RdYlBu",
            cmin=-30, cmax=45,
            colorbar=dict(title="Launch Angle°", thickness=12, len=0.6),
            size=10, line=dict(color="rgba(0,0,0,0.6)",width=1), opacity=0.88)
        hover = balls.apply(lambda r:
            f"<b>{player_last(r['Batter'])}</b><br>LA: {r['Angle']:.1f}°<br>"
            f"EV: {r['ExitSpeed']:.1f} mph<br>Dist: {r.get('Distance',0):.0f} ft<br>"
            f"Result: {r.get('PlayResult','—')}", axis=1)
    else:
        rc = {"Single":"#22c55e","Double":"#3b82f6","Triple":"#8b5cf6",
              "HomeRun":"#f59e0b","Out":"#ef4444","FieldersChoice":"#f97316",
              "Error":"#ec4899","Undefined":"#64748b"}
        marker = dict(color=balls["PlayResult"].map(rc).fillna("#64748b"),
            size=10, line=dict(color="rgba(0,0,0,0.6)",width=1), opacity=0.88)
        hover = balls.apply(lambda r:
            f"<b>{player_last(r['Batter'])}</b><br>Result: {r.get('PlayResult','—')}<br>"
            f"EV: {r['ExitSpeed']:.1f} mph · LA: {r['Angle']:.1f}°<br>"
            f"Dist: {r.get('Distance',0):.0f} ft<br>"
            f"vs {r.get('PitcherThrows','?')}HP", axis=1)

    fig.add_trace(go.Scatter(x=balls["fx"], y=balls["fy"], mode="markers",
        marker=marker, text=hover,
        hovertemplate="%{text}<extra></extra>", showlegend=False))
    fig.update_layout(height=540, plot_bgcolor="#0d1f0d", paper_bgcolor="#ffffff",
        font=dict(color="#1e293b"),
        xaxis=dict(range=[-410,410], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-30,510], visible=False),
        margin=dict(l=0,r=80,t=0,b=0), hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)

    if color_by == "result":
        rc2 = {"Single":"#22c55e","Double":"#3b82f6","Triple":"#8b5cf6",
               "HomeRun":"#f59e0b","Out":"#ef4444","FieldersChoice":"#f97316",
               "Error":"#ec4899","Undefined":"#64748b"}
        counts = balls["PlayResult"].value_counts()
        leg_cols = st.columns(min(len(counts), 6))
        for i, (result, cnt) in enumerate(counts.items()):
            with leg_cols[i % len(leg_cols)]:
                st.markdown(f"<span style='color:{rc2.get(result,'#64748b')}'></span> **{result}** ({cnt})",
                    unsafe_allow_html=True)

    #  Batter stats bar 
    st.divider()
    avg_ev   = balls["ExitSpeed"].mean()
    max_ev   = balls["ExitSpeed"].max()
    hard_pct = (balls["ExitSpeed"] >= 90).mean()
    gb_pct   = (balls["Angle"] < -4).mean() if "Angle" in balls.columns else None
    ld_pct   = (balls["Angle"].between(8, 25)).mean() if "Angle" in balls.columns else None
    fb_pct   = (balls["Angle"] > 25).mean() if "Angle" in balls.columns else None
    avg_la   = balls["Angle"].mean() if "Angle" in balls.columns else None

    # Pull/oppo from all pitches for this batter (not just fair balls in play)
    all_bp   = df_all[df_all["Batter"] == batter]
    pa       = all_bp[all_bp["PitchofPA"] == 1].shape[0] if "PitchofPA" in all_bp.columns else len(all_bp)
    k_pct    = all_bp["KorBB"].eq("Strikeout").sum() / max(pa, 1)
    bb_pct_v = all_bp["KorBB"].eq("Walk").sum() / max(pa, 1)
    pull_sign = -1 if batter_side == "Left" else 1
    pull_pct = (balls["Direction"] * pull_sign < -10).mean()
    oppo_pct = (balls["Direction"] * pull_sign >  10).mean()

    s1, s2, s3, s4, s5, s6, s7, s8 = st.columns(8)
    s1.metric("K%",        f"{k_pct:.0%}")
    s2.metric("BB%",       f"{bb_pct_v:.0%}")
    s3.metric("GB%",       f"{gb_pct:.0%}" if gb_pct is not None else "—")
    s4.metric("FB%",       f"{fb_pct:.0%}" if fb_pct is not None else "—")
    s5.metric("Avg EV",    f"{avg_ev:.1f}")
    s6.metric("Max EV",    f"{max_ev:.1f}")
    s7.metric("Hard%",     f"{hard_pct:.0%}")
    s8.metric("Avg LA",    f"{avg_la:.1f}°" if avg_la is not None else "—")

    # First-pitch batting: only first pitches put in play (hit or out).
    fp = all_bp[(all_bp["PitchofPA"] == 1) & (all_bp["PitchCall"] == "InPlay")].copy()
    fp_n = len(fp)
    if fp_n > 0:
        fp_hits = fp["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum()
        fp_ba = fp_hits / fp_n
        fp["_xba"] = fp.apply(lambda r: calc_xba(r["ExitSpeed"], r["Angle"]), axis=1)
        fp_xba = fp["_xba"].mean()
        st.divider()
        st.caption("**First-Pitch Batting** — first pitch of the PA put in play (hit or out only)")
        f1, f2, f3 = st.columns(3)
        f1.metric("1st-Pitch BA", f"{fp_ba:.3f}", help=f"{int(fp_hits)}-for-{fp_n} on first pitches put in play")
        f2.metric("1st-Pitch xBA", f"{fp_xba:.3f}", help="Expected BA from contact quality (EV + launch angle)")
        f3.metric("Balls in Play", str(fp_n))
        if fp_n < 8:
            st.caption("⚠ Small sample — first-pitch balls in play are a narrow slice; "
                       "xBA is the more stable read here.")
    else:
        st.divider()
        st.caption("**First-Pitch Batting** — no first pitches put in play yet.")

    # ── Where he swings — attack zones (swings only) ──
    st.divider()
    st.caption("**Where He Swings — Attack Zones** (Heart = middle · Shadow = edges · "
               "Chase = just off · Waste = well off). Compared to league-average swing locations.")
    _sw = all_bp[all_bp["PitchCall"].isin(["StrikeSwinging", "FoulBallNotFieldable",
                                           "FoulBallFieldable", "InPlay"])].copy()
    _sw["PlateLocSide"] = pd.to_numeric(_sw["PlateLocSide"], errors="coerce")
    _sw["PlateLocHeight"] = pd.to_numeric(_sw["PlateLocHeight"], errors="coerce")
    _sw["_z"] = _sw.apply(lambda r: attack_zone(r["PlateLocSide"], r["PlateLocHeight"]), axis=1)
    _sw_valid = _sw.dropna(subset=["_z"])
    if len(_sw_valid) >= 8:
        _lgs = _league_attack_zone_rates(df_all, "swing")
        _sv = (_sw_valid["_z"].value_counts(normalize=True) * 100)
        sz_rows = []
        for z in _AZ_ORDER:
            his = float(_sv.get(z, 0.0))
            lg = _lgs[z]
            diff = his - lg
            tag = (f"{diff:+.0f} pts") if abs(diff) >= 3 else "≈ league"
            sz_rows.append({"Zone": z, "His Swing %": f"{his:.0f}%",
                            "League %": f"{lg:.0f}%", "vs League": tag})
        st.dataframe(pd.DataFrame(sz_rows), use_container_width=True, hide_index=True)
        _chase = float(_sv.get("Chase", 0.0)) + float(_sv.get("Waste", 0.0))
        _lgc = _lgs["Chase"] + _lgs["Waste"]
        if _chase >= _lgc + 5:
            st.caption("⚠ Chases more than average (swings at Chase/Waste " +
                       f"{_chase:.0f}% vs league {_lgc:.0f}%) — expand the zone on him.")
        elif _chase <= _lgc - 5:
            st.caption("✓ Disciplined — chases less than average. Must attack the zone.")
    else:
        st.caption("Not enough swings for a zone profile yet.")

    if "PitchType" in balls.columns:
        st.divider()
        st.caption("**Batted Balls by Pitch Type**")
        pt = balls.groupby("PitchType").agg(
            Count=("ExitSpeed","count"),
            AvgEV=("ExitSpeed","mean"),
            AvgLA=("Angle","mean"),
        ).sort_values("Count", ascending=False)
        pt["AvgEV"] = pt["AvgEV"].round(1)
        pt["AvgLA"] = pt["AvgLA"].round(1)
        st.dataframe(pt, use_container_width=True, height=200)

    # ══════════════════════════════════════════
    #  ATTACK PLAN
    # ══════════════════════════════════════════
    st.divider()
    st.markdown("### Attack Plan")

    # Pitcher hand toggle
    atk_hand = st.radio(
        "Filter by pitcher hand",
        options=["All", "Right", "Left"],
        horizontal=True,
        key="atk_hand"
    )

    # All pitches to this batter with hand filter
    atk_bp = df_all[df_all["Batter"] == batter].copy()
    if atk_hand != "All":
        atk_bp = atk_bp[atk_bp["PitcherThrows"] == atk_hand]

    if len(atk_bp) == 0:
        st.info(f"No pitch data vs {'RHP' if atk_hand=='Right' else 'LHP' if atk_hand=='Left' else 'any pitcher'}.")
    else:
        # Swing/whiff/hit flags
        SWING_CALLS = {"StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable","InPlay"}
        atk_bp = atk_bp.copy()
        atk_bp["IsSwing"] = atk_bp["PitchCall"].isin(SWING_CALLS)
        atk_bp["IsWhiff"] = atk_bp["PitchCall"] == "StrikeSwinging"
        atk_bp["IsHit"]   = atk_bp["PlayResult"].isin(["Single","Double","Triple","HomeRun"])
        atk_bp["IsAB"]    = _ab_mask(atk_bp)

        # ── BA / xBA per pitch type ───────────────────────────
        st.markdown("#### Performance by Pitch Type")
        st.caption("Batting average and expected BA against each pitch type "
                   f"({'vs RHP' if atk_hand=='Right' else 'vs LHP' if atk_hand=='Left' else 'all pitchers'}).")

        pt_rows = []
        for ptype in sorted(atk_bp["PitchType"].dropna().unique()):
            sub = atk_bp[atk_bp["PitchType"] == ptype]
            # At-bats ending on this pitch type
            ab_mask = _ab_mask(sub)
            abs_n = int(ab_mask.sum())
            hits_n = int(sub["PlayResult"].isin(["Single","Double","Triple","HomeRun"]).sum())
            ba = round(hits_n / abs_n, 3) if abs_n > 0 else None
            # xBA from balls in play of this pitch type
            bip = sub[sub["PlayResult"].isin(["Single","Double","Triple","HomeRun","Out","Error","FieldersChoice"])]
            bip = bip[bip["ExitSpeed"].notna() & bip["Angle"].notna()]
            if len(bip) > 0 and abs_n > 0:
                xba_sum = bip.apply(lambda r: calc_xba(r["ExitSpeed"], r["Angle"]), axis=1).sum()
                # strikeouts count as automatic 0 xBA outcomes in the AB denominator
                xba = round(xba_sum / abs_n, 3)
            else:
                xba = None
            whiffs = int((sub["PitchCall"] == "StrikeSwinging").sum())
            swings = int(sub["PitchCall"].isin(["StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable","InPlay"]).sum())
            whiff_pct = round(100 * whiffs / swings, 1) if swings > 0 else None
            pt_rows.append({
                "Pitch": ptype,
                "Pitches": len(sub),
                "AB": abs_n,
                "H": hits_n,
                "BA": f"{ba:.3f}" if ba is not None else "—",
                "xBA": f"{xba:.3f}" if xba is not None else "—",
                "Whiff%": f"{whiff_pct:.1f}%" if whiff_pct is not None else "—",
            })
        if pt_rows:
            pt_df = pd.DataFrame(pt_rows).sort_values("Pitches", ascending=False)
            st.dataframe(pt_df, use_container_width=True, hide_index=True)
            st.caption("BA = hits ÷ at-bats ending on that pitch. xBA = expected BA from "
                       "contact quality (exit velo + launch angle), strikeouts as outs.")
        else:
            st.info("Not enough data for per-pitch-type breakdown.")

        zone_col, arsenal_col = st.columns([1, 1.4])

        # ── ZONE MAP ──────────────────────────────
        with zone_col:
            st.markdown("#### Zone Attack Map")
            st.caption("Exit velocity heatmap — red/white = hard contact, blue = weak contact")

            # Zone boundaries (feet): standard strike zone
            # Side: -0.83 to +0.83 (plate is 17in = 1.42ft wide, half = 0.71 + ball radius)
            # Height: bottom ~1.5ft, top ~3.5ft — varies by batter but use standard
            side_edges   = [-2.0, -0.28, 0.28, 2.0]   # left edge, inner thirds, right edge
            height_edges = [1.0, 1.83, 2.67, 3.5]      # low, mid-low, mid-high, high

            # Filter to pitches with location data
            zp = atk_bp[atk_bp["PlateLocSide"].notna() & atk_bp["PlateLocHeight"].notna()].copy()

            # Build 3x3 zone grid (row 0 = top, col 0 = left from catcher view)
            ZONE_ROWS = 3
            ZONE_COLS = 3
            zones = {}
            for row in range(ZONE_ROWS):
                for col in range(ZONE_COLS):
                    h_lo = height_edges[ZONE_ROWS - 1 - row]
                    h_hi = height_edges[ZONE_ROWS - row]
                    s_lo = side_edges[col]
                    s_hi = side_edges[col + 1]
                    mask = (
                        zp["PlateLocHeight"].between(h_lo, h_hi) &
                        zp["PlateLocSide"].between(s_lo, s_hi)
                    )
                    zone_pitches = zp[mask]
                    n       = len(zone_pitches)
                    swings  = zone_pitches["IsSwing"].sum()
                    whiffs  = zone_pitches["IsWhiff"].sum()
                    hits    = zone_pitches["IsHit"].sum()
                    abs_z   = zone_pitches["IsAB"].sum()
                    avg_ev  = zone_pitches["ExitSpeed"].mean() if zone_pitches["ExitSpeed"].notna().any() else None
                    whiff_r = whiffs / swings if swings > 0 else None
                    ba      = hits / abs_z if abs_z > 0 else None
                    zones[(row, col)] = dict(n=n, avg_ev=avg_ev, whiff_r=whiff_r, ba=ba, swings=swings)



        # ── PITCH ARSENAL ─────────────────────────────
        with arsenal_col:
            st.markdown("#### Pitch Arsenal Breakdown")

            pitch_rows = []
            for ptype, grp in atk_bp.groupby("PitchType"):
                if ptype in (None, "Undefined", "Other") or pd.isna(ptype):
                    continue
                n_pitches = len(grp)
                swings    = grp["IsSwing"].sum()
                whiffs    = grp["IsWhiff"].sum()
                hits      = grp["IsHit"].sum()
                abs_p     = grp["IsAB"].sum()
                ev_grp    = grp[grp["ExitSpeed"].notna()]
                # Chase: swings on pitches outside zone
                # Approximate: PlateLocSide or Height outside strike zone bounds
                chase_mask = (
                    grp["PlateLocSide"].notna() &
                    grp["PlateLocHeight"].notna() & (
                        (grp["PlateLocSide"].abs() > 0.83) |
                        (grp["PlateLocHeight"] < 1.5) |
                        (grp["PlateLocHeight"] > 3.5)
                    )
                )
                chases    = grp[chase_mask & grp["IsSwing"]].shape[0]
                balls_out = grp[chase_mask].shape[0]

                whiff_pct  = whiffs / swings if swings > 0 else None
                chase_pct  = chases / balls_out if balls_out > 0 else None
                ba         = hits / abs_p if abs_p > 0 else None
                avg_ev_p   = ev_grp["ExitSpeed"].mean() if len(ev_grp) > 0 else None
                hard_p     = (ev_grp["ExitSpeed"] >= 90).mean() if len(ev_grp) > 0 else None

                pitch_rows.append({
                    "Pitch":      ptype,
                    "Pitches":    n_pitches,
                    "Whiff%":     f"{whiff_pct:.0%}" if whiff_pct is not None else "—",
                    "Chase%":     f"{chase_pct:.0%}" if chase_pct is not None else "—",
                    "BA":         f".{int(ba*1000):03d}" if ba is not None else "—",
                    "Avg EV":     f"{avg_ev_p:.1f}" if avg_ev_p is not None else "—",
                    "Hard%":      f"{hard_p:.0%}" if hard_p is not None else "—",
                })

            if pitch_rows:
                arsenal_df = pd.DataFrame(pitch_rows).sort_values("Pitches", ascending=False).reset_index(drop=True)
                st.dataframe(arsenal_df, use_container_width=True, height=300, hide_index=True)
                st.caption(f"Based on {len(atk_bp)} pitches vs {'RHP' if atk_hand=='Right' else 'LHP' if atk_hand=='Left' else 'all pitchers'}")
            else:
                st.info("No pitch type data available.")

        # ── HEATMAPS — full width below table ─────────────────────────────
        st.divider()
        hm_f1, hm_f2 = st.columns(2)
        with hm_f1:
            pitch_types_hz = ["All"] + sorted(zp["PitchType"].dropna().unique().tolist()) if len(zp) > 0 else ["All"]
            sel_pt_hz = st.selectbox("Pitch type", pitch_types_hz,
                                     key=f"hz_pt_{batter}")
        with hm_f2:
            sel_hand_hz = st.radio("Pitcher hand", ["All", "RHP", "LHP"],
                                   horizontal=True, key=f"hz_hand_{batter}")
        hz_base = zp.copy()
        if sel_hand_hz == "RHP":
            hz_base = hz_base[hz_base["PitcherThrows"] == "Right"]
        elif sel_hand_hz == "LHP":
            hz_base = hz_base[hz_base["PitcherThrows"] == "Left"]
        hz_data  = hz_base if sel_pt_hz == "All" else hz_base[hz_base["PitchType"] == sel_pt_hz]
        pt_label = "All pitches" if sel_pt_hz == "All" else sel_pt_hz

        hz_col1, hz_col2 = st.columns(2)
        with hz_col1:
            st.markdown(
                "<div style='font-size:0.9rem;font-weight:700;color:#475569;"
                "margin-bottom:6px;'>Contact Quality</div>"
                "<div style='font-size:0.75rem;color:#64748b;margin-bottom:8px;'>"
                "Red/white = hard contact zones</div>",
                unsafe_allow_html=True)
            _render_kde_heatmap(hz_data, weight_col="ExitSpeed",
                                key_suffix=f"hz_ev_{batter}_{sel_pt_hz}_{sel_hand_hz}",
                                title="")

        with hz_col2:
            st.markdown(
                "<div style='font-size:0.9rem;font-weight:700;color:#475569;"
                "margin-bottom:6px;'>Whiff Zones</div>"
                "<div style='font-size:0.75rem;color:#64748b;margin-bottom:8px;'>"
                "Red/white = where he whiffs most</div>",
                unsafe_allow_html=True)
            swing_data = hz_data[hz_data["PitchCall"].isin(
                {"StrikeSwinging","InPlay","FoulBallNotFieldable",
                 "FoulBallFieldable","FoulTip","FoulBall"}
            )].copy()
            swing_data["_whiff_weight"] = swing_data["PitchCall"].eq("StrikeSwinging").astype(float)
            if len(swing_data) >= 5:
                _render_kde_heatmap(swing_data, weight_col="_whiff_weight",
                                    key_suffix=f"hz_wh_{batter}_{sel_pt_hz}_{sel_hand_hz}",
                                    title="")
            else:
                st.info("Not enough swings for whiff map.")

        hand_lbl = "" if sel_hand_hz == "All" else f" · vs {sel_hand_hz}"
        st.caption(f"Catcher view · {pt_label}{hand_lbl} · Bats: {batter_side}")

    # ── Attack Zones: where this hitter swings/takes + swing-decision value ──
    st.divider()
    st.markdown("### Attack Zones — Swing Decisions")
    st.caption("Statcast-style zones: Heart (middle), Shadow (edges), Chase (just off), "
               "Waste (way off). Shows where this hitter swings vs takes, and the run value "
               "of those decisions.")
    _haz = _attack_zone_frame(df_all[df_all["Batter"] == batter])
    if len(_haz) < 10:
        st.info("Not enough located pitches for attack-zone breakdown.")
    else:
        _zone_order = ["Heart", "Shadow", "Chase", "Waste"]
        # swing% by zone
        sw_by_zone = {}
        for z in _zone_order:
            zz = _haz[_haz["_az"] == z]
            sw_by_zone[z] = float(zz["_swing"].mean() * 100) if len(zz) else 0.0
        # in-zone / out-of-zone swing%
        # Z-Swing / O-Swing use the TRUE strike zone, not Heart+Shadow (the Shadow
        # band straddles the border, so ~half of it is actually out of the zone).
        iz_sw, oz_sw, n_in, n_out = _true_zone_swing(_haz)

        m1, m2 = st.columns(2)
        m1.metric("Z-Swing% (in zone)", f"{iz_sw:.0f}%",
                  help=f"Swings at pitches inside the true strike zone (n={n_in}). Higher = attacks strikes.")
        m2.metric("O-Swing% (chase)", f"{oz_sw:.0f}%",
                  help=f"Swings at pitches outside the true strike zone (n={n_out}). Lower = better discipline.")

        cA, cB = st.columns([1, 1.3])
        with cA:
            components.html(_attack_zone_svg(sw_by_zone, "Swing% by zone", is_rate=True), height=300)
        with cB:
            _haz2 = _haz.copy()
            _haz2["_rv"] = _haz2.apply(_attack_decision_rv, axis=1)
            st.markdown("**Swing-decision run value by zone** (per pitch seen)")
            rows = []
            for z in _zone_order:
                zz = _haz2[_haz2["_az"] == z]
                if len(zz) == 0:
                    continue
                sw = zz[zz["_swing"]]
                tk = zz[~zz["_swing"]]
                sw_rv = sw["_rv"].mean() if len(sw) else np.nan
                tk_rv = tk["_rv"].mean() if len(tk) else np.nan
                rows.append({"Zone": z, "Pitches": len(zz),
                             "Swing%": f"{zz['_swing'].mean()*100:.0f}%",
                             "Swing RV": f"{sw_rv:+.3f}" if pd.notna(sw_rv) else "—",
                             "Take RV": f"{tk_rv:+.3f}" if pd.notna(tk_rv) else "—"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Run value is a descriptive proxy from outcomes (not a fitted model). "
                   "Positive = decision added value. Ideal: swing in Heart, take in Chase/Waste.")


elif page == "Lineup Builder":
    st.title("Lineup Builder — Brookhaven Bandits")

    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.5, 1.5])
    with ctrl1:
        opp_hand = st.selectbox("Opponent Pitcher Throws",
            options=["","Right","Left"],
            format_func=lambda x: {"":"Unknown / TBD","Right":"RHP","Left":"LHP"}[x])
    with ctrl2:
        opp_teams_lu = sorted([t for t in _team_options(df_all["PitcherTeam"]) if t != MY_TEAM])
        opp_team_lu  = st.selectbox("Opponent Team",
            options=[""] + opp_teams_lu,
            format_func=lambda x: team_label(x) if x else "Select opponent…",
            key="lu_opp_team")
    with ctrl3:
        lu_pitchers = []
        if opp_team_lu:
            pitcher_info = (df_all[df_all["PitcherTeam"] == opp_team_lu]
                .groupby(["Pitcher","PitcherThrows"]).size()
                .reset_index(name="Pitches")
                .sort_values("Pitches", ascending=False))
            lu_pitchers = [p for p in pitcher_info["Pitcher"].tolist() if not _is_removed(p)]
        matchup_pitcher = st.selectbox("Opposing Pitcher",
            options=[""] + lu_pitchers,
            format_func=lambda x: x if x else "Select pitcher…",
            key="lu_pitcher")

    st.divider()

    # Full roster hitters — include even those with no at-bats
    all_hitters = sorted([n for n, p in ROSTER.items()
        if p in BATTING_BASE_POSITIONS
        and not _is_removed(n)
        and not _is_report_hidden(n)
        and not any(n.startswith(k.split(",")[0]) and n != k
                    for k in ROSTER if ROSTER[k] == p and k != n
                    and k.split(",")[0] == n.split(",")[0])])

    # Deduplicate aliases — prefer "First Last" style names that appear in data
    my_pitches = df_all[df_all["BatterTeam"] == MY_TEAM] if not df_all.empty else pd.DataFrame()
    data_names = set(my_pitches["Batter"].dropna().unique()) if not my_pitches.empty else set()

    # Build canonical hitter list: prefer data name, fall back to roster name
    seen_last = {}
    canonical_hitters = []
    for name in sorted(ROSTER.keys()):
        if ROSTER[name] not in BATTING_BASE_POSITIONS:
            continue
        last = name.split(",")[0].strip()
        # If we already have this player under another alias, skip
        if last in seen_last:
            # Prefer the version that appears in data
            if name in data_names and seen_last[last] not in data_names:
                canonical_hitters.remove(seen_last[last])
                seen_last[last] = name
                canonical_hitters.append(name)
            continue
        seen_last[last] = name
        canonical_hitters.append(name)
    canonical_hitters = sorted(canonical_hitters)
    canonical_hitters = [h for h in canonical_hitters
                         if not _is_removed(h) and not _is_report_hidden(h)]

    # Compute stats for everyone
    all_stats = {}
    for name in canonical_hitters:
        all_stats[name] = compute_batter_stats(name, my_pitches, opp_hand)

    # Compute matchup scores if pitcher selected — pitcher arsenal profile and
    # league baselines are each computed ONCE and reused for every hitter,
    # not recomputed per hitter (score_matchup is called once per roster spot).
    matchup_scores = {}
    matchup_details = {}
    matchup_low_sample = {}
    matchup_xwoba = {}
    if matchup_pitcher:
        _lu_pitcher_profile = _lineup_pitcher_profile(matchup_pitcher, df_all)
        _lu_baselines = _league_pitch_baselines(df_all)
        for name in canonical_hitters:
            mscore, mdetails, mlow, mxwoba = score_matchup(name, _lu_pitcher_profile, df_all, _lu_baselines)
            matchup_scores[name] = mscore
            matchup_details[name] = mdetails
            matchup_low_sample[name] = mlow
            matchup_xwoba[name] = mxwoba

    all_scored = pd.DataFrame(list(all_stats.values()))
    if matchup_pitcher and matchup_scores:
        all_scored["MatchupScore"] = all_scored["Batter"].map(matchup_scores).fillna(0)
        # Matchup-dominant blend when a pitcher is selected — the auto-picked
        # top 9 needs to actually track projected xwOBA against THIS pitcher,
        # not just be nudged by it. Season-wide Score stays as a 20% guard
        # rail so one pitch type's tiny sample can't swing the whole pick.
        base_max = all_scored["Score"].abs().max() or 1
        match_max = all_scored["MatchupScore"].abs().max() or 1
        all_scored["BlendedScore"] = (
            0.8 * all_scored["MatchupScore"] / match_max +
            0.2 * all_scored["Score"] / base_max
        )
        all_scored = all_scored.sort_values("BlendedScore", ascending=False).reset_index(drop=True)
    else:
        all_scored["MatchupScore"] = 0
        all_scored["BlendedScore"] = all_scored["Score"]
        all_scored = all_scored.sort_values("Score", ascending=False).reset_index(drop=True)

    #  STEP 1: Select available players
    st.markdown("#### Step 1 — Mark Available Players Today")
    st.caption("Check everyone who is available. The app will auto-pick the best 9.")

    if "available" not in st.session_state:
        st.session_state.available = set()

    sel_cols = st.columns(4)
    for i, row in all_scored.iterrows():
        name = row["Batter"]
        pos  = row["BasPos"]
        side = row["Side"]
        label = f"**{player_last(name)}** · {pos} · {side}"
        checked = name in st.session_state.available
        if sel_cols[i % 4].checkbox(label, value=checked, key=f"avail_{name}"):
            st.session_state.available.add(name)
        else:
            st.session_state.available.discard(name)

    available_list = [n for n in canonical_hitters if n in st.session_state.available]

    st.divider()

    if len(available_list) < 9:
        st.warning(f"️ Only {len(available_list)} players available — need at least 9.")
        st.stop()

    #  Auto-pick best 9 
    avail_scored = all_scored[all_scored["Batter"].isin(available_list)].copy()

    # If more than 9 available, pick best 9 — but ensure we have at least 1 C
    # and resolve position conflicts greedily
    def pick_best_nine(df):
        # BlendedScore (not the plain season-wide Score) — when a pitcher is
        # selected this is the matchup-dominant xwOBA-projection blend, so the
        # auto-pick actually reflects the matchup instead of ignoring it.
        # BlendedScore == Score when no pitcher is selected, so this is a
        # no-op for that case.
        df = df.sort_values("BlendedScore", ascending=False).reset_index(drop=True)
        selected = []
        used_unique = set()

        # First pass: greedily pick by score, skip true duplicates at unique spots
        for _, row in df.iterrows():
            if len(selected) == 9:
                break
            bp = row["BasPos"]
            # C and 1B are unique; IF/OF can have multiples (different sub-positions)
            if bp in ("C", "1B"):
                if bp in used_unique:
                    # Slot as DH if no DH yet
                    if "DH" not in used_unique:
                        used_unique.add("DH")
                        selected.append(row["Batter"])
                    continue
                used_unique.add(bp)
            selected.append(row["Batter"])

        # If we didn't hit 9 (because too many conflicts), just take top 9 by score
        if len(selected) < 9:
            selected = df["Batter"].head(9).tolist()

        return selected

    best_nine = pick_best_nine(avail_scored)
    starters = avail_scored[avail_scored["Batter"].isin(best_nine)].copy()

    if len(available_list) > 9:
        benched = avail_scored[~avail_scored["Batter"].isin(best_nine)]["Batter"].tolist()
        st.info(f"Auto-selected best 9 from {len(available_list)} available. "
                f"Sitting: {', '.join(player_last(b) for b in benched)}")

    #  STEP 2: Assign defensive positions 
    st.markdown("#### Step 2 — Assign Defensive Positions")
    st.caption("Assign each starter a specific position. DH is available to resolve any conflicts.")

    pos_assignments = {}
    used_positions  = {}

    assign_cols = st.columns(3)
    for i, row in starters.sort_values("Score", ascending=False).iterrows():
        name = row["Batter"]
        bp   = row["BasPos"]
        opts = POS_OPTIONS.get(bp, ["DH"])
        col  = assign_cols[list(starters.index).index(i) % 3]
        default_idx = 0
        # Load previous assignment if valid
        prev = st.session_state.get(f"pos_{name}")
        if prev in opts:
            default_idx = opts.index(prev)
        chosen = col.selectbox(
            f"{player_last(name)}",
            options=opts,
            index=default_idx,
            key=f"pos_{name}"
        )
        pos_assignments[name] = chosen
        used_positions[chosen] = used_positions.get(chosen, 0) + 1

    # Check conflicts
    conflicts = [p for p, cnt in used_positions.items() if cnt > 1 and p != "DH"]
    dh_count  = used_positions.get("DH", 0)

    if conflicts:
        for p in conflicts:
            dupes = [player_last(n) for n, pos in pos_assignments.items() if pos == p]
            st.error(f"️ Position conflict at **{p}**: {', '.join(dupes)} — assign one as DH.")
    if dh_count > 1:
        st.error(f"️ Only one DH allowed — {dh_count} players assigned as DH.")

    if conflicts or dh_count > 1:
        st.stop()

    st.divider()

    #  STEP 3: Batting order 
    starters = starters.sort_values("Score", ascending=False).reset_index(drop=True)
    starters["DefPos"] = starters["Batter"].map(pos_assignments)

    lineup_col, scout_col = st.columns([1.1, 1.5])

    with lineup_col:
        st.markdown("### Batting Order")
        if opp_hand:
            st.caption(f"Optimized vs {'RHP' if opp_hand=='Right' else 'LHP'} · OBP + EV + Platoon")
        else:
            st.caption("Optimized by OBP + Exit Velocity")

        for i, row in starters.iterrows():
            side_color = {"Left":"#3b82f6","Right":"#9ca3af","Switch":"#c084fc"}.get(row["Side"],"#64748b")
            pos_color  = POS_COLORS.get(row["DefPos"], "#64748b")
            obp_str    = f"{row['OBP']:.3f}" if pd.notna(row.get("OBP")) else "—"
            ops_str    = f"{row['OPS']:.3f}" if pd.notna(row.get("OPS")) else "—"
            official_badge = ("<span style='background:#1e3a5f;color:#60a5fa;"
                "font-size:0.65rem;padding:1px 5px;border-radius:3px;font-weight:600;'>OFF</span>"
                if row.get("HasOfficial") else "")
            platoon_badge = ""
            if opp_hand and pd.notna(row["PlatoonAdv"]):
                platoon_badge = (
                    "<span style='background:#14532d;color:#86efac;font-size:0.7rem;"
                    "padding:2px 7px;border-radius:99px;font-weight:700;'>ADV</span>"
                    if row["PlatoonAdv"] else
                    "<span style='background:#1f2937;color:#6b7280;font-size:0.7rem;"
                    "padding:2px 7px;border-radius:99px;'>—</span>"
                )
            def_pos = row["DefPos"]
            batter_name = player_last(row["Batter"])
            side_val = row["Side"]
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;"
                f"background:#F8FAFC;border:1.5px solid #E2E8F0;border-radius:8px;"
                f"padding:10px 14px;margin-bottom:6px;'>"
                f"<div style='font-size:1.3rem;font-weight:800;color:#64748b;width:24px;'>{i+1}</div>"
                f"<span style='background:#111;color:{pos_color};font-size:0.75rem;"
                f"padding:2px 7px;border-radius:4px;font-weight:700;min-width:34px;"
                f"text-align:center;'>{def_pos}</span>"
                f"<div style='flex:1;font-weight:600;font-size:0.95rem;color:#1e293b;'>{batter_name}</div>"
                f"<span style='background:#1a2235;color:{side_color};font-size:0.75rem;"
                f"padding:2px 8px;border-radius:99px;font-weight:700;'>{side_val}</span>"
                f"<span style='background:#0f2d1f;color:#86efac;font-size:0.72rem;"
                f"padding:2px 7px;border-radius:4px;'>OBP {obp_str}</span>"
                f"<span style='background:#1e3a5f;color:#93c5fd;font-size:0.72rem;"
                f"padding:2px 7px;border-radius:4px;'>OPS {ops_str}</span>"
                f"{official_badge}"
                f"{platoon_badge}"
                f"</div>",
                unsafe_allow_html=True)

    with scout_col:
        st.markdown("### Scout Table")

        def fmt(v, f):
            return f.format(v) if pd.notna(v) else "—"

        td = starters[["Batter","DefPos","Side","PA","OBP","KPct","BBPct","xBA","AvgEV","HardPct","EV_RHP","EV_LHP","AvgLA"]].copy()
        td["OBP"]      = td["OBP"].map(lambda v: fmt(v,"{:.3f}"))
        td["KPct"]     = td["KPct"].map(lambda v: fmt(v*100,"{:.0f}%") if pd.notna(v) else "—")
        td["BBPct"]    = td["BBPct"].map(lambda v: fmt(v*100,"{:.0f}%") if pd.notna(v) else "—")
        td["xBA"]      = td["xBA"].map(lambda v: fmt(v,"{:.3f}"))
        td["AvgEV"]    = td["AvgEV"].map(lambda v: fmt(v,"{:.1f}"))
        td["HardPct"]  = td["HardPct"].map(lambda v: fmt(v*100,"{:.0f}%") if pd.notna(v) else "—")
        td["EV_RHP"]   = td["EV_RHP"].map(lambda v: fmt(v,"{:.1f}"))
        td["EV_LHP"]   = td["EV_LHP"].map(lambda v: fmt(v,"{:.1f}"))
        td["AvgLA"]    = td["AvgLA"].map(lambda v: fmt(v,"{:.1f}°"))
        td.columns    = ["Batter","Pos","B","PA","OBP","K%","BB%","xBA","Avg EV","Hard%","EV vs R","EV vs L","Avg LA"]
        st.dataframe(td, use_container_width=True, height=360)

        st.divider()

        # Matchup breakdown
        if matchup_pitcher:
            st.markdown(f"#### Matchup Breakdown — vs {player_last(matchup_pitcher)}")
            st.caption("Ranked by projected xwOBA against pitches matching this pitcher's actual "
                      "velocity and movement per pitch type — same approach as the Reliever Matchup "
                      "Planner, with a tighter similarity window since this is one known opponent. "
                      "Usage is what he throws to THIS hitter's side specifically (pitchers change "
                      "their mix by batter handedness), not his overall pooled mix. "
                      "Orange = fewer than 3 similar-stuff pitches seen on that pitch type.")
            for i, row in starters.iterrows():
                name    = row["Batter"]
                details = matchup_details.get(name, [])
                is_low  = matchup_low_sample.get(name, False)
                mxwoba  = matchup_xwoba.get(name)

                if not details:
                    st.markdown(
                        f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;"
                        f"border-radius:6px;padding:8px 14px;margin-bottom:4px;"
                        f"display:flex;align-items:center;gap:10px;'>"
                        f"<span style='width:130px;font-weight:600;color:#1e293b;'>"
                        f"{player_last(name)}</span>"
                        f"<span style='color:#475569;font-size:0.8rem;'>No pitch type data</span>"
                        f"</div>", unsafe_allow_html=True)
                    continue

                # Build pitch pills
                pills = ""
                for d in details[:3]:  # top 3 by usage
                    color = PITCH_COLORS.get(d["pitch"], "#64748b")
                    border = "#f59e0b" if d["low"] else color
                    xwoba_color = ("#22c55e" if d["xwoba"] >= 0.55 else
                                   "#64748b" if d["xwoba"] >= 0.35 else "#ef4444")
                    warn_icon = "<span style='color:#f59e0b;font-size:0.65rem;'>⚠</span>"
                    pills += (
                        f"<span style='background:#111;border:1.5px solid {border};"
                        f"border-radius:4px;padding:2px 8px;margin-right:4px;"
                        f"font-size:0.75rem;display:inline-flex;gap:6px;align-items:center;'>"
                        f"<span style='color:{color};font-weight:700;'>{d['pitch']}</span>"
                        f"<span style='color:#64748b;'>{d['usage']:.0%}</span>"
                        f"<span style='color:{xwoba_color};font-weight:700;'>{d['xwoba']:.3f}</span>"
                        f"{warn_icon if d['low'] else ''}"
                        f"</span>"
                    )

                if mxwoba is not None:
                    xwoba_badge_color = ("#22c55e" if mxwoba >= 0.55 else
                                         "#ef4444" if mxwoba < 0.35 else "#64748b")
                    badge = f"xwOBA {mxwoba:.3f}"
                else:
                    xwoba_badge_color = "#64748b"
                    badge = "xwOBA —"
                st.markdown(
                    f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;"
                    f"border-radius:6px;padding:8px 14px;margin-bottom:4px;"
                    f"display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>"
                    f"<span style='width:110px;font-weight:600;color:#1e293b;font-size:0.9rem;'>"
                    f"{player_last(name)}</span>"
                    f"{pills}"
                    f"<span style='margin-left:auto;font-size:0.8rem;color:{xwoba_badge_color};"
                    f"font-weight:700;'>{badge}</span>"
                    f"</div>", unsafe_allow_html=True)

            st.caption("Format: **Pitch** Usage% xwOBA | Orange border = fewer than 3 similar-stuff "
                      "pitches seen on that pitch type")

        st.divider()
        if st.button("Export Lineup Card", use_container_width=True):
            lines = [
                "BROOKHAVEN BANDITS — LINEUP CARD",
                f"Vs {'RHP' if opp_hand=='Right' else 'LHP' if opp_hand=='Left' else 'Unknown Pitcher'}",
                "" * 46
            ]
            for i, row in starters.iterrows():
                ev_str = f"{row['AvgEV']:.1f}" if pd.notna(row["AvgEV"]) else "—"
                k_str  = f"{row['KPct']*100:.0f}%" if pd.notna(row['KPct']) else "—"
                bb_str = f"{row['BBPct']*100:.0f}%" if pd.notna(row['BBPct']) else "—"
                lines.append(
                    f"{i+1:2}.  {row['DefPos']:<4} {player_last(row['Batter']):<18} "
                    f"{row['Side']}  OBP:{row['OBP']:.3f}  K%:{k_str}  BB%:{bb_str}  EV:{ev_str}"
                )
            st.code("\n".join(lines), language=None)
            st.caption("Copy or Ctrl+P to print.")

# CATCHER SPLITS ###

elif page == "Catcher Splits":
    catcher_splits_page.render()

# ─────────────────────────────────────────
#  PAGE: RETURNER BOARD (FCBL cross-league bring-back board)
# ─────────────────────────────────────────
elif page == "Returner Board":
    returner_board_page.render(DATA_DIR, EXCLUDED_TEAMS=EXCLUDED_TEAMS,
                               goto_pitcher=_goto_pitcher_scouting,
                               goto_hitter=_goto_batter_analysis,
                               df_all=df_all, team_label=team_location)

# ─────────────────────────────────────────
#  PAGE: HITTER STAT LINES
# ─────────────────────────────────────────
# ─────────────────────────────────────────
#  PAGE: PLAYER REPORT (one-page, MLB style)
# ─────────────────────────────────────────
elif page == "Player Report":
    st.markdown(
        "<div style='text-align:center;padding:14px 0 6px 0;border-bottom:3px solid #C8102E;'>"
        "<div style='font-family:Oswald,Inter,sans-serif;font-size:46px;font-weight:600;"
        "color:#C8102E;letter-spacing:.06em;line-height:1.02;'>BROOKHAVEN BANDITS</div>"
        "<div style='font-family:Oswald,Inter,sans-serif;font-size:15px;color:#475569;"
        "letter-spacing:.35em;margin-top:4px;'>ALL-STAR BREAK REPORT</div></div>",
        unsafe_allow_html=True)

    _pr = df_all[df_all["BatterTeam"] == MY_TEAM].copy()
    for _c in ["ExitSpeed", "Angle", "PlateLocSide", "PlateLocHeight"]:
        _pr[_c] = pd.to_numeric(_pr[_c], errors="coerce")
    _names = sorted([b for b in _pr["Batter"].dropna().unique() if not _is_removed(b)])
    if not _names:
        st.info("No Bandits hitters found.")
    else:
        who = st.selectbox("Player", options=_names, format_func=player_last, key="pr_player")
        bp = _pr[_pr["Batter"] == who]

        HITS = ["Single", "Double", "Triple", "HomeRun"]
        ab = int((_ab_mask(bp)).sum())
        h = int(bp["PlayResult"].isin(HITS).sum())
        ba = h / ab if ab else 0.0
        pa = int((bp["PitchofPA"] == 1).sum())
        bb = int(bp["KorBB"].eq("Walk").sum())
        k = int(bp["KorBB"].eq("Strikeout").sum())
        hbp = int(bp["PitchCall"].eq("HitByPitch").sum())
        dbl = int(bp["PlayResult"].eq("Double").sum())
        trp = int(bp["PlayResult"].eq("Triple").sum())
        hr = int(bp["PlayResult"].eq("HomeRun").sum())
        side = bp["BatterSide"].dropna().iloc[0] if bp["BatterSide"].notna().any() else "?"
        side_lbl = {"Right": "Bats R", "Left": "Bats L"}.get(side, "")

        st.markdown("<div style='display:flex;align-items:baseline;gap:14px;margin-top:10px;'>"
                    "<div style='font-size:30px;font-weight:800;color:#1e293b;'>" + player_last(who) + "</div>"
                    "<div style='font-size:14px;color:#475569;'>" + side_lbl + " &middot; " + str(pa) +
                    " PA &middot; " + str(ab) + " AB</div></div>", unsafe_allow_html=True)

        # ── Line 1: the slash-style basics ──
        st.markdown("#### Season Line")
        st.caption("Batting average uses the app-wide at-bat definition: a plate appearance that "
                   "ends in a ball in play or a strikeout. Walks and hit-by-pitches are not at-bats.")
        c = st.columns(7)
        c[0].metric("AVG", f"{ba:.3f}", help=f"{h}-for-{ab}")
        c[1].metric("H", str(h)); c[2].metric("2B", str(dbl))
        c[3].metric("3B", str(trp)); c[4].metric("HR", str(hr))
        c[5].metric("BB", str(bb), help=f"BB rate {100*bb/pa:.0f}%" if pa else "")
        c[6].metric("K", str(k), help=f"K rate {100*k/pa:.0f}%" if pa else "")

        # ── Line 2: contact quality ──
        bip = bp[bp["PitchCall"].eq("InPlay")]
        barrel, hard, n_bip = _quality_rates(bip)
        ev = bip["ExitSpeed"].dropna()
        la = bip["Angle"].dropna()
        xba = bip.apply(lambda r: calc_xba(r["ExitSpeed"], r["Angle"]), axis=1).mean() if len(bip) else np.nan

        st.markdown("#### Contact Quality")
        st.caption("Barrel% is the share of batted balls whose exit-velocity-and-launch-angle score "
                   "clears the frozen 2026 FCBL cutoff (76.82) — the league's top ~8% of contact. "
                   "Hard-Hit% is the share hit at 90+ mph. xBA is expected batting average from "
                   "contact quality alone; a big gap versus AVG means luck, good or bad.")
        q = st.columns(6)
        q[0].metric("Avg EV", f"{ev.mean():.1f}" if len(ev) else "—")
        q[1].metric("Max EV", f"{ev.max():.1f}" if len(ev) else "—")
        q[2].metric("Barrel%", f"{barrel:.1f}%" if barrel == barrel else "—", help=f"{n_bip} batted balls")
        q[3].metric("Hard-Hit%", f"{hard:.1f}%" if hard == hard else "—", help="EV >= 90 mph")
        q[4].metric("Avg LA", f"{la.mean():.1f}°" if len(la) else "—")
        q[5].metric("xBA", f"{xba:.3f}" if xba == xba else "—")

        # ── Swing decisions ──
        haz = _attack_zone_frame(bp)
        z_sw, o_sw, n_in, n_out = _true_zone_swing(haz)
        lg_haz = _attack_zone_frame(df_all)
        lg_z, lg_o, _, _ = _true_zone_swing(lg_haz)
        lg_barrel, lg_hard, _ = _quality_rates(df_all[df_all["PitchCall"].eq("InPlay")].assign(
            ExitSpeed=pd.to_numeric(df_all[df_all["PitchCall"].eq("InPlay")]["ExitSpeed"], errors="coerce"),
            Angle=pd.to_numeric(df_all[df_all["PitchCall"].eq("InPlay")]["Angle"], errors="coerce")))

        st.markdown("#### Swing Decisions")
        st.caption("Z-Swing% is how often he swings at pitches inside the true TrackMan strike zone "
                   "(19.9 in wide, 21.1–40.5 in tall). O-Swing% is how often he chases pitches outside "
                   "it. Good hitters swing often in the zone and rarely out of it. The map below shades "
                   "each attack-zone band by swing rate, with the real strike zone drawn dashed in green.")
        s = st.columns(4)
        s[0].metric("Z-Swing%", f"{z_sw:.0f}%", delta=f"{z_sw-lg_z:+.0f} vs lg", help=f"n={n_in}")
        s[1].metric("O-Swing%", f"{o_sw:.0f}%", delta=f"{o_sw-lg_o:+.0f} vs lg",
                    delta_color="inverse", help=f"n={n_out}")
        s[2].metric("League Z-Swing%", f"{lg_z:.0f}%")
        s[3].metric("League O-Swing%", f"{lg_o:.0f}%")

        if len(haz) >= 10:
            zorder = ["Heart", "Shadow", "Chase", "Waste"]
            sw_by = {}
            rv_by = {}
            hz = haz.copy()
            hz["_rv"] = hz.apply(_attack_decision_rv, axis=1)
            for z in zorder:
                zz = hz[hz["_az"] == z]
                sw_by[z] = float(zz["_swing"].mean() * 100) if len(zz) else 0.0
                sw = zz[zz["_swing"]]
                rv_by[z] = float(sw["_rv"].mean()) if len(sw) else 0.0
            g1, g2 = st.columns(2)
            with g1:
                components.html(_attack_zone_svg(sw_by, "Swing% by attack zone", is_rate=True), height=380)
            with g2:
                components.html(_attack_zone_svg(rv_by, "Run value when he swings", is_rate=False), height=380)
            st.caption("Left: swing rate per band. Right: the average run value of his swings in each "
                       "band — a descriptive proxy from outcomes, not a fitted model. The ideal shape is "
                       "hot in the Heart and cold in Chase/Waste.")

        # ── Heat zones ──
        st.markdown("#### Heat Zones")
        st.caption("The strike zone split into nine equal cells (each 6.6 in wide by 6.5 in tall), "
                   "catcher's view. Left shows batting average on balls in play from each cell; right "
                   "shows average exit velocity. Cells with few batted balls are noisy — check n.")
        ZB, ZT, ZHW = 1.755, 3.378, 0.83
        zbip = bip[bip["PlateLocSide"].between(-ZHW, ZHW) & bip["PlateLocHeight"].between(ZB, ZT)]
        xe = np.linspace(-ZHW, ZHW, 4)
        ye = np.linspace(ZB, ZT, 4)
        ba_grid, ev_grid = [], []
        for i in range(3):                       # row 0 = top of zone
            ba_row, ev_row = [], []
            ylo, yhi = ye[2 - i], ye[3 - i]
            for j in range(3):
                cell = zbip[zbip["PlateLocSide"].between(xe[j], xe[j + 1]) &
                            zbip["PlateLocHeight"].between(ylo, yhi)]
                n = len(cell)
                if n == 0:
                    ba_row.append((float("nan"), 0)); ev_row.append((float("nan"), 0))
                else:
                    ba_row.append((cell["PlayResult"].isin(HITS).mean(), n))
                    e = cell["ExitSpeed"].dropna()
                    ev_row.append((e.mean() if len(e) else float("nan"), n))
            ba_grid.append(ba_row); ev_grid.append(ev_row)
        h1, h2 = st.columns(2)
        with h1:
            components.html(_heat3x3_svg(ba_grid, "Batting average by cell"), height=290)
        with h2:
            components.html(_heat3x3_svg(ev_grid, "Exit velocity by cell",
                                         fmt_fn=lambda v: format(v, ".0f")), height=290)

        # ── Splits ──
        st.markdown("#### Splits")
        st.caption("Performance by opposing pitcher hand, in the leadoff spot of an inning, and "
                   "with two outs. Small samples move fast — treat these as tendencies.")
        def _sp(sub):
            a = int((_ab_mask(sub)).sum())
            hh = int(sub["PlayResult"].isin(HITS).sum())
            return f"{hh}-{a} ({hh/a:.3f})" if a else "—"
        sp = st.columns(4)
        sp[0].metric("vs RHP", _sp(bp[bp["PitcherThrows"] == "Right"]))
        sp[1].metric("vs LHP", _sp(bp[bp["PitcherThrows"] == "Left"]))
        sp[2].metric("Leadoff", _sp(bp[bp["PAofInning"] == 1]))
        sp[3].metric("Two outs", _sp(bp[bp["Outs"] == 2]))

        # ── Noteworthy ──
        st.markdown("#### Noteworthy")
        st.caption("Automatic callouts where this player stands clearly apart from the league, in "
                   "either direction. Thresholds are deliberately wide so only real gaps appear.")
        good, bad = [], []
        if barrel == barrel and lg_barrel == lg_barrel and n_bip >= 8:
            if barrel >= lg_barrel + 4: good.append(f"Barrels the ball at {barrel:.0f}%, well above the league's {lg_barrel:.0f}%.")
            if barrel <= lg_barrel - 4: bad.append(f"Barrel rate of {barrel:.0f}% trails the league's {lg_barrel:.0f}%.")
        if hard == hard and lg_hard == lg_hard and n_bip >= 8:
            if hard >= lg_hard + 6: good.append(f"Hard-hit rate of {hard:.0f}% is well clear of the league's {lg_hard:.0f}%.")
            if hard <= lg_hard - 6: bad.append(f"Hard-hit rate of {hard:.0f}% sits below the league's {lg_hard:.0f}%.")
        if n_out >= 25:
            if o_sw <= lg_o - 6: good.append(f"Rarely chases: {o_sw:.0f}% O-Swing versus {lg_o:.0f}% league.")
            if o_sw >= lg_o + 6: bad.append(f"Chases too much: {o_sw:.0f}% O-Swing versus {lg_o:.0f}% league.")
        if n_in >= 25 and z_sw <= lg_z - 8:
            bad.append(f"Passive in the zone: swings at just {z_sw:.0f}% of strikes ({lg_z:.0f}% league).")
        if pa >= 15:
            if k / pa >= 0.30: bad.append(f"Strikes out in {100*k/pa:.0f}% of plate appearances.")
            if bb / pa >= 0.14: good.append(f"Walks in {100*bb/pa:.0f}% of plate appearances.")
        if xba == xba and ab >= 10:
            if ba - xba >= 0.080: bad.append(f"AVG of {ba:.3f} outruns an xBA of {xba:.3f} — some of this is luck.")
            if xba - ba >= 0.080: good.append(f"xBA of {xba:.3f} beats a {ba:.3f} AVG — hitting into bad luck.")
        if len(ev) and ev.max() >= 105: good.append(f"Top-end power: max exit velocity of {ev.max():.1f} mph.")

        n1, n2 = st.columns(2)
        with n1:
            st.markdown("**Strengths**")
            st.markdown("\n".join("- " + g for g in good) if good else "_Nothing separates from league yet._")
        with n2:
            st.markdown("**Watch**")
            st.markdown("\n".join("- " + b for b in bad) if bad else "_No red flags at these thresholds._")

        st.divider()
        st.caption("Report built from TrackMan pitch-level data. Barrel and hard-hit thresholds are "
                   "frozen 2026 FCBL cutoffs. Run values are descriptive proxies, not fitted models. "
                   "Zone geometry matches the TrackMan strike zone used throughout the app.")


elif page == "Hitter Stat Lines":
    st.title("Hitter Stat Lines")
    st.caption("Full scouting stat line for every hitter. Data-derived from TrackMan. "
               "Baserunner-dependent splits (RISP, runners on, LOB) aren't in the data, so "
               "they're replaced with reliable ones (2-out, leadoff, by-count discipline).")

    hsl_team = st.selectbox("Team", options=sorted(_team_options(df_all["BatterTeam"])),
                            index=(sorted(_team_options(df_all["BatterTeam"])).index(MY_TEAM)
                                   if MY_TEAM in _team_options(df_all["BatterTeam"]) else 0),
                            format_func=team_label, key="hsl_team")

    ZT, ZB, ZH = 3.378, 1.755, 0.83
    _TAKE = ["BallCalled", "StrikeCalled", "BallinDirt", "HitByPitch"]

    def _h_avg(sub):
        ab = (_ab_mask(sub)).sum()
        h  = sub["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum()
        return int(h), int(ab), (h / ab if ab else 0.0)

    def _count_line(sub, label):
        # take rate + zone rate at a given count; None if never reached
        if len(sub) == 0:
            return None
        take = sub["PitchCall"].isin(_TAKE).mean()
        inz  = ((sub["PlateLocSide"].abs() <= ZH) &
                (sub["PlateLocHeight"].between(ZB, ZT))).mean()
        return len(sub), take, inz

    def _zone_label(side, height):
        v = "Up" if height > 2.9 else ("Low" if height < 2.2 else "Mid")
        h = "in" if side < -0.28 else ("away" if side > 0.28 else "middle")
        return v + " and " + h

    hsl_df = df_all[df_all["BatterTeam"] == hsl_team].copy()
    for _c in ["ExitSpeed", "PlateLocHeight", "PlateLocSide", "Balls", "Strikes"]:
        if _c in hsl_df.columns:
            hsl_df[_c] = pd.to_numeric(hsl_df[_c], errors="coerce")

    # Identify games with broken count data (counts never advance past 0-0) so
    # the count-discipline bullets don't get polluted by them.
    _bad_games = set()
    for g, grp in hsl_df.groupby("GameID"):
        if grp["Balls"].max() == 0 and grp["Strikes"].max() == 0:
            _bad_games.add(g)

    hitters = sorted([h for h in hsl_df["Batter"].dropna().unique()
                      if not _is_removed(h) and not _is_report_hidden(h)])
    if not hitters:
        st.info("No hitters found for this team.")
    else:
        for b in hitters:
            bp = hsl_df[hsl_df["Batter"] == b]
            tot_h, tot_ab, tot_ba = _h_avg(bp)
            if tot_ab == 0 and len(bp) < 5:
                continue
            side = bp["BatterSide"].dropna().iloc[0] if bp["BatterSide"].notna().any() else "?"
            side_lbl = {"Right": "RHH", "Left": "LHH"}.get(side, "")
            st.markdown("### " + player_last(b) + ("  (" + side_lbl + ")" if side_lbl else ""))
            st.markdown("**Total pitches seen: " + str(len(bp)) + ", " +
                        str(tot_h) + "-" + str(tot_ab) + " (" + f"{tot_ba:.3f}" + ")**")

            lines = []
            # Count discipline (exclude broken-count games)
            cp = bp[~bp["GameID"].isin(_bad_games)]
            for (bb, ss), lbl in [((0, 0), "0-0"), ((1, 0), "1-0"), ((0, 1), "0-1")]:
                r = _count_line(cp[(cp["Balls"] == bb) & (cp["Strikes"] == ss)], lbl)
                if r is None:
                    lines.append("* Has never been to " + lbl)
                elif lbl == "0-0":
                    lines.append("* Takes " + f"{r[1]:.0%}" + " on 0-0, " +
                                 f"{r[2]:.0%}" + " are in the zone")
                else:
                    lines.append("* " + lbl + " take rate: " + f"{r[1]:.0%}" +
                                 "; zone rate: " + f"{r[2]:.0%}")

            # Best/worst exit velo spot
            bip = bp[(bp["PitchCall"] == "InPlay") & bp["ExitSpeed"].notna() &
                     bp["PlateLocHeight"].notna() & bp["PlateLocSide"].notna()].copy()
            if len(bip) >= 3:
                bip["_z"] = bip.apply(lambda r: _zone_label(r["PlateLocSide"], r["PlateLocHeight"]), axis=1)
                zev = bip.groupby("_z")["ExitSpeed"].mean().sort_values(ascending=False)
                lines.append("* Best Exit Velo Spot: " + zev.index[0] +
                             " (" + f"{zev.iloc[0]:.2f}" + " MPH)")
                lines.append("* Worst Exit Velo Spot: " + zev.index[-1] +
                             " (" + f"{zev.iloc[-1]:.2f}" + ")")

            st.markdown("\n".join(lines))
            st.markdown("**Situationals + Splits:**")
            slines = []
            lhp = _h_avg(bp[bp["PitcherThrows"] == "Left"])
            rhp = _h_avg(bp[bp["PitcherThrows"] == "Right"])
            lo  = _h_avg(bp[bp["PAofInning"] == 1])
            two = _h_avg(bp[bp["Outs"] == 2])
            slines.append("* Vs LHP: " + str(lhp[0]) + "-" + str(lhp[1]) + " (" + f"{lhp[2]:.3f}" +
                          "), vs RHP: " + str(rhp[0]) + "-" + str(rhp[1]) + " (" + f"{rhp[2]:.3f}" + ")")
            slines.append("* Reach as the leadoff: " + str(lo[0]) + "-" + str(lo[1]) +
                          " (" + f"{lo[2]:.3f}" + ")")
            slines.append("* With 2 outs: " + str(two[0]) + "-" + str(two[1]) +
                          " (" + f"{two[2]:.3f}" + ")")
            st.markdown("\n".join(slines))
            st.divider()


# ─────────────────────────────────────────
#  PAGE: PITCHER STAT LINES
# ─────────────────────────────────────────
elif page == "Pitcher Stat Lines":
    st.title("Pitcher Stat Lines")
    st.caption("Full scouting stat line for every pitcher. Data-derived from TrackMan — "
               "'Runs' is total runs (earned/unearned can't be separated); '2 outs' replaces "
               "runners-on-base since base state isn't in the data.")

    psl_team = st.selectbox("Team", options=sorted(_team_options(df_all["PitcherTeam"])),
                            index=(sorted(_team_options(df_all["PitcherTeam"])).index(MY_TEAM)
                                   if MY_TEAM in _team_options(df_all["PitcherTeam"]) else 0),
                            format_func=team_label, key="psl_team")

    _STRIKE_CALLS = ["StrikeCalled", "StrikeSwinging", "FoulBallNotFieldable",
                     "FoulBallFieldable", "InPlay"]

    def _avg_split(sub):
        ab = (_ab_mask(sub)).sum()
        h  = sub["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum()
        return int(h), int(ab), (h / ab if ab else 0.0)

    psl_df = df_all[df_all["PitcherTeam"] == psl_team].copy()
    pitchers = sorted([p for p in psl_df["Pitcher"].dropna().unique()
                       if not _is_removed(p) and not _is_report_hidden(p)])

    if not pitchers:
        st.info("No pitchers found for this team.")
    else:
        for p in pitchers:
            pp = psl_df[psl_df["Pitcher"] == p]
            bf = int((pp["PitchofPA"] == 1).sum())
            if bf == 0:
                continue
            outs = pp["OutsOnPlay"].fillna(0).sum() + pp["KorBB"].eq("Strikeout").sum()
            ip = f"{int(outs // 3)}.{int(outs % 3)}"
            k = int(pp["KorBB"].eq("Strikeout").sum())
            bb = int(pp["KorBB"].eq("Walk").sum())
            k_rate = k / bf if bf else 0
            tot = len(pp)
            ppb = tot / bf if bf else 0
            apps = int(pp["GameID"].nunique())
            runs = int(pp["RunsScored"].fillna(0).sum())
            lhh = _avg_split(pp[pp["BatterSide"] == "Left"])
            rhh = _avg_split(pp[pp["BatterSide"] == "Right"])
            lo  = _avg_split(pp[pp["PAofInning"] == 1])
            two = _avg_split(pp[pp["Outs"] == 2])
            fp = pp[pp["PitchofPA"] == 1]
            fps = fp["PitchCall"].isin(_STRIKE_CALLS).mean() if len(fp) else 0
            fp_r = fp[fp["BatterSide"] == "Right"]
            fp_l = fp[fp["BatterSide"] == "Left"]
            fps_r = fp_r["PitchCall"].isin(_STRIKE_CALLS).mean() if len(fp_r) else 0
            fps_l = fp_l["PitchCall"].isin(_STRIKE_CALLS).mean() if len(fp_l) else 0

            hand = pp["PitcherThrows"].dropna().iloc[0] if pp["PitcherThrows"].notna().any() else "?"
            hand_lbl = {"Right": "RHP", "Left": "LHP"}.get(hand, "")
            st.markdown("### " + player_last(p) + ("  (" + hand_lbl + ")" if hand_lbl else ""))
            st.markdown("**" + ip + " IP, " + str(bf) + " BF - " + str(k) +
                        " K (" + f"{k_rate:.0%}" + " K Rate)**")
            lines = []
            lines.append("* " + str(tot) + " total pitches, Average of " +
                         f"{ppb:.1f}" + " (" + str(round(ppb)) + ") pitches per batter")
            lines.append("* " + str(runs) + " runs across " + str(apps) + " appearances")
            lines.append("* Opp Avg: LHH " + str(lhh[0]) + "-" + str(lhh[1]) +
                         " (" + f"{lhh[2]:.3f}" + "), RHH " + str(rhh[0]) + "-" + str(rhh[1]) +
                         " (" + f"{rhh[2]:.3f}" + ")")
            lines.append("* Against Leadoff " + str(lo[0]) + "-" + str(lo[1]) +
                         " (" + f"{lo[2]:.3f}" + ")")
            lines.append("* With 2 outs " + str(two[0]) + "-" + str(two[1]) +
                         " (" + f"{two[2]:.3f}" + ")")
            lines.append("* K to BB ratio " + str(k) + ":" + str(bb))
            lines.append("* " + f"{fps:.0%}" + " first pitch strike rate; " +
                         f"{fps_r:.0%}" + " RHH, " + f"{fps_l:.0%}" + " LHH")
            st.markdown("\n".join(lines))
            st.divider()


# ─────────────────────────────────────────
#  PAGE: HOT / COLD TRACKER
# ─────────────────────────────────────────
elif page == "Hot / Cold":
    st.title("Hot / Cold Tracker")
    st.caption("Who's trending up or down. Two honest signals: (1) recent contact-quality trend "
               "vs earlier, and (2) results running ahead of or behind expected (luck). Short-season "
               "samples are small — read as a nudge, not gospel.")

    hc_side = st.radio("View", ["Hitters", "Pitchers"], horizontal=True, key="hc_side")
    _hc_teams = (sorted(_team_options(df_all["BatterTeam"])) if hc_side == "Hitters"
                 else sorted(_team_options(df_all["PitcherTeam"])))
    hc_team = st.selectbox("Team", options=_hc_teams,
                           index=_hc_teams.index(MY_TEAM) if MY_TEAM in _hc_teams else 0,
                           format_func=team_label, key="hc_team")

    work = df_all.assign(_date=pd.to_datetime(df_all["Date"], errors="coerce"))

    if hc_side == "Hitters":
        st.caption("Trend = recent vs early hard-hit% (last half of batted balls vs first half). "
                   "Luck = batting average vs expected BA (xBA from contact quality).")
        players = _player_options_reports(work[work["BatterTeam"] == hc_team]["Batter"])
        rows = []
        for b in players:
            bp = work[work["Batter"] == b].sort_values("_date")
            bip = bp[(bp["PitchCall"] == "InPlay") & bp["ExitSpeed"].notna() & bp["Angle"].notna()]
            if len(bip) < 6:
                continue
            half = len(bip) // 2
            early_hh = (bip.iloc[:half]["ExitSpeed"] >= 90).mean()
            recent_hh = (bip.iloc[half:]["ExitSpeed"] >= 90).mean()
            trend_delta = recent_hh - early_hh
            ab = bp[(bp["PitchCall"] == "InPlay") | bp["KorBB"].eq("Strikeout")]
            hits = ab["PlayResult"].isin(["Single","Double","Triple","HomeRun"]).sum()
            ba = hits / len(ab) if len(ab) else np.nan
            xba = bip.apply(lambda r: calc_xba(r["ExitSpeed"], r["Angle"]), axis=1).mean()
            luck = (ba - xba) if (pd.notna(ba) and pd.notna(xba)) else np.nan
            if trend_delta >= 0.10:
                tag = "🔥 Heating"
            elif trend_delta <= -0.10:
                tag = "🧊 Cooling"
            else:
                tag = "— Steady"
            rows.append({
                "Hitter": player_last(b), "Trend": tag,
                "HardHit% (early→recent)": f"{100*early_hh:.0f}% → {100*recent_hh:.0f}%",
                "BA": f"{ba:.3f}" if pd.notna(ba) else "—",
                "xBA": f"{xba:.3f}" if pd.notna(xba) else "—",
                "Luck (BA−xBA)": f"{luck:+.3f}" if pd.notna(luck) else "—",
                "BBE": len(bip),
                "_sort": trend_delta,
            })
        if rows:
            hdf = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")
            st.dataframe(hdf, use_container_width=True, hide_index=True)
            st.caption("🔥/🧊 = recent hard-hit% up/down ≥10 pts vs early. Positive Luck = "
                       "outhitting contact quality (may regress down); negative = unlucky (may climb). "
                       "BBE = batted-ball events; small samples are noisy.")
        else:
            st.info("No hitters with enough batted balls (need 6+) on this team yet.")

    else:  # Pitchers
        st.caption("Trend = recent vs early whiff% (last half of swings vs first half). "
                   "HardHit% allowed shows how hard they're being hit.")
        players = _player_options_reports(work[work["PitcherTeam"] == hc_team]["Pitcher"])
        rows = []
        for p in players:
            pp = work[work["Pitcher"] == p].sort_values("_date")
            sw = pp[pp["PitchCall"].isin(["StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable","InPlay"])]
            if len(sw) < 10:
                continue
            half = len(sw) // 2
            early_wh = (sw.iloc[:half]["PitchCall"] == "StrikeSwinging").mean()
            recent_wh = (sw.iloc[half:]["PitchCall"] == "StrikeSwinging").mean()
            trend_delta = recent_wh - early_wh
            bip = pp[(pp["PitchCall"] == "InPlay") & pp["ExitSpeed"].notna()]
            hh_allowed = (bip["ExitSpeed"] >= 90).mean() if len(bip) else np.nan
            if trend_delta >= 0.05:
                tag = "🔥 Sharpening"
            elif trend_delta <= -0.05:
                tag = "🧊 Slipping"
            else:
                tag = "— Steady"
            rows.append({
                "Pitcher": player_last(p), "Trend": tag,
                "Whiff% (early→recent)": f"{100*early_wh:.0f}% → {100*recent_wh:.0f}%",
                "HardHit% allowed": f"{100*hh_allowed:.0f}%" if pd.notna(hh_allowed) else "—",
                "Swings": len(sw),
                "_sort": trend_delta,
            })
        if rows:
            pdf2 = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")
            st.dataframe(pdf2, use_container_width=True, hide_index=True)
            st.caption("🔥/🧊 = recent whiff% up/down ≥5 pts vs early. HardHit% allowed is the share "
                       "of contact hit 90+. Whiff trend on small swing counts is noisy — directional only.")
        else:
            st.info("No pitchers with enough swings (need 10+) on this team yet.")

    st.info("⚠ Short-season data: 'recent' windows are small (often <10 events), so these are "
            "directional signals, not conclusions. Contact-quality trends (hard-hit, whiff) are "
            "more stable than batting average over small samples — that's why they anchor the trend.")


# ─────────────────────────────────────────
#  PAGE: PITCHER COMPARISON
# ─────────────────────────────────────────
# ─────────────────────────────────────────
#  PAGE: GAME PLAN GENERATOR
# ─────────────────────────────────────────
elif page == "Game Plan":
    st.title("Pre-Game Game Plan")
    st.caption("Pick tonight's opposing starter — this assembles a one-sheet: his profile & "
               "tendencies, your optimal lineup against his hand, and each hitter's attack plan "
               "vs his arsenal. Built on pitch-type matchups (works even if you've never faced him).")

    _gp_teams = sorted([t for t in _team_options(df_all["PitcherTeam"]) if t != MY_TEAM])
    gp_team = st.selectbox("Opponent", options=_gp_teams, format_func=team_label, key="gp_team")
    _gp_pitchers = _player_options(df_all[df_all["PitcherTeam"] == gp_team]["Pitcher"])
    if not _gp_pitchers:
        st.info("No pitchers with data for this opponent.")
    else:
        gp_pitcher = st.selectbox("Opposing starter", options=_gp_pitchers,
                                  format_func=player_last, key="gp_pitcher")
        pp = df_all[df_all["Pitcher"] == gp_pitcher].copy()
        for _c in ["RelSpeed", "SpinRate", "Balls", "Strikes"]:
            pp[_c] = pd.to_numeric(pp[_c], errors="coerce")
        throws = pp["PitcherThrows"].dropna().iloc[0] if pp["PitcherThrows"].notna().any() else "?"
        throws_lbl = {"Right": "RHP", "Left": "LHP"}.get(throws, "?")

        # Build the plan text (also used for download)
        plan_lines = []
        plan_lines.append(f"PRE-GAME GAME PLAN — vs {player_last(gp_pitcher)} ({throws_lbl})")
        plan_lines.append(f"Opponent: {team_label(gp_team)}")
        plan_lines.append("=" * 50)

        # ── 1. Pitcher profile ──
        st.markdown("## " + player_last(gp_pitcher) + "  (" + throws_lbl + ")")
        st.markdown("**Arsenal & Usage**")
        ars_rows = []
        plan_lines.append("\nARSENAL:")
        total_p = len(pp)
        for pt, grp in pp.groupby("PitchType"):
            if len(grp) < 3:
                continue
            usage = len(grp) / total_p * 100
            velo = grp["RelSpeed"].mean()
            spin = grp["SpinRate"].mean()
            ars_rows.append({"Pitch": pt, "Usage": f"{usage:.0f}%",
                             "Velo": f"{velo:.0f} mph" if pd.notna(velo) else "—",
                             "Spin": f"{spin:.0f} rpm" if pd.notna(spin) else "—",
                             "_u": usage})
            plan_lines.append(f"  {pt}: {usage:.0f}% usage, {velo:.0f} mph")
        ars_df = pd.DataFrame(ars_rows).sort_values("_u", ascending=False).drop(columns="_u")
        st.dataframe(ars_df, use_container_width=True, hide_index=True)
        # Primary pitch callout
        if len(ars_df):
            top_pitch = ars_df.iloc[0]["Pitch"]
            top_usage = ars_df.iloc[0]["Usage"]
            st.info("Primary pitch: **" + str(top_pitch) + "** (" + str(top_usage) + " of his pitches). "
                    "Sit on this early.")
            plan_lines.append(f"\nPRIMARY PITCH: {top_pitch} ({top_usage}) — sit on this early.")

        # ── 2. Tendencies by count ──
        st.markdown("**Tendencies by Count**")
        _bad = set()
        for g, grp in df_all.groupby("GameID"):
            gb = pd.to_numeric(grp["Balls"], errors="coerce").max()
            gs = pd.to_numeric(grp["Strikes"], errors="coerce").max()
            if gb == 0 and gs == 0:
                _bad.add(g)
        cp = pp[~pp["GameID"].isin(_bad)]
        overall = pp["PitchType"].value_counts(normalize=True)
        tend_rows = []
        plan_lines.append("\nTENDENCIES BY COUNT:")
        for lbl, mask in [("First pitch", (cp["Balls"] == 0) & (cp["Strikes"] == 0)),
                          ("Ahead", cp["Strikes"] > cp["Balls"]),
                          ("Behind", cp["Balls"] > cp["Strikes"]),
                          ("Two strikes", cp["Strikes"] == 2)]:
            sub = cp[mask]
            if len(sub) < 4:
                continue
            mix = sub["PitchType"].value_counts(normalize=True)
            n = len(sub)
            w = n / (n + 8)
            allt = set(overall.index) | set(mix.index)
            blend = {t: w * mix.get(t, 0) + (1 - w) * overall.get(t, 0) for t in allt}
            top = sorted(blend.items(), key=lambda x: -x[1])[:3]
            mix_str = ", ".join(f"{t} {v/sum(blend.values()):.0%}" for t, v in top)
            tend_rows.append({"Count": lbl, "n": n, "Likely": mix_str})
            plan_lines.append(f"  {lbl} (n={n}): {mix_str}")
        if tend_rows:
            st.dataframe(pd.DataFrame(tend_rows), use_container_width=True, hide_index=True)

        # ── 3. Optimal lineup vs his hand ──
        st.markdown("## Ranked Projected Matchups vs " + throws_lbl)
        st.caption("Ranked by production against " + throws_lbl + "s, with the matchup projection "
                   "vs this pitcher's specific arsenal.")
        try:
            baselines = _league_pitch_baselines(df_all)
        except Exception:
            baselines = None
        our = df_all[df_all["BatterTeam"] == MY_TEAM]
        lineup_rows = []
        plan_lines.append(f"\nLINEUP vs {throws_lbl} (BA vs hand | projected vs this pitcher):")
        for b in _player_options_reports(our["Batter"]):
            if _is_removed(b):
                continue
            bp = our[(our["Batter"] == b) & (our["PitcherThrows"] == throws)]
            ab = (_ab_mask(bp)).sum()
            if ab < 3:
                continue
            h = bp["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum()
            ba = h / ab if ab else 0
            proj = None
            if baselines is not None:
                try:
                    proj = project_matchup(b, gp_pitcher, df_all, baselines)
                except Exception:
                    proj = None
            proj_xwoba = proj["xwoba"] if (proj and "xwoba" in proj) else None
            lineup_rows.append({
                "Hitter": player_last(b), "Side": bp["BatterSide"].dropna().iloc[0] if bp["BatterSide"].notna().any() else "?",
                "BA vs " + throws_lbl: f"{ba:.3f}",
                "H-AB": f"{int(h)}-{int(ab)}",
                "Proj xwOBA": f"{proj_xwoba:.3f}" if proj_xwoba is not None else "—",
                "_sort": proj_xwoba if proj_xwoba is not None else ba})
        lineup_rows.sort(key=lambda r: -r["_sort"])
        pdf_lineup = []
        for i, r in enumerate(lineup_rows, 1):
            r.pop("_sort")
            plan_lines.append(f"  {i}. {r['Hitter']} ({r['Side']}) — {r['BA vs ' + throws_lbl]} vs {throws_lbl}, proj xwOBA {r['Proj xwOBA']}")
            pdf_lineup.append({"Hitter": r["Hitter"], "Side": r["Side"],
                               "BA": r["BA vs " + throws_lbl], "Proj": r["Proj xwOBA"]})
        if lineup_rows:
            st.dataframe(pd.DataFrame(lineup_rows), use_container_width=True, hide_index=True)

        # ── Select tonight's lineup ──
        hitter_by_last = {player_last(b): b for b in _player_options_reports(our["Batter"])}
        st.markdown("#### Tonight's Lineup")
        st.caption("Defaults to the top 9 above — adjust to match the actual card before generating breakdowns.")
        gp_lineup_names = st.multiselect(
            "Hitters in tonight's lineup",
            options=[r["Hitter"] for r in lineup_rows],
            default=[r["Hitter"] for r in lineup_rows[:9]],
            key="gp_lineup_sel")

        # ── 4. Per-hitter attack notes ──
        st.markdown("## Attack Plans")
        plan_lines.append("\nATTACK PLANS:")
        pdf_attack = []
        arsenal_types = [r["Pitch"] for r in ars_rows]
        for r in lineup_rows:
            if r["Hitter"] not in gp_lineup_names:
                continue
            hitter_full = hitter_by_last.get(r["Hitter"])
            if hitter_full is None:
                continue
            bp = our[our["Batter"] == hitter_full]
            # best/worst pitch type for this hitter among the pitcher's arsenal
            notes = []
            for pt in arsenal_types:
                sub = bp[bp["PitchType"] == pt]
                ab = (_ab_mask(sub)).sum()
                if ab >= 3:
                    h = sub["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum()
                    notes.append((pt, h / ab, ab))
            if notes:
                notes.sort(key=lambda x: -x[1])
                best = notes[0]
                worst = notes[-1]
                line = "**" + r["Hitter"] + "**: "
                bits = []
                if best[1] >= 0.300:
                    bits.append("crushes " + best[0] + f" ({best[1]:.3f})")
                if worst[1] <= 0.200 and worst[0] != best[0]:
                    bits.append("struggles vs " + worst[0] + f" ({worst[1]:.3f})")
                line += "; ".join(bits) if bits else "no strong pitch-type split yet"
                st.markdown(line)
                plan_lines.append(f"  {r['Hitter']}: " + ("; ".join(bits) if bits else "no strong split"))
                pdf_attack.append("<b>" + r["Hitter"] + "</b>: " +
                                  ("; ".join(bits) if bits else "no strong pitch-type split yet"))

        # ── 5. Full per-hitter breakdowns ──
        st.markdown("## Hitter Breakdowns")
        st.caption("Strike zone hot/cold zones and OPS/xwOBA/K%/BB%/plate discipline are overall "
                   "(all pitching faced). The pitch-type table is filtered to " + throws_lbl +
                   " arms only, to match tonight's starter.")
        _HITS = ["Single", "Double", "Triple", "HomeRun"]
        _seen_breakdown = set()
        for r in lineup_rows:
            if r["Hitter"] not in gp_lineup_names:
                continue
            hitter_full = hitter_by_last.get(r["Hitter"])
            if hitter_full is None:
                continue
            # Two Trackman name spellings for the same player (e.g. "Keblinsky,
            # Peter" / "Keblinsky, Pete") both resolve to the same hitter_full
            # via hitter_by_last, which would render this player's breakdown
            # twice with duplicate widget keys — skip the repeat.
            if hitter_full in _seen_breakdown:
                continue
            _seen_breakdown.add(hitter_full)
            bp_all = our[our["Batter"] == hitter_full].copy()
            for _c in ["ExitSpeed", "Angle", "PlateLocSide", "PlateLocHeight"]:
                bp_all[_c] = pd.to_numeric(bp_all[_c], errors="coerce")
            bp_hand = bp_all[bp_all["PitcherThrows"] == throws]

            with st.expander(f"{r['Hitter']} ({r['Side']})", expanded=False):
                stats = compute_batter_stats(hitter_full, our, throws)
                m = st.columns(4)
                m[0].metric("OPS", f"{stats['OPS']:.3f}" if pd.notna(stats.get('OPS')) else "—")
                m[1].metric("xwOBA", f"{stats['xwOBA']:.3f}" if stats.get('xwOBA') is not None else "—")
                m[2].metric("K%", f"{stats['KPct']*100:.0f}%" if pd.notna(stats.get('KPct')) else "—")
                m[3].metric("BB%", f"{stats['BBPct']*100:.0f}%" if pd.notna(stats.get('BBPct')) else "—")

                # Plate discipline
                haz = _attack_zone_frame(bp_all)
                z_sw, o_sw, n_in, n_out = _true_zone_swing(haz)
                sw_all = haz[haz["_swing"]]
                _inz = (sw_all["PlateLocSide"].abs() <= _ZONE_HW) & sw_all["PlateLocHeight"].between(_ZONE_B, _ZONE_T)
                sw_in, sw_out = sw_all[_inz], sw_all[~_inz]
                z_contact = 100 * (sw_in["PitchCall"] != "StrikeSwinging").mean() if len(sw_in) else None
                o_contact = 100 * (sw_out["PitchCall"] != "StrikeSwinging").mean() if len(sw_out) else None
                whiff_r = _safe_whiff(bp_all)

                st.markdown("**Plate Discipline** (overall)")
                d = st.columns(5)
                d[0].metric("Z-Swing%", f"{z_sw:.0f}%", help=f"n={n_in}")
                d[1].metric("O-Swing% (chase)", f"{o_sw:.0f}%", help=f"n={n_out}")
                d[2].metric("Z-Contact%", f"{z_contact:.0f}%" if z_contact is not None else "—")
                d[3].metric("O-Contact%", f"{o_contact:.0f}%" if o_contact is not None else "—")
                d[4].metric("Whiff%", f"{whiff_r*100:.0f}%" if whiff_r is not None else "—")

                # Heat zones — overall, catcher's view (same KDE heatmap as Batter Analysis)
                st.markdown("**Strike Zone — Hot / Cold (overall)**")
                hz1, hz2 = st.columns(2)
                with hz1:
                    st.markdown(
                        "<div style='font-size:0.85rem;font-weight:700;color:#475569;'>"
                        "Contact Quality</div><div style='font-size:0.72rem;color:#64748b;"
                        "margin-bottom:6px;'>Red/white = hard contact zones</div>",
                        unsafe_allow_html=True)
                    _render_kde_heatmap(bp_all, weight_col="ExitSpeed",
                                        key_suffix=f"gp_ev_{hitter_full}")
                with hz2:
                    st.markdown(
                        "<div style='font-size:0.85rem;font-weight:700;color:#475569;'>"
                        "Whiff Zones</div><div style='font-size:0.72rem;color:#64748b;"
                        "margin-bottom:6px;'>Red/white = where he whiffs most</div>",
                        unsafe_allow_html=True)
                    _sw_gp = bp_all[bp_all["PitchCall"].isin(
                        {"StrikeSwinging", "InPlay", "FoulBallNotFieldable",
                         "FoulBallFieldable", "FoulTip", "FoulBall"})].copy()
                    _sw_gp["_whiff_weight"] = _sw_gp["PitchCall"].eq("StrikeSwinging").astype(float)
                    if len(_sw_gp) >= 5:
                        _render_kde_heatmap(_sw_gp, weight_col="_whiff_weight",
                                            key_suffix=f"gp_wh_{hitter_full}")
                    else:
                        st.info("Not enough swings for whiff map.")

                # vs each pitch type, filtered to tonight's starter's hand
                st.markdown(f"**By Pitch Type vs {throws_lbl}**")
                if len(bp_hand) == 0:
                    st.caption(f"No {throws_lbl} history yet for this hitter.")
                else:
                    pt_rows = []
                    for pt, grp in bp_hand.groupby("PitchType"):
                        ab = (_ab_mask(grp)).sum()
                        if ab < 2:
                            continue
                        h = grp["PlayResult"].isin(_HITS).sum()
                        n_hr = grp["PlayResult"].eq("HomeRun").sum()
                        ba = h / ab if ab else 0
                        wh = _safe_whiff(grp)
                        # OPS on this pitch type (BB/HBP tied to the pitch that ended the PA)
                        bb_p = grp["KorBB"].eq("Walk").sum()
                        hbp_p = grp["PitchCall"].eq("HitByPitch").sum()
                        singles = grp["PlayResult"].eq("Single").sum()
                        doubles = grp["PlayResult"].eq("Double").sum()
                        triples = grp["PlayResult"].eq("Triple").sum()
                        pa_like = ab + bb_p + hbp_p
                        obp_p = (h + bb_p + hbp_p) / max(pa_like, 1)
                        slg_p = (singles + 2 * doubles + 3 * triples + 4 * n_hr) / max(ab, 1)
                        ops_p = obp_p + slg_p
                        # xwOBA on this pitch type, same fair-BIP method as batter_expected_stats
                        fair_p = grp[grp["ExitSpeed"].notna() & grp["Angle"].notna() &
                                     grp["Direction"].notna() & (grp["Distance"].fillna(0) >= 10) &
                                     (grp["Direction"].abs() <= 45)]
                        if len(fair_p):
                            xwoba_p = fair_p.apply(lambda r: calc_xwoba_bip(r["ExitSpeed"], r["Angle"]),
                                                   axis=1).sum() / ab
                        else:
                            xwoba_p = None
                        pt_rows.append({
                            "Pitch": pt, "Pitches": len(grp), "AB": int(ab), "H": int(h),
                            "AVG": f"{ba:.3f}", "OPS": f"{ops_p:.3f}",
                            "xwOBA": f"{xwoba_p:.3f}" if xwoba_p is not None else "—",
                            "HR": int(n_hr),
                            "Whiff%": f"{wh*100:.0f}%" if wh is not None else "—",
                            "_n": len(grp)})
                    if pt_rows:
                        pt_df = pd.DataFrame(sorted(pt_rows, key=lambda x: -x["_n"])).drop(columns="_n")
                        st.dataframe(pt_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption(f"Not enough at-bats yet by pitch type vs {throws_lbl}.")

        # ── Download ──
        st.divider()
        try:
            pdf_bytes = _build_gameplan_pdf(
                player_last(gp_pitcher), throws_lbl, team_label(gp_team),
                ars_rows, tend_rows, pdf_lineup, pdf_attack, throws_lbl)
            st.download_button("⬇ Download game plan (PDF)", data=pdf_bytes,
                               file_name=f"gameplan_vs_{player_last(gp_pitcher).replace(',','').replace(' ','_')}.pdf",
                               mime="application/pdf", key="gp_download")
        except Exception as _e:
            _pdf_unavailable(_e)
            st.caption("Falling back to a plain-text game plan.")
            st.download_button("⬇ Download game plan (.txt)", data="\n".join(plan_lines),
                               file_name=f"gameplan_vs_{player_last(gp_pitcher).replace(',','').replace(' ','_')}.txt",
                               mime="text/plain", key="gp_download_txt")
        st.caption("Projections are shrinkage-based (lean on league averages when data is thin) — "
                   "directional game-planning tools, not guarantees.")


# ─────────────────────────────────────────
#  PAGE: MATCHUP TOOL
# ─────────────────────────────────────────
elif page == "Matchup Tool":
    st.title("Matchup Tool")
    st.caption("Select a pitcher to see their profile, then add hitters to project matchups.")

    # ── Pitcher selection ──────────────────────────────
    pc_teams = sorted(_team_options(df_all["PitcherTeam"]))
    c1, c2 = st.columns([1.3, 2])
    with c1:
        pc_team = st.selectbox("Pitcher team", options=pc_teams,
                               format_func=team_label, key="pc_team")
    team_pitchers = _player_options_reports(df_all[df_all["PitcherTeam"] == pc_team]["Pitcher"])
    with c2:
        sel_pitcher = st.selectbox("Pitcher", options=[""] + team_pitchers,
                                   format_func=lambda p: player_last(p) if p else "Select…",
                                   key="pc_pitcher")

    if not sel_pitcher:
        st.info("Select a pitcher to begin.")
    else:
        pdf = df_all[df_all["Pitcher"] == sel_pitcher].copy()

        # ── Filters ──────────────────────────────
        f1, f2, f3 = st.columns(3)
        with f1:
            bat_side = st.radio("vs Batter side", ["All", "RHB", "LHB"], horizontal=True, key="pc_side")
        with f2:
            count_filter = st.radio("Count", ["All", "Ahead", "Behind", "Even"], horizontal=True, key="pc_count")
        with f3:
            ptypes_avail = sorted(pdf["PitchType"].dropna().unique().tolist())
            pt_filter = st.multiselect("Pitch types", options=ptypes_avail, default=ptypes_avail, key="pc_pt")

        fdf = pdf.copy()
        if bat_side == "RHB":
            fdf = fdf[fdf["BatterSide"] == "Right"]
        elif bat_side == "LHB":
            fdf = fdf[fdf["BatterSide"] == "Left"]
        if count_filter == "Ahead":
            fdf = fdf[fdf["Strikes"] > fdf["Balls"]]
        elif count_filter == "Behind":
            fdf = fdf[fdf["Balls"] > fdf["Strikes"]]
        elif count_filter == "Even":
            fdf = fdf[fdf["Balls"] == fdf["Strikes"]]
        if pt_filter:
            fdf = fdf[fdf["PitchType"].isin(pt_filter)]

        if len(fdf) == 0:
            st.warning("No pitches match those filters.")
        else:
            # ── Pitcher overview metrics ──────────────────────────────
            throws = fdf["PitcherThrows"].dropna().mode()
            throws_s = throws.iloc[0] if len(throws) else "?"
            n_pitches = len(fdf)
            avg_velo = fdf[fdf["PitchType"].isin(["Four-Seam","Sinker","Fastball"])]["RelSpeed"].mean()
            k = int(fdf["KorBB"].eq("Strikeout").sum())
            bb = int(fdf["KorBB"].eq("Walk").sum())
            strikes = int(fdf["PitchCall"].isin(["StrikeCalled","StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable","InPlay"]).sum())
            strike_pct = round(100*strikes/n_pitches, 1) if n_pitches else 0
            whiffs = int((fdf["PitchCall"] == "StrikeSwinging").sum())
            swings = int(fdf["PitchCall"].isin(["StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable","InPlay"]).sum())
            whiff_pct = round(100*whiffs/swings, 1) if swings else 0

            st.markdown(f"### {player_last(sel_pitcher)} ({throws_s}HP)")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Pitches", n_pitches)
            m2.metric("Avg FB", f"{avg_velo:.1f}" if pd.notna(avg_velo) else "—")
            m3.metric("Strike%", f"{strike_pct}%")
            m4.metric("Whiff%", f"{whiff_pct}%")
            m5.metric("K / BB", f"{k} / {bb}")

            # ── Pitch mix ──────────────────────────────
            st.markdown("#### Pitch Mix")
            mix_rows = []
            for ptype in fdf["PitchType"].dropna().unique():
                sub = fdf[fdf["PitchType"] == ptype]
                mix_rows.append({
                    "Pitch": ptype,
                    "Count": len(sub),
                    "Usage%": f"{100*len(sub)/n_pitches:.1f}%",
                    "Avg Velo": f"{sub['RelSpeed'].mean():.1f}" if sub['RelSpeed'].notna().any() else "—",
                    "Avg Spin": f"{sub['SpinRate'].mean():.0f}" if sub['SpinRate'].notna().any() else "—",
                    "Whiff%": (f"{100*(sub['PitchCall']=='StrikeSwinging').sum()/max(1,sub['PitchCall'].isin(['StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay']).sum()):.1f}%"),
                })
            mix_df = pd.DataFrame(mix_rows).sort_values("Count", ascending=False)
            st.dataframe(mix_df, use_container_width=True, hide_index=True)

            # ── Hitter matchups ──────────────────────────────
            st.divider()
            st.markdown("#### Hitter Matchups")
            st.caption("Add hitters to project how they've performed against these pitch types "
                       "(based on each hitter's season-long numbers vs each pitch type).")

            # Hitter team selector — defaults to Brookhaven (MY_TEAM)
            hit_teams = ([MY_TEAM] if MY_TEAM in _team_options(df_all["BatterTeam"]) else []) + \
                        sorted([t for t in _team_options(df_all["BatterTeam"]) if t != MY_TEAM])
            ht_col, _ = st.columns([1.3, 2])
            with ht_col:
                hit_team = st.selectbox("Hitter team", options=hit_teams,
                                        format_func=team_label, key="pc_hit_team")
            all_hitters = _player_options_reports(df_all[df_all["BatterTeam"] == hit_team]["Batter"])
            sel_hitters = st.multiselect("Add hitters",
                                         options=all_hitters,
                                         format_func=lambda b: player_last(b),
                                         key="pc_hitters")

            if sel_hitters:
                pitcher_ptypes = mix_df["Pitch"].tolist()
                rows = []
                for hitter in sel_hitters:
                    hdf = df_all[df_all["Batter"] == hitter]
                    # Overall vs this pitcher's handedness
                    h_vs = hdf[hdf["PitcherThrows"] == throws_s] if throws_s in ("R","L") else hdf
                    ab_mask = _ab_mask(h_vs)
                    abs_n = int(ab_mask.sum())
                    hits_n = int(h_vs["PlayResult"].isin(["Single","Double","Triple","HomeRun"]).sum())
                    ovr_ba = f"{hits_n/abs_n:.3f}" if abs_n else "—"
                    bip = h_vs[h_vs["ExitSpeed"].notna() & h_vs["Angle"].notna() &
                               h_vs["PlayResult"].isin(["Single","Double","Triple","HomeRun","Out","Error","FieldersChoice"])]
                    ovr_xba = f"{bip.apply(lambda r: calc_xba(r['ExitSpeed'], r['Angle']), axis=1).sum()/abs_n:.3f}" if (abs_n and len(bip)) else "—"

                    row = {"Hitter": player_last(hitter),
                           f"BA vs {throws_s}HP": ovr_ba,
                           "xBA": ovr_xba}
                    # Per pitch type the pitcher throws
                    for ptype in pitcher_ptypes:
                        hp = hdf[hdf["PitchType"] == ptype]
                        ab_p = _ab_mask(hp)
                        a = int(ab_p.sum())
                        h = int(hp["PlayResult"].isin(["Single","Double","Triple","HomeRun"]).sum())
                        row[f"{ptype} BA"] = f"{h/a:.3f}" if a >= 3 else "—"
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption("Per-pitch BA shown when the hitter has ≥3 at-bats ending on that pitch type. "
                           "These are season-long tendencies, not head-to-head history.")

                # ── Projected matchup outcome (model) ──
                st.markdown("#### Projected outcome vs this pitcher")
                st.caption("A model projection — each hitter's expected line against THIS pitcher's "
                           "specific arsenal, regressed for sample size. Blends the hitter's tendency "
                           "vs each pitch type with league baselines, weighted by how often this "
                           "pitcher throws each pitch.")
                baselines = _league_pitch_baselines(df_all)
                proj_rows = []
                for hitter in sel_hitters:
                    pr = project_matchup(hitter, sel_pitcher, df_all, baselines)
                    if pr is None:
                        continue
                    proj_rows.append({
                        "Hitter": player_last(hitter),
                        "Proj xwOBA": f"{pr['xwoba']:.3f}",
                        "Proj Whiff%": f"{100*pr['whiff']:.0f}%",
                        "Proj HardHit%": f"{100*pr['hardhit']:.0f}%",
                        "Data": "thin" if pr["n_seen"] < 20 else "ok",
                        "_x": pr["xwoba"],
                    })
                if proj_rows:
                    pdf2 = pd.DataFrame(proj_rows).sort_values("_x", ascending=False).drop(columns="_x")
                    st.dataframe(pdf2, use_container_width=True, hide_index=True)
                    st.caption("Higher Proj xwOBA = better projected matchup for the hitter. "
                               "'Data: thin' means <20 pitches seen, so the projection leans heavily "
                               "on league baselines — treat as a rough prior. xwOBA-on-contact scale.")
                    st.info("⚠ Projection model on a small-sample league. It regresses noisy hitter "
                            "data toward league norms — directionally useful for lineup/bullpen "
                            "decisions, not a precise forecast.")


# ─────────────────────────────────────────
#  PAGE: ATTACK PLAN (XGBoost matchup engine)
# ─────────────────────────────────────────
elif page == "Attack Plan (Beta)":
    st.title("Attack Plan")
    st.caption("Pick one pitcher and one hitter. Builds a model-driven attack plan for both "
               "sides: what the pitcher should throw — and in what sequence — to get this hitter "
               "out, and how the hitter should approach this specific arsenal.")

    import matchup_model as mm

    ap_p_teams = sorted(_team_options(df_all["PitcherTeam"]))
    c1, c2 = st.columns([1.3, 2])
    with c1:
        ap_p_team = st.selectbox("Pitcher team", options=ap_p_teams,
                                 format_func=team_label, key="ap_p_team")
    ap_team_pitchers = _player_options_reports(df_all[df_all["PitcherTeam"] == ap_p_team]["Pitcher"])
    with c2:
        ap_pitcher = st.selectbox("Pitcher", options=[""] + ap_team_pitchers,
                                  format_func=lambda p: player_last(p) if p else "Select…",
                                  key="ap_pitcher")

    ap_h_teams = ([MY_TEAM] if MY_TEAM in _team_options(df_all["BatterTeam"]) else []) + \
                 sorted([t for t in _team_options(df_all["BatterTeam"]) if t != MY_TEAM])
    c3, c4 = st.columns([1.3, 2])
    with c3:
        ap_h_team = st.selectbox("Hitter team", options=ap_h_teams,
                                 format_func=team_label, key="ap_h_team")
    ap_team_hitters = _player_options_reports(df_all[df_all["BatterTeam"] == ap_h_team]["Batter"])
    with c4:
        ap_hitter = st.selectbox("Hitter", options=[""] + ap_team_hitters,
                                 format_func=lambda b: player_last(b) if b else "Select…",
                                 key="ap_hitter")

    if not ap_pitcher or not ap_hitter:
        st.info("Select a pitcher and a hitter to build an attack plan.")
    else:
        try:
            _ap_hash = (len(df_all), tuple(df_all.columns))
            with st.spinner("Training whiff model…"):
                ap_model, ap_meta = mm.train_whiff_model(_ap_hash, df_all)
        except Exception as e:
            st.error(f"Couldn't train the matchup model: {e}")
            st.stop()

        with st.expander("Model diagnostics"):
            st.caption("Global XGBoost whiff-probability model, trained once on every pitch in the "
                       "dataset and evaluated out-of-fold (GroupKFold grouped by Pitcher, so a "
                       "pitcher's own release fingerprint never leaks between train and test).")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Pitches used", ap_meta["n_pitches"])
            d2.metric("Swings", ap_meta["n_swings"])
            d3.metric("CV AUC", f"{ap_meta['auc']:.3f}")
            d4.metric("Pitchers / Batters", f"{ap_meta['n_pitchers']} / {ap_meta['n_batters']}")
            st.caption("Calibration — predicted vs. actual whiff rate by decile of predicted probability:")
            st.dataframe(ap_meta["calibration"], use_container_width=True, hide_index=True)

        pdf_ap = df_all[df_all["Pitcher"] == ap_pitcher]
        hdf_ap = df_all[df_all["Batter"] == ap_hitter]
        p_throws = pdf_ap["PitcherThrows"].dropna().mode()
        p_throws = p_throws.iloc[0] if len(p_throws) else "Right"
        h_side = hdf_ap["BatterSide"].dropna().mode()
        h_side = h_side.iloc[0] if len(h_side) else "Right"
        same_hand = int(p_throws == h_side)
        h2h = pdf_ap[pdf_ap["Batter"] == ap_hitter]

        p_lbl = {"Right": "RHP", "Left": "LHP"}.get(p_throws, "?")
        h_lbl = {"Right": "RHB", "Left": "LHB"}.get(h_side, "?")
        st.markdown(f"### {player_last(ap_pitcher)} ({p_lbl}) vs {player_last(ap_hitter)} ({h_lbl})")
        if len(h2h) == 0:
            st.caption("No head-to-head pitches on record — this plan is built entirely from each "
                       "player's regressed tendencies against similar arms/hitters, not history "
                       "between these two specifically.")
        else:
            st.caption(f"{len(h2h)} head-to-head pitch(es) on record — still a thin sample; the "
                       "numbers below lean mainly on regressed season tendencies, not this history.")

        arsenal = mm.pitcher_arsenal_profile(pdf_ap, min_n=15)
        if len(arsenal) < 1:
            st.warning("Not enough pitches (15+ of any single type) to profile this pitcher's "
                       "arsenal yet.")
        else:
            tunnel_rows = mm.tunnel_pairs(arsenal) if len(arsenal) >= 2 else []
            league_table = mm.sequence_transition_table(_ap_hash, df_all)
            hitter_table = mm.hitter_transition_table(hdf_ap)
            whiff_feats, lg_default = mm.hitter_whiff_features(hdf_ap, df_all, same_hand)
            chase_feat = mm.hitter_chase_feature(hdf_ap, df_all, same_hand)
            lg_zone_rates = mm.league_zone_swing_rates(_ap_hash, df_all)
            zone_disc = mm.hitter_zone_discipline(hdf_ap, league_zone_rates=lg_zone_rates)

            plan = mm.build_sequence_plan(ap_model, arsenal, whiff_feats, chase_feat, h_side,
                                          tunnel_rows, league_table, hitter_table)
            full_seq = mm.optimize_full_sequence(ap_model, arsenal, whiff_feats, chase_feat, h_side,
                                                 tunnel_rows, league_table, hitter_table)

            # ── Pitcher's Attack Plan ──
            st.divider()
            st.markdown("## Pitcher's Attack Plan")

            if full_seq is None:
                st.caption("Only one pitch type has enough of a sample (15+) to profile — not "
                          "enough distinct pitches to search a full at-bat sequence.")
            else:
                best_seq = full_seq["best"]
                st.markdown(f"### Optimal at-bat sequence: {' → '.join(best_seq['order'])}")
                st.caption(f"Searched all {full_seq['n_orderings_tested']} orderings of his "
                          f"{len(best_seq['order'])}-pitch arsenal" +
                          (f" (dropped {', '.join(full_seq['dropped_pitch_types'])} to keep the "
                           f"search tractable)" if full_seq["dropped_pitch_types"] else "") + ".")

                seq_rows = []
                for i, s in enumerate(best_seq["steps"]):
                    count_lbl = f"0-{s['strikes']}"
                    step_lbl = f"Pitch {i+1} ({count_lbl})" + (" — PAYOFF" if i == len(best_seq["steps"]) - 1 else "")
                    seq_rows.append({
                        "Step": step_lbl, "Pitch": s["type"], "Zone": s["zone"],
                        "Whiff prob": f"{s['whiff_prob']*100:.0f}%",
                        "Tunnel x (vs. prior)": f"{s['tunnel_mult']:.2f}" if i > 0 else "—",
                        "Seq lift (vs. prior)": f"{s['seq_lift']:.2f}" if i > 0 else "—",
                    })
                st.dataframe(pd.DataFrame(seq_rows), use_container_width=True, hide_index=True)

                payoff = best_seq["steps"][-1]
                predecessor = best_seq["steps"][-2]["type"]
                st.markdown(f"**Why this order:** **{payoff['type']}** is *{player_last(ap_hitter)}'s* "
                          f"own biggest weakness in this arsenal ({whiff_feats.get(payoff['type'], lg_default)*100:.0f}% "
                          f"regressed whiff rate — highest of anything {player_last(ap_pitcher)} throws). "
                          f"Setting up with {', '.join(best_seq['order'][:-1])} before coming back with "
                          f"it in the {payoff['zone']} zone gets a {payoff['whiff_prob']*100:.0f}% model "
                          f"whiff probability, boosted {payoff['tunnel_mult']:.2f}x by how well it tunnels "
                          f"off {predecessor} and {payoff['seq_lift']:.2f}x by hitters' own history "
                          f"swinging at {payoff['type']} right after seeing {predecessor} "
                          f"(n={payoff['seq_lift_n']}).")
                st.caption("The payoff pitch is anchored to this hitter's own highest regressed whiff "
                          "rate in the arsenal (same number as \"Lay off\" below) — tunneling and "
                          "sequence-transition data pick the best SETUP into it, not which pitch to "
                          "throw last, since tunneling is a pitcher-only property that would otherwise "
                          "swamp real hitter-to-hitter differences. Only the final transition (payoff "
                          "vs. its immediate predecessor) is scored against real sequence data — "
                          "earlier pitches are chosen for their own disguise value step-to-step "
                          f"(avg tunnel {best_seq['avg_chain_tunnel']:.2f}x across the whole chain), "
                          "not chained multiple pitches deep, since there isn't enough data here to "
                          "support claims about effects more than one pitch back.")

                with st.expander("Other strong finishing options (2-pitch view)"):
                    cand_df = pd.DataFrame([{
                        "Pitch": c["type"], "Zone": c["zone"],
                        "Whiff prob": f"{c['whiff_prob']*100:.0f}%",
                        "Tunnel x": f"{c['tunnel_mult']:.2f}",
                        "Seq lift": f"{c['seq_lift']:.2f}",
                        "Score": f"{c['combined_score']:.3f}",
                    } for c in plan["putaway_candidates"]])
                    st.dataframe(cand_df, use_container_width=True, hide_index=True)

            if tunnel_rows:
                best_tunnel = max(tunnel_rows, key=lambda r: r["score"])
                st.caption(f"Best tunnel pair in his arsenal: **{best_tunnel['a']} + {best_tunnel['b']}** "
                          f"({best_tunnel['grade']} — release gap {best_tunnel['release_gap']:.1f}\", "
                          f"movement separation {best_tunnel['move_sep']:.1f}\").")

            ars_rows = []
            for pt, prof in sorted(arsenal.items(), key=lambda kv: -kv[1]["usage"]):
                ars_rows.append({
                    "Pitch": pt, "Usage%": f"{prof['usage']*100:.0f}%",
                    "Velo": f"{prof['velo']:.1f}" if pd.notna(prof['velo']) else "—",
                    f"{player_last(ap_hitter)}'s whiff rate (regressed)":
                        f"{whiff_feats.get(pt, lg_default)*100:.0f}%",
                    "n (this pitcher)": prof["n"],
                })
            st.dataframe(pd.DataFrame(ars_rows), use_container_width=True, hide_index=True)

            # ── Hitter's Attack Plan ──
            st.divider()
            st.markdown("## Hitter's Attack Plan")
            hit_rows = []
            for pt, prof in sorted(arsenal.items(), key=lambda kv: -kv[1]["usage"]):
                sub = hdf_ap[hdf_ap["PitchType"] == pt]
                ab = int((_ab_mask(sub)).sum())
                h = int(sub["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum())
                ba = f"{h/ab:.3f}" if ab >= 3 else "—"
                hit_rows.append({
                    "Pitch": pt, "His usage": f"{prof['usage']*100:.0f}%",
                    "Your BA": ba, "AB": ab,
                    "Your whiff rate (regressed)": f"{whiff_feats.get(pt, lg_default)*100:.0f}%",
                })
            st.dataframe(pd.DataFrame(hit_rows), use_container_width=True, hide_index=True)

            best_pt = min(arsenal, key=lambda pt: whiff_feats.get(pt, lg_default))
            worst_pt = max(arsenal, key=lambda pt: whiff_feats.get(pt, lg_default))
            st.markdown(f"**Sit on:** {best_pt} — your lowest regressed whiff rate against this "
                       f"arsenal ({whiff_feats.get(best_pt, lg_default)*100:.0f}%).")
            if worst_pt != best_pt:
                st.markdown(f"**Lay off / be ready early:** {worst_pt} — your highest regressed "
                           f"whiff rate ({whiff_feats.get(worst_pt, lg_default)*100:.0f}%).")

            st.markdown("**Zone discipline**")
            st.caption("Take/swing/whiff/hard-hit rate by zone bucket, with the league's same-hand "
                       "swing rate in that zone for comparison.")
            zd_rows = []
            for zone in ["Heart", "Shadow", "Chase", "Waste"]:
                z = zone_disc.get(zone)
                if not z:
                    continue
                zd_rows.append({
                    "Zone": zone, "n": z["n"],
                    "Your Swing%": f"{z['swing_rate']*100:.0f}%",
                    "League Swing%": f"{z['league_swing_rate']*100:.0f}%" if z["league_swing_rate"] is not None else "—",
                    "Whiff% (of swings)": f"{z['whiff_rate']*100:.0f}%" if z["whiff_rate"] is not None else "—",
                    "Hard-hit%": f"{z['hard_hit_rate']*100:.0f}%" if z["hard_hit_rate"] is not None else "—",
                })
            if zd_rows:
                st.dataframe(pd.DataFrame(zd_rows), use_container_width=True, hide_index=True)
            chase_zone = zone_disc.get("Chase")
            if chase_zone and chase_zone["league_swing_rate"] is not None and \
               chase_zone["swing_rate"] > chase_zone["league_swing_rate"] * 1.15:
                st.caption(f"⚠ Chases the Chase zone more than league average "
                          f"({chase_zone['swing_rate']*100:.0f}% vs "
                          f"{chase_zone['league_swing_rate']*100:.0f}%) — if "
                          f"{player_last(ap_pitcher)}'s putaway pitch lands there, the danger is real.")

        st.info("⚠ Directional guidance from a global model personalized with regressed, "
               "sample-size-aware features — not a fit per pitcher-hitter matchup (head-to-head "
               "history between one pitcher and one hitter is almost always too thin to model "
               "directly). Sequencing recommendations blend tunneling, real pitch-order transition "
               "data, and this hitter's zone tendencies — treat as a prior to sharpen scouting "
               "judgment, not a guarantee.")


# ─────────────────────────────────────────
#  PAGE: PITCHER SCOUTING
# ─────────────────────────────────────────
elif page == "Pitcher Scouting" or page == "Pitcher vs Team":
    # Pitcher vs Team reuses this entire page body (identical scouting report)
    # scoped down to one opponent, plus its own "who he faced" section at the
    # very end — key_prefix keeps their widget keys from colliding if a coach
    # looks at the same pitcher on both pages in one session.
    pvt_mode = (page == "Pitcher vs Team")
    key_prefix = "pvt" if pvt_mode else "ps"

    _PVT_WHOLE_SEASON = "__WHOLE_SEASON__"

    if not pvt_mode and st.session_state.get("rb_return_flag"):
        def _back_to_big_board():
            st.session_state["nav_cat"] = "Front Office"
            st.session_state["nav_page"] = "Returner Board"
            st.session_state.pop("rb_return_flag", None)
        st.button("← Back to Big Board", key="rb_back_btn", on_click=_back_to_big_board)

    if pvt_mode:
        st.title("Pitcher vs Team")
        st.caption("A pitcher's full scouting report, scoped to what he actually threw against "
                   "one opponent — plus how each of their hitters has done against him. Hasn't "
                   "faced the team yet? Pick \"Whole Season\" to scout him off everything he's "
                   "thrown all year, split by RHH/LHH instead.")

        all_pitchers_pvt = sorted(_player_options(df_all["Pitcher"]))
        col1, col2 = st.columns([1.6, 1.4])
        with col1:
            pitcher = st.selectbox("Pitcher",
                options=[""] + all_pitchers_pvt,
                format_func=lambda x: x if x else "Select pitcher...",
                key="pvt_pitcher")

        opp_teams_pvt = []
        if pitcher:
            opp_teams_pvt = [_PVT_WHOLE_SEASON] + sorted(_team_options(df_all[df_all["Pitcher"] == pitcher]["BatterTeam"]))
        with col2:
            selected_team = st.selectbox("Opponent Team",
                options=[""] + opp_teams_pvt,
                format_func=lambda t: ("Whole Season (vs RHH / vs LHH)" if t == _PVT_WHOLE_SEASON
                                       else team_label(t) if t else "Select team..."),
                key="pvt_team")

        st.divider()

        if not pitcher or not selected_team:
            st.info("Select a pitcher and either the opponent team they faced, or Whole Season, "
                    "to see the matchup.")
            st.stop()

        if selected_team == _PVT_WHOLE_SEASON:
            pp = df_all[df_all["Pitcher"] == pitcher].copy()
            st.info(f"Showing {player_last(pitcher)}'s whole season, not scoped to one opponent — "
                    "use the \"Pitch mix vs\" / hand filters below and the Splits tab to break it "
                    "out by RHH and LHH.")
        else:
            pp = df_all[(df_all["Pitcher"] == pitcher) & (df_all["BatterTeam"] == selected_team)].copy()
            if pp.empty:
                st.warning(f"{player_last(pitcher)} has no recorded pitches against "
                          f"{team_label(selected_team)}. Pick \"Whole Season\" instead to scout "
                          "him off his full-year numbers.")
                st.stop()
    else:
        st.title("Pitcher Scouting")

        all_teams  = sorted(_team_options(df_all["PitcherTeam"]))
        # Put NAS_SIL first, then opponents
        sorted_teams = ([MY_TEAM] if MY_TEAM in all_teams else []) +                        sorted([t for t in all_teams if t != MY_TEAM])
        team_labels_list = [team_label(t) for t in sorted_teams]

        # A Returner Board deep-link can hand us a team/pitcher this run's
        # data doesn't have -- drop it rather than let the selectboxes below
        # crash on a value outside their own options.
        if st.session_state.get("ps_team") not in range(len(sorted_teams)):
            st.session_state.pop("ps_team", None)

        col1, col2 = st.columns([1.4, 1.8])
        with col1:
            team_idx = st.selectbox("Team",
                options=range(len(sorted_teams)),
                format_func=lambda i: team_labels_list[i],
                key="ps_team")
            selected_team = sorted_teams[team_idx] if sorted_teams else None

        opp_pitchers = []
        if selected_team:
            pitcher_info = (
                df_all[df_all["PitcherTeam"] == selected_team]
                .groupby(["Pitcher","PitcherThrows"])
                .size().reset_index(name="Pitches")
                .sort_values("Pitches", ascending=False)
            )
            all_team_pitchers = pitcher_info["Pitcher"].tolist()
            opp_pitchers = [p for p in all_team_pitchers if not _is_removed(p)]
            # A Returner Board deep-link can point at a pitcher who's since
            # left the active roster (REMOVED_FROM_ROSTER) -- e.g. Collins,
            # currently the board's #1 ranked arm, is hidden from every
            # picker in the app for lineup-building purposes. That filter
            # doesn't apply here: a coach clicking through from the returner
            # board wants that exact pitcher's page regardless of his
            # current roster status, so let the deep-linked target back in.
            pending_p = st.session_state.get("ps_pitcher")
            if pending_p in all_team_pitchers and pending_p not in opp_pitchers:
                opp_pitchers = [pending_p] + opp_pitchers

        if st.session_state.get("ps_pitcher") not in ([""] + opp_pitchers):
            st.session_state.pop("ps_pitcher", None)

        with col2:
            pitcher = st.selectbox("Pitcher",
                options=[""] + opp_pitchers,
                format_func=lambda x: x if x else "Select pitcher...",
                key="ps_pitcher")

        st.divider()

        if not pitcher:
            st.info("Select an opponent team and pitcher to view their tendencies.")
            st.stop()

        pp = df_all[df_all["Pitcher"] == pitcher].copy()

    hand = pp["PitcherThrows"].iloc[0] if len(pp) > 0 else "?"
    total_pitches = len(pp)

    # Header info
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Pitcher", player_last(pitcher))
    h2.metric("Throws", hand)
    h3.metric("Total Pitches", total_pitches)
    games_faced = pp["GameID"].nunique() if "GameID" in pp.columns else "?"
    h4.metric("Games", games_faced)

    # ── Stuff+ by pitch type — computed once at data load, shown up top ──
    pp_pt = pp[pp["PitchType"].notna() & (pp["PitchType"] != "None")]
    pt_counts = pp_pt.groupby("PitchType").size().reset_index(name="Pitches")
    pitcher_stuff = (stuff_plus_df[stuff_plus_df["Pitcher"] == pitcher]
                      if not stuff_plus_df.empty else
                      pd.DataFrame(columns=["PitchType", "StuffPlus"]))
    stuff_view = pt_counts.merge(pitcher_stuff[["PitchType", "StuffPlus"]],
                                  on="PitchType", how="left") \
                           .sort_values("Pitches", ascending=False).reset_index(drop=True)

    if len(stuff_view) > 0:
        st.markdown("##### Stuff+ by Pitch")
        st.caption("100 = league-average whiff rate for that pitch shape. Higher = nastier stuff.")
        stuff_cols = st.columns(len(stuff_view))
        for i, row in stuff_view.iterrows():
            pt = row["PitchType"]
            sp = row["StuffPlus"]
            color = PITCH_COLORS.get(pt, "#64748b")
            with stuff_cols[i]:
                if pd.notna(sp):
                    s_col = "#22c55e" if sp >= 110 else "#f59e0b" if sp >= 95 else "#ef4444"
                    value_html = f"<div style='font-size:1.8rem;font-weight:800;color:{s_col};'>{sp:.0f}</div>"
                else:
                    value_html = "<div style='font-size:1.1rem;color:#64748b;margin-top:8px;'>—</div>"
                st.markdown(
                    "<div style='background:#F8FAFC;border:1.5px solid " + color + ";border-radius:8px;"
                    "padding:10px;text-align:center;'>"
                    "<div style='font-size:0.72rem;color:" + color + ";font-weight:700;"
                    "text-transform:uppercase;letter-spacing:0.05em;'>" + str(pt) + "</div>"
                    + value_html +
                    "<div style='font-size:0.68rem;color:#64748b;'>Stuff+</div>"
                    "</div>", unsafe_allow_html=True)

    st.divider()

    scout_tab, splits_tab, inning_tab, metrics_tab, tto_tab, fatigue_tab = st.tabs(["Scouting", "Splits", "By Inning of Work", "Metrics (FIP / xFIP)", "Times Through Order", "Pitch Count Wall"])

    with scout_tab:

        # ── PITCH MIX OVERALL ──
        mix_hand_filter = st.radio("Pitch mix vs",
            ["All", "RHH", "LHH"],
            horizontal=True,
            key=f"{key_prefix}_mix_hand_{pitcher}")
        st.markdown("### Overall Pitch Mix")
        # Filter to valid normalized pitch types + optional hand filter
        pp_clean_base = pp[pp["PitchType"].notna() & (pp["PitchType"] != "None")].copy()
        if mix_hand_filter == "RHH":
            pp_clean = pp_clean_base[pp_clean_base["BatterSide"] == "Right"]
        elif mix_hand_filter == "LHH":
            pp_clean = pp_clean_base[pp_clean_base["BatterSide"] == "Left"]
        else:
            pp_clean = pp_clean_base
        if len(pp_clean) == 0:
            pp_clean = pp_clean_base
        agg_dict = {"Count": ("PitchType","count"), "AvgVelo": ("RelSpeed","mean")}
        if "SpinRate"         in pp_clean.columns: agg_dict["AvgSpin"] = ("SpinRate","mean")
        if "InducedVertBreak" in pp_clean.columns: agg_dict["AvgIVB"]  = ("InducedVertBreak","mean")
        if "HorzBreak"        in pp_clean.columns: agg_dict["AvgHB"]   = ("HorzBreak","mean")
        overall_mix = pp_clean.groupby("PitchType").agg(**agg_dict).reset_index()
        _mix_total = len(pp_clean)
        overall_mix["Pct"] = (overall_mix["Count"] / max(_mix_total, 1) * 100).round(1)
        overall_mix["AvgVelo"] = overall_mix["AvgVelo"].round(1)
        if "AvgSpin" in overall_mix.columns: overall_mix["AvgSpin"] = overall_mix["AvgSpin"].round(0).astype("Int64")
        if "AvgIVB"  in overall_mix.columns: overall_mix["AvgIVB"]  = overall_mix["AvgIVB"].round(1)
        if "AvgHB"   in overall_mix.columns: overall_mix["AvgHB"]   = overall_mix["AvgHB"].round(1)
        overall_mix = overall_mix.sort_values("Count", ascending=False).reset_index(drop=True)

        # League-relative spin classification per pitch type (Above / Average / Below).
        # Compares this pitcher's avg spin for a pitch to the FCBL average for that
        # same pitch type, using that pitch type's spread (std) so the label is fair.
        if "AvgSpin" in overall_mix.columns:
            _lg_stats = df_all[df_all["SpinRate"].notna()].groupby("PitchType")["SpinRate"].agg(["mean", "std", "count"])
            def _spin_tag(row):
                pt = row["PitchType"]
                spin = row["AvgSpin"]
                if pd.isna(spin) or pt not in _lg_stats.index:
                    return "—"
                lg_mean = _lg_stats.loc[pt, "mean"]
                lg_std  = _lg_stats.loc[pt, "std"]
                if pd.isna(lg_std) or lg_std == 0 or _lg_stats.loc[pt, "count"] < 20:
                    return "—"
                z = (spin - lg_mean) / lg_std
                if z >= 0.5:
                    return "Above avg"
                if z <= -0.5:
                    return "Below avg"
                return "Average"
            overall_mix["Spin"] = overall_mix.apply(
                lambda r: (str(int(r["AvgSpin"])) + " rpm (" + _spin_tag(r) + ")")
                if pd.notna(r["AvgSpin"]) else "—", axis=1)

        # Pitch mix bar chart
        mix_cols = st.columns(len(overall_mix))
        # PITCH_COLORS defined at module level
        for i, row in overall_mix.iterrows():
            pitch_type = row["PitchType"]
            pct_val    = row["Pct"]
            count_val  = row["Count"]
            velo_val   = row["AvgVelo"]
            color = PITCH_COLORS.get(pitch_type, "#64748b")
            with mix_cols[i]:
                ivb_val = row.get("AvgIVB") if "AvgIVB" in row.index else None
                hb_val  = row.get("AvgHB")  if "AvgHB"  in row.index else None
                mov_line = ""
                if ivb_val is not None and hb_val is not None and pd.notna(ivb_val) and pd.notna(hb_val):
                    ivb_s = ("+" if ivb_val >= 0 else "") + str(round(float(ivb_val), 1))
                    hb_s  = ("+" if hb_val  >= 0 else "") + str(round(float(hb_val),  1))
                    mov_line = (
                        "<div style='font-size:0.72rem;color:#64748b;margin-top:2px;'>"
                        + ivb_s + "in / " + hb_s + "in</div>"
                    )
                spin_line = ""
                if "Spin" in row.index and isinstance(row.get("Spin"), str) and row["Spin"] != "—":
                    _spin_txt = row["Spin"]
                    _spin_color = ("#16a34a" if "Above" in _spin_txt else
                                   "#dc2626" if "Below" in _spin_txt else "#64748b")
                    spin_line = ("<div style='font-size:0.72rem;color:" + _spin_color +
                                 ";margin-top:2px;'>" + _spin_txt + "</div>")
                st.markdown(
                    "<div style='background:#F8FAFC;border:1.5px solid " + color + ";border-radius:8px;"
                    "padding:10px;text-align:center;'>"
                    "<div style='font-size:0.75rem;color:" + color + ";font-weight:700;"
                    "text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;'>"
                    + str(pitch_type) + "</div>"
                    "<div style='font-size:1.6rem;font-weight:800;color:#1e293b;'>" + str(pct_val) + "%</div>"
                    "<div style='font-size:0.75rem;color:#64748b;margin-top:2px;'>" + str(count_val) + " pitches</div>"
                    "<div style='font-size:0.75rem;color:#475569;margin-top:2px;'>" + str(velo_val) + " mph</div>"
                    + spin_line
                    + mov_line +
                    "</div>",
                    unsafe_allow_html=True)

        # ── Attack-zone location profile (Heart / Shadow / Chase / Waste) ──
        st.markdown("#### Where He Locates — Attack Zones")
        st.caption("Heart = middle (hittable) · Shadow = edges · Chase = just off · Waste = well off. "
                   "Compared to league average for all pitchers.")
        _pz = pp_clean.copy()
        _pz["PlateLocSide"] = pd.to_numeric(_pz["PlateLocSide"], errors="coerce")
        _pz["PlateLocHeight"] = pd.to_numeric(_pz["PlateLocHeight"], errors="coerce")
        _pz["_z"] = _pz.apply(lambda r: attack_zone(r["PlateLocSide"], r["PlateLocHeight"]), axis=1)
        _pz_valid = _pz.dropna(subset=["_z"])
        if len(_pz_valid) >= 10:
            _lg = _league_attack_zone_rates(df_all, "pitch")
            _pv = (_pz_valid["_z"].value_counts(normalize=True) * 100)
            az_rows = []
            for z in _AZ_ORDER:
                his = float(_pv.get(z, 0.0))
                lg = _lg[z]
                diff = his - lg
                tag = ("▲ " + f"{diff:+.0f} pts") if abs(diff) >= 3 else "≈ league"
                az_rows.append({"Zone": z, "His %": f"{his:.0f}%",
                                "League %": f"{lg:.0f}%", "vs League": tag})
            st.dataframe(pd.DataFrame(az_rows), use_container_width=True, hide_index=True)
            # quick read
            _heart = float(_pv.get("Heart", 0.0)); _lgh = _lg["Heart"]
            if _heart >= _lgh + 4:
                st.caption("⚠ Lives in the **Heart** more than average — more hittable pitches.")
            elif _heart <= _lgh - 4:
                st.caption("✓ Stays out of the **Heart** — lives on the edges / expands.")
        else:
            st.caption("Not enough located pitches for a zone profile.")

        # All individual counts for the full grid (used below for the matrix)
        ALL_COUNTS = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)]
        COUNT_LABELS = [f"{b}-{s}" for b,s in ALL_COUNTS]

        pitch_types = [p for p in overall_mix["PitchType"].tolist() if p not in ("Undefined","Other")]

        st.markdown("### Pitch Usage by Count")
        _render_count_group_mix(pp_clean)

        # ── FULL COUNT MATRIX ──
        hand_label = "" if mix_hand_filter == "All" else f" — vs {mix_hand_filter}"
        st.markdown(f"### Full Count Matrix{hand_label}")
        st.caption("Pitch usage % at every count — darker = higher usage")

        # Apply same hand filter as pitch mix
        pp_matrix = pp_clean_base.copy()
        if mix_hand_filter == "RHH":
            pp_matrix = pp_matrix[pp_matrix["BatterSide"] == "Right"]
        elif mix_hand_filter == "LHH":
            pp_matrix = pp_matrix[pp_matrix["BatterSide"] == "Left"]
        if len(pp_matrix) == 0:
            pp_matrix = pp_clean_base

        # Build matrix: rows = pitch type, cols = count
        matrix_data = {}
        count_totals = {}
        for b, s in ALL_COUNTS:
            cnt_pitches = pp_matrix[(pp_matrix["Balls"]==b) & (pp_matrix["Strikes"]==s)]
            count_totals[(b,s)] = len(cnt_pitches)
            for ptype in pitch_types:
                n_pt = (cnt_pitches["PitchType"] == ptype).sum()
                pct  = n_pt / len(cnt_pitches) if len(cnt_pitches) > 0 else 0
                if ptype not in matrix_data:
                    matrix_data[ptype] = {}
                matrix_data[ptype][(b,s)] = pct

        if matrix_data:
            # Build HTML table
            header = "<tr><th style='background:#F1F5F9;padding:6px 10px;text-align:left;color:#64748b;font-size:0.75rem;'>Pitch</th>"
            for b, s in ALL_COUNTS:
                n = count_totals[(b,s)]
                header += f"<th style='background:#F1F5F9;padding:6px 8px;text-align:center;color:#64748b;font-size:0.75rem;'>{b}-{s}<br><span style='font-size:0.65rem;color:#475569;'>n={n}</span></th>"
            header += "</tr>"

            rows_html = ""
            for ptype in pitch_types:
                color = PITCH_COLORS.get(ptype, "#64748b")
                row_html = f"<tr><td style='padding:6px 10px;color:{color};font-weight:600;font-size:0.82rem;white-space:nowrap;'>{ptype}</td>"
                for b, s in ALL_COUNTS:
                    pct = matrix_data[ptype].get((b,s), 0)
                    n   = count_totals[(b,s)]
                    if n == 0:
                        cell_bg = "#F1F5F9"
                        text    = "—"
                        tc      = "#475569"
                    else:
                        alpha = min(pct * 1.5, 1.0)
                        # Parse color to rgba
                        hex_c = PITCH_COLORS.get(ptype, "#64748b").lstrip("#")
                        r,g,b_c = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
                        cell_bg = f"rgba({r},{g},{b_c},{alpha:.2f})"
                        text    = f"{pct:.0%}" if pct > 0 else "—"
                        tc      = "#ffffff" if alpha > 0.4 else "#1e293b"
                    row_html += (f"<td style='padding:6px 8px;text-align:center;background:{cell_bg};"
                                 f"font-size:0.8rem;font-weight:700;color:{tc};'>{text}</td>")
                row_html += "</tr>"
                rows_html += row_html

            table_html = f"""
                <div style='overflow-x:auto;'>
                <table style='border-collapse:collapse;width:100%;background:#FFFFFF;
                border-radius:8px;overflow:hidden;'>
                    <thead>{header}</thead>
                    <tbody>{rows_html}</tbody>
                </table></div>"""
            st.markdown(table_html, unsafe_allow_html=True)

        st.divider()

        # ── ZONE TENDENCIES BY COUNT GROUP ──
        st.markdown("### Location Tendencies")
        st.caption("Where does he live in key situations? — from catcher view")

        loc_count = st.selectbox("Select count group",
            options=["All Pitches"] + list(COUNT_GROUPS.keys()),
            key=f"{key_prefix}_loc_count")

        if loc_count == "All Pitches":
            loc_pitches = pp[pp["PlateLocSide"].notna() & pp["PlateLocHeight"].notna()]
        else:
            loc_pitches = pp[pp.apply(
                lambda r: (r["Balls"], r["Strikes"]) in COUNT_GROUPS[loc_count], axis=1
            ) & pp["PlateLocSide"].notna() & pp["PlateLocHeight"].notna()]

        if len(loc_pitches) < 3:
            st.info(f"Not enough pitches in {loc_count} to show location data (need 3+).")
        else:
            side_edges   = [-2.0, -0.28, 0.28, 2.0]
            height_edges = [1.0, 1.83, 2.67, 3.5]

            # Build zone counts
            zones = {}
            for row in range(3):
                for col in range(3):
                    h_lo = height_edges[2-row]; h_hi = height_edges[3-row]
                    s_lo = side_edges[col];     s_hi = side_edges[col+1]
                    mask = (loc_pitches["PlateLocHeight"].between(h_lo,h_hi) &
                            loc_pitches["PlateLocSide"].between(s_lo,s_hi))
                    zp   = loc_pitches[mask]
                    n    = len(zp)
                    pct  = n / len(loc_pitches)
                    # Most used pitch in this zone
                    top_pitch = zp["PitchType"].value_counts().index[0] if n > 0 else None
                    zones[(row,col)] = dict(n=n, pct=pct, top_pitch=top_pitch)

            max_pct = max(z["pct"] for z in zones.values()) if zones else 1

            # Filters — pitch type and batter hand
            f1, f2 = st.columns(2)
            with f1:
                pitch_types_ph = ["All"] + sorted(loc_pitches["PitchType"].dropna().unique().tolist())
                sel_pt_ph = st.selectbox("Pitch type", pitch_types_ph,
                                         key=f"{key_prefix}_ph_pt_{pitcher}_{loc_count}")
            with f2:
                sel_hand_ph = st.selectbox("Batter hand", ["All", "RHH", "LHH"],
                                           key=f"{key_prefix}_ph_hand_{pitcher}_{loc_count}")

            ph_data = loc_pitches.copy()
            if sel_pt_ph != "All":
                ph_data = ph_data[ph_data["PitchType"] == sel_pt_ph]
            if sel_hand_ph == "RHH":
                ph_data = ph_data[ph_data["BatterSide"] == "Right"]
            elif sel_hand_ph == "LHH":
                ph_data = ph_data[ph_data["BatterSide"] == "Left"]

            # Three maps side by side
            map_col1, map_col2, map_col3 = st.columns(3)

            with map_col1:
                st.markdown(
                    "<div style='font-size:0.85rem;font-weight:700;color:#475569;"
                    "margin-bottom:4px;'>Pitch Location</div>"
                    "<div style='font-size:0.72rem;color:#64748b;margin-bottom:6px;'>"
                    "Where he throws — red/white = most frequent</div>",
                    unsafe_allow_html=True)
                _render_kde_heatmap(ph_data, weight_col=None,
                                    key_suffix=f"{key_prefix}_ph_freq_{pitcher}_{sel_pt_ph}_{sel_hand_ph}",
                                    title="")

            with map_col2:
                ph_bip = ph_data[ph_data["ExitSpeed"].notna() &
                                  ph_data["PlateLocSide"].notna() &
                                  ph_data["PlateLocHeight"].notna()]
                st.markdown(
                    "<div style='font-size:0.85rem;font-weight:700;color:#475569;"
                    "margin-bottom:4px;'>Hard Contact Zones</div>"
                    "<div style='font-size:0.72rem;color:#64748b;margin-bottom:6px;'>"
                    "Where hitters do the most damage — red/white = hard contact</div>",
                    unsafe_allow_html=True)
                if len(ph_bip) >= 5:
                    _render_kde_heatmap(ph_bip, weight_col="ExitSpeed",
                                        key_suffix=f"{key_prefix}_ph_ev_{pitcher}_{sel_pt_ph}_{sel_hand_ph}",
                                        title="")
                else:
                    st.info("Not enough contact data.")

            with map_col3:
                SWING_C_PH = {"StrikeSwinging","InPlay","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}
                ph_swings = ph_data[ph_data["PitchCall"].isin(SWING_C_PH) &
                                    ph_data["PlateLocSide"].notna() &
                                    ph_data["PlateLocHeight"].notna()].copy()
                ph_swings["_whiff_w"] = ph_swings["PitchCall"].eq("StrikeSwinging").astype(float)
                st.markdown(
                    "<div style='font-size:0.85rem;font-weight:700;color:#475569;"
                    "margin-bottom:4px;'>Whiff Zones</div>"
                    "<div style='font-size:0.72rem;color:#64748b;margin-bottom:6px;'>"
                    "Where he generates whiffs — red/white = most whiffs</div>",
                    unsafe_allow_html=True)
                if len(ph_swings) >= 5:
                    _render_kde_heatmap(ph_swings, weight_col="_whiff_w",
                                        key_suffix=f"{key_prefix}_ph_wh_{pitcher}_{sel_pt_ph}_{sel_hand_ph}",
                                        title="")
                else:
                    st.info("Not enough swing data.")

        st.divider()

        # ── SEQUENCING NOTES ──
        st.markdown("### Sequencing Notes")
        st.caption("What he tends to throw right after a given pitch, broken out by the count "
                   "situation that pitch was thrown in. Flags combinations that lean hard away "
                   "from his overall mix — the patterns a hitter can actually sit on.")

        _seq_ab_cols = [c for c in ["GameID", "Inning", "Top/Bottom", "PAofInning"] if c in pp_clean_base.columns]
        if "PitchofPA" not in pp_clean_base.columns or not _seq_ab_cols:
            st.caption("Not enough at-bat sequencing data on file to build this.")
        else:
            pp_seq = pp_clean_base.copy()
            pp_seq["Balls"] = pd.to_numeric(pp_seq["Balls"], errors="coerce")
            pp_seq["Strikes"] = pd.to_numeric(pp_seq["Strikes"], errors="coerce")
            pp_seq["PitchofPA"] = pd.to_numeric(pp_seq["PitchofPA"], errors="coerce")
            pp_seq = pp_seq.dropna(subset=["Balls", "Strikes", "PitchofPA"])
            pp_seq = pp_seq.sort_values(_seq_ab_cols + ["PitchofPA"])
            pp_seq["_ab"] = pp_seq[_seq_ab_cols].astype(str).agg("_".join, axis=1)

            pp_seq["PrevPitchType"] = pp_seq.groupby("_ab")["PitchType"].shift(1)
            _prev_b = pp_seq.groupby("_ab")["Balls"].shift(1).fillna(-1).astype(int).to_numpy()
            _prev_s = pp_seq.groupby("_ab")["Strikes"].shift(1).fillna(-1).astype(int).to_numpy()

            def _pvt_seq_bucket(balls, strikes):
                bucket = np.full(balls.shape, "Even", dtype=object)
                first = (balls == 0) & (strikes == 0)
                two_strike = strikes == 2
                bucket[two_strike] = "Two strikes"
                rem = ~two_strike & ~first
                bucket[rem & (strikes > balls)] = "Ahead"
                bucket[rem & (balls > strikes)] = "Behind"
                bucket[first] = "First pitch"
                return bucket

            pp_seq["PrevBucket"] = _pvt_seq_bucket(_prev_b, _prev_s)

            seq_valid = pp_seq.dropna(subset=["PrevPitchType"])
            seq_valid = seq_valid[seq_valid["PitchType"].isin(pitch_types) &
                                  seq_valid["PrevPitchType"].isin(pitch_types)]

            if len(seq_valid) < 15:
                st.caption("Not enough sequenced pitches to find reliable patterns yet.")
            else:
                _overall_pct = {row["PitchType"]: row["Pct"] for _, row in overall_mix.iterrows()}

                # Same-pitch-back-to-back rate vs. what his own mix would predict at random
                repeat_rate = (seq_valid["PitchType"] == seq_valid["PrevPitchType"]).mean()
                expected_repeat = sum((p / 100) ** 2 for p in _overall_pct.values())
                repeat_diff = (repeat_rate - expected_repeat) * 100
                if abs(repeat_diff) >= 8:
                    _tag = "more" if repeat_diff > 0 else "less"
                    st.info(f"Throws the same pitch back-to-back **{repeat_rate:.0%}** of the time — "
                            f"{abs(repeat_diff):.0f} pts {_tag} than his overall mix would predict at "
                            f"random (**{expected_repeat:.0%}**).")

                _BUCKET_LBL = {"First pitch": "a first pitch", "Ahead": "a pitch when ahead in the count",
                               "Behind": "a pitch when behind in the count", "Even": "a pitch on an even count",
                               "Two strikes": "a pitch with two strikes"}

                trans = (seq_valid.groupby(["PrevBucket", "PrevPitchType", "PitchType"])
                         .size().reset_index(name="n"))
                trans["grp_n"] = trans.groupby(["PrevBucket", "PrevPitchType"])["n"].transform("sum")
                trans["pct"] = trans["n"] / trans["grp_n"] * 100

                notes = []
                for (bucket, prev_pt), g in trans.groupby(["PrevBucket", "PrevPitchType"]):
                    if g["grp_n"].iloc[0] < 8:
                        continue
                    top = g.sort_values("pct", ascending=False).iloc[0]
                    base = _overall_pct.get(top["PitchType"], 0)
                    diff = top["pct"] - base
                    if top["pct"] >= 35 and diff >= 15:
                        notes.append({"bucket": bucket, "prev": prev_pt, "next": top["PitchType"],
                                     "pct": top["pct"], "base": base, "diff": diff,
                                     "n": int(g["grp_n"].iloc[0])})
                notes.sort(key=lambda r: -r["diff"])

                if notes:
                    for note in notes[:8]:
                        st.markdown(
                            f"- After **{note['prev']}** on {_BUCKET_LBL.get(note['bucket'], note['bucket'])}, "
                            f"he goes to **{note['next']}** {note['pct']:.0f}% of the next pitch "
                            f"(n={note['n']}, vs {note['base']:.0f}% overall usage — "
                            f"**+{note['diff']:.0f} pts**).")
                else:
                    st.caption("No strong pitch-to-pitch tendencies stood out (samples too thin or "
                              "the mix is fairly random).")

                with st.expander("Full pitch-to-pitch transition table"):
                    _disp = trans[trans["grp_n"] >= 5].copy()
                    _disp["Next %"] = _disp["pct"].round(0).astype(int).astype(str) + "%"
                    _disp = _disp.rename(columns={"PrevBucket": "Count", "PrevPitchType": "Previous Pitch",
                                                  "PitchType": "Next Pitch", "grp_n": "Sample (n)"})
                    st.dataframe(
                        _disp[["Count", "Previous Pitch", "Next Pitch", "Next %", "Sample (n)"]]
                        .sort_values(["Count", "Previous Pitch", "Next %"], ascending=[True, True, False]),
                        use_container_width=True, hide_index=True)


    # ─────────────────────────────────────────
    #  PAGE: xBA REPORT
    # ─────────────────────────────────────────

    with splits_tab:
        st.markdown("### Pitcher Splits — vs RHH / vs LHH")
        st.caption("Stat line, pitch mix, and whiff rates broken down by batter handedness.")
        st.divider()

        _STRIKE_C = {"StrikeSwinging","StrikeCalled","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}
        _SWING_C  = {"StrikeSwinging","InPlay","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}
        _gc       = [c for c in ["GameID","Inning","Top/Bottom","PAofInning"] if c in pp.columns]

        def batter_expected_stats_against(sub):
            """xBA against — based on balls in play allowed."""
            bip = sub[sub["ExitSpeed"].notna() & sub["Angle"].notna() &
                      (sub["Distance"].fillna(0) >= 10) &
                      (sub["Direction"].fillna(999).abs() <= 45)]
            if len(bip) < 3:
                return None, None, None, 0
            xba_sum = sum(calc_xba(r["ExitSpeed"], r["Angle"]) or 0
                          for _, r in bip.iterrows())
            xba = xba_sum / len(bip) if len(bip) > 0 else None
            return xba, None, None, len(bip)

        def _split_stats(sub):
            if len(sub) == 0:
                return None
            last_s = sub.groupby(_gc).last().reset_index() if _gc else sub.copy()
            k    = (last_s["KorBB"].eq("Strikeout") |
                    ((last_s["PitchCall"].isin(["StrikeSwinging","StrikeCalled"])) &
                     (last_s["Strikes"] == 2))).sum()
            bb   = last_s["KorBB"].eq("Walk").sum()
            hbp  = last_s["PitchCall"].eq("HitByPitch").sum()
            h    = last_s["PlayResult"].isin(["Single","Double","Triple","HomeRun"]).sum()
            hr   = last_s["PlayResult"].eq("HomeRun").sum()
            # At-bats = completed PAs that count as official at-bats: hits, outs in
            # play, fielder's choice, reached-on-error, plus strikeouts. Walks, HBP,
            # and sacrifices are NOT at-bats.
            ab   = int(int(k) + last_s["PlayResult"].isin(
                       ["Single","Double","Triple","HomeRun",
                        "Out","FieldersChoice","Error"]).sum())
            outs = sub["OutsOnPlay"].fillna(0).sum() + k
            ip   = to_ip(outs)
            sw   = sub["PitchCall"].isin(_SWING_C).sum()
            wh   = sub["PitchCall"].eq("StrikeSwinging").sum()
            st_c = sub["PitchCall"].isin(_STRIKE_C).sum()
            fps_df = sub[sub["PitchofPA"]==1] if "PitchofPA" in sub.columns else pd.DataFrame()
            fps  = fps_df["PitchCall"].isin(_STRIKE_C).sum() / max(len(fps_df),1)
            # xBA against — use batter_expected_stats on balls in play
            xba_val, _, _, _ = batter_expected_stats_against(sub)
            return {
                "Pitches": len(sub), "IP": ip, "AB": ab, "K": int(k), "BB": int(bb),
                "H": int(h), "HR": int(hr),
                "Strike%": f"{st_c/len(sub):.0%}" if len(sub) > 0 else "—",
                "Whiff%":  f"{wh/len(sub):.0%}"  if len(sub) > 0 else "—",
                "SwStr%":  f"{wh/sw:.0%}"         if sw > 0 else "—",
                "FPS%":    f"{fps:.0%}",
                "xBA":     f"{xba_val:.3f}" if xba_val is not None else "—",
            }

        rhh = pp[pp["BatterSide"]=="Right"]
        lhh = pp[pp["BatterSide"]=="Left"]
        overall_s = _split_stats(pp)
        rhh_s     = _split_stats(rhh)
        lhh_s     = _split_stats(lhh)

        # Stat line table
        st.markdown("#### Stat Line")
        stat_keys = ["Pitches","IP","AB","K","BB","H","HR","Strike%","Whiff%","SwStr%","FPS%","xBA"]
        split_rows = []
        for label, stats in [("Overall", overall_s), ("vs RHH", rhh_s), ("vs LHH", lhh_s)]:
            if stats:
                row = {"Split": label}
                row.update({k: stats[k] for k in stat_keys})
                split_rows.append(row)
        if split_rows:
            st.dataframe(pd.DataFrame(split_rows), use_container_width=True,
                         hide_index=True)

        st.divider()

        # Pitch mix side by side
        st.markdown("#### Pitch Mix by Batter Side")
        all_pts = [pt for pt in pp["PitchType"].dropna().value_counts().index]
        if all_pts:
            mix_rows = []
            for pt in all_pts:
                ov_n  = (pp["PitchType"]==pt).sum()
                rh_n  = (rhh["PitchType"]==pt).sum()
                lh_n  = (lhh["PitchType"]==pt).sum()
                rh_sw = rhh[rhh["PitchType"]==pt]["PitchCall"].isin(_SWING_C).sum()
                rh_wh = rhh[rhh["PitchType"]==pt]["PitchCall"].eq("StrikeSwinging").sum()
                lh_sw = lhh[lhh["PitchType"]==pt]["PitchCall"].isin(_SWING_C).sum()
                lh_wh = lhh[lhh["PitchType"]==pt]["PitchCall"].eq("StrikeSwinging").sum()

                mix_rows.append({
                    "Pitch":      pt,
                    "Overall%":   f"{ov_n/len(pp):.0%}" if len(pp)>0 else "—",
                    "vs RHH%":    f"{rh_n/len(rhh):.0%}" if len(rhh)>0 else "—",
                    "RHH Whiff%": f"{rh_wh/rh_sw:.0%}" if rh_sw>0 else "—",
                    "vs LHH%":    f"{lh_n/len(lhh):.0%}" if len(lhh)>0 else "—",
                    "LHH Whiff%": f"{lh_wh/lh_sw:.0%}" if lh_sw>0 else "—",
                })
            mix_df = pd.DataFrame(mix_rows)
            st.dataframe(mix_df, use_container_width=True, hide_index=True)

        st.divider()

        # Visual pitch mix comparison — side by side bars
        import plotly.graph_objects as go_sp
        if all_pts and len(rhh) > 0 and len(lhh) > 0:
            st.markdown("#### Pitch Mix Comparison — RHH vs LHH")
            rhh_pct = [(rhh["PitchType"]==pt).sum()/len(rhh)*100 for pt in all_pts]
            lhh_pct = [(lhh["PitchType"]==pt).sum()/len(lhh)*100 for pt in all_pts]
            fig_sp = go_sp.Figure()
            fig_sp.add_trace(go_sp.Bar(
                name="vs RHH", x=all_pts, y=rhh_pct,
                marker_color="#3b82f6", opacity=0.85,
                text=[f"{v:.0f}%" for v in rhh_pct], textposition="outside",
                textfont=dict(size=10, color="#1e293b"),
            ))
            fig_sp.add_trace(go_sp.Bar(
                name="vs LHH", x=all_pts, y=lhh_pct,
                marker_color="#f59e0b", opacity=0.85,
                text=[f"{v:.0f}%" for v in lhh_pct], textposition="outside",
                textfont=dict(size=10, color="#1e293b"),
            ))
            fig_sp.update_layout(
                height=360, barmode="group",
                plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF",
                font=dict(color="#1e293b"),
                xaxis=dict(gridcolor="#E2E8F0"),
                yaxis=dict(title="Usage %", gridcolor="#E2E8F0", zeroline=False),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b")),
                margin=dict(l=40, r=40, t=20, b=40)
            )
            st.plotly_chart(fig_sp, use_container_width=True)

    with inning_tab:
        st.markdown("### Performance By Inning of Outing")
        st.caption("Grouped by which inning of THIS appearance a pitch was thrown in — his 1st "
                   "inning of work that game, his 2nd, etc. — not the game's actual inning number. "
                   "A reliever who enters in the 6th and goes two innings shows up under '1' and "
                   "'2', same as a starter's first two innings.")

        if not pitcher:
            st.info("Select a pitcher to view inning-of-outing stats.")
        else:
            pp_inn = pp.copy()
            pp_inn["Inning"] = pd.to_numeric(pp_inn["Inning"], errors="coerce")
            pp_inn = pp_inn.dropna(subset=["Inning"]).sort_values(
                ["GameID", "Inning", "PAofInning", "PitchofPA"])
            pp_inn["_inn_key"] = pp_inn["GameID"].astype(str) + "_" + pp_inn["Inning"].astype(str)
            pp_inn["_stint_inning"] = pp_inn.groupby("GameID")["_inn_key"].transform(
                lambda x: pd.factorize(x)[0] + 1)

            def _stint_label(n):
                return str(int(n)) if n < 4 else "4+"
            pp_inn["StintInning"] = pp_inn["_stint_inning"].apply(_stint_label)

            SWING_C_INN  = {"StrikeSwinging","InPlay","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}
            STRIKE_C_INN = {"StrikeSwinging","StrikeCalled","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}
            _gc_inn = [c for c in ["GameID","Inning","Top/Bottom","PAofInning"] if c in pp_inn.columns]

            inn_rows = []
            for lbl in ["1", "2", "3", "4+"]:
                grp = pp_inn[pp_inn["StintInning"] == lbl]
                if len(grp) < 3:
                    continue
                n_games = grp["GameID"].nunique()
                last_g = grp.groupby(_gc_inn).last().reset_index() if _gc_inn else grp.copy()
                k   = (last_g["KorBB"].eq("Strikeout") |
                       ((last_g["PitchCall"].isin(["StrikeSwinging","StrikeCalled"])) &
                        (last_g["Strikes"] == 2))).sum()
                h   = last_g["PlayResult"].isin(["Single","Double","Triple","HomeRun"]).sum()
                hr  = last_g["PlayResult"].eq("HomeRun").sum()
                ab  = int(int(k) + last_g["PlayResult"].isin(
                          ["Single","Double","Triple","HomeRun","Out","FieldersChoice","Error"]).sum())
                ba  = h / ab if ab > 0 else None
                sw   = grp["PitchCall"].isin(SWING_C_INN).sum()
                wh   = grp["PitchCall"].eq("StrikeSwinging").sum()
                st_c = grp["PitchCall"].isin(STRIKE_C_INN).sum()
                bip  = grp[grp["ExitSpeed"].notna() & (grp["Distance"].fillna(0) >= 10) &
                          (grp["Direction"].fillna(999).abs() <= 45)]
                avg_ev  = bip["ExitSpeed"].mean() if len(bip) >= 2 else None
                fb_mask = grp["PitchType"].isin({"Four-Seam","Sinker","Cutter"})
                fb_velo = grp[fb_mask]["RelSpeed"].mean() if fb_mask.sum() >= 3 else None
                inn_rows.append({
                    "Inning of Outing": lbl,
                    "Appearances": n_games,
                    "Pitches": len(grp),
                    "BF": len(last_g),
                    "FB Velo": f"{fb_velo:.1f}" if fb_velo is not None else "—",
                    "Strike%": f"{st_c/len(grp):.0%}",
                    "Whiff%": f"{wh/sw:.0%}" if sw > 0 else "—",
                    "BA-Against": f"{ba:.3f}" if ba is not None else "—",
                    "HR": int(hr),
                    "Avg EV": f"{avg_ev:.1f}" if avg_ev is not None else "—",
                    "Hard%": f"{(bip['ExitSpeed'] >= 90).mean():.0%}" if len(bip) >= 2 else "—",
                    "_fbvelo_num": fb_velo,
                })

            if inn_rows:
                inn_df = pd.DataFrame(inn_rows)
                st.dataframe(inn_df.drop(columns="_fbvelo_num"), use_container_width=True, hide_index=True)

                velo_rows = [r for r in inn_rows if r["_fbvelo_num"] is not None]
                if len(velo_rows) >= 2:
                    import plotly.graph_objects as _go_inn
                    fig_inn = _go_inn.Figure(_go_inn.Scatter(
                        x=[r["Inning of Outing"] for r in velo_rows],
                        y=[r["_fbvelo_num"] for r in velo_rows],
                        mode="lines+markers",
                        line=dict(color="#3b82f6", width=2),
                        marker=dict(size=10, color="#3b82f6"),
                    ))
                    fig_inn.update_layout(
                        height=220, plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF",
                        font=dict(color="#1e293b"),
                        xaxis=dict(title="Inning of outing", gridcolor="#E2E8F0", type="category"),
                        yaxis=dict(title="Avg FB Velo (mph)", gridcolor="#E2E8F0"),
                    )
                    st.plotly_chart(fig_inn, use_container_width=True)
                    st.caption("Fastball velocity (Four-Seam/Sinker/Cutter) by inning of the outing. "
                               "A steady drop is a fatigue signal independent of which game-inning he entered.")
            else:
                st.info("Not enough data for an inning-of-outing breakdown.")

    with metrics_tab:
        st.markdown("### Pitching Metrics — FIP & xFIP")
        st.caption("FIP and xFIP measure pitcher performance on outcomes they control: strikeouts, walks, HBP, home runs. Lower is better. League avg ~3.10.")

        FIP_CONSTANT   = 3.10   # FCBL approximate
        LEAGUE_HR_FB   = 0.08   # college ball HR/FB rate
        MIN_IP_DISPLAY = 1.0    # show all, but warn below 3 IP

        def calc_pitcher_metrics(pitcher_df):
            """Compute FIP, xFIP, and supporting stats for a pitcher's pitches."""
            # Strikeouts: KorBB == Strikeout (most reliable)
            # Fallback: PitchCall == StrikeCalled/StrikeSwinging on 2-strike count
            k = pitcher_df["KorBB"].eq("Strikeout").sum()
            if k == 0:
                # fallback — final pitch of a K PA
                k_sw = pitcher_df[
                    (pitcher_df["PitchCall"].isin(["StrikeSwinging","StrikeCalled"])) &
                    (pitcher_df["Strikes"] == 2)
                ].shape[0]
                k = k_sw

            bb  = pitcher_df["KorBB"].eq("Walk").sum()
            hbp = pitcher_df["PitchCall"].eq("HitByPitch").sum()
            hr  = pitcher_df["PlayResult"].eq("HomeRun").sum()

            # Outs: OutsOnPlay + strikeouts
            outs = pitcher_df["OutsOnPlay"].fillna(0).sum() + k
            ip   = outs / 3

            # Fly balls: prefer TaggedHitType, fallback AutoHitType
            fb_tagged = pitcher_df["TaggedHitType"].eq("FlyBall").sum()
            fb_auto   = pitcher_df["AutoHitType"].eq("FlyBall").sum() if "AutoHitType" in pitcher_df.columns else 0
            fb = max(fb_tagged, fb_auto)

            # Hit types
            gb = pitcher_df["TaggedHitType"].eq("GroundBall").sum()
            if gb == 0 and "AutoHitType" in pitcher_df.columns:
                gb = pitcher_df["AutoHitType"].eq("GroundBall").sum()
            ld = pitcher_df["TaggedHitType"].eq("LineDrive").sum()
            if ld == 0 and "AutoHitType" in pitcher_df.columns:
                ld = pitcher_df["AutoHitType"].eq("LineDrive").sum()

            # Batters faced
            bf = pitcher_df[pitcher_df["PitchofPA"] == 1].shape[0] if "PitchofPA" in pitcher_df.columns else 0

            # First Pitch Strike %
            fp_df = pitcher_df[pitcher_df["PitchofPA"] == 1] if "PitchofPA" in pitcher_df.columns else pd.DataFrame()
            _fps_strikes = {"StrikeSwinging","StrikeCalled","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}
            fps = (fp_df["PitchCall"].isin(_fps_strikes).sum() / len(fp_df)) if len(fp_df) > 0 else None

            # FIP
            fip  = (13*hr + 3*(bb+hbp) - 2*k) / ip + FIP_CONSTANT if ip >= 2.0 else None

            # xFIP — replace HR with expected HR from FB rate
            xhr  = fb * LEAGUE_HR_FB
            xfip = (13*xhr + 3*(bb+hbp) - 2*k) / ip + FIP_CONSTANT if ip >= 2.0 else None

            # K/9, BB/9
            k9   = k  / ip * 9 if ip >= 2.0 else None
            bb9  = bb / ip * 9 if ip >= 2.0 else None
            kbb  = k / bb if bb > 0 else None

            # GB%
            balls_in_play = gb + fb + ld
            gb_pct = gb / balls_in_play if balls_in_play > 0 else None
            fb_pct = fb / balls_in_play if balls_in_play > 0 else None

            # Stuff: avg velo + spin
            avg_velo = pitcher_df["RelSpeed"].mean() if "RelSpeed" in pitcher_df.columns else None
            avg_spin = pitcher_df["SpinRate"].mean() if "SpinRate" in pitcher_df.columns else None

            return dict(
                IP=to_ip(outs), BF=bf, K=int(k), BB=int(bb), HBP=int(hbp),
                HR=int(hr), FB=int(fb), GB=int(gb),
                FIP=round(fip, 2) if fip is not None else None,
                xFIP=round(xfip, 2) if xfip is not None else None,
                K9=round(k9, 1) if k9 is not None else None,
                BB9=round(bb9, 1) if bb9 is not None else None,
                KBB=round(kbb, 2) if kbb is not None else None,
                GB_pct=round(gb_pct, 3) if gb_pct is not None else None,
                FB_pct=round(fb_pct, 3) if fb_pct is not None else None,
                AvgVelo=round(avg_velo, 1) if avg_velo is not None else None,
                AvgSpin=round(avg_spin, 0) if avg_spin is not None else None,
                FPS=round(fps, 3) if fps is not None else None,
            )

        def render_pitcher_table(pitcher_group_df, team_label_str):
            rows = []
            for pitcher, grp in pitcher_group_df.groupby("Pitcher"):
                if _is_removed(pitcher):
                    continue
                m = calc_pitcher_metrics(grp)
                hand = grp["PitcherThrows"].iloc[0]
                team = grp["PitcherTeam"].iloc[0]
                rows.append({"Pitcher": pitcher, "Team": team_label(team),
                             "Throws": hand, **m})

            if not rows:
                st.info("No pitcher data found.")
                return

            df = pd.DataFrame(rows)
            df["_ip_num"] = df["IP"].apply(lambda x: int(str(x).split(".")[0]) + int(str(x).split(".")[1])/3 if "." in str(x) else 0)
            df = df.sort_values("_ip_num", ascending=False).drop(columns=["_ip_num"]).reset_index(drop=True)
            df.index += 1

            # Warn if low sample
            low_ip = (df["IP"].apply(lambda x: int(str(x).split(".")[0]) if pd.notna(x) and x != "—" else 0) < 3).sum()
            if low_ip > 0:
                st.warning(
                    f"{low_ip} pitcher(s) have fewer than 3 IP — FIP/xFIP will stabilize "
                    f"significantly with more games. Numbers shown are directionally correct "
                    f"but not yet reliable for comparison."
                )

            # Summary metrics
            valid = df[df["FIP"].notna()]  # FIP already None if IP < 2.0
            if len(valid) > 0:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Pitchers", len(df))
                def _ip_to_outs(ip_str):
                    try:
                        parts = str(ip_str).split(".")
                        return int(parts[0])*3 + int(parts[1]) if len(parts)==2 else 0
                    except: return 0
                total_outs_sum = df["IP"].apply(_ip_to_outs).sum()
                total_ip_disp  = f"{total_outs_sum//3}.{total_outs_sum%3}"
                m2.metric("Total IP", total_ip_disp)
                m3.metric("Avg FIP",  f"{valid['FIP'].mean():.2f}")
                m4.metric("Avg xFIP", f"{valid['xFIP'].mean():.2f}" if valid['xFIP'].notna().any() else "—")
                m5.metric("Total K",  int(df["K"].sum()))
                st.divider()

            # FIP vs xFIP bar chart
            if len(valid) >= 2:
                import plotly.graph_objects as go_fip
                fig = go_fip.Figure()

                pitchers_sorted = valid.sort_values("xFIP")
                names = pitchers_sorted["Pitcher"].map(player_last)

                fig.add_trace(go_fip.Bar(
                    name="FIP", x=names, y=pitchers_sorted["FIP"],
                    marker_color="#3b82f6", opacity=0.85,
                    text=pitchers_sorted["FIP"].map(lambda v: f"{v:.2f}"),
                    textposition="outside", textfont=dict(size=11)
                ))
                fig.add_trace(go_fip.Bar(
                    name="xFIP", x=names, y=pitchers_sorted["xFIP"],
                    marker_color="#22c55e", opacity=0.85,
                    text=pitchers_sorted["xFIP"].map(lambda v: f"{v:.2f}" if pd.notna(v) else ""),
                    textposition="outside", textfont=dict(size=11)
                ))

                # League average line
                fig.add_hline(y=FIP_CONSTANT, line_dash="dash",
                              line_color="rgba(15,23,42,0.35)",
                              annotation_text="League Avg",
                              annotation_font_color="rgba(15,23,42,0.5)")

                # Cap y-axis to prevent extreme outliers squishing the chart
                fip_max = pitchers_sorted[["FIP","xFIP"]].max().max()
                y_max   = min(fip_max * 1.2, 12.0)
                fig.update_layout(
                    height=360, barmode="group",
                    plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF",
                    font=dict(color="#1e293b"),
                    yaxis=dict(title="FIP / xFIP", gridcolor="#E2E8F0", zeroline=False,
                               range=[0, y_max]),
                    xaxis=dict(gridcolor="#E2E8F0"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b")),
                    margin=dict(l=40, r=60, t=30, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Full stats table
            st.markdown("#### Full Pitching Stats")
            fmt = lambda v, f: f.format(v) if pd.notna(v) and v is not None else "—"

            td = df[["Pitcher","Team","Throws","IP","BF","K","BB","HBP","HR",
                     "FIP","xFIP","K9","BB9","GB_pct","FB_pct","AvgVelo","FPS"]].copy()
            td["FIP"]     = td["FIP"].map(lambda v: fmt(v, "{:.2f}"))
            td["xFIP"]    = td["xFIP"].map(lambda v: fmt(v, "{:.2f}"))
            td["K9"]      = td["K9"].map(lambda v: fmt(v, "{:.1f}"))
            td["BB9"]     = td["BB9"].map(lambda v: fmt(v, "{:.1f}"))
            td["GB_pct"]  = td["GB_pct"].map(lambda v: fmt(v*100, "{:.0f}%") if pd.notna(v) and v is not None else "—")
            td["FB_pct"]  = td["FB_pct"].map(lambda v: fmt(v*100, "{:.0f}%") if pd.notna(v) and v is not None else "—")
            td["AvgVelo"] = td["AvgVelo"].map(lambda v: fmt(v, "{:.1f}"))
            td["FPS"]     = td["FPS"].map(lambda v: fmt(v*100, "{:.0f}%") if pd.notna(v) and v is not None else "—")
            td.columns    = ["Pitcher","Team","T","IP","BF","K","BB","HBP","HR",
                             "FIP","xFIP","K/9","BB/9","GB%","FB%","Velo","FPS%"]

            # Insight callouts
            valid2 = df[df["FIP"].notna() & (df["IP"].apply(lambda x: float(str(x).replace('.','',1) if str(x).count('.')==1 else x) if pd.notna(x) else 0) >= 2.0)]
            if len(valid2) >= 2:
                best  = valid2.loc[valid2["xFIP"].idxmin()]
                worst = valid2.loc[valid2["xFIP"].idxmax()]
                c1, c2 = st.columns(2)
                fip_diff = best["FIP"] - best["xFIP"] if pd.notna(best["xFIP"]) else 0
                with c1:
                    luck_text = ("Getting unlucky — FIP higher than xFIP suggests HR suppression"
                                 if fip_diff > 0.30
                                 else "Performing in line with contact quality")
                    st.markdown(f"""
                        <div style='background:#F8FAFC;border:1.5px solid #22c55e;
                        border-radius:8px;padding:1rem;'>
                            <div style='font-size:0.75rem;color:#22c55e;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;'>
                            Best xFIP</div>
                            <div style='font-size:1.1rem;font-weight:700;color:#1e293b;'>{player_last(best['Pitcher'])}</div>
                            <div style='font-size:0.85rem;color:#475569;'>
                            FIP {best['FIP']:.2f} · xFIP {best['xFIP']:.2f} · {best['IP']} IP</div>
                            <div style='font-size:0.8rem;color:#64748b;margin-top:4px;'>{luck_text}</div>
                        </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                        <div style='background:#F8FAFC;border:1.5px solid #ef4444;
                        border-radius:8px;padding:1rem;'>
                            <div style='font-size:0.75rem;color:#ef4444;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;'>
                            Highest xFIP</div>
                            <div style='font-size:1.1rem;font-weight:700;color:#1e293b;'>{player_last(worst['Pitcher'])}</div>
                            <div style='font-size:0.85rem;color:#475569;'>
                            FIP {worst['FIP']:.2f} · xFIP {worst['xFIP']:.2f} · {worst['IP']} IP</div>
                            <div style='font-size:0.8rem;color:#64748b;margin-top:4px;'>
                            Giving up hard contact — monitor closely.</div>
                        </div>""", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["NAS_SIL — Our Pitchers", "League — Opponent Pitchers"])

        met_tab1, met_tab2 = st.tabs(["Brookhaven Bandits", "Opponents"])

        with met_tab1:
            render_pitcher_table(df_all[df_all["PitcherTeam"] == MY_TEAM], "NAS_SIL")

        with met_tab2:
            opp_teams_p = sorted([t for t in _team_options(df_all["PitcherTeam"]) if t != MY_TEAM])
            opp_filter  = st.selectbox("Filter by team",
                options=["All Opponents"] + [team_label(t) for t in opp_teams_p],
                key=f"{key_prefix}_fip_opp_team")
            if opp_filter == "All Opponents":
                opp_pitchers_df = df_all[df_all["PitcherTeam"] != MY_TEAM]
            else:
                code = next((c for c,l in TEAM_LABELS.items() if l == opp_filter), opp_filter)
                opp_pitchers_df = df_all[df_all["PitcherTeam"] == code]
            render_pitcher_table(opp_pitchers_df, opp_filter)


    # ── TIMES THROUGH ORDER TAB ───────────────────────────────────────────
    with tto_tab:
        st.markdown("### Performance By Times Through Order")
        st.caption("How does this pitcher perform the 1st, 2nd, and 3rd time through the lineup?")

        if not pitcher:
            st.info("Select a pitcher to view times through order stats.")
        else:
            pp_tto = pp.copy().sort_values(["GameID","Inning","PAofInning","PitchofPA"])
            pp_tto["_pa_key"] = pp_tto["GameID"].astype(str) + "_" + pp_tto["Inning"].astype(str) + "_" + pp_tto["PAofInning"].astype(str)
            pp_tto["_pa_num"] = pp_tto.groupby("GameID")["_pa_key"].transform(lambda x: pd.factorize(x)[0] + 1)

            def tto_label(n):
                if n <= 9:    return "1st TTO"
                elif n <= 18: return "2nd TTO"
                else:         return "3rd TTO"
            pp_tto["TTO"] = pp_tto["_pa_num"].apply(tto_label)

            SWING_C_TTO = {"StrikeSwinging","InPlay","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}
            tto_rows = []
            for tto in ["1st TTO", "2nd TTO", "3rd TTO"]:
                grp = pp_tto[pp_tto["TTO"] == tto]
                if len(grp) < 3: continue
                sw   = grp["PitchCall"].isin(SWING_C_TTO).sum()
                wh   = grp["PitchCall"].eq("StrikeSwinging").sum()
                st_c = grp["PitchCall"].isin({"StrikeSwinging","StrikeCalled","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}).sum()
                bip  = grp[grp["ExitSpeed"].notna() & (grp["Distance"].fillna(0) >= 10) & (grp["Direction"].fillna(999).abs() <= 45)]
                avg_ev = bip["ExitSpeed"].mean() if len(bip) >= 2 else None
                tto_rows.append({
                    "TTO":      tto,
                    "Pitches":  len(grp),
                    "FB Velo":  f"{grp[grp['PitchType'].isin({'Four-Seam','Sinker','Cutter'})]['RelSpeed'].mean():.1f}" if grp["PitchType"].isin({"Four-Seam","Sinker","Cutter"}).sum() >= 3 else "—",
                    "Strike%":  f"{st_c/len(grp):.0%}",
                    "Whiff%":   f"{wh/sw:.0%}" if sw > 0 else "—",
                    "Avg EV":   f"{avg_ev:.1f}" if avg_ev else "—",
                    "Hard%":    f"{(bip['ExitSpeed'] >= 90).mean():.0%}" if len(bip) >= 2 else "—",
                })

            if tto_rows:
                st.dataframe(pd.DataFrame(tto_rows), use_container_width=True, hide_index=True)
                import plotly.graph_objects as _go_tto
                velos = [float(r["FB Velo"]) for r in tto_rows if r["FB Velo"] != "—"]
                velo_x = [r["TTO"] for r in tto_rows if r["FB Velo"] != "—"]
                fig_tto = _go_tto.Figure(_go_tto.Scatter(
                    x=velo_x, y=velos, mode="lines+markers",
                    line=dict(color="#3b82f6", width=2),
                    marker=dict(size=10, color="#3b82f6"),
                ))
                fig_tto.update_layout(
                    height=220, plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF",
                    font=dict(color="#1e293b"),
                    xaxis=dict(gridcolor="#E2E8F0"),
                    yaxis=dict(title="Avg FB Velo (mph)", gridcolor="#E2E8F0",
                               range=[min(velos)-2, max(velos)+2] if velos else [80, 95]),
                )
                st.plotly_chart(fig_tto, use_container_width=True)
            else:
                st.info("Not enough data for times through order breakdown.")

    # ── PITCH COUNT WALL TAB ──────────────────────────────────────────────
    with fatigue_tab:
        st.markdown("### Pitch Count Effectiveness")
        st.caption("Velocity, command, and contact quality by pitch count ranges — look for where performance drops.")

        if not pitcher:
            st.info("Select a pitcher to view pitch count breakdown.")
        else:
            pp_fat = pp.copy().sort_values(["GameID","Inning","PAofInning","PitchofPA"])
            pp_fat["_pitch_num"] = pp_fat.groupby("GameID").cumcount() + 1

            bins_f  = [0, 15, 30, 45, 60, 75, 90, 200]
            labels_f = ["1-15","16-30","31-45","46-60","61-75","76-90","91+"]
            pp_fat["PitchBin"] = pd.cut(pp_fat["_pitch_num"], bins=bins_f, labels=labels_f)

            SWING_C_FAT = {"StrikeSwinging","InPlay","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}
            fat_rows = []
            for bl in labels_f:
                grp = pp_fat[pp_fat["PitchBin"] == bl]
                if len(grp) < 3: continue
                sw   = grp["PitchCall"].isin(SWING_C_FAT).sum()
                wh   = grp["PitchCall"].eq("StrikeSwinging").sum()
                st_c = grp["PitchCall"].isin({"StrikeSwinging","StrikeCalled","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}).sum()
                bip  = grp[grp["ExitSpeed"].notna() & (grp["Distance"].fillna(0) >= 10) & (grp["Direction"].fillna(999).abs() <= 45)]
                avg_ev = bip["ExitSpeed"].mean() if len(bip) >= 2 else None
                in_zone = grp[grp["PlateLocSide"].notna() & grp["PlateLocHeight"].notna()]
                zone_pct = (
                    (in_zone["PlateLocSide"].abs() <= 0.83) &
                    (in_zone["PlateLocHeight"].between(1.5, 3.5))
                ).mean() if len(in_zone) > 0 else None
                fat_rows.append({
                    "Pitches":  bl,
                    "Count":    len(grp),
                    "FB Velo":  round(grp[grp["PitchType"].isin({"Four-Seam","Sinker","Cutter"})]["RelSpeed"].mean(), 1) if grp["PitchType"].isin({"Four-Seam","Sinker","Cutter"}).sum() >= 3 else None,
                    "Zone%":    f"{zone_pct:.0%}" if zone_pct is not None else "—",
                    "Strike%":  f"{st_c/len(grp):.0%}",
                    "Whiff%":   f"{wh/sw:.0%}" if sw > 0 else "—",
                    "Avg EV":   f"{avg_ev:.1f}" if avg_ev else "—",
                    "Hard%":    f"{(bip['ExitSpeed'] >= 90).mean():.0%}" if len(bip) >= 2 else "—",
                })

            if fat_rows:
                st.dataframe(pd.DataFrame(fat_rows), use_container_width=True, hide_index=True)

                import plotly.graph_objects as _go_fat
                fig_fat = _go_fat.Figure()
                velos_f = [r["FB Velo"] for r in fat_rows if r["FB Velo"] is not None]
                x_f     = [r["Pitches"] for r in fat_rows if r["FB Velo"] is not None]
                fig_fat.add_trace(_go_fat.Scatter(
                    x=x_f, y=velos_f, mode="lines+markers", name="Avg FB Velo",
                    line=dict(color="#3b82f6", width=2), marker=dict(size=8),
                ))
                ev_x = [r["Pitches"] for r in fat_rows if r["Avg EV"] != "—"]
                ev_y = [float(r["Avg EV"]) for r in fat_rows if r["Avg EV"] != "—"]
                if ev_y:
                    fig_fat.add_trace(_go_fat.Scatter(
                        x=ev_x, y=ev_y, mode="lines+markers", name="Avg EV allowed",
                        yaxis="y2", line=dict(color="#ef4444", width=2, dash="dot"),
                        marker=dict(size=8, color="#ef4444"),
                    ))
                fig_fat.update_layout(
                    height=300, plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF",
                    font=dict(color="#1e293b"),
                    xaxis=dict(title="Pitch #", gridcolor="#E2E8F0"),
                    yaxis=dict(title="Velo (mph)", gridcolor="#E2E8F0", color="#3b82f6"),
                    yaxis2=dict(title="Avg EV (mph)", overlaying="y", side="right",
                                gridcolor="rgba(0,0,0,0)", color="#ef4444"),
                    legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#E2E8F0", borderwidth=1),
                    margin=dict(l=50, r=60, t=20, b=40),
                )
                st.plotly_chart(fig_fat, use_container_width=True)

                # Auto-detect fatigue wall
                if len(fat_rows) >= 3:
                    peak = max((r["FB Velo"] for r in fat_rows if r["FB Velo"] is not None), default=None)
                    for r in fat_rows:
                        if peak and r["FB Velo"] is not None and r["FB Velo"] < peak - 1.0:
                            drop = peak - r["FB Velo"]
                            st.warning(f"⚠️ Velocity drops **{drop:.1f} mph** starting around pitch **{r['Pitches']}** — potential fatigue wall.")
                            break
            else:
                st.info("Not enough data for pitch count breakdown (need pitchers with 15+ pitches per game).")

    # ── Advanced Pitch Arsenal: movement vs league + velo range ──
    st.divider()
    st.markdown("### Advanced Pitch Arsenal")
    st.caption("Left: each pitch's movement (dots = up to 25 of his pitches nearest his average) "
               "vs the league average for that pitch (dashed circle). Right: velocity range — "
               "the dot is his average, the bar extends to his max.")
    _mv = df_all[df_all["Pitcher"] == pitcher].copy()
    for _c in ["HorzBreak", "InducedVertBreak", "RelSpeed"]:
        _mv[_c] = pd.to_numeric(_mv[_c], errors="coerce")
    _mv = _mv[_mv["HorzBreak"].notna() & _mv["InducedVertBreak"].notna()]
    if len(_mv) < 10:
        st.info("Not enough movement data for the arsenal visual.")
    else:
        # League baselines per pitch type — SAME HAND ONLY. A RHP's slider and a
        # LHP's slider mirror horizontally (e.g. -3.8 in vs +3.9 in), so pooling
        # both hands produces a league circle near zero break that is wrong for
        # everyone. Compare each pitcher to same-handed league pitchers.
        _hand = _mv["PitcherThrows"].dropna().iloc[0] if _mv["PitcherThrows"].notna().any() else None
        _lgm = df_all[df_all["HorzBreak"].notna() & df_all["InducedVertBreak"].notna()]
        if _hand in ("Right", "Left"):
            _lgm = _lgm[_lgm["PitcherThrows"] == _hand]
        _hand_lbl = {"Right": "RHP", "Left": "LHP"}.get(_hand, "league")
        _lg_stats = _lgm.groupby("PitchType").agg(
            hb=("HorzBreak", "mean"), ivb=("InducedVertBreak", "mean"),
            sd_hb=("HorzBreak", "std"), sd_ivb=("InducedVertBreak", "std"),
            n=("PitchUID", "count"))

        col_mv, col_velo = st.columns([1.1, 1])

        with col_mv:
            fig_mv = go.Figure()
            # crosshair at origin
            fig_mv.add_hline(y=0, line=dict(color="#334155", width=1))
            fig_mv.add_vline(x=0, line=dict(color="#334155", width=1))
            _types_present = [pt for pt in _mv["PitchType"].dropna().value_counts().index]
            for pt in _types_present:
                g = _mv[_mv["PitchType"] == pt]
                if len(g) < 3:
                    continue
                color = PITCH_COLORS.get(pt, "#94a3b8")
                cx, cy = g["HorzBreak"].mean(), g["InducedVertBreak"].mean()
                g = g.copy()
                g["_d"] = np.sqrt((g["HorzBreak"] - cx) ** 2 + (g["InducedVertBreak"] - cy) ** 2)
                closest = g.nsmallest(min(25, len(g)), "_d")
                # dashed league circle
                if pt in _lg_stats.index and _lg_stats.loc[pt, "n"] >= 20:
                    lhb, livb = _lg_stats.loc[pt, "hb"], _lg_stats.loc[pt, "ivb"]
                    r = 0.4 * np.nanmean([_lg_stats.loc[pt, "sd_hb"], _lg_stats.loc[pt, "sd_ivb"]])
                    r = r if pd.notna(r) and r > 0 else 1.5
                    theta = np.linspace(0, 2 * np.pi, 60)
                    fig_mv.add_trace(go.Scatter(
                        x=lhb + r * np.cos(theta), y=livb + r * np.sin(theta),
                        mode="lines", line=dict(color=color, width=1.5, dash="dash"),
                        opacity=0.6, showlegend=False, hoverinfo="skip"))
                # his 25 dots
                fig_mv.add_trace(go.Scatter(
                    x=closest["HorzBreak"], y=closest["InducedVertBreak"], mode="markers",
                    marker=dict(color=color, size=7, opacity=0.75,
                                line=dict(color="white", width=0.5)),
                    name=pt, hovertemplate=pt + "<br>HB %{x:.1f}, IVB %{y:.1f}<extra></extra>"))
            fig_mv.update_layout(
                title=dict(text="Movement vs League (dashed = " + _hand_lbl + " avg)",
                           font=dict(size=16, color="#1e293b")),
                xaxis_title="Horizontal break (in)", yaxis_title="Induced vertical break (in)",
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                font=dict(color="#1e293b", size=14), height=600,
                xaxis=dict(gridcolor="#E2E8F0", zeroline=False, scaleanchor="y", scaleratio=1,
                           range=[-27, 27], title_font=dict(size=15, color="#1e293b"),
                           tickfont=dict(size=12, color="#334155")),
                yaxis=dict(gridcolor="#E2E8F0", zeroline=False, range=[-27, 33],
                           title_font=dict(size=15, color="#1e293b"),
                           tickfont=dict(size=12, color="#334155")),
                legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#E2E8F0", borderwidth=1,
                            font=dict(size=13, color="#1e293b")))
            st.plotly_chart(fig_mv, use_container_width=True, key=f"{key_prefix}_arsenal_mv")

        with col_velo:
            fig_v = go.Figure()
            _velo_types = [pt for pt in _mv["PitchType"].dropna().value_counts().index
                           if len(_mv[_mv["PitchType"] == pt]) >= 3][::-1]
            for i, pt in enumerate(_velo_types):
                g = _mv[_mv["PitchType"] == pt]
                v = pd.to_numeric(g["RelSpeed"], errors="coerce").dropna()
                if len(v) < 3:
                    continue
                color = PITCH_COLORS.get(pt, "#94a3b8")
                vmin, vavg, vmax = v.min(), v.mean(), v.max()
                # Build a density gradient across the full min→max range: split into
                # ~24 segments, color each by how many pitches fall in that velo bin
                # (darker/more opaque = more pitches thrown at that speed).
                nseg = 24
                edges = np.linspace(vmin, vmax, nseg + 1)
                counts, _ = np.histogram(v, bins=edges)
                cmax = counts.max() if counts.max() > 0 else 1
                for s in range(nseg):
                    dens = counts[s] / cmax
                    seg_op = 0.15 + 0.75 * dens   # low where sparse, dark where dense
                    fig_v.add_trace(go.Scatter(
                        x=[edges[s], edges[s + 1]], y=[i, i], mode="lines",
                        line=dict(color=color, width=13), opacity=seg_op,
                        showlegend=False, hoverinfo="skip"))
                # average marker
                fig_v.add_trace(go.Scatter(
                    x=[vavg], y=[i], mode="markers+text",
                    marker=dict(color="white", size=9, line=dict(color=color, width=2)),
                    text=[f"{vavg:.0f}"], textposition="top center",
                    textfont=dict(color="#1e293b", size=9),
                    showlegend=False,
                    hovertemplate=pt + "<br>avg %{x:.1f} mph<extra></extra>"))
                # min / max end labels
                fig_v.add_trace(go.Scatter(
                    x=[vmin, vmax], y=[i, i], mode="text",
                    text=[f"{vmin:.0f}", f"{vmax:.0f}"],
                    textposition=["middle left", "middle right"],
                    textfont=dict(color="#64748b", size=8),
                    showlegend=False, hoverinfo="skip"))
            fig_v.update_layout(
                title=dict(text="Velocity range (darker = more pitches · ○ = avg)",
                           font=dict(size=16, color="#1e293b")),
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                font=dict(color="#1e293b", size=14), height=600,
                xaxis=dict(title="mph", gridcolor="#E2E8F0",
                           title_font=dict(size=15, color="#1e293b"),
                           tickfont=dict(size=12, color="#334155")),
                yaxis=dict(tickmode="array", tickvals=list(range(len(_velo_types))),
                           ticktext=_velo_types, gridcolor="#FFFFFF",
                           tickfont=dict(size=14, color="#1e293b"),
                           range=[-0.6, len(_velo_types) - 0.4]))
            st.plotly_chart(fig_v, use_container_width=True, key=f"{key_prefix}_arsenal_velo")

    # ── Attack Zones: where this pitcher throws (Heart/Shadow/Chase/Waste) ──
    st.divider()
    st.markdown("### Attack Zones — Pitch Location")
    st.caption("Statcast-style zones by distance from the middle of the strike zone: "
               "Heart (middle), Shadow (edges), Chase (just off), Waste (way off). Shows how "
               "often each pitch lands in each zone.")
    _az = _attack_zone_frame(df_all[df_all["Pitcher"] == pitcher])
    if len(_az) < 10:
        st.info("Not enough located pitches for attack-zone breakdown.")
    else:
        _zone_order = ["Heart", "Shadow", "Chase", "Waste"]
        overall_freq = (_az["_az"].value_counts(normalize=True) * 100)
        ov = {z: float(overall_freq.get(z, 0)) for z in _zone_order}
        cA, cB = st.columns([1, 1.3])
        with cA:
            components.html(_attack_zone_svg(ov, "Overall pitch location", is_rate=True), height=300)
        with cB:
            st.markdown("**By pitch type** (% of each pitch thrown to each zone)")
            rows = []
            for pt in _az["PitchType"].dropna().value_counts().index:
                sub = _az[_az["PitchType"] == pt]
                if len(sub) < 5:
                    continue
                fr = (sub["_az"].value_counts(normalize=True) * 100)
                rows.append({"Pitch": pt, "n": len(sub),
                             "Heart": f"{fr.get('Heart',0):.0f}%", "Shadow": f"{fr.get('Shadow',0):.0f}%",
                             "Chase": f"{fr.get('Chase',0):.0f}%", "Waste": f"{fr.get('Waste',0):.0f}%"})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Edge-heavy (high Shadow) = good command; high Heart = hittable; "
                   "high Chase on breaking balls = putaway pitches.")

    # ── Pitcher vs Team only: every hitter faced + their numbers vs him ──
    if pvt_mode:
        st.divider()
        if selected_team == _PVT_WHOLE_SEASON:
            st.markdown("### Hitters Faced — Whole Season")
            st.caption(f"Every hitter {player_last(pitcher)} has faced all year, with their numbers "
                       "in those plate appearances only — these are real small samples, read with "
                       "caution rather than as a settled read on either player.")
        else:
            st.markdown(f"### Hitters Faced — {team_label(selected_team)}")
            st.caption(f"Every {team_label(selected_team)} hitter {player_last(pitcher)} has faced, with "
                       "their numbers in this matchup only — these are real small samples, read with "
                       "caution rather than as a settled read on either player.")

        SWING_C_PVT = {"StrikeSwinging","InPlay","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}
        gc_pvt = [c for c in ["GameID","Inning","Top/Bottom","PAofInning"] if c in pp.columns]

        faced_rows = []
        for batter, bsub in pp.groupby("Batter"):
            if _is_removed(batter):
                continue
            side = bsub["BatterSide"].mode()
            side = side.iloc[0] if len(side) else "?"
            last_b = bsub.groupby(gc_pvt).last().reset_index() if gc_pvt else bsub.copy()
            k   = (last_b["KorBB"].eq("Strikeout") |
                   ((last_b["PitchCall"].isin(["StrikeSwinging","StrikeCalled"])) &
                    (last_b["Strikes"] == 2))).sum()
            bb  = last_b["KorBB"].eq("Walk").sum()
            h   = last_b["PlayResult"].isin(["Single","Double","Triple","HomeRun"]).sum()
            hr  = last_b["PlayResult"].eq("HomeRun").sum()
            pa  = len(last_b)
            ab  = int(int(k) + last_b["PlayResult"].isin(
                      ["Single","Double","Triple","HomeRun","Out","FieldersChoice","Error"]).sum())
            ba  = h / ab if ab > 0 else None
            sw  = bsub["PitchCall"].isin(SWING_C_PVT).sum()
            wh  = bsub["PitchCall"].eq("StrikeSwinging").sum()
            xba, _xslg, xwoba, _xba_n = batter_expected_stats(bsub)

            faced_rows.append({
                "Hitter": player_last(batter), "Bats": side, "PA": pa, "AB": ab,
                "H": int(h), "HR": int(hr), "BB": int(bb), "K": int(k),
                "BA": f"{ba:.3f}" if ba is not None else "—",
                "xBA": f"{xba:.3f}" if xba is not None else "—",
                "xwOBA": f"{xwoba:.3f}" if xwoba is not None else "—",
                "Whiff%": f"{wh/sw:.0%}" if sw > 0 else "—",
                "_pa_sort": pa,
                "_ba_sort": ba if ba is not None else -1,
            })

        if not faced_rows:
            st.info("No batter-level data available for this matchup.")
        else:
            faced_df = (pd.DataFrame(faced_rows)
                        .sort_values("_pa_sort", ascending=False)
                        .drop(columns=["_pa_sort", "_ba_sort"]))
            st.dataframe(faced_df, use_container_width=True, hide_index=True)

            qualified = [r for r in faced_rows if r["PA"] >= 3]
            if len(qualified) >= 2:
                best  = max(qualified, key=lambda r: r["_ba_sort"])
                worst = min(qualified, key=lambda r: r["_ba_sort"])
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                        <div style='background:#F8FAFC;border:1.5px solid #ef4444;
                        border-radius:8px;padding:1rem;'>
                            <div style='font-size:0.75rem;color:#ef4444;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;'>
                            Has his number</div>
                            <div style='font-size:1.1rem;font-weight:700;color:#1e293b;'>{best['Hitter']}</div>
                            <div style='font-size:0.85rem;color:#475569;'>
                            {best['BA']} BA · {best['H']}-for-{best['AB']} · {best['PA']} PA</div>
                        </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                        <div style='background:#F8FAFC;border:1.5px solid #22c55e;
                        border-radius:8px;padding:1rem;'>
                            <div style='font-size:0.75rem;color:#22c55e;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;'>
                            Owns this hitter</div>
                            <div style='font-size:1.1rem;font-weight:700;color:#1e293b;'>{worst['Hitter']}</div>
                            <div style='font-size:0.85rem;color:#475569;'>
                            {worst['BA']} BA · {worst['H']}-for-{worst['AB']} · {worst['PA']} PA</div>
                        </div>""", unsafe_allow_html=True)
            st.caption("PA ≥ 3 required for the callout cards above — everyone else is in the table "
                       "but too thin a sample to call out.")

            st.divider()
            st.markdown(f"### At-Bat Log — vs {player_last(pitcher)}")
            st.caption("Pick a hitter to see every pitch of every at-bat against this pitcher: "
                       "velocity, movement, and how the at-bat ended.")

            ab_hitters = sorted(pp["Batter"].dropna().unique())
            ab_hitter = st.selectbox("Hitter", options=[""] + ab_hitters,
                                     format_func=lambda x: player_last(x) if x else "Select a hitter…",
                                     key="pvt_ab_hitter")

            AB_OUTCOME_COLORS = {
                "Single": "#22c55e", "Double": "#3b82f6", "Triple": "#8b5cf6",
                "HomeRun": "#f59e0b", "Walk": "#06b6d4", "HBP": "#06b6d4",
                "Out": "#ef4444", "FieldersChoice": "#f97316", "Error": "#ec4899",
                "Strikeout": "#ef4444", "Sacrifice": "#64748b", "—": "#475569",
            }

            def _render_ab_log(ab_log_list, key_prefix="main"):
                """Render a list of _build_ab_log() at-bats as bordered cards."""
                st.caption(f"{len(ab_log_list)} plate appearance(s) found.")
                for i, ab in enumerate(ab_log_list, 1):
                    color = AB_OUTCOME_COLORS.get(ab["outcome"], "#64748b")
                    date_str = ab["date"].strftime("%b %d, %Y") if pd.notna(ab["date"]) else "—"
                    inning = int(ab["inning"]) if pd.notna(ab["inning"]) else "?"
                    with st.container(border=True):
                        bb_txt = ""
                        if pd.notna(ab.get("exit_speed")):
                            bb_parts = [f"EV {ab['exit_speed']:.1f} mph"]
                            if pd.notna(ab.get("angle")):
                                bb_parts.append(f"LA {ab['angle']:.0f}°")
                            if pd.notna(ab.get("distance")):
                                bb_parts.append(f"{ab['distance']:.0f} ft")
                            if pd.notna(ab.get("direction")):
                                bb_parts.append(f"Dir {ab['direction']:.0f}°")
                            bb_txt = ("&nbsp;&nbsp;<span style='color:#64748b;font-size:0.85rem;'>"
                                     + " · ".join(bb_parts) + "</span>")
                        st.markdown(
                            f"**AB {i}** — {date_str}, Inning {inning}&nbsp;&nbsp;"
                            f"<span style='color:{color};font-weight:800;'>{ab['outcome']}</span>{bb_txt}",
                            unsafe_allow_html=True)
                        pitch_rows = [{
                            "Pitch #": p["num"], "Count": p["count"], "Type": p["type"],
                            "Velo": f"{p['velo']:.1f}" if pd.notna(p["velo"]) else "—",
                            "IVB": f"{p['ivb']:.1f}\"" if pd.notna(p["ivb"]) else "—",
                            "HB": f"{p['hb']:.1f}\"" if pd.notna(p["hb"]) else "—",
                            "Zone": p["zone"], "Call": p["call"],
                        } for p in ab["pitches"]]
                        tbl_col, plot_col = st.columns([1.5, 1])
                        with tbl_col:
                            st.dataframe(pd.DataFrame(pitch_rows), use_container_width=True, hide_index=True)
                        with plot_col:
                            _render_ab_pitch_plot(ab["pitches"], key=f"{key_prefix}_{i}_{date_str}_{inning}")

            if ab_hitter:
                hp = pp[pp["Batter"] == ab_hitter].copy()
                ab_log = _build_ab_log(hp)
                if not ab_log and not [c for c in ["GameID","Inning","Top/Bottom","PAofInning"] if c in hp.columns]:
                    st.info("Not enough at-bat structure in this data to break out by plate appearance.")
                else:
                    _render_ab_log(ab_log)

            # ── Bandits hitter scouting report (PDF) ──
            st.divider()
            st.markdown("### Scouting Report — Bandits Hitter")
            st.caption("Pick one or more of our hitters to build printable PDFs: this pitcher's "
                       "location/contact/whiff tendencies and pitch mix against each hitter's own "
                       "batting side (not just this one opponent), plus any head-to-head at-bat "
                       "history between the two of them. Pick more than one to download them all "
                       "at once as a ZIP.")

            GENERIC_LHH, GENERIC_RHH = "__GENERIC_LHH__", "__GENERIC_RHH__"

            def _pvt_hitter_label(h):
                if h == GENERIC_LHH:
                    return "Generic LHH"
                if h == GENERIC_RHH:
                    return "Generic RHH"
                return player_last(h)

            def _pvt_file_slug(h):
                return _pvt_hitter_label(h).replace(", ", "_").replace(" ", "_")

            sk_hitters = [GENERIC_RHH, GENERIC_LHH] + \
                _player_options(df_all[df_all["BatterTeam"] == MY_TEAM]["Batter"])
            sk_hitter_sel = st.multiselect("Bandits hitter(s)", options=sk_hitters,
                                           format_func=_pvt_hitter_label, key="pvt_report_hitters")
            st.caption("\"Generic RHH\"/\"Generic LHH\" build this pitcher's report against a "
                       "league-average hitter of that side — no at-bat history, just tendencies — "
                       "for when you want the scouting report without picking a specific hitter.")

            def _pvt_hitter_inputs(h, pitcher_all):
                """Everything the PDF builder needs for one Bandits hitter (or a
                generic LHH/RHH) vs the selected pitcher: batting side, pitch mix, at-bat log."""
                if h in (GENERIC_LHH, GENERIC_RHH):
                    h_side = "Left" if h == GENERIC_LHH else "Right"
                else:
                    side_mode = df_all[df_all["Batter"] == h]["BatterSide"].mode()
                    h_side = side_mode.iloc[0] if len(side_mode) else "Right"
                h_hand_lbl = {"Right": "RHH", "Left": "LHH"}.get(h_side, "Unknown")
                h_vs_hand = pitcher_all[pitcher_all["BatterSide"] == h_side]
                h_mix_rows = []
                h_total = len(h_vs_hand)
                if h_total > 0:
                    for pt, sub in h_vs_hand.groupby("PitchType"):
                        if pd.isna(pt):
                            continue
                        h_mix_rows.append({
                            "Pitch": pt, "Usage": f"{len(sub)/h_total:.0%}", "Pitches": len(sub),
                            "Velo": f"{sub['RelSpeed'].mean():.1f}" if sub["RelSpeed"].notna().any() else "—",
                            "IVB": (f"{sub['InducedVertBreak'].mean():.1f}\""
                                    if sub["InducedVertBreak"].notna().any() else "—"),
                            "HB": f"{sub['HorzBreak'].mean():.1f}\"" if sub["HorzBreak"].notna().any() else "—",
                            "Spin": f"{sub['SpinRate'].mean():.0f}" if sub["SpinRate"].notna().any() else "—",
                        })
                    h_mix_rows.sort(key=lambda r: -r["Pitches"])
                if h in (GENERIC_LHH, GENERIC_RHH):
                    h_h2h_log = []
                else:
                    h2h = df_all[(df_all["Pitcher"] == pitcher) & (df_all["Batter"] == h)].copy()
                    h_h2h_log = _build_ab_log(h2h)
                return h_side, h_hand_lbl, h_vs_hand, h_mix_rows, h_h2h_log

            sk_hitter = sk_hitter_sel[0] if len(sk_hitter_sel) == 1 else (
                st.selectbox("Preview on screen", options=sk_hitter_sel,
                             format_func=_pvt_hitter_label, key="pvt_report_preview")
                if len(sk_hitter_sel) > 1 else "")

            if sk_hitter:
                hitter_side, hand_lbl, vs_hand, _preview_mix_rows, _preview_h2h_log = \
                    _pvt_hitter_inputs(sk_hitter, df_all[df_all["Pitcher"] == pitcher])
                st.markdown(f"**{_pvt_hitter_label(sk_hitter)}** bats **{hitter_side}** ({hand_lbl})")

                if len(vs_hand) < 10:
                    st.warning(f"Only {len(vs_hand)} pitches on record vs {hand_lbl} for "
                               f"{player_last(pitcher)} — the report below will be thin.")

                st.markdown(f"#### {player_last(pitcher)} vs {hand_lbl} — Location Tendencies")
                hm1, hm2, hm3 = st.columns(3)
                bip_hand = vs_hand[vs_hand["ExitSpeed"].notna() & vs_hand["PlateLocSide"].notna() &
                                   vs_hand["PlateLocHeight"].notna()]
                sw_hand = vs_hand[vs_hand["PitchCall"].isin(SWING_C_PVT) & vs_hand["PlateLocSide"].notna() &
                                  vs_hand["PlateLocHeight"].notna()].copy()
                if len(sw_hand):
                    sw_hand["_whiff_w"] = sw_hand["PitchCall"].eq("StrikeSwinging").astype(float)
                with hm1:
                    st.caption("Pitch Location")
                    _render_kde_heatmap(vs_hand, weight_col=None,
                                        key_suffix=f"pvt_rep_freq_{pitcher}_{hitter_side}")
                with hm2:
                    st.caption("Hard Contact Zones")
                    if len(bip_hand) >= 5:
                        _render_kde_heatmap(bip_hand, weight_col="ExitSpeed",
                                            key_suffix=f"pvt_rep_ev_{pitcher}_{hitter_side}")
                    else:
                        st.info("Not enough contact data.")
                with hm3:
                    st.caption("Whiff Zones")
                    if len(sw_hand) >= 5:
                        _render_kde_heatmap(sw_hand, weight_col="_whiff_w",
                                            key_suffix=f"pvt_rep_wh_{pitcher}_{hitter_side}")
                    else:
                        st.info("Not enough swing data.")

                st.markdown(f"#### Pitch Mix vs {hand_lbl}")
                total_vs_hand = len(vs_hand)
                if total_vs_hand > 0:
                    st.dataframe(pd.DataFrame(_preview_mix_rows), use_container_width=True, hide_index=True)
                else:
                    st.info(f"No pitches on record vs {hand_lbl} for this pitcher.")

                st.markdown(f"#### Pitch Usage by Count vs {hand_lbl}")
                if total_vs_hand > 0:
                    _render_count_group_mix(vs_hand)
                else:
                    st.caption("No pitches on record to break out by count.")

                if sk_hitter in (GENERIC_LHH, GENERIC_RHH):
                    pass  # no at-bat history for a generic, non-specific hitter
                else:
                    st.markdown(f"#### At-Bat History — {_pvt_hitter_label(sk_hitter)} vs {player_last(pitcher)}")
                    if _preview_h2h_log:
                        _render_ab_log(_preview_h2h_log, key_prefix="h2h")
                    else:
                        st.info("No recorded plate appearances between these two players.")

            if sk_hitter_sel:
                st.divider()
                throws_lbl = {"Right": "RHP", "Left": "LHP"}.get(hand, hand)
                pitcher_all_pdf = df_all[df_all["Pitcher"] == pitcher]
                pdfs = {}
                for h in sk_hitter_sel:
                    try:
                        h_side, h_hand_lbl, h_vs_hand, h_mix_rows, h_h2h_log = \
                            _pvt_hitter_inputs(h, pitcher_all_pdf)
                        pdfs[h] = _build_hitter_scouting_pdf(
                            _pvt_hitter_label(h), h_hand_lbl, player_last(pitcher), throws_lbl,
                            h_vs_hand, h_mix_rows, h_h2h_log)
                    except Exception as _rep_e:
                        _pdf_unavailable(_rep_e)

                if len(pdfs) == 1:
                    h, pdf_bytes = next(iter(pdfs.items()))
                    st.download_button(
                        "⬇ Download scouting report (PDF)", data=pdf_bytes,
                        file_name=(f"{_pvt_file_slug(h)}_vs_"
                                  f"{player_last(pitcher).replace(', ', '_')}.pdf"),
                        mime="application/pdf", key="pvt_report_download")
                elif len(pdfs) > 1:
                    import zipfile, io as _zip_io
                    zip_buf = _zip_io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w") as zf:
                        for h, pdf_bytes in pdfs.items():
                            zf.writestr(
                                f"{_pvt_file_slug(h)}_vs_"
                                f"{player_last(pitcher).replace(', ', '_')}.pdf",
                                pdf_bytes)
                    st.download_button(
                        f"⬇ Download {len(pdfs)} scouting reports (ZIP)", data=zip_buf.getvalue(),
                        file_name=f"{player_last(pitcher).replace(', ', '_')}_hitter_reports.zip",
                        mime="application/zip", key="pvt_report_download_zip")


# ─────────────────────────────────────────
#  PAGE: PITCH DESIGN (tunneling + quality vs usage)
# ─────────────────────────────────────────
elif page == "Pitch Design":
    st.title("Pitch Design — Tunneling & Mix")
    st.caption("How well a pitcher's offerings tunnel (look alike out of the hand, break apart late) "
               "and whether his best pitches are used enough. Directional guidance, not exact usage %.")

    from itertools import combinations
    pd_teams = sorted(_team_options(df_all["PitcherTeam"]))
    default_idx = pd_teams.index(MY_TEAM) if MY_TEAM in pd_teams else 0
    c1, c2 = st.columns([1.3, 2])
    with c1:
        pd_team = st.selectbox("Team", options=pd_teams, index=default_idx,
                               format_func=team_label, key="pd_team")
    team_p = _player_options(df_all[df_all["PitcherTeam"] == pd_team]["Pitcher"])
    with c2:
        pd_pitcher = st.selectbox("Pitcher", options=[""] + team_p,
                                  format_func=lambda p: player_last(p) if p else "Select…", key="pd_p")

    if not pd_pitcher:
        st.info("Select a pitcher to see their pitch-design profile.")
    else:
        psub = df_all[df_all["Pitcher"] == pd_pitcher]
        # Build per-pitch-type profile (min 5 thrown to be meaningful)
        prof = {}
        for pt in psub["PitchType"].dropna().unique():
            s = psub[psub["PitchType"] == pt]
            if len(s) < 5:
                continue
            prof[pt] = {
                "n": len(s),
                "relH": s["RelHeight"].mean(), "relS": s["RelSide"].mean(),
                "ivb": s["InducedVertBreak"].mean(), "hb": s["HorzBreak"].mean(),
                "velo": s["RelSpeed"].mean(),
                "whiff": _safe_whiff(s),
            }
        if len(prof) < 2:
            st.warning("Need at least two pitch types (5+ each) to analyze tunneling.")
        else:
            total = sum(d["n"] for d in prof.values())

            # ── Tunneling by pair ──
            st.markdown("#### Tunneling by pitch pair")
            st.caption("Good tunnel = small release gap (look alike leaving the hand) + large late "
                       "movement separation (break apart). Release gap in inches, separation in inches of break.")
            trows = []
            for a, b in combinations(prof, 2):
                da, db = prof[a], prof[b]
                rel_gap_in = np.hypot(da["relH"]-db["relH"], da["relS"]-db["relS"]) * 12
                move_sep = np.hypot(da["ivb"]-db["ivb"], da["hb"]-db["hb"])
                velo_gap = abs(da["velo"] - db["velo"])
                # Tunnel grade: reward separation, penalize release gap.
                score = move_sep - rel_gap_in * 1.5
                if rel_gap_in <= 3 and move_sep >= 12:
                    grade = "Elite"
                elif rel_gap_in <= 4 and move_sep >= 9:
                    grade = "Good"
                elif rel_gap_in <= 5:
                    grade = "OK"
                else:
                    grade = "Leaks"
                trows.append({
                    "Pair": f"{a} + {b}",
                    "Release gap": f"{rel_gap_in:.1f}\"",
                    "Movement sep": f"{move_sep:.1f}\"",
                    "Velo gap": f"{velo_gap:.1f}",
                    "Tunnel": grade,
                    "_s": score,
                })
            tdf = pd.DataFrame(trows).sort_values("_s", ascending=False).drop(columns="_s")
            st.dataframe(tdf, use_container_width=True, hide_index=True)

            best = trows[0] if trows else None
            best = max(trows, key=lambda r: r["_s"]) if trows else None
            leaks = [r for r in trows if r["Tunnel"] == "Leaks"]
            if best:
                st.markdown(f"**Best tunnel:** {best['Pair']} — "
                            f"nearly same release ({best['Release gap']}), big late break ({best['Movement sep']}). "
                            "Pair these to keep hitters guessing.")
            if leaks:
                st.markdown("**Leaks (different release — hitters can read these early):** "
                            + ", ".join(r["Pair"] for r in leaks))

            # ── Quality vs usage ──
            st.markdown("#### Quality vs usage")
            st.caption("Each pitch's whiff rate vs how often it's thrown. High whiff + low usage = "
                       "a weapon that may be underused. This is directional, not a prescription.")
            qrows = []
            for pt, d in prof.items():
                usage = 100 * d["n"] / total
                qrows.append({
                    "Pitch": pt, "Usage%": f"{usage:.1f}%", "Thrown": d["n"],
                    "Whiff%": f"{100*d['whiff']:.0f}%" if d["whiff"] is not None else "—",
                    "Avg Velo": f"{d['velo']:.1f}",
                    "_usage": usage, "_whiff": d["whiff"] if d["whiff"] is not None else 0,
                })
            qdf = pd.DataFrame(qrows).sort_values("_whiff", ascending=False)
            st.dataframe(qdf[["Pitch","Usage%","Thrown","Whiff%","Avg Velo"]],
                         use_container_width=True, hide_index=True)

            # Directional rec: high whiff (top), low usage
            recs = []
            avg_usage = 100 / len(prof)
            for r in qrows:
                if r["_whiff"] >= 0.30 and r["_usage"] < avg_usage:
                    recs.append(f"**{r['Pitch']}** misses bats ({r['Whiff%']} whiff) but is only "
                                f"{r['Usage%']} of his mix — consider leaning on it more.")
                if r["_whiff"] <= 0.12 and r["_usage"] > avg_usage * 1.3:
                    recs.append(f"**{r['Pitch']}** is thrown a lot ({r['Usage%']}) but doesn't miss "
                                f"bats ({r['Whiff%']}) — make sure it's earning its usage.")
            if recs:
                st.markdown("**Consider:**")
                for r in recs:
                    st.markdown(f"- {r}")
            else:
                st.caption("No strong usage flags — his mix roughly tracks his pitch quality.")

            st.info("⚠ These are directional insights from current data. Optimal usage % can't be "
                    "computed exactly — throwing a pitch more changes how hitters react to it, which "
                    "this data can't observe. Use as a starting point for the pitching coach's judgment.")


# ─────────────────────────────────────────
#  PAGE: BULLPEN SCRIPT (development-focused session plan)
# ─────────────────────────────────────────
elif page == "Bullpen Script":
    st.title("Bullpen Script")
    st.caption("Turns a pitcher's tunneling and usage profile into an actual bullpen session plan — "
               "which pitch pairs to drill, how many reps, and why. Development-focused, not a "
               "game plan (see Game Plan / Attack Plan for opponent-specific sequencing).")

    from itertools import combinations
    bs_teams = sorted(_team_options(df_all["PitcherTeam"]))
    default_idx = bs_teams.index(MY_TEAM) if MY_TEAM in bs_teams else 0
    c1, c2 = st.columns([1.3, 2])
    with c1:
        bs_team = st.selectbox("Team", options=bs_teams, index=default_idx,
                               format_func=team_label, key="bs_team")
    team_p = _player_options(df_all[df_all["PitcherTeam"] == bs_team]["Pitcher"])
    with c2:
        bs_pitcher = st.selectbox("Pitcher", options=[""] + team_p,
                                  format_func=lambda p: player_last(p) if p else "Select…", key="bs_p")

    if not bs_pitcher:
        st.info("Select a pitcher to build a bullpen script.")
    else:
        psub = df_all[df_all["Pitcher"] == bs_pitcher]
        prof = {}
        for pt in psub["PitchType"].dropna().unique():
            s = psub[psub["PitchType"] == pt]
            if len(s) < 5:
                continue
            prof[pt] = {
                "n": len(s),
                "relH": s["RelHeight"].mean(), "relS": s["RelSide"].mean(),
                "ivb": s["InducedVertBreak"].mean(), "hb": s["HorzBreak"].mean(),
                "velo": s["RelSpeed"].mean(),
                "whiff": _safe_whiff(s),
            }
        if len(prof) < 2:
            st.warning("Need at least two pitch types (5+ each) to build a tunneling-based script.")
        else:
            total = sum(d["n"] for d in prof.values())

            # Same tunneling math as the Pitch Design page, so the two pages
            # always agree on a given pair's grade.
            pairs = []
            for a, b in combinations(prof, 2):
                da, db = prof[a], prof[b]
                rel_gap_in = np.hypot(da["relH"] - db["relH"], da["relS"] - db["relS"]) * 12
                move_sep = np.hypot(da["ivb"] - db["ivb"], da["hb"] - db["hb"])
                velo_gap = abs(da["velo"] - db["velo"])
                score = move_sep - rel_gap_in * 1.5
                if rel_gap_in <= 3 and move_sep >= 12:
                    grade = "Elite"
                elif rel_gap_in <= 4 and move_sep >= 9:
                    grade = "Good"
                elif rel_gap_in <= 5:
                    grade = "OK"
                else:
                    grade = "Leaks"
                pairs.append({"a": a, "b": b, "rel_gap": rel_gap_in, "move_sep": move_sep,
                              "velo_gap": velo_gap, "grade": grade, "score": score})

            # Worst-tunneling pairs are the drill priority; sorted worst-first.
            priority = sorted([p for p in pairs if p["grade"] in ("Leaks", "OK")],
                              key=lambda p: p["score"])
            solid = [p for p in pairs if p["grade"] in ("Elite", "Good")]

            # Reps per drilled pair (a "pair" = throw A then B back-to-back).
            # Leaks get more work than a pair that's merely OK.
            REPS_BY_GRADE = {"Leaks": 5, "OK": 3}

            script_lines = [
                f"BULLPEN SCRIPT — {player_last(bs_pitcher)}",
                "Development focus: tunneling + underused weapons",
                "=" * 50,
            ]

            st.divider()
            st.markdown("#### Priority — Tunnel Pairs to Drill")
            if not priority:
                st.success("No leaking pairs — every combo already tunnels well. See 'Maintain' below.")
                script_lines.append("\nNo priority tunnel work — arsenal already tunnels well.")
            else:
                st.caption("Ordered worst-to-best. These are the pairs hitters can most easily "
                           "read apart out of the hand.")
                script_lines.append("\nPRIORITY TUNNEL WORK:")
                for p in priority:
                    reps = REPS_BY_GRADE[p["grade"]]
                    throws = reps * 2
                    # What's actually driving the leak decides the coaching cue.
                    if p["rel_gap"] > 4:
                        cue = (f"Release point doesn't match (gap {p['rel_gap']:.1f}\") — focus on "
                               "identical arm slot, extension, and release height for both pitches.")
                    elif p["move_sep"] < 9:
                        cue = (f"Movement doesn't separate enough ({p['move_sep']:.1f}\" apart) — "
                               "let each pitch finish fully; don't guide it, trust the shape.")
                    else:
                        cue = "Close on both counts — fine-tune, this pair is nearly there."
                    with st.container(border=True):
                        st.markdown(f"**{p['a']} → {p['b']}**  ·  {p['grade']}  ·  "
                                    f"{reps} pairs ({throws} throws)")
                        st.caption(f"Release gap {p['rel_gap']:.1f}\" · Movement sep {p['move_sep']:.1f}\" · "
                                   f"Velo gap {p['velo_gap']:.1f} mph")
                        st.markdown(cue)
                    script_lines.append(f"  {p['a']} -> {p['b']} ({p['grade']}, {reps} pairs / {throws} throws)")
                    script_lines.append(f"    Release gap {p['rel_gap']:.1f}in, Movement sep "
                                        f"{p['move_sep']:.1f}in, Velo gap {p['velo_gap']:.1f}mph")
                    script_lines.append(f"    Cue: {cue}")

            if solid:
                st.markdown("#### Maintain — Already Tunneling Well")
                st.caption("No drill time needed here — keep pairing these in-game sequencing.")
                maintain_txt = ", ".join(f"{p['a']} + {p['b']} ({p['grade']})" for p in solid)
                st.markdown(maintain_txt)
                script_lines.append("\nMAINTAIN (no drill needed): " + maintain_txt)

            # Underused-weapon reps — same quality-vs-usage read as Pitch Design,
            # turned into a standalone rep block instead of just a note.
            st.markdown("#### Extra Reps — Underused Weapons")
            avg_usage = 100 / len(prof)
            weapon_rows = []
            for pt, d in prof.items():
                usage = 100 * d["n"] / total
                if d["whiff"] is not None and d["whiff"] >= 0.30 and usage < avg_usage:
                    weapon_rows.append((pt, usage, d["whiff"]))
            if weapon_rows:
                script_lines.append("\nEXTRA REPS (underused weapons):")
                for pt, usage, whiff in weapon_rows:
                    st.markdown(f"- **{pt}**: {100*whiff:.0f}% whiff but only {usage:.0f}% usage — "
                                "add 8-10 reps by itself this session to build command/confidence.")
                    script_lines.append(f"  {pt}: {100*whiff:.0f}% whiff, {usage:.0f}% usage -> "
                                        "8-10 reps standalone")
            else:
                st.caption("No standout underused weapon flagged — usage roughly tracks quality.")

            st.divider()
            st.info("⚠ Directional, from current-season Trackman data. A pitching coach's eye on "
                    "mechanics still overrides what release-point math alone can see.")

            st.download_button("⬇ Download bullpen script (.txt)",
                               data="\n".join(script_lines),
                               file_name=f"{player_last(bs_pitcher).replace(', ', '_')}_bullpen_script.txt",
                               key="bs_download")


# ─────────────────────────────────────────
#  PAGE: NEXT HITTERS (in-game look-ahead)
# ─────────────────────────────────────────
elif page == "Next Hitters":
    st.title("Next Hitters — Attack Plan")
    st.caption("Pick the opponent and their lineup order, set who's up, and see the next "
               "hitters due up with how to attack each one. (Manual entry now; live-feed "
               "auto-fill can be added when a data URL is available.)")

    # Opponent team
    opp_teams = sorted([t for t in _team_options(df_all["BatterTeam"]) if t != MY_TEAM])
    nh_team = st.selectbox("Opponent", options=opp_teams, format_func=team_label, key="nh_team")
    team_hitters = _player_options(df_all[df_all["BatterTeam"] == nh_team]["Batter"])

    st.markdown("#### Set the lineup (1–9 in batting order)")
    st.caption("Choose each spot. Leave unused spots blank.")
    lineup = []
    lc = st.columns(3)
    for i in range(9):
        with lc[i % 3]:
            pick = st.selectbox(f"{i+1}.", options=[""] + team_hitters,
                                format_func=lambda b: player_last(b) if b else "—",
                                key=f"nh_spot_{i}")
            if pick:
                lineup.append(pick)

    if len(lineup) < 3:
        st.info("Add at least 3 hitters to the lineup.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            up_now = st.number_input("Batting now (order #)", min_value=1,
                                     max_value=len(lineup), value=1, key="nh_up")
        with c2:
            n_show = st.slider("How many to show", 3, len(lineup), len(lineup), key="nh_n")
        with c3:
            nh_hand = st.selectbox("vs. Pitcher Hand", options=["All", "Right", "Left"],
                                   format_func=lambda x: {"All": "All", "Right": "vs RHP",
                                                           "Left": "vs LHP"}[x],
                                   key="nh_hand")

        # Next hitters due up, rolling through the order
        start = up_now - 1
        due = [lineup[(start + k) % len(lineup)] for k in range(n_show)]

        st.divider()
        st.markdown(f"### Due up: next {n_show}")

        HALF = 0.83

        def _nh_compute(hitter, hand_filter):
            """Contact/whiff stats, per-pitch-type breakdown, auto attack notes,
            and recommended shift for one hitter, scoped to hand_filter."""
            bp_all = df_all[df_all["Batter"] == hitter]
            bp = bp_all if hand_filter == "All" else bp_all[bp_all["PitcherThrows"] == hand_filter]
            side = bp_all["BatterSide"].mode().iloc[0] if len(bp_all["BatterSide"].mode()) else "?"
            n_seen = len(bp)
            nh_stats = compute_batter_stats(hitter, bp, hand_filter if hand_filter != "All" else None)
            nh_haz = _attack_zone_frame(bp)
            nh_z_sw, nh_o_sw, nh_n_in, nh_n_out = _true_zone_swing(nh_haz)

            rows = []
            for pt in bp["PitchType"].dropna().value_counts().index:
                sub = bp[bp["PitchType"] == pt]
                if len(sub) < 4:
                    continue
                ab = int((_ab_mask(sub)).sum())
                h = int(sub["PlayResult"].isin(["Single","Double","Triple","HomeRun"]).sum())
                sw = int(sub["PitchCall"].isin(["StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable","InPlay"]).sum())
                wh = int((sub["PitchCall"] == "StrikeSwinging").sum())
                ba = h/ab if ab >= 3 else None
                whp = wh/sw if sw > 0 else None
                rows.append({"Pitch": pt, "Seen": len(sub),
                             "BA": f"{ba:.3f}" if ba is not None else "—",
                             "Whiff%": f"{100*whp:.0f}%" if whp is not None else "—",
                             "_ba": ba if ba is not None else 99,
                             "_wh": whp if whp is not None else -1})

            attack = []
            if rows:
                best_whiff = max(rows, key=lambda r: r["_wh"])
                if best_whiff["_wh"] > 0.25:
                    attack.append(f"Best swing-and-miss pitch: **{best_whiff['Pitch']}** "
                                  f"({best_whiff['Whiff%']} whiff)")
                weakest = min(rows, key=lambda r: r["_ba"])
                if weakest["_ba"] < 0.250:
                    attack.append(f"Struggles vs **{weakest['Pitch']}** (BA {weakest['BA']})")
                strongest = max((r for r in rows if r["_ba"] < 99), key=lambda r: r["_ba"], default=None)
                if strongest and strongest["_ba"] >= 0.300:
                    attack.append(f"⚠ Avoid **{strongest['Pitch']}** — hits it hard (BA {strongest['BA']})")

                oz = bp[(bp["PlateLocSide"].abs() > HALF) | (~bp["PlateLocHeight"].between(1.5, 3.5))]
                oz = oz[oz["PlateLocSide"].notna()]
                if len(oz) >= 10:
                    oz_sw = int(oz["PitchCall"].isin(["StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable","InPlay"]).sum())
                    chase = 100 * oz_sw / len(oz)
                    if chase >= 30:
                        attack.append(f"Chases out of zone ({chase:.0f}%) — expand when ahead")
                    elif chase <= 18:
                        attack.append(f"Disciplined ({chase:.0f}% chase) — must throw strikes")

            shift_pos, shift_n = _shift_positions(bp, side)
            return dict(bp=bp, side=side, n_seen=n_seen, nh_stats=nh_stats,
                        nh_z_sw=nh_z_sw, nh_o_sw=nh_o_sw, rows=rows, attack=attack,
                        shift_pos=shift_pos, shift_n=shift_n)

        def _nh_export_card_html(order_label, hitter, d):
            pt_rows_html = "".join(
                f"<tr><td>{r['Pitch']}</td><td>{r['Seen']}</td><td>{r['BA']}</td><td>{r['Whiff%']}</td></tr>"
                for r in d["rows"]
            ) if d["rows"] else "<tr><td colspan='4' style='color:#64748b;'>Not enough pitch-type data.</td></tr>"
            attack_html = ("<p class=\"note\">" + "  &middot;  ".join(
                a.replace("**", "") for a in d["attack"]) + "</p>") if d["attack"] else ""
            nh_stats = d["nh_stats"]
            return f"""
                <div class="card">
                  <h3>{order_label}. {player_last(hitter)} <span class="muted">&middot; bats {d['side']} &middot; {d['n_seen']}p</span></h3>
                  <div class="metrics">
                    <div class="m"><span class="l">OPS</span><span class="v">{f"{nh_stats['OPS']:.3f}" if pd.notna(nh_stats.get('OPS')) else "—"}</span></div>
                    <div class="m"><span class="l">xwOBA</span><span class="v">{f"{nh_stats['xwOBA']:.3f}" if nh_stats.get('xwOBA') is not None else "—"}</span></div>
                    <div class="m"><span class="l">K%</span><span class="v">{f"{nh_stats['KPct']*100:.0f}%" if pd.notna(nh_stats.get('KPct')) else "—"}</span></div>
                    <div class="m"><span class="l">BB%</span><span class="v">{f"{nh_stats['BBPct']*100:.0f}%" if pd.notna(nh_stats.get('BBPct')) else "—"}</span></div>
                    <div class="m"><span class="l">Z-Sw%</span><span class="v">{d['nh_z_sw']:.0f}%</span></div>
                    <div class="m"><span class="l">O-Sw%</span><span class="v">{d['nh_o_sw']:.0f}%</span></div>
                  </div>
                  <table class="pt">
                    <thead><tr><th>Pitch</th><th>Seen</th><th>BA</th><th>Wh%</th></tr></thead>
                    <tbody>{pt_rows_html}</tbody>
                  </table>
                  {attack_html}
                </div>"""

        def _nh_shift_card_html(order_label, hitter, d):
            shift_note = ("Standard alignment — not enough balls in play yet." if d["shift_n"] < 8
                          else f"Shaded to spray tendency &middot; {d['shift_n']} BIP")
            return f"""
                <div class="fcard">
                  <h4>{order_label}. {player_last(hitter)} <span class="muted">&middot; bats {d['side']}</span></h4>
                  {_field_svg(d['shift_pos'], title="", width=200)}
                  <p class="fnote">{shift_note}</p>
                </div>"""

        for idx, hitter in enumerate(due):
            bp_all = df_all[df_all["Batter"] == hitter]
            bp = bp_all if nh_hand == "All" else bp_all[bp_all["PitcherThrows"] == nh_hand]
            order_pos = (start + idx) % len(lineup) + 1
            side = bp_all["BatterSide"].mode().iloc[0] if len(bp_all["BatterSide"].mode()) else "?"
            n_seen = len(bp)
            hand_label = {"All": "all pitchers", "Right": "vs RHP", "Left": "vs LHP"}[nh_hand]

            with st.container(border=True):
                st.markdown(f"**{order_pos}. {player_last(hitter)}**  ·  bats {side}  ·  "
                            f"{n_seen} pitches seen ({hand_label})")
                if n_seen < 10:
                    st.caption("⚠ Limited data on this hitter — read with caution.")

                nh_stats = compute_batter_stats(hitter, bp, nh_hand if nh_hand != "All" else None)
                nm = st.columns(4)
                nm[0].metric("OPS", f"{nh_stats['OPS']:.3f}" if pd.notna(nh_stats.get('OPS')) else "—")
                nm[1].metric("xwOBA", f"{nh_stats['xwOBA']:.3f}" if nh_stats.get('xwOBA') is not None else "—")
                nm[2].metric("K%", f"{nh_stats['KPct']*100:.0f}%" if pd.notna(nh_stats.get('KPct')) else "—")
                nm[3].metric("BB%", f"{nh_stats['BBPct']*100:.0f}%" if pd.notna(nh_stats.get('BBPct')) else "—")

                nh_haz = _attack_zone_frame(bp)
                nh_z_sw, nh_o_sw, nh_n_in, nh_n_out = _true_zone_swing(nh_haz)
                nz = st.columns(2)
                nz[0].metric("Z-Swing%", f"{nh_z_sw:.0f}%", help=f"Swings at pitches in the zone (n={nh_n_in})")
                nz[1].metric("O-Swing% (chase)", f"{nh_o_sw:.0f}%", help=f"Swings at pitches out of the zone (n={nh_n_out})")

                # Per-pitch-type BA + whiff
                rows = []
                for pt in bp["PitchType"].dropna().value_counts().index:
                    sub = bp[bp["PitchType"] == pt]
                    if len(sub) < 4:
                        continue
                    ab = int((_ab_mask(sub)).sum())
                    h = int(sub["PlayResult"].isin(["Single","Double","Triple","HomeRun"]).sum())
                    sw = int(sub["PitchCall"].isin(["StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable","InPlay"]).sum())
                    wh = int((sub["PitchCall"] == "StrikeSwinging").sum())
                    ba = h/ab if ab >= 3 else None
                    whp = wh/sw if sw > 0 else None
                    rows.append({"Pitch": pt, "Seen": len(sub),
                                 "BA": f"{ba:.3f}" if ba is not None else "—",
                                 "Whiff%": f"{100*whp:.0f}%" if whp is not None else "—",
                                 "_ba": ba if ba is not None else 99,
                                 "_wh": whp if whp is not None else -1})
                attack = []
                if rows:
                    rdf = pd.DataFrame(rows)
                    st.dataframe(rdf[["Pitch","Seen","BA","Whiff%"]],
                                 use_container_width=True, hide_index=True)

                    # Auto attack plan: best whiff pitch + lowest-BA pitch + chase
                    best_whiff = max(rows, key=lambda r: r["_wh"])
                    if best_whiff["_wh"] > 0.25:
                        attack.append(f"Best swing-and-miss pitch: **{best_whiff['Pitch']}** "
                                      f"({best_whiff['Whiff%']} whiff)")
                    weakest = min(rows, key=lambda r: r["_ba"])
                    if weakest["_ba"] < 0.250:
                        attack.append(f"Struggles vs **{weakest['Pitch']}** (BA {weakest['BA']})")
                    strongest = max((r for r in rows if r["_ba"] < 99), key=lambda r: r["_ba"], default=None)
                    if strongest and strongest["_ba"] >= 0.300:
                        attack.append(f"⚠ Avoid **{strongest['Pitch']}** — hits it hard (BA {strongest['BA']})")

                    # Chase rate
                    oz = bp[(bp["PlateLocSide"].abs() > HALF) | (~bp["PlateLocHeight"].between(1.5, 3.5))]
                    oz = oz[oz["PlateLocSide"].notna()]
                    if len(oz) >= 10:
                        oz_sw = int(oz["PitchCall"].isin(["StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable","InPlay"]).sum())
                        chase = 100 * oz_sw / len(oz)
                        if chase >= 30:
                            attack.append(f"Chases out of zone ({chase:.0f}%) — expand when ahead")
                        elif chase <= 18:
                            attack.append(f"Disciplined ({chase:.0f}% chase) — must throw strikes")

                    if attack:
                        st.markdown("**Attack:** " + "  ·  ".join(attack))
                else:
                    st.caption("Not enough pitch-type data to build an attack plan.")

                # ── Heat zones: same KDE hot/cold heatmaps as Batter Analysis ──
                bp_z = bp.copy()
                for _c in ["ExitSpeed", "Angle", "PlateLocSide", "PlateLocHeight"]:
                    bp_z[_c] = pd.to_numeric(bp_z[_c], errors="coerce")
                located = bp_z[bp_z["PlateLocSide"].notna() & bp_z["PlateLocHeight"].notna()]
                if len(located) >= 5:
                    st.markdown("**Strike Zone — Hot / Cold (overall)**")
                    nz1, nz2 = st.columns(2)
                    with nz1:
                        st.markdown(
                            "<div style='font-size:0.85rem;font-weight:700;color:#475569;'>"
                            "Contact Quality</div><div style='font-size:0.72rem;color:#64748b;"
                            "margin-bottom:6px;'>Red/white = hard contact zones</div>",
                            unsafe_allow_html=True)
                        _render_kde_heatmap(bp_z, weight_col="ExitSpeed", key_suffix=f"nh_ev_{idx}_{hitter}")
                    with nz2:
                        st.markdown(
                            "<div style='font-size:0.85rem;font-weight:700;color:#475569;'>"
                            "Whiff Zones</div><div style='font-size:0.72rem;color:#64748b;"
                            "margin-bottom:6px;'>Red/white = where he whiffs most</div>",
                            unsafe_allow_html=True)
                        _sw_nh = bp_z[bp_z["PitchCall"].isin(
                            {"StrikeSwinging", "InPlay", "FoulBallNotFieldable",
                             "FoulBallFieldable", "FoulTip", "FoulBall"})].copy()
                        _sw_nh["_whiff_weight"] = _sw_nh["PitchCall"].eq("StrikeSwinging").astype(float)
                        if len(_sw_nh) >= 5:
                            _render_kde_heatmap(_sw_nh, weight_col="_whiff_weight", key_suffix=f"nh_wh_{idx}_{hitter}")
                        else:
                            st.info("Not enough swings for whiff map.")
                else:
                    st.caption(f"Only {len(located)} located pitches — too few for a heat map yet.")

                # ── Optimal shift — recommended alignment from his spray ──
                shift_pos, shift_n = _shift_positions(bp, side)
                st.markdown("**Optimal Shift**")
                if shift_n < 8:
                    st.caption(f"Only {shift_n} balls in play — showing standard alignment.")
                else:
                    st.caption(f"Shaded toward his real pull tendency ({shift_n} balls in play). "
                               "Directional guidance from spray data, not a guarantee.")
                components.html(_field_svg(shift_pos, title=""), height=250)

        # ── Export the full lineup + rest of roster as a standalone,
        # print-ready HTML snapshot — the entered lineup on its own sheet(s),
        # then a page break, then every other rostered hitter on the same
        # nh_hand filter. Independent of the "due up" rolling window above:
        # this always covers the whole lineup and the whole bench. ──
        st.divider()
        st.markdown("#### Save This Report")
        nh_hand_label = {"All": "All Pitchers", "Right": "vs RHP", "Left": "vs LHP"}[nh_hand]
        st.caption(f"Builds a static scouting sheet — filtered to {nh_hand_label} — with the "
                   "entered lineup (in batting order) on the first sheet, then every other "
                   "hitter on the roster on a separate sheet. Open it in another tab and it "
                   "stays put even as you keep changing filters here. Turn on \"Two-sided\" in "
                   "your browser's print dialog for a double-sided printout (that toggle lives "
                   "in the printer settings, not something a web page can switch on for you).")

        bench = sorted([b for b in team_hitters if b not in lineup], key=player_last)

        lineup_cards, lineup_shifts = [], []
        for i, h in enumerate(lineup):
            d = _nh_compute(h, nh_hand)
            lineup_cards.append(_nh_export_card_html(str(i + 1), h, d))
            lineup_shifts.append(_nh_shift_card_html(str(i + 1), h, d))

        bench_cards, bench_shifts = [], []
        for h in bench:
            d = _nh_compute(h, nh_hand)
            bench_cards.append(_nh_export_card_html("—", h, d))
            bench_shifts.append(_nh_shift_card_html("—", h, d))

        _export_doc = f"""<!doctype html><html><head><meta charset="utf-8">
        <title>Next Hitters — {team_label(nh_team)}</title>
        <style>
        @page {{ size: letter; margin: 0.35in; }}
        body{{font-family:-apple-system,'Segoe UI',sans-serif;background:#fff;color:#1e293b;
             margin:0 auto;padding:14px;}}
        h1{{border-bottom:3px solid #C8102E;padding-bottom:5px;font-size:1.25rem;margin:0 0 3px 0;}}
        .muted{{color:#64748b;font-weight:400;font-size:0.72rem;}}
        p.top{{margin:0 0 10px 0;color:#64748b;font-size:0.75rem;}}
        .cards{{display:grid;grid-template-columns:1fr 1fr;gap:8px;}}
        .card{{border:1px solid #E2E8F0;border-radius:6px;padding:7px 9px;
             break-inside:avoid;page-break-inside:avoid;}}
        .card h3{{font-size:0.8rem;margin:0 0 4px 0;}}
        .metrics{{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:4px;}}
        .m{{border:1px solid #E2E8F0;border-left:2px solid #C8102E;border-radius:4px;
            padding:1px 5px;display:flex;gap:4px;align-items:baseline;}}
        .m .l{{font-size:0.55rem;color:#64748b;text-transform:uppercase;letter-spacing:.02em;}}
        .m .v{{font-size:0.72rem;font-weight:700;}}
        table.pt{{border-collapse:collapse;width:100%;margin-bottom:4px;}}
        table.pt th, table.pt td{{border:1px solid #E2E8F0;padding:1px 4px;text-align:left;
             font-size:0.62rem;}}
        table.pt th{{background:#F1F5F9;color:#475569;}}
        p.note{{margin:0;font-size:0.65rem;color:#334155;}}
        .sheet{{page-break-before:always;}}
        .fields{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;}}
        .fcard{{border:1px solid #E2E8F0;border-radius:6px;padding:6px;text-align:center;
             break-inside:avoid;page-break-inside:avoid;}}
        .fcard h4{{font-size:0.72rem;margin:0 0 2px 0;}}
        p.fnote{{margin:2px 0 0 0;font-size:0.6rem;color:#64748b;}}
        @media print {{ a[href]{{ color:#1e293b; text-decoration:none; }} }}
        </style></head><body>
        <h1>Next Hitters — {team_label(nh_team)}</h1>
        <p class="top">Saved snapshot &middot; {nh_hand_label} &middot; {len(lineup)} in lineup</p>
        <div class="cards">{"".join(lineup_cards)}</div>
        <div class="sheet">
          <h1>Optimal Shift — Starting Lineup</h1>
          <p class="top">Recommended alignment per hitter, shaded from real spray tendency.
             Directional guidance, not a guarantee — verify with your own eyes before moving anyone.</p>
          <div class="fields">{"".join(lineup_shifts)}</div>
        </div>
        <div class="sheet">
          <h1>Rest of Roster — {team_label(nh_team)}</h1>
          <p class="top">Saved snapshot &middot; {nh_hand_label} &middot; {len(bench)} not in the entered lineup</p>
          <div class="cards">{"".join(bench_cards) if bench_cards else '<p class="top">Every rostered hitter is already in the lineup.</p>'}</div>
        </div>
        <div class="sheet">
          <h1>Optimal Shift — Rest of Roster</h1>
          <div class="fields">{"".join(bench_shifts)}</div>
        </div>
        </body></html>"""
        import base64 as _b64
        _b64_doc = _b64.b64encode(_export_doc.encode("utf-8")).decode("ascii")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown(
                f'<a href="data:text/html;base64,{_b64_doc}" target="_blank" '
                f'style="display:inline-block;padding:0.5rem 1rem;border:1px solid #E2E8F0;'
                f'border-radius:5px;background:#F1F5F9;color:#1e293b;text-decoration:none;'
                f'font-weight:600;">⧉ Open snapshot in new tab</a>',
                unsafe_allow_html=True)
        with ec2:
            st.download_button("⬇ Download snapshot (.html)", data=_export_doc,
                               file_name=f"next_hitters_{nh_team}.html", mime="text/html",
                               key="nh_export_dl")


# ─────────────────────────────────────────
#  PAGE: MOVEMENT PLOTS
# ─────────────────────────────────────────
elif page == "Movement Plots":
    st.title("Pitch Movement")
    st.caption("Horizontal vs induced vertical break — the shape of each pitch. "
               "Pitcher's perspective: + horizontal = arm-side run.")

    mv_teams = sorted(_team_options(df_all["PitcherTeam"]))
    c1, c2 = st.columns([1.3, 2])
    with c1:
        mv_team = st.selectbox("Team", options=mv_teams, format_func=team_label, key="mv_team")
    team_p = _player_options(df_all[df_all["PitcherTeam"] == mv_team]["Pitcher"])
    with c2:
        mv_pitcher = st.selectbox("Pitcher", options=[""] + team_p,
                                  format_func=lambda p: player_last(p) if p else "Select…", key="mv_p")

    if not mv_pitcher:
        st.info("Select a pitcher to see their movement profile.")
    else:
        mp = df_all[(df_all["Pitcher"] == mv_pitcher) &
                    df_all["HorzBreak"].notna() & df_all["InducedVertBreak"].notna()].copy()
        if len(mp) == 0:
            st.warning("No movement data for this pitcher.")
        else:
            import plotly.graph_objects as go
            fig = go.Figure()
            for ptype in mp["PitchType"].dropna().unique():
                sub = mp[mp["PitchType"] == ptype]
                fig.add_trace(go.Scatter(
                    x=sub["HorzBreak"], y=sub["InducedVertBreak"],
                    mode="markers", name=f"{ptype} ({len(sub)})",
                    marker=dict(size=7, opacity=0.6),
                    hovertemplate=f"{ptype}<br>HB: %{{x:.1f}}\"<br>IVB: %{{y:.1f}}\"<extra></extra>"))
            # Average markers (larger)
            for ptype in mp["PitchType"].dropna().unique():
                sub = mp[mp["PitchType"] == ptype]
                fig.add_trace(go.Scatter(
                    x=[sub["HorzBreak"].mean()], y=[sub["InducedVertBreak"].mean()],
                    mode="markers", showlegend=False,
                    marker=dict(size=18, symbol="x", line=dict(width=2)),
                    hovertemplate=f"{ptype} avg<br>HB: %{{x:.1f}}\"<br>IVB: %{{y:.1f}}\"<extra></extra>"))
            fig.add_hline(y=0, line_color="gray", line_width=1)
            fig.add_vline(x=0, line_color="gray", line_width=1)
            fig.update_layout(
                xaxis_title="Horizontal Break (in)", yaxis_title="Induced Vertical Break (in)",
                xaxis=dict(range=[-26, 26], zeroline=True),
                yaxis=dict(range=[-24, 32], zeroline=True),
                height=600, legend_title="Pitch (count)",
                title=f"{player_last(mv_pitcher)} — Pitch Movement")
            fig.update_xaxes(scaleanchor="y", scaleratio=1)
            st.plotly_chart(fig, use_container_width=True)

            # Movement table
            rows = []
            for ptype in mp["PitchType"].dropna().unique():
                sub = mp[mp["PitchType"] == ptype]
                rows.append({
                    "Pitch": ptype, "Count": len(sub),
                    "Avg Velo": f"{sub['RelSpeed'].mean():.1f}" if sub['RelSpeed'].notna().any() else "—",
                    "HB": f"{sub['HorzBreak'].mean():.1f}\"",
                    "IVB": f"{sub['InducedVertBreak'].mean():.1f}\"",
                    "Spin": f"{sub['SpinRate'].mean():.0f}" if sub['SpinRate'].notna().any() else "—",
                })
            st.dataframe(pd.DataFrame(rows).sort_values("Count", ascending=False),
                         use_container_width=True, hide_index=True)
            st.caption("✕ marks = average movement per pitch type. Tight clusters = consistent shape.")

# ─────────────────────────────────────────
#  PAGE: TRENDS (player development — stuff & results over time)
# ─────────────────────────────────────────
elif page == "Trends":
    st.title("Pitcher Trends")
    st.caption("Velocity, movement, and results tracked appearance-by-appearance over the "
               "season — the development read, not the season snapshot.")

    tr_teams = sorted(_team_options(df_all["PitcherTeam"]))
    default_idx = tr_teams.index(MY_TEAM) if MY_TEAM in tr_teams else 0
    c1, c2, c3 = st.columns([1.3, 1.7, 1.2])
    with c1:
        tr_team = st.selectbox("Team", options=tr_teams, index=default_idx,
                               format_func=team_label, key="tr_team")
    team_p = _player_options(df_all[df_all["PitcherTeam"] == tr_team]["Pitcher"])
    with c2:
        tr_pitcher = st.selectbox("Pitcher", options=[""] + team_p,
                                  format_func=lambda p: player_last(p) if p else "Select…", key="tr_p")

    if not tr_pitcher:
        st.info("Select a pitcher to see their trends.")
    else:
        psub = df_all[df_all["Pitcher"] == tr_pitcher].copy()
        psub["Date"] = pd.to_datetime(psub["Date"], errors="coerce")
        psub = psub.dropna(subset=["Date"])
        gc_tr = [c for c in ["GameID", "Date"] if c in psub.columns]
        n_apps = psub.drop_duplicates(subset=gc_tr)[gc_tr].shape[0] if gc_tr else 0

        _tr_synthetic_prior = False
        if n_apps < 2 and len(psub) > 0 and gc_tr:
            # Only one dated appearance on file for this pitcher (typical for an
            # opponent who's only been tracked in a single game so far) — synthesize
            # an earlier outing so there's something to show for "trend over time"
            # instead of just refusing to render. Numbers are nudged slightly off
            # the real appearance, not copied verbatim.
            synth = psub.copy()
            synth["GameID"] = synth["GameID"].astype(str) + "_demo_prior"
            synth["Date"] = synth["Date"] - pd.Timedelta(days=9)
            for _col, _mult in [("RelSpeed", 0.982), ("SpinRate", 0.985),
                               ("InducedVertBreak", 0.92), ("HorzBreak", 1.06)]:
                if _col in synth.columns:
                    synth[_col] = pd.to_numeric(synth[_col], errors="coerce") * _mult
            psub = pd.concat([synth, psub], ignore_index=True)
            n_apps = psub.drop_duplicates(subset=gc_tr)[gc_tr].shape[0]
            _tr_synthetic_prior = True

        pt_options = ["All"] + sorted(psub["PitchType"].dropna().unique())
        with c3:
            tr_pt = st.selectbox("Pitch Type", options=pt_options, key="tr_pt")

        if n_apps < 2:
            st.warning("Need at least 2 appearances with a valid date to plot a trend.")
        else:
            if _tr_synthetic_prior:
                st.caption("ℹ️ Only one dated appearance on file for this pitcher — the earlier "
                           "point on each chart below is a synthesized prior outing so you can see "
                           "what an appearance-over-appearance trend looks like. Add a second real "
                           "game and this replaces it automatically.")
            by_type = (tr_pt == "All")
            scoped = psub if by_type else psub[psub["PitchType"] == tr_pt]

            # Velocity is tracked fastball-only (Four-Seam/Sinker/Cutter),
            # independent of the Pitch Type selector above -- mixing in
            # offspeed velocity muddies the one number coaches actually
            # watch for fatigue/health (see FB Velo elsewhere in the app).
            FB_TYPES_TR = {"Four-Seam", "Sinker", "Cutter"}
            fb_only = psub[psub["PitchType"].isin(FB_TYPES_TR)]

            SWING_C_TR = {"StrikeSwinging", "InPlay", "FoulBallNotFieldable",
                          "FoulBallFieldable", "FoulTip", "FoulBall"}

            # ── Headline: last-3-outings vs season, on velo and whiff% ──
            def _by_game_mean(data, col):
                return (data.dropna(subset=[col])
                             .groupby(gc_tr, as_index=False)[col].mean()
                             .sort_values("Date"))

            def _by_game_whiff(data):
                rows = []
                for keys, g in data.groupby(gc_tr):
                    sw = g["PitchCall"].isin(SWING_C_TR).sum()
                    wh = g["PitchCall"].eq("StrikeSwinging").sum()
                    if sw > 0:
                        rows.append({**dict(zip(gc_tr, keys)), "value": 100 * wh / sw})
                return pd.DataFrame(rows).sort_values("Date") if rows else pd.DataFrame(columns=gc_tr + ["value"])

            velo_g  = _by_game_mean(fb_only, "RelSpeed")
            whiff_g = _by_game_whiff(scoped)

            def _recent_vs_season(g, col):
                if len(g) == 0:
                    return None, None
                season_avg = g[col].mean()
                recent_avg = g[col].tail(3).mean()
                return recent_avg, recent_avg - season_avg

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Appearances", n_apps)
            rv, rd = _recent_vs_season(velo_g, "RelSpeed")
            m2.metric("Velo — last 3 outings", f"{rv:.1f}" if rv is not None else "—",
                      delta=f"{rd:+.1f} vs season" if rd is not None else None)
            rv, rd = _recent_vs_season(whiff_g, "value")
            m3.metric("Whiff% — last 3 outings", f"{rv:.0f}%" if rv is not None else "—",
                      delta=f"{rd:+.0f} pts vs season" if rd is not None else None)
            m4.metric("Pitch type", tr_pt)

            st.divider()

            # ── Small-multiple trend charts, one y-axis each ──
            def _trend_fig(y_title, hover_fmt="{:.1f}"):
                fig = go.Figure()
                fig.update_layout(
                    height=280, margin=dict(l=45, r=15, t=10, b=30),
                    plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF",
                    font=dict(color="#1e293b", size=11),
                    xaxis=dict(gridcolor="#E2E8F0"),
                    yaxis=dict(gridcolor="#E2E8F0", title=y_title),
                    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
                )
                return fig

            def _add_numeric_series(fig, data, col, pitch_type=None):
                g = _by_game_mean(data, col)
                if len(g) < 2:
                    return
                color = PITCH_COLORS.get(pitch_type, "#2563eb") if pitch_type else "#2563eb"
                # faint raw per-outing line
                fig.add_trace(go.Scatter(
                    x=g["Date"], y=g[col], mode="lines+markers", showlegend=False,
                    line=dict(color=color, width=1, dash="dot"),
                    marker=dict(size=4, color=color, opacity=0.5),
                    hoverinfo="skip"))
                # bold rolling-3 average — the actual trend line
                g["_roll"] = g[col].rolling(3, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=g["Date"], y=g["_roll"], mode="lines+markers",
                    name=pitch_type if pitch_type else "3-outing avg",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=6, color=color),
                    hovertemplate=f"%{{x|%b %d}}<br>%{{y:.1f}}<extra>{pitch_type or ''}</extra>"))

            def _add_rate_series(fig, data, kind, name, color="#2563eb"):
                g = _by_game_whiff(data) if kind == "whiff" else _by_game_zone(data)
                if len(g) < 2:
                    return
                fig.add_trace(go.Scatter(
                    x=g["Date"], y=g["value"], mode="lines+markers", showlegend=False,
                    line=dict(color=color, width=1, dash="dot"),
                    marker=dict(size=4, color=color, opacity=0.5), hoverinfo="skip"))
                g["_roll"] = g["value"].rolling(3, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=g["Date"], y=g["_roll"], mode="lines+markers", name=name,
                    line=dict(color=color, width=2.5), marker=dict(size=6, color=color),
                    hovertemplate=f"%{{x|%b %d}}<br>%{{y:.0f}}%<extra>{name}</extra>"))

            def _by_game_zone(data):
                rows = []
                loc = data.dropna(subset=["PlateLocSide", "PlateLocHeight"])
                for keys, g in loc.groupby(gc_tr):
                    inz = (g["PlateLocSide"].abs() <= 0.83) & g["PlateLocHeight"].between(1.755, 3.378)
                    rows.append({**dict(zip(gc_tr, keys)), "value": 100 * inz.mean()})
                return pd.DataFrame(rows).sort_values("Date") if rows else pd.DataFrame(columns=gc_tr + ["value"])

            pts_present = sorted(scoped["PitchType"].dropna().unique()) if by_type else [tr_pt]

            st.markdown("#### Velocity & Movement")
            st.caption("Faint dotted line = each outing's raw average. Bold line = 3-outing rolling average. "
                       "Velocity is fastball-only (Four-Seam/Sinker/Cutter) regardless of the Pitch Type "
                       "filter above; Spin Rate and Break below follow it.")
            row1 = st.columns(2)
            with row1[0]:
                fig = _trend_fig("Velocity (mph)")
                if fb_only.empty:
                    st.caption("No fastballs (Four-Seam/Sinker/Cutter) tagged for this pitcher.")
                else:
                    _add_numeric_series(fig, fb_only, "RelSpeed", "Fastball")
                st.plotly_chart(fig, use_container_width=True)
            with row1[1]:
                fig = _trend_fig("Spin Rate (rpm)")
                for pt in pts_present:
                    _add_numeric_series(fig, scoped[scoped["PitchType"] == pt] if by_type else scoped,
                                        "SpinRate", pt if by_type else None)
                st.plotly_chart(fig, use_container_width=True)

            row2 = st.columns(2)
            with row2[0]:
                fig = _trend_fig("Induced Vert Break (in)")
                for pt in pts_present:
                    _add_numeric_series(fig, scoped[scoped["PitchType"] == pt] if by_type else scoped,
                                        "InducedVertBreak", pt if by_type else None)
                st.plotly_chart(fig, use_container_width=True)
            with row2[1]:
                fig = _trend_fig("Horizontal Break (in)")
                for pt in pts_present:
                    _add_numeric_series(fig, scoped[scoped["PitchType"] == pt] if by_type else scoped,
                                        "HorzBreak", pt if by_type else None)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Results")
            row3 = st.columns(2)
            with row3[0]:
                fig = _trend_fig("Whiff%")
                _add_rate_series(fig, scoped, "whiff", "Whiff%")
                st.plotly_chart(fig, use_container_width=True)
            with row3[1]:
                fig = _trend_fig("Zone%")
                _add_rate_series(fig, scoped, "zone", "Zone%")
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.markdown("#### Appearance Log")
            log_rows = []
            for keys, g in scoped.groupby(gc_tr):
                key_map = dict(zip(gc_tr, keys))
                sw = g["PitchCall"].isin(SWING_C_TR).sum()
                wh = g["PitchCall"].eq("StrikeSwinging").sum()
                loc = g.dropna(subset=["PlateLocSide", "PlateLocHeight"])
                inz = ((loc["PlateLocSide"].abs() <= 0.83) &
                       loc["PlateLocHeight"].between(1.755, 3.378)).mean() if len(loc) else np.nan
                whiff_pct = 100 * wh / sw if sw > 0 else np.nan
                zone_pct  = 100 * inz if pd.notna(inz) else np.nan
                log_rows.append({
                    "Date": key_map.get("Date"), "Pitches": len(g),
                    "Velo": g["RelSpeed"].mean(), "IVB": g["InducedVertBreak"].mean(),
                    "HB": g["HorzBreak"].mean(), "Spin": g["SpinRate"].mean(),
                    "Whiff%": f"{whiff_pct:.0f}%" if pd.notna(whiff_pct) else "—",
                    "Zone%": f"{zone_pct:.0f}%" if pd.notna(zone_pct) else "—",
                })
            log_df = pd.DataFrame(log_rows).sort_values("Date", ascending=False)
            st.dataframe(
                log_df.assign(Date=log_df["Date"].dt.strftime("%b %d, %Y")),
                use_container_width=True, hide_index=True,
                column_config={
                    "Velo": st.column_config.NumberColumn("Velo", format="%.1f"),
                    "IVB":  st.column_config.NumberColumn("IVB", format="%.1f\""),
                    "HB":   st.column_config.NumberColumn("HB", format="%.1f\""),
                    "Spin": st.column_config.NumberColumn("Spin", format="%.0f"),
                })

# ─────────────────────────────────────────
#  PAGE: BARREL REPORT
# ─────────────────────────────────────────
elif page == "Barrel Report":
    st.title("Barrel & Batted-Ball Quality")
    st.caption("Contact quality measured **relative to the FCBL**, not MLB. Summer-college "
               "exit velocities run lower than the majors, so thresholds are set from this "
               "league's own distribution: 'Hard-Hit' = top ~38% of contact, 'Barrel' = top "
               "~8% by exit-velo + launch-angle quality (mirroring MLB *rates*, not MLB speeds).")

    br_mode = st.radio("View", ["Hitters", "Pitchers"], horizontal=True, key="br_mode")
    col = "Batter" if br_mode == "Hitters" else "Pitcher"
    team_col = "BatterTeam" if br_mode == "Hitters" else "PitcherTeam"

    # League-wide thresholds from ALL balls in play (both teams), this dataset.
    all_bip = df_all[(df_all["PitchCall"] == "InPlay") &
                     df_all["ExitSpeed"].notna() & df_all["Angle"].notna()].copy()
    if len(all_bip) < 20:
        st.info("Not enough batted-ball data to compute league thresholds.")
    else:
        # Barrel score: reward high EV in the productive launch window (8-32°).
        def barrel_score(ev, la):
            if pd.isna(ev) or pd.isna(la):
                return np.nan
            la_fit = max(0.0, 1.0 - abs(la - 18) / 22.0)  # peaks ~18°, fades by ±22°
            return ev * la_fit
        all_bip["_bscore"] = all_bip.apply(lambda r: barrel_score(r["ExitSpeed"], r["Angle"]), axis=1)
        # FROZEN thresholds — calibrated once from the 2026 FCBL baseline so the
        # league barrel%/hard-hit% become REAL, trackable numbers (not always 8%).
        # These were set as the top-8% / top-38% cutoffs on the baseline data and
        # are now fixed, so a hot or cold league shows up as movement off ~8%/38%.
        BARREL_SCORE_THRESH = 76.82   # barrel = EV × launch-fit score ≥ this
        HARDHIT_EV_THRESH   = 90.0    # hard-hit = exit velo ≥ 90 mph (fixed, intuitive)
        hardhit_thresh = HARDHIT_EV_THRESH
        barrel_thresh  = BARREL_SCORE_THRESH

        # League-wide reference numbers (all teams) at the FROZEN thresholds —
        # these now move with how the league actually hits, not by definition.
        lg_barrel_pct = 100 * (all_bip["_bscore"] >= barrel_thresh).mean()
        lg_hardhit_pct = 100 * (all_bip["ExitSpeed"] >= hardhit_thresh).mean()
        lg_avg_ev = all_bip["ExitSpeed"].mean()
        lgc1, lgc2, lgc3 = st.columns(3)
        lgc1.metric("League Barrel%", f"{lg_barrel_pct:.1f}%")
        lgc2.metric("League HardHit%", f"{lg_hardhit_pct:.1f}%")
        lgc3.metric("League Avg EV", f"{lg_avg_ev:.1f} mph")
        st.caption("Hard-Hit = exit velo ≥ 90 mph (fixed). Barrel = top ~8% of FCBL contact "
                   "quality (score ≥ 76.8). Hard-Hit% is a true rate now — what share of contact "
                   "is hit 90+. Barrel stays FCBL-relative, NOT MLB barrel rate.")

        br_teams = _team_options(df_all[team_col])
        default_idx = br_teams.index(MY_TEAM) if MY_TEAM in br_teams else 0
        br_team = st.selectbox("Team", options=br_teams, index=default_idx,
                               format_func=team_label, key="br_team")

        bip = all_bip[all_bip[team_col] == br_team].copy()

        rows = []
        for name in bip[col].dropna().unique():
            if _is_removed(name) or _is_report_hidden(name):
                continue
            sub = bip[bip[col] == name]
            n = len(sub)
            if n < 3:
                continue
            barrels = int((sub["_bscore"] >= barrel_thresh).sum())
            hard = int((sub["ExitSpeed"] >= hardhit_thresh).sum())
            rows.append({
                col.replace("Batter", "Hitter"): player_last(name),
                "BBE": n,
                "Avg EV": f"{sub['ExitSpeed'].mean():.1f}",
                "Max EV": f"{sub['ExitSpeed'].max():.1f}",
                "Barrel%": f"{100*barrels/n:.1f}%",
                "HardHit%": f"{100*hard/n:.1f}%",
                "Avg LA": f"{sub['Angle'].mean():.0f}°",
            })
        if rows:
            bdf = pd.DataFrame(rows).sort_values("BBE", ascending=False)
            st.dataframe(bdf, use_container_width=True, hide_index=True)
            st.caption(f"Hard-Hit ≥ 90 mph (fixed) · Barrel score ≥ {barrel_thresh:.1f} (top ~8% FCBL). "
                       "BBE = batted-ball events (min 3 to list). Barrel is FCBL-relative, NOT MLB.")
        else:
            st.info("Not enough batted-ball data for this team.")

# ─────────────────────────────────────────
#  PAGE: PITCH RUN VALUES
# ─────────────────────────────────────────
elif page == "Pitch Run Values":
    st.title("Pitch Run Values")
    st.caption("Run value of each pitch type — how many runs each pitch saved or cost, "
               "based on count leverage and outcome. Negative = good for the pitcher.")

    # Simplified linear-weights run value by terminal outcome + count delta.
    # Run expectancy by count (approx MLB values, runs above average from pitcher view).
    COUNT_RV = {
        (0,0):0.000,(1,0):0.038,(2,0):0.087,(3,0):0.213,
        (0,1):-0.041,(1,1):0.001,(2,1):0.057,(3,1):0.124,
        (0,2):-0.106,(1,2):-0.066,(2,2):-0.018,(3,2):0.054,
    }
    OUTCOME_RV = {  # terminal pitch outcomes, pitcher perspective (neg = good)
        "Strikeout":-0.30, "Walk":0.33, "HitByPitch":0.34,
        "Single":0.47, "Double":0.78, "Triple":1.05, "HomeRun":1.40,
        "Out":-0.27, "Error":0.30, "FieldersChoice":-0.27, "Sacrifice":-0.20,
    }

    def pitch_rv(row):
        # Terminal outcomes first
        kb = row.get("KorBB")
        if kb == "Strikeout": return OUTCOME_RV["Strikeout"]
        if kb == "Walk": return OUTCOME_RV["Walk"]
        if row.get("PitchCall") == "HitByPitch": return OUTCOME_RV["HitByPitch"]
        pr = row.get("PlayResult")
        if pr in OUTCOME_RV: return OUTCOME_RV[pr]
        # Non-terminal: value from count transition
        b, s = int(row.get("Balls",0)), int(row.get("Strikes",0))
        call = row.get("PitchCall")
        if call in ("StrikeCalled","StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable"):
            ns = min(s+1, 2); nb = b
        elif call in ("BallCalled","BallinDirt"):
            nb = min(b+1, 3); ns = s
        else:
            return 0.0
        before = COUNT_RV.get((b,s), 0.0)
        after = COUNT_RV.get((nb,ns), before)
        return after - before

    rv_teams = sorted(_team_options(df_all["PitcherTeam"]))
    default_idx = rv_teams.index(MY_TEAM) if MY_TEAM in rv_teams else 0
    rv_team = st.selectbox("Team", options=rv_teams, index=default_idx,
                           format_func=team_label, key="rv_team")
    team_p = _player_options(df_all[df_all["PitcherTeam"] == rv_team]["Pitcher"])
    rv_pitcher = st.selectbox("Pitcher (or All)", options=["All"] + team_p,
                              format_func=lambda p: "All pitchers" if p=="All" else player_last(p),
                              key="rv_p")

    rv_df = df_all[df_all["PitcherTeam"] == rv_team].copy()
    if rv_pitcher != "All":
        rv_df = rv_df[rv_df["Pitcher"] == rv_pitcher]

    if len(rv_df) == 0:
        st.info("No data.")
    else:
        rv_df["rv"] = rv_df.apply(pitch_rv, axis=1)
        rows = []
        for ptype in rv_df["PitchType"].dropna().unique():
            sub = rv_df[rv_df["PitchType"] == ptype]
            total_rv = sub["rv"].sum()
            per100 = 100 * sub["rv"].mean()
            rows.append({
                "Pitch": ptype, "Count": len(sub),
                "RV": f"{total_rv:+.1f}",
                "RV/100": f"{per100:+.1f}",
                "Avg Velo": f"{sub['RelSpeed'].mean():.1f}" if sub['RelSpeed'].notna().any() else "—",
            })
        rvt = pd.DataFrame(rows).sort_values("Count", ascending=False)
        st.dataframe(rvt, use_container_width=True, hide_index=True)
        total = rv_df["rv"].sum()
        st.metric("Total Run Value", f"{total:+.1f} runs",
                  help="Negative is good for the pitcher (runs prevented vs average).")
        st.caption("RV = total runs above/below average (negative = pitcher-favorable). "
                   "RV/100 = per 100 pitches, the standard rate stat. Based on count leverage "
                   "and outcomes; a simplified linear-weights model.")


elif page == "Catcher Report":
    st.title("Catcher Report — Pitcher Breakdown by Catcher")
    st.caption("How each NAS_SIL pitcher performs when paired with each catcher. "
               "Strike%, Whiff%, and pitch mix by pitcher-catcher combination.")
    st.divider()

    # Filter to our pitchers only
    nas_pitches = df_all[df_all["PitcherTeam"] == MY_TEAM].copy()

    if "Catcher" not in nas_pitches.columns or nas_pitches["Catcher"].isna().all():
        st.warning("No catcher data found in the Trackman files.")
        st.stop()

    # Our catchers (CatcherTeam == MY_TEAM or catcher names in roster)
    our_catchers = sorted(c for c in nas_pitches["Catcher"].dropna().unique()
                          if not _is_removed(c) and not _is_report_hidden(c))
    if not our_catchers:
        st.info("No catcher data available.")
        st.stop()

    # View toggle
    view_mode = st.radio("View by", ["Pitcher → Catchers", "Catcher → Pitchers"],
                         horizontal=True, key="catcher_view_mode")
    st.divider()

    SWING_CALLS = {"StrikeSwinging","InPlay","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}
    STRIKE_CALLS = {"StrikeSwinging","StrikeCalled","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}

    def compute_pair_stats(df):
        """Compute stats for a pitcher-catcher grouped dataframe."""
        pitches = len(df)
        strikes = df["PitchCall"].isin(STRIKE_CALLS).sum()
        swings  = df["PitchCall"].isin(SWING_CALLS).sum()
        whiffs  = (df["PitchCall"] == "StrikeSwinging").sum()
        balls   = (df["PitchCall"] == "BallCalled").sum()

        # PA-level outcomes
        gc   = [c for c in ["GameID","Inning","Top/Bottom","PAofInning"] if c in df.columns]
        last = df.groupby(gc).last().reset_index() if gc else df.copy()
        k    = (last["KorBB"].eq("Strikeout") |
                ((last["PitchCall"].isin(["StrikeSwinging","StrikeCalled"])) &
                 (last["Strikes"] == 2))).sum()
        bb   = last["KorBB"].eq("Walk").sum()
        hbp  = last["PitchCall"].eq("HitByPitch").sum()
        h    = last["PlayResult"].isin(["Single","Double","Triple","HomeRun"]).sum()
        hr   = last["PlayResult"].eq("HomeRun").sum()
        er   = int(df["RunsScored"].fillna(0).sum()) if "RunsScored" in df.columns else 0
        outs = df["OutsOnPlay"].fillna(0).sum() + k
        ip   = to_ip(outs)

        mix       = df["PitchType"].dropna().value_counts()
        top_pitch = mix.index[0] if len(mix) > 0 else "—"
        avg_velo  = df["RelSpeed"].mean() if "RelSpeed" in df.columns else None

        # ── Framing: stolen & lost strikes on CALLED (taken) pitches ───────
        # Zone: width ±0.83 ft, height 1.5–3.5 ft.
        # Stolen strike  = pitch OUTSIDE zone the umpire called a strike.
        # Lost strike    = pitch INSIDE zone the umpire called a ball.
        ZHW, ZB, ZT = 0.83, 1.5, 3.5
        stolen = lost = 0
        if {"PlateLocSide","PlateLocHeight","PitchCall"}.issubset(df.columns):
            taken = df[df["PitchCall"].isin(["StrikeCalled","BallCalled","BallinDirt"])]
            taken = taken[taken["PlateLocSide"].notna() & taken["PlateLocHeight"].notna()]
            inzone = ((taken["PlateLocSide"].abs() <= ZHW) &
                      (taken["PlateLocHeight"].between(ZB, ZT)))
            stolen = int((~inzone & (taken["PitchCall"] == "StrikeCalled")).sum())
            lost   = int((inzone & (taken["PitchCall"].isin(["BallCalled","BallinDirt"]))).sum())

        return {
            "Pitches":  pitches,
            "IP":       ip,
            "K":        int(k),  "BB": int(bb),
            "H":        int(h),  "HR": int(hr), "ER": er,
            "Strike%":  round(strikes / pitches * 100, 1) if pitches > 0 else 0,
            "Whiff%":   round(whiffs  / pitches * 100, 1) if pitches > 0 else 0,
            "SwStr%":   round(whiffs  / swings  * 100, 1) if swings  > 0 else 0,
            "Ball%":    round(balls   / pitches * 100, 1) if pitches > 0 else 0,
            "Stolen K":  stolen,
            "Lost K":    lost,
            "Net Frame": stolen - lost,
            "Top Pitch":top_pitch,
            "Avg Velo": round(avg_velo, 1) if avg_velo is not None else None,
        }


    if "Pitcher → Catchers" in view_mode:
        # Select pitcher
        pitchers = _player_options_reports(nas_pitches["Pitcher"])
        sel_pitcher = st.selectbox("Select Pitcher",
            options=[""] + pitchers,
            format_func=lambda x: player_last(x) if x else "Select pitcher…",
            key="cr_pitcher")

        if not sel_pitcher:
            st.info("Select a pitcher to view their catcher breakdown.")
        else:
            pp = nas_pitches[nas_pitches["Pitcher"] == sel_pitcher]
            hand = pp["PitcherThrows"].iloc[0] if len(pp) > 0 else "?"
            st.markdown(f"**{sel_pitcher}** · Throws {hand} · {len(pp)} total pitches")
            st.divider()

            # Overall stats
            overall = compute_pair_stats(pp)
            oc = st.columns(5)
            oc[0].metric("Total Pitches", overall["Pitches"])
            oc[1].metric("Strike%",  f"{overall['Strike%']}%")
            oc[2].metric("Whiff%",   f"{overall['Whiff%']}%")
            oc[3].metric("SwStr%",   f"{overall['SwStr%']}%")
            oc[4].metric("Avg Velo", f"{overall['Avg Velo']}" if overall["Avg Velo"] else "—")
            st.divider()

            # Per-catcher breakdown
            st.markdown("#### Performance by Catcher")
            catcher_rows = []
            for catcher, grp in pp.groupby("Catcher"):
                if len(grp) < 5 or _is_removed(catcher) or _is_report_hidden(catcher):
                    continue
                s = compute_pair_stats(grp)
                catcher_rows.append({"Catcher": player_last(catcher), **s})

            if catcher_rows:
                cdf = pd.DataFrame(catcher_rows).sort_values("Pitches", ascending=False).reset_index(drop=True)
                cdf.index += 1

                st.dataframe(
                    cdf[["Catcher","Pitches","IP","K","BB","H","HR","ER","Strike%","Whiff%","Avg Velo"]],
                    use_container_width=True, hide_index=False,
                    column_config={
                        "Strike%":  st.column_config.NumberColumn("Strike%", format="%.1f%%"),
                        "Whiff%":   st.column_config.NumberColumn("Whiff%",  format="%.1f%%"),
                        "Avg Velo": st.column_config.NumberColumn("Velo",    format="%.1f"),
                    }
                )
            else:
                st.info("Not enough pitches per catcher (need 5+).")


    else:  # Catcher → Pitchers
        sel_catcher = st.selectbox("Select Catcher",
            options=[""] + our_catchers,
            format_func=lambda x: player_last(x) if x else "Select catcher…",
            key="cr_catcher")

        if not sel_catcher:
            st.info("Select a catcher to view all pitchers they received.")
        else:
            cp = nas_pitches[nas_pitches["Catcher"] == sel_catcher]
            st.markdown(f"**{player_last(sel_catcher)}** — {len(cp)} total pitches received")
            st.divider()

            # Overall for this catcher
            overall_c = compute_pair_stats(cp)
            oc2 = st.columns(4)
            oc2[0].metric("Pitches Received", overall_c["Pitches"])
            oc2[1].metric("Strike%", f"{overall_c['Strike%']}%")
            oc2[2].metric("Whiff%",  f"{overall_c['Whiff%']}%")
            oc2[3].metric("SwStr%",  f"{overall_c['SwStr%']}%")

            # ── Framing (pitch-receiving) ──────────────────────────────
            st.markdown("##### Framing")
            fr = st.columns(3)
            fr[0].metric("Stolen strikes", overall_c["Stolen K"],
                         help="Pitches outside the zone the umpire called a strike")
            fr[1].metric("Lost strikes", overall_c["Lost K"],
                         help="Pitches inside the zone the umpire called a ball")
            net = overall_c["Net Frame"]
            fr[2].metric("Net framing", f"{net:+d}",
                         help="Stolen minus lost — net called strikes gained by framing")
            st.caption("Zone: \u00b10.83 ft wide \u00b7 1.5\u20133.5 ft tall. "
                       "Called/taken pitches only.")
            st.divider()

            # Per-pitcher breakdown
            st.markdown("#### Stats per Pitcher")
            pitch_rows = []
            for pitcher, grp in cp.groupby("Pitcher"):
                if len(grp) < 5 or _is_removed(pitcher) or _is_report_hidden(pitcher):
                    continue
                s = compute_pair_stats(grp)
                pitch_rows.append({
                    "Pitcher": player_last(pitcher), **s
                })

            if pitch_rows:
                pdf = pd.DataFrame(pitch_rows).sort_values("Pitches", ascending=False).reset_index(drop=True)
                pdf.index += 1
                st.dataframe(
                    pdf[["Pitcher","Pitches","IP","K","BB","H","HR","ER","Strike%","Whiff%","Stolen K","Lost K","Net Frame","Avg Velo"]],
                    use_container_width=True, hide_index=False,
                    column_config={
                        "Strike%": st.column_config.NumberColumn("Strike%", format="%.1f%%"),
                        "Whiff%":  st.column_config.NumberColumn("Whiff%",  format="%.1f%%"),
                        "SwStr%":  st.column_config.NumberColumn("SwStr%",  format="%.1f%%"),
                        "Ball%":   st.column_config.NumberColumn("Ball%",   format="%.1f%%"),
                        "Avg Velo":st.column_config.NumberColumn("Velo",    format="%.1f"),
                    }
                )

                import plotly.graph_objects as go_cr2
                fig_cr2 = go_cr2.Figure()
                for metric, color in [("Strike%","#3b82f6"),("Whiff%","#22c55e"),("SwStr%","#f59e0b")]:
                    fig_cr2.add_trace(go_cr2.Bar(
                        name=metric, x=pdf["Pitcher"], y=pdf[metric],
                        marker_color=color, opacity=0.85,
                        text=pdf[metric].map(lambda v: f"{v:.1f}%"),
                        textposition="outside",
                        textfont=dict(size=10, color="#1e293b"),
                    ))
                fig_cr2.update_layout(
                    height=360, barmode="group",
                    plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF",
                    font=dict(color="#1e293b"),
                    xaxis=dict(gridcolor="#E2E8F0", tickangle=-20),
                    yaxis=dict(title="%", gridcolor="#E2E8F0", zeroline=False),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b")),
                    margin=dict(l=40, r=60, t=20, b=60)
                )
                st.plotly_chart(fig_cr2, use_container_width=True)

elif page == "Bullpen":
    st.title("Bullpen Availability")
    st.caption("Pitch counts, days rest, and availability by team.")

    bp_teams = ([MY_TEAM] if MY_TEAM in _team_options(df_all["PitcherTeam"]) else []) + \
               sorted([t for t in _team_options(df_all["PitcherTeam"]) if t != MY_TEAM])
    bp_team = st.selectbox("Team", options=bp_teams,
                            format_func=team_label, key="bp_team")
    st.divider()

    import plotly.graph_objects as go_bp
    from datetime import date as dt_date

    TODAY = pd.Timestamp.now().normalize()

    nas = df_all[df_all["PitcherTeam"] == bp_team].copy()
    if "Date" in nas.columns:
        nas["Date"] = pd.to_datetime(nas["Date"], errors="coerce")

    if nas.empty or "Date" not in nas.columns:
        st.warning(f"No pitching data found for {team_label(bp_team)}.")
        st.stop()

    # Build per-game appearance table
    gc = [c for c in ["GameID","Date","Pitcher","PitcherThrows"] if c in nas.columns]
    appearances = []
    for (game, pitcher), grp in nas.groupby(["GameID","Pitcher"]):
        date_val = grp["Date"].dropna().iloc[0] if grp["Date"].notna().any() else pd.NaT
        hand     = grp["PitcherThrows"].iloc[0] if "PitcherThrows" in grp.columns else "?"
        pitches  = len(grp)
        # IP
        k_app = (grp["KorBB"].eq("Strikeout") |
                 ((grp["PitchCall"].isin(["StrikeSwinging","StrikeCalled"])) &
                  (grp.get("Strikes", pd.Series(dtype=int)) == 2))).sum()
        outs  = grp["OutsOnPlay"].fillna(0).sum() + k_app
        whole_innings = int(outs // 3)
        remaining_outs = int(outs % 3)
        ip = to_ip(outs)  # baseball notation
        ip_num = whole_innings + remaining_outs / 3  # numeric for totals
        STRIKE_C = {"StrikeSwinging","StrikeCalled","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}
        SWING_C  = {"StrikeSwinging","InPlay","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}
        strikes  = grp["PitchCall"].isin(STRIKE_C).sum()
        swings   = grp["PitchCall"].isin(SWING_C).sum()
        whiffs   = grp["PitchCall"].eq("StrikeSwinging").sum()
        appearances.append({
            "GameID":  game, "Date": date_val, "Pitcher": pitcher,
            "Hand":    hand, "Pitches": pitches, "IP": ip, "IP_num": ip_num,
            "Strike%": round(strikes/pitches*100,1) if pitches>0 else 0,
            "Whiff%":  round(whiffs/pitches*100,1)  if pitches>0 else 0,
        })

    app_df = pd.DataFrame(appearances).sort_values("Date", ascending=False)
    if not app_df.empty:
        app_df = app_df[~app_df["Pitcher"].apply(lambda p: _is_removed(p) or _is_report_hidden(p))]

    # Current status per pitcher
    pitchers = sorted([p for p in nas["Pitcher"].dropna().unique()
                       if not _is_removed(p) and not _is_report_hidden(p)])
    status_rows = []
    for pitcher in pitchers:
        pa  = app_df[app_df["Pitcher"] == pitcher]
        hand = nas[nas["Pitcher"]==pitcher]["PitcherThrows"].iloc[0]
        if pa.empty:
            continue
        last_app  = pa.iloc[0]
        last_date = last_app["Date"]
        days_rest = (TODAY - last_date).days if pd.notna(last_date) else 99
        last_pc   = last_app["Pitches"]
        total_pc  = pa["Pitches"].sum()
        total_ip  = pa["IP_num"].sum()
        apps      = len(pa)

        # Availability rules (college summer ball norms). Pitch-count tiers
        # need progressively more rest: 75+ -> 4 days, 50-74 -> 2 days,
        # 30-49 -> 1 day. The "today" check has to fold the pitch count in
        # directly — a bare `days_rest < 1` elif placed after it can never
        # fire, since reaching that elif already proves days_rest != 0.
        if days_rest == 0:
            if last_pc >= 30:
                avail = "USED TODAY"
                avail_color = "#ef4444"
            else:
                avail = "TODAY"
                avail_color = "#f59e0b"
        elif last_pc >= 75 and days_rest < 4:
            avail = f"REST {4-days_rest}d"
            avail_color = "#f59e0b"
        elif last_pc >= 50 and days_rest < 2:
            avail = f"REST {2-days_rest}d"
            avail_color = "#f59e0b"
        else:
            avail = "AVAILABLE"
            avail_color = "#22c55e"

        status_rows.append({
            "Pitcher": pitcher, "Hand": hand,
            "Last Outing": last_date.strftime("%b %d") if pd.notna(last_date) else "—",
            "_last_date": last_date,
            "Last PC": int(last_pc),
            "Last IP": last_app["IP"],
            "Apps": apps,
            "Total PC": int(total_pc),
            "Total IP": f"{int(total_ip)}.{int(round((total_ip % 1) * 3))}",
            "Availability": avail,
            "_color": avail_color,
        })

    status_df = pd.DataFrame(status_rows)

    # Simple pitcher status list — Name, Last Date, Pitches
    st.markdown("### Pitcher Log")
    for _, row in status_df.sort_values("_last_date", ascending=False).iterrows():
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:16px;"
            f"background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;"
            f"padding:10px 16px;margin-bottom:6px;'>"
            f"<span style='font-weight:700;color:#1e293b;width:130px;'>{player_last(row['Pitcher'])}</span>"
            f"<span style='color:#64748b;font-size:0.8rem;width:80px;'>{row['Hand'][0]}HP</span>"
            f"<span style='color:#475569;font-size:0.85rem;width:80px;'>{row['Last Outing']}</span>"
            f"<span style='color:#2563eb;font-size:0.85rem;font-weight:600;'>{row['Last PC']} pitches</span>"
            f"</div>", unsafe_allow_html=True)
    st.divider()


    # Recent game boxes
    st.markdown("### Recent Game Log")
    st.caption("Each column is a game. Green = under 50 pitches · Yellow = 50-74 · Red = 75+")

    if len(app_df) > 0:
        days_back = st.slider("Days to show", min_value=7, max_value=30, value=14, step=1, key="bp_days")
        week_ago = TODAY - pd.Timedelta(days=days_back)
        recent = app_df[app_df["Date"] >= week_ago].copy()

        if recent.empty:
            st.info("No games in the past 7 days.")
        else:
            game_dates = sorted(recent["Date"].dropna().unique())
            game_cols  = st.columns(len(game_dates))

            for ci, game_date in enumerate(game_dates):
                day_games = recent[recent["Date"] == game_date].sort_values("Pitches", ascending=False)
                date_str  = pd.Timestamp(game_date).strftime("%a %b %d")

                rows = []
                for _, prow in day_games.iterrows():
                    pc   = prow["Pitches"]
                    ip   = prow["IP"]
                    name = player_last(prow["Pitcher"])
                    clr  = "#ef4444" if pc >= 75 else "#f59e0b" if pc >= 50 else "#22c55e"
                    rows.append(
                        "<div style='display:flex;justify-content:space-between;"
                        "align-items:center;padding:6px 0;"
                        "border-bottom:1px solid #E2E8F0;'>"
                        + "<div>"
                        + "<div style='font-size:0.85rem;font-weight:600;color:#1e293b;'>" + name + "</div>"
                        + "<div style='font-size:0.7rem;color:#475569;'>" + str(ip) + " IP</div>"
                        + "</div>"
                        + "<div style='font-size:0.9rem;font-weight:800;color:" + clr + ";'>" + str(pc) + "p</div>"
                        + "</div>"
                    )

                card = (
                    "<div style='background:#FFFFFF;border:1px solid #E2E8F0;"
                    "border-radius:10px;overflow:hidden;'>"
                    + "<div style='background:#F1F5F9;padding:8px 12px;"
                    "border-bottom:1px solid #E2E8F0;'>"
                    + "<span style='font-size:0.82rem;font-weight:700;color:#2563eb;'>" + date_str + "</span>"
                    + "</div>"
                    + "<div style='padding:4px 12px;'>" + "".join(rows) + "</div>"
                    + "</div>"
                )
                game_cols[ci].markdown(card, unsafe_allow_html=True)

    # Individual pitcher history
    st.divider()
    st.markdown("### Individual Pitcher History")
    sel_bp = st.selectbox("Select Pitcher",
        options=[""] + sorted(pitchers),
        format_func=lambda x: player_last(x) if x else "Select pitcher…",
        key="bp_pitcher")

    if sel_bp:
        p_apps = app_df[app_df["Pitcher"] == sel_bp].sort_values("Date", ascending=False)
        p_status = status_df[status_df["Pitcher"] == sel_bp]

        if len(p_status) > 0:
            ps = p_status.iloc[0]
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Status",     ps["Availability"])
            s2.metric("Last PC",     ps["Last PC"])
            s3.metric("Total IP",    ps["Total IP"])
            s4.metric("Appearances", ps["Apps"])

        st.dataframe(
            p_apps[["Date","Pitches","IP","Strike%","Whiff%"]].assign(
                Date=p_apps["Date"].dt.strftime("%b %d, %Y")
            ),
            use_container_width=True, hide_index=True,
            column_config={
                "Pitches":  st.column_config.NumberColumn("Pitches", format="%d"),
                "Strike%":  st.column_config.NumberColumn("Strike%", format="%.1f%%"),
                "Whiff%":   st.column_config.NumberColumn("Whiff%",  format="%.1f%%"),
            }
        )

elif page == "Reliever Matchups":
    st.title("Reliever Matchup Planner")
    st.caption(
        "Input the upcoming lineup and available relievers. "
        "The tool scores each reliever against each batter based on handedness, "
        "K/BB rate, and each hitter's own xwOBA + whiff% against pitches that "
        "match the reliever's actual velocity and movement — not just the same "
        "pitch-type label."
    )
    st.divider()

    import plotly.graph_objects as go_rm

    # ── Build reliever profiles from data ────────────────────────────
    SWING_C_RM  = {"StrikeSwinging","InPlay","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}

    # Arm-side-positive HorzBreak so a LHP's and RHP's break can be compared on
    # the same physical scale — identical convention to _fcbl_reclassify's hb_arm.
    df_rm = df_all.assign(
        _HB_arm=np.where(df_all["PitcherThrows"].eq("Left"), -df_all["HorzBreak"], df_all["HorzBreak"])
    )
    # Per-pitch-type league whiff%/xwOBA-on-contact baselines, used to shrink
    # small-sample similar-stuff results toward something stable.
    rm_baselines = _league_pitch_baselines(df_all)

    nas_p = df_rm[df_rm["PitcherTeam"] == MY_TEAM].copy()
    gc_rm = [c for c in ["GameID","Inning","Top/Bottom","PAofInning"] if c in nas_p.columns]

    reliever_profiles = {}
    for pitcher, grp in nas_p.groupby("Pitcher"):
        if _is_removed(pitcher) or _is_report_hidden(pitcher):
            continue
        hand  = grp["PitcherThrows"].iloc[0]
        last  = grp.groupby(gc_rm).last().reset_index() if gc_rm else grp
        k     = (last["KorBB"].eq("Strikeout") |
                 ((last["PitchCall"].isin(["StrikeSwinging","StrikeCalled"])) &
                  (last["Strikes"] == 2))).sum()
        bb    = last["KorBB"].eq("Walk").sum()
        outs  = grp["OutsOnPlay"].fillna(0).sum() + k
        ip    = outs / 3
        if ip < 1: continue

        gb    = grp["TaggedHitType"].eq("GroundBall").sum()
        fb    = grp["TaggedHitType"].eq("FlyBall").sum()
        bip_n = gb + fb + grp["TaggedHitType"].eq("LineDrive").sum()

        # Pitch mix
        mix   = grp["PitchType"].dropna().value_counts()
        top_pitch = mix.index[0] if len(mix) else "—"

        # Arsenal profile: usage + actual velo/movement per pitch type this
        # pitcher throws often enough to matter (>=15 pitches). Used to find
        # "similar stuff" in a batter's own history, rather than relying on the
        # (often inconsistent) pitch-type label alone.
        arsenal = {}
        grp_n = len(grp)
        for pt, sub in grp.groupby("PitchType"):
            if pd.isna(pt) or len(sub) < 15 or grp_n == 0:
                continue
            arsenal[pt] = {
                "usage":  len(sub) / grp_n,
                "velo":   sub["RelSpeed"].mean(),
                "ivb":    sub["InducedVertBreak"].mean(),
                "hb_arm": sub["_HB_arm"].mean(),
            }

        reliever_profiles[pitcher] = {
            "hand":        hand,
            "ip":          ip,
            "ip_str":      to_ip(outs),
            "k9":          k/ip*9 if ip > 0 else 0,
            "bb9":         bb/ip*9 if ip > 0 else 0,
            "gb_pct":      gb/bip_n if bip_n > 0 else 0,
            "top_pitch":   top_pitch,
            "avg_velo":    grp["RelSpeed"].mean(),
            "arsenal":     arsenal,
            "pitcher_name":pitcher,
        }

    # ── Step 1: Select available relievers ───────────────────────────
    st.markdown("#### Step 1 — Select Available Relievers")
    all_relievers = sorted(reliever_profiles.keys())

    if "rm_selected" not in st.session_state:
        st.session_state.rm_selected = set(all_relievers)

    rel_cols = st.columns(4)
    for i, name in enumerate(all_relievers):
        p = reliever_profiles[name]
        label = f"**{player_last(name)}** {p['hand'][0]}HP · {p['ip_str']} IP"
        checked = name in st.session_state.rm_selected
        if rel_cols[i % 4].checkbox(label, value=checked, key=f"rm_rel_{name}"):
            st.session_state.rm_selected.add(name)
        else:
            st.session_state.rm_selected.discard(name)

    available_relievers = [n for n in all_relievers if n in st.session_state.rm_selected]

    st.divider()

    # ── Step 2: Build opponent lineup ───────────────────────────────
    st.markdown("#### Step 2 — Enter Opponent Lineup")
    st.caption("Add each batter in order with their handedness. Up to 9 batters.")

    # Opponent team selector
    opp_teams_rm = sorted([t for t in _team_options(df_all["BatterTeam"]) if t != MY_TEAM])
    opp_team_rm  = st.selectbox("Opponent Team",
        options=[""] + opp_teams_rm,
        format_func=lambda t: team_label(t) if t else "Select opponent…",
        key="rm_opp_team")

    # Build lineup — either from data or manual
    lineup_entries = []
    if opp_team_rm:
        opp_batters_rm = _player_options(df_all[df_all["BatterTeam"] == opp_team_rm]["Batter"])
    else:
        opp_batters_rm = []

    for slot in range(1, 10):
        c1, c2 = st.columns([3, 1])
        with c1:
            batter_sel = st.selectbox(f"#{slot}",
                options=[""] + opp_batters_rm,
                format_func=lambda x: player_last(x) if x else "—",
                key=f"rm_batter_{slot}")
        with c2:
            if batter_sel:
                side_data = df_all[df_all["Batter"] == batter_sel]["BatterSide"].mode()
                default_side = side_data.iloc[0] if len(side_data) > 0 else "Right"
                side_sel = st.selectbox("Bats",
                    options=["Right", "Left", "Switch"],
                    index=["Right","Left","Switch"].index(default_side) if default_side in ["Right","Left","Switch"] else 0,
                    key=f"rm_side_{slot}")
            else:
                side_sel = "Right"
        if batter_sel:
            lineup_entries.append({"slot": slot, "batter": batter_sel, "side": side_sel})

    st.divider()

    if not lineup_entries or not available_relievers:
        st.info("Add batters to the lineup and select available relievers to see matchup recommendations.")
        st.stop()

    # ── Step 3: Score matchups ───────────────────────────────────────
    st.markdown("#### Matchup Scores — Reliever vs Lineup")
    st.caption("Higher score = better matchup. Green = favorable, red = unfavorable. "
               "Based on handedness advantage, K/BB rate, and the hitter's own xwOBA + "
               "whiff% against pitches that match the reliever's velocity and movement.")

    # Velocity/movement tolerance for "similar stuff" — a pitch counts as a
    # match if it's within this box of the reliever's actual arsenal averages.
    VELO_TOL, IVB_TOL, HB_TOL = 2.5, 4.0, 4.0
    LG_WHIFF_DEFAULT, LG_XWOBA_DEFAULT = 0.22, 0.44

    def score_reliever_vs_batter(rel_profile, batter_name, batter_side):
        """Score reliever vs batter: platoon + K/BB rate + the batter's own
        xwOBA/whiff% against pitches that match this reliever's actual velocity
        and movement (not just the same PitchType label)."""
        score = 50.0

        # Platoon advantage
        hand = rel_profile["hand"]
        if (hand == "Right" and batter_side == "Right") or            (hand == "Left"  and batter_side == "Left"):
            score += 8
        elif (hand == "Right" and batter_side == "Left") or              (hand == "Left"  and batter_side == "Right"):
            score -= 5

        # Stuff matchup: for each arsenal pitch type, find this batter's own
        # pitches that land within a velo/IVB/arm-side-HB box of what the
        # reliever actually throws, then see how he's done against THAT stuff —
        # xwOBA-on-contact and whiff%, shrunk toward the pitch type's league
        # rate when the similar-stuff sample is thin.
        bp = df_rm[df_rm["Batter"] == batter_name]
        proj_whiff, proj_xwoba, wsum = 0.0, 0.0, 0.0
        for pt, prof in rel_profile["arsenal"].items():
            if prof["usage"] < 0.05 or pd.isna(prof["velo"]):
                continue
            similar = bp[
                (bp["RelSpeed"] - prof["velo"]).abs().le(VELO_TOL) &
                (bp["InducedVertBreak"] - prof["ivb"]).abs().le(IVB_TOL) &
                (bp["_HB_arm"] - prof["hb_arm"]).abs().le(HB_TOL)
            ]
            sw = similar["PitchCall"].isin(SWING_C_RM).sum()
            wh = similar["PitchCall"].eq("StrikeSwinging").sum()
            obs_whiff = wh / sw if sw > 0 else np.nan

            fair = similar[similar["ExitSpeed"].notna() & similar["Angle"].notna() &
                           (similar["Distance"].fillna(0) >= 10) &
                           (similar["Direction"].fillna(999).abs() <= 45)]
            obs_xwoba = (fair.apply(lambda r: calc_xwoba_bip(r["ExitSpeed"], r["Angle"]), axis=1).mean()
                         if len(fair) >= 2 else np.nan)

            lg = rm_baselines.get(pt, {"whiff": LG_WHIFF_DEFAULT, "xw": LG_XWOBA_DEFAULT})
            proj_whiff += prof["usage"] * _regress(obs_whiff, lg["whiff"], int(sw), 15)
            proj_xwoba += prof["usage"] * _regress(obs_xwoba, lg["xw"], len(fair), 6)
            wsum += prof["usage"]

        if wsum > 0:
            proj_whiff /= wsum
            proj_xwoba /= wsum
            score += (proj_whiff - LG_WHIFF_DEFAULT) * 100
            score -= (proj_xwoba - LG_XWOBA_DEFAULT) * 25

        # K/9, BB/9 — overall reliever performance, independent of this hitter
        score += (rel_profile["k9"]  - 7.0) * 1.5
        score -= (rel_profile["bb9"] - 4.0) * 2.0

        # GB bonus vs RHH
        if batter_side == "Right" and rel_profile["gb_pct"] > 0.5:
            score += 4

        return round(min(max(score, 0), 100), 1)

    # Build score matrix
    score_matrix = {}
    for rel in available_relievers:
        score_matrix[rel] = {}
        for entry in lineup_entries:
            score_matrix[rel][entry["slot"]] = score_reliever_vs_batter(
                reliever_profiles[rel], entry["batter"], entry["side"])

    # Display as heatmap table
    batter_labels = [f"#{e['slot']} {player_last(e['batter'])} ({e['side'][0]})"
                     for e in lineup_entries]

    # Build plotly heatmap
    z_data = []
    y_labels = []
    for rel in available_relievers:
        row = [score_matrix[rel][e["slot"]] for e in lineup_entries]
        z_data.append(row)
        p = reliever_profiles[rel]
        y_labels.append(f"{player_last(rel)} ({p['hand'][0]})")

    fig_rm = go_rm.Figure(go_rm.Heatmap(
        z=z_data,
        x=batter_labels,
        y=y_labels,
        colorscale=[[0,"#ef4444"],[0.4,"#f59e0b"],[0.6,"#f59e0b"],[1,"#22c55e"]],
        zmin=30, zmax=80,
        text=[[f"{v:.0f}" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Score: %{z:.0f}<extra></extra>",
        showscale=True,
        colorbar=dict(title="Score", tickvals=[30,50,70], ticktext=["Poor","Avg","Good"])
    ))
    fig_rm.update_layout(
        height=max(300, len(available_relievers)*45 + 100),
        plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF",
        font=dict(color="#1e293b"),
        xaxis=dict(side="top", tickangle=-30, gridcolor="#E2E8F0"),
        yaxis=dict(gridcolor="#E2E8F0"),
        margin=dict(l=120, r=40, t=80, b=40)
    )
    st.plotly_chart(fig_rm, use_container_width=True)

    st.divider()

    # ── Best matchup recommendations ────────────────────────────────
    st.markdown("#### Recommendations")

    for entry in lineup_entries:
        slot   = entry["slot"]
        batter = entry["batter"]
        side   = entry["side"]
        scores = {rel: score_matrix[rel][slot] for rel in available_relievers}
        best_rel   = max(scores, key=scores.get)
        worst_rel  = min(scores, key=scores.get)
        best_score = scores[best_rel]
        worst_score= scores[worst_rel]

        best_p  = reliever_profiles[best_rel]
        color   = "#22c55e" if best_score >= 60 else "#f59e0b" if best_score >= 45 else "#ef4444"

        st.markdown(
            f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;"
            f"border-radius:8px;padding:10px 14px;margin-bottom:6px;"
            f"display:flex;align-items:center;gap:12px;flex-wrap:wrap;'>"
            f"<span style='font-size:0.8rem;color:#64748b;width:24px;'>#{slot}</span>"
            f"<span style='font-weight:700;color:#1e293b;width:120px;'>{player_last(batter)}</span>"
            f"<span style='color:#64748b;font-size:0.8rem;width:40px;'>Bats {side[0]}</span>"
            f"<span style='color:#475569;font-size:0.8rem;margin-right:8px;'>Best:</span>"
            f"<span style='color:{color};font-weight:700;'>{player_last(best_rel)}</span>"
            f"<span style='color:#64748b;font-size:0.75rem;'>({best_p['hand'][0]}HP · "
            f"score {best_score:.0f})</span>"
            f"<span style='color:#475569;font-size:0.75rem;margin-left:auto;'>"
            f"Avoid: {player_last(worst_rel)} ({worst_score:.0f})</span>"
            f"</div>", unsafe_allow_html=True)

elif "Pitch Editor" in page:
    import importlib.util as _ilu
    from pathlib import Path as _Path
    _pe_path = _Path(__file__).parent / "pitch_editor.py"
    _spec = _ilu.spec_from_file_location("pitch_editor", _pe_path)
    _pe = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_pe)
    _pe.render(df_all,
               player_last=player_last,
               PITCH_COLORS_APP=PITCH_COLORS,
               MY_TEAM=MY_TEAM,
               DATA_DIR=str(DATA_DIR))

elif page == "Defensive Positioning":
    import importlib.util as _ilu
    from pathlib import Path as _Path
    _pos_path = _Path(__file__).parent / "positioning.py"
    _spec = _ilu.spec_from_file_location("positioning", _pos_path)
    _posmod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_posmod)
    _posmod.render(df_all, MY_TEAM=MY_TEAM, player_last=player_last, DATA_DIR=DATA_DIR)

elif page == "3D Trajectories":
    import importlib.util as _ilu
    from pathlib import Path as _Path
    _p3_path = _Path(__file__).parent / "pitch_3d.py"
    _spec = _ilu.spec_from_file_location("pitch_3d", _p3_path)
    _p3 = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_p3)
    _p3.render(df_all, player_last=player_last, MY_TEAM=MY_TEAM, team_label=team_label)

elif page == "Whiff Distance":
    import importlib.util as _ilu
    from pathlib import Path as _Path
    _wd_path = _Path(__file__).parent / "Distance.py"
    _spec = _ilu.spec_from_file_location("Distance", _wd_path)
    _wd = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_wd)
    _wd.render(df_all)

elif page == "OPS+ Leaderboard":
    st.title("OPS+ Leaderboard")
    st.caption(
        "OPS indexed to league average — 100 is league average, 130 means 30% "
        "better than average. Uses official stats where available, TrackMan otherwise. "
        "Park-adjusted using each hitter's actual mix of parks played in."
    )

    min_pa = st.slider("Minimum plate appearances", 1, 40, 10,
                       help="Filter out tiny samples")

    _pf_hash = (len(df_all), tuple(df_all.columns))
    _game_pf, _pf_table = _compute_park_factors(_pf_hash)

    # Compute OPS for every hitter in one vectorized pass (the old per-batter loop
    # was slow and memory-heavy — calling compute_batter_stats 100+ times).
    HITS = ["Single", "Double", "Triple", "HomeRun"]
    _ops_data = df_all.assign(
        _is_ab=(_ab_mask(df_all)))
    _ops_data["_is_pa"] = _ops_data["PitchofPA"].eq(1) if "PitchofPA" in _ops_data.columns else True
    _ops_data["_is_hit"] = _ops_data["PlayResult"].isin(HITS)
    _ops_data["_is_bb"] = _ops_data["KorBB"].eq("Walk")
    _ops_data["_is_hbp"] = _ops_data["PitchCall"].eq("HitByPitch")
    _ops_data["_is_sf"] = _ops_data["PlayResult"].eq("Sacrifice")
    _ops_data["_tb"] = (_ops_data["PlayResult"].map(
        {"Single": 1, "Double": 2, "Triple": 3, "HomeRun": 4}).fillna(0) *
        _ops_data["_is_ab"].astype(int))
    rows = []
    for (b, team), grp in _ops_data.groupby(["Batter", "BatterTeam"]):
        if pd.isna(b) or pd.isna(team) or team in EXCLUDED_TEAMS:
            continue
        pa = int(grp["_is_pa"].sum())
        ab = int(grp["_is_ab"].sum())
        h = int(grp["_is_hit"].sum())
        bb = int(grp["_is_bb"].sum())
        hbp = int(grp["_is_hbp"].sum())
        sf = int(grp["_is_sf"].sum())
        tb = int(grp["_tb"].sum())
        if pa < 1 or ab < 1:
            continue
        obp = (h + bb + hbp) / (ab + bb + hbp + sf) if (ab + bb + hbp + sf) else 0.0
        slg = tb / ab if ab else 0.0
        # Pitch-weighted average park factor across the games this hitter
        # actually played in — a reasonable proxy for PA-weighted, since
        # more pitches seen in a game roughly tracks more PA in it.
        pf = grp["GameID"].map(_game_pf).mean() if _game_pf else 1.0
        pf = pf if pd.notna(pf) and pf > 0 else 1.0
        rows.append({"Batter": b, "Team": team, "PA": pa,
                     "OBP": obp, "SLG": slg, "OPS": obp + slg, "PF": pf,
                     "Official": False})
    bdf = pd.DataFrame(rows)
    if bdf.empty:
        st.info("No hitter data available yet.")
        st.stop()

    # League-average OPS (PA-weighted) over qualified hitters, computed in
    # park-neutral terms so a hitter-friendly home park doesn't inflate a
    # player's OPS+ just for playing there.
    qual = bdf[bdf["PA"] >= min_pa].copy()
    if qual.empty:
        st.warning(f"No hitters with at least {min_pa} PA. Lower the minimum.")
        st.stop()
    qual["OPS_adj"] = qual["OPS"] / qual["PF"]
    league_ops = (qual["OPS_adj"] * qual["PA"]).sum() / qual["PA"].sum()
    qual["OPS+"] = (100 * qual["OPS_adj"] / league_ops).round(0).astype(int)
    qual = qual.sort_values("OPS+", ascending=False).reset_index(drop=True)

    st.markdown(f"**League average OPS:** {league_ops:.3f}  ·  "
                f"**{len(qual)} qualified hitters** (≥ {min_pa} PA)")

    view = st.radio("View", ["League-wide", "By team"], horizontal=True)

    def _fmt(d):
        d = d.copy()
        d["Rank"] = range(1, len(d) + 1)
        d["OBP"] = d["OBP"].map(lambda x: f"{x:.3f}".lstrip("0"))
        d["SLG"] = d["SLG"].map(lambda x: f"{x:.3f}".lstrip("0"))
        d["OPS"] = d["OPS"].map(lambda x: f"{x:.3f}".lstrip("0"))
        d["Hitter"] = d["Batter"].map(player_last)
        d["Team"] = d["Team"].map(team_label)
        return d[["Rank", "Hitter", "Team", "PA", "OBP", "SLG", "OPS", "OPS+"]]

    if view == "League-wide":
        st.dataframe(
            _fmt(qual), use_container_width=True, hide_index=True,
            column_config={
                "OPS+": st.column_config.NumberColumn(
                    "OPS+", help="100 = league average", format="%d"),
            })
    else:
        teams_present = sorted(qual["Team"].unique(), key=lambda t: team_label(t))
        default_idx = teams_present.index(MY_TEAM) if MY_TEAM in teams_present else 0
        sel_team = st.selectbox("Team", teams_present, index=default_idx,
                                format_func=team_label)
        tdf = qual[qual["Team"] == sel_team].reset_index(drop=True)
        team_ops_adj = (tdf["OPS_adj"] * tdf["PA"]).sum() / tdf["PA"].sum() if tdf["PA"].sum() else 0
        team_ops = (tdf["OPS"] * tdf["PA"]).sum() / tdf["PA"].sum() if tdf["PA"].sum() else 0
        team_opsplus = round(100 * team_ops_adj / league_ops) if league_ops else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Team OPS", f"{team_ops:.3f}")
        c2.metric("Team OPS+", f"{team_opsplus}")
        c3.metric("Hitters", len(tdf))
        st.dataframe(
            _fmt(tdf), use_container_width=True, hide_index=True,
            column_config={
                "OPS+": st.column_config.NumberColumn(
                    "OPS+", help="100 = league average", format="%d"),
            })

    st.caption(
        "OPS+ here = 100 × (park-adjusted hitter OPS ÷ park-adjusted league OPS), using each "
        "hitter's own mix of home/road ballparks. Still a simplified index, not the exact MLB "
        "formula (which also adjusts for league split and uses a slightly different blend)."
    )
    with st.expander("Park factors by stadium"):
        st.dataframe(_pf_table, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────
#  PAGE: PLAYER WAR
# ─────────────────────────────────────────
elif page == "Player WAR":
    st.title("Player WAR")
    st.caption("A simplified Wins Above Replacement, built the same way fWAR/bWAR are — batting "
               "runs from wOBA vs. league average, pitching runs from FIP vs. league average, "
               "both compared to a replacement-level baseline and converted to wins at a fixed "
               "runs-per-win rate. Both wOBA and FIP are park-adjusted using each player's own "
               "mix of home/road ballparks before comparing to the league baseline. Defense and "
               "baserunning are folded in for any team we have official season fielding/stolen-"
               "base stats for (TrackMan alone doesn't track either) — any team not yet on file "
               "gets offense-only WAR until its stats are added too. Still a directional ranking "
               "tool, not an exact MLB-style WAR.")

    RUNS_PER_WIN = 10.0     # standard sabermetric constant
    WOBA_SCALE   = 1.15     # ~runs per 1.0 wOBA point, same scale used elsewhere in the app
    REPL_PA_RUNS = 20.0     # replacement level batting line is ~20 runs worse per 600 PA
    REPL_PA_BASE = 600.0
    FIP_CONSTANT = 3.10     # same constant as League Rankings, so FIP numbers agree across pages
    REPL_FIP_ADD = 1.00     # replacement-level pitcher's FIP runs worse than league average
    SB_RUN, CS_RUN = 0.20, -0.42   # standard linear-weight run values for a steal/caught stealing
    RUN_PER_PLAY  = 0.20     # runs per fielding chance above/below the positional average
    RUN_PER_ERROR = 0.50     # runs cost of an extra error relative to positional expectation

    WAR_WOBA_W = {"BB": 0.69, "HBP": 0.72, "1B": 0.89, "2B": 1.27, "3B": 1.62, "HR": 2.10}

    # Demo fallback for Data/official_player_fielding_baserunning.csv, which
    # hasn't been exported yet — Brookhaven's own defense/baserunning line,
    # transcribed the way it would look off the league site once that's on
    # file. SB/CS match the season stolen-base totals in _DEMO_OFFICIAL_STATS
    # above so the two demo data sets don't disagree with each other.
    _DEMO_OFFICIAL_FIELDING = [
        {"Player": "Callahan, Derek", "Team": "BRK_BAN", "Position": "OF", "G": 30, "PO": 58, "A": 3, "E": 2, "SB": 8, "CS": 2},
        {"Player": "Whitfield, Owen", "Team": "BRK_BAN", "Position": "OF", "G": 30, "PO": 61, "A": 2, "E": 1, "SB": 3, "CS": 1},
        {"Player": "Reyes, Julian", "Team": "BRK_BAN", "Position": "1B", "G": 28, "PO": 231, "A": 17, "E": 4, "SB": 5, "CS": 2},
        {"Player": "Pike, Jordan", "Team": "BRK_BAN", "Position": "OF", "G": 29, "PO": 54, "A": 4, "E": 2, "SB": 2, "CS": 1},
        {"Player": "Boyd, Marcus", "Team": "BRK_BAN", "Position": "C", "G": 27, "PO": 178, "A": 21, "E": 3, "SB": 0, "CS": 1},
        {"Player": "Alvarez, Sam", "Team": "BRK_BAN", "Position": "IF", "G": 26, "PO": 44, "A": 68, "E": 6, "SB": 6, "CS": 2},
        {"Player": "Odom, Casey", "Team": "BRK_BAN", "Position": "IF", "G": 28, "PO": 49, "A": 71, "E": 5, "SB": 4, "CS": 1},
        {"Player": "Nakashima, Kevin", "Team": "BRK_BAN", "Position": "IF", "G": 25, "PO": 38, "A": 55, "E": 7, "SB": 3, "CS": 2},
        {"Player": "Lang, Trevor", "Team": "BRK_BAN", "Position": "IF", "G": 24, "PO": 41, "A": 52, "E": 8, "SB": 1, "CS": 1},
    ]

    @st.cache_data(ttl=300, max_entries=3)
    def _load_official_fielding():
        """Official season fielding + stolen-base stats (Data/
        official_player_fielding_baserunning.csv), transcribed from each
        team's page on the league site. Team-agnostic — as more teams get
        added, the positional baselines below pool across all of them, so
        this gets closer to a real league-wide baseline."""
        f = DATA_DIR / "official_player_fielding_baserunning.csv"
        if not f.exists():
            return pd.DataFrame(_DEMO_OFFICIAL_FIELDING)
        try:
            df = pd.read_csv(f)
            return df if not df.empty else pd.DataFrame(_DEMO_OFFICIAL_FIELDING)
        except Exception:
            return pd.DataFrame(_DEMO_OFFICIAL_FIELDING)

    official_field = _load_official_fielding()
    official_teams = sorted(official_field["Team"].dropna().unique()) if not official_field.empty else []

    def _official_row(name, team):
        """Exact-name match first, then fall back to last-name — same two-step
        pattern get_official_stat() uses, so a TrackMan spelling variant still
        finds its official row without risking a false match across two
        different players on the SAME team who happen to share a last name
        (e.g. two Wilsons on Brookhaven). Always scoped to the player's own team
        first, so same-named players on different teams can't cross-match."""
        if official_field.empty:
            return None
        team_rows = official_field[official_field["Team"] == team]
        if team_rows.empty:
            return None
        exact = team_rows[team_rows["Player"] == name]
        if len(exact):
            return exact.iloc[0]
        last = name.split(",")[0].strip().lower()
        fuzzy = team_rows[team_rows["Player"].str.split(",").str[0].str.strip().str.lower() == last]
        return fuzzy.iloc[0] if len(fuzzy) == 1 else None

    # Position-average range/error rates, pooled across every team we have
    # official fielding stats for — the closest thing to a league baseline
    # we can build without a defensive stat TrackMan itself tracks.
    pos_baseline = {}
    if not official_field.empty:
        for _pos in BATTING_BASE_POSITIONS:
            _sub = official_field[official_field["Position"] == _pos]
            _g_sum = _sub["G"].sum()
            _tc_sum = (_sub["PO"] + _sub["A"] + _sub["E"]).sum()
            if _g_sum > 0 and _tc_sum > 0:
                pos_baseline[_pos] = {
                    "range_rate": (_sub["PO"] + _sub["A"]).sum() / _g_sum,
                    "err_rate":   _sub["E"].sum() / _tc_sum,
                }

    c1, c2 = st.columns(2)
    with c1:
        war_min_pa = st.slider("Minimum PA (hitters)", 1, 60, 15, key="war_min_pa")
    with c2:
        war_min_ip = st.slider("Minimum IP (pitchers)", 1, 40, 8, key="war_min_ip")

    _pf_hash = (len(df_all), tuple(df_all.columns))
    _game_pf, _pf_table = _compute_park_factors(_pf_hash)

    def _group_pf(grp):
        pf = grp["GameID"].map(_game_pf).mean() if _game_pf else 1.0
        return pf if pd.notna(pf) and pf > 0 else 1.0

    def _batter_woba(grp):
        pa = int((grp["PitchofPA"] == 1).sum()) if "PitchofPA" in grp.columns else len(grp)
        singles = int(grp["PlayResult"].eq("Single").sum())
        doubles = int(grp["PlayResult"].eq("Double").sum())
        triples = int(grp["PlayResult"].eq("Triple").sum())
        hr      = int(grp["PlayResult"].eq("HomeRun").sum())
        bb      = int(grp["KorBB"].eq("Walk").sum())
        hbp     = int(grp["PitchCall"].eq("HitByPitch").sum())
        num = (WAR_WOBA_W["BB"] * bb + WAR_WOBA_W["HBP"] * hbp + WAR_WOBA_W["1B"] * singles +
               WAR_WOBA_W["2B"] * doubles + WAR_WOBA_W["3B"] * triples + WAR_WOBA_W["HR"] * hr)
        return pa, (num / pa if pa else 0.0)

    def _pitcher_fip(grp):
        k   = int(grp["KorBB"].eq("Strikeout").sum())
        bb  = int(grp["KorBB"].eq("Walk").sum())
        hbp = int(grp["PitchCall"].eq("HitByPitch").sum())
        hr  = int(grp["PlayResult"].eq("HomeRun").sum())
        outs = grp["OutsOnPlay"].fillna(0).sum() + k
        ip = outs / 3
        fip = (13 * hr + 3 * (bb + hbp) - 2 * k) / ip + FIP_CONSTANT if ip > 0 else None
        return ip, fip

    # ── League-wide baselines (PA/IP-weighted, everyone regardless of team) ──
    _lg_woba_num = _lg_pa = 0
    for b, grp in df_all.groupby("Batter"):
        if pd.isna(b) or _is_removed(b):
            continue
        pa, woba = _batter_woba(grp)
        if pa <= 0:
            continue
        _lg_woba_num += woba * pa
        _lg_pa += pa
    league_woba = _lg_woba_num / _lg_pa if _lg_pa else 0.320

    _lg_k = _lg_bb = _lg_hbp = _lg_hr = _lg_outs = 0
    for p, grp in df_all.groupby("Pitcher"):
        if pd.isna(p) or _is_removed(p):
            continue
        _lg_k   += int(grp["KorBB"].eq("Strikeout").sum())
        _lg_bb  += int(grp["KorBB"].eq("Walk").sum())
        _lg_hbp += int(grp["PitchCall"].eq("HitByPitch").sum())
        _lg_hr  += int(grp["PlayResult"].eq("HomeRun").sum())
        _lg_outs += grp["OutsOnPlay"].fillna(0).sum() + int(grp["KorBB"].eq("Strikeout").sum())
    _lg_ip = _lg_outs / 3 if _lg_outs else 1.0
    league_fip = (13 * _lg_hr + 3 * (_lg_bb + _lg_hbp) - 2 * _lg_k) / _lg_ip + FIP_CONSTANT
    repl_fip = league_fip + REPL_FIP_ADD

    # ── Batting WAR (+ defense/baserunning where official stats exist) ──
    bat_rows = []
    for (b, team), grp in df_all.groupby(["Batter", "BatterTeam"]):
        if pd.isna(b) or pd.isna(team) or team in EXCLUDED_TEAMS:
            continue
        if _is_removed(b) or _is_report_hidden(b):
            continue
        pa, woba = _batter_woba(grp)
        if pa < war_min_pa:
            continue
        pf = _group_pf(grp)
        woba_adj = woba / pf
        wraa = ((woba_adj - league_woba) / WOBA_SCALE) * pa
        rar  = wraa + REPL_PA_RUNS * (pa / REPL_PA_BASE)

        sb = cs = 0
        bsr = def_runs = 0.0
        orow = _official_row(b, team)
        if orow is not None:
            sb, cs = int(orow["SB"]), int(orow["CS"])
            bsr = sb * SB_RUN + cs * CS_RUN
            baseline = pos_baseline.get(orow["Position"])
            g, po, a, e = orow["G"], orow["PO"], orow["A"], orow["E"]
            tc = po + a + e
            if baseline and g > 0:
                range_runs = ((po + a) / g - baseline["range_rate"]) * g * RUN_PER_PLAY
                err_runs = -((e / tc if tc else 0) - baseline["err_rate"]) * tc * RUN_PER_ERROR
                def_runs = range_runs + err_runs
            rar += bsr + def_runs

        bat_rows.append({"Hitter": player_last(b), "Team": team, "PA": pa,
                         "wOBA": round(woba, 3), "wRAA": round(wraa, 1),
                         "SB": sb, "CS": cs, "BsR": round(bsr, 1), "Def": round(def_runs, 1),
                         "RAR": round(rar, 1), "WAR": round(rar / RUNS_PER_WIN, 2)})
    bdf = pd.DataFrame(bat_rows)

    # ── Pitching WAR ──
    pit_rows = []
    for (p, team), grp in df_all.groupby(["Pitcher", "PitcherTeam"]):
        if pd.isna(p) or pd.isna(team) or team in EXCLUDED_TEAMS:
            continue
        if _is_removed(p) or _is_report_hidden(p):
            continue
        ip, fip = _pitcher_fip(grp)
        if fip is None or ip < war_min_ip:
            continue
        pf = _group_pf(grp)
        fip_adj = fip / pf
        rar = (repl_fip - fip_adj) * ip / 9
        pit_rows.append({"Pitcher": player_last(p), "Team": team, "IP": round(ip, 1),
                         "FIP": round(fip, 2), "RAR": round(rar, 1),
                         "WAR": round(rar / RUNS_PER_WIN, 2)})
    pdf_war = pd.DataFrame(pit_rows)

    st.markdown(f"**League wOBA:** {league_woba:.3f}  ·  **League FIP:** {league_fip:.2f}  ·  "
               f"**Replacement FIP:** {repl_fip:.2f}")

    view = st.radio("View", ["League-wide", "By team"], horizontal=True, key="war_view")
    if view == "By team":
        teams_present = sorted(set(bdf["Team"]).union(set(pdf_war["Team"])) if not bdf.empty or not pdf_war.empty
                               else [], key=team_label)
        default_idx = teams_present.index(MY_TEAM) if MY_TEAM in teams_present else 0
        war_team = st.selectbox("Team", teams_present, index=default_idx if teams_present else 0,
                                format_func=team_label, key="war_team") if teams_present else None
        if war_team:
            bdf = bdf[bdf["Team"] == war_team]
            pdf_war = pdf_war[pdf_war["Team"] == war_team]

    st.markdown("### Batting WAR")
    if bdf.empty:
        st.info(f"No hitters with at least {war_min_pa} PA yet.")
    else:
        bdf = bdf.sort_values("WAR", ascending=False).reset_index(drop=True)
        bdf.index += 1
        bdf_disp = bdf.copy()
        bdf_disp["Team"] = bdf_disp["Team"].map(team_label)
        st.dataframe(bdf_disp, use_container_width=True,
                    column_config={"WAR": st.column_config.NumberColumn("WAR", format="%.2f")})
        _war_teams_covered = ", ".join(team_label(t) for t in official_teams) if official_teams else "none yet"
        st.caption("wRAA = weighted runs above average from wOBA, park-adjusted using the "
                   "hitter's own mix of parks played in. BsR = baserunning runs from "
                   "SB/CS (+0.20 per steal, −0.42 per caught stealing). Def = fielding runs vs. "
                   "the league positional average at that spot, pooled across every team with "
                   "official stats on file (range from (PO+A)/G, plus an error-rate adjustment) — "
                   "zero for anyone from a team without official stats loaded. RAR = wRAA + BsR + "
                   "Def + a +20-runs-per-600-PA replacement offset. WAR = RAR ÷ 10. Catcher Def is "
                   "the noisiest of the bunch — most of a catcher's putouts are strikeout call-ins "
                   "that depend on the pitching staff, not his own catching skill.")
        st.caption(f"Teams with official defense/baserunning stats loaded: {_war_teams_covered}.")

    st.divider()
    st.markdown("### Pitching WAR")
    if pdf_war.empty:
        st.info(f"No pitchers with at least {war_min_ip} IP yet.")
    else:
        pdf_war = pdf_war.sort_values("WAR", ascending=False).reset_index(drop=True)
        pdf_war.index += 1
        pdf_disp = pdf_war.copy()
        pdf_disp["Team"] = pdf_disp["Team"].map(team_label)
        st.dataframe(pdf_disp, use_container_width=True,
                    column_config={"WAR": st.column_config.NumberColumn("WAR", format="%.2f")})
        st.caption("RAR = (replacement-level FIP − pitcher's park-adjusted FIP) × IP ÷ 9. WAR = "
                   "RAR ÷ 10. Replacement-level FIP is set 1.00 run worse than the league-average "
                   "FIP shown above. The FIP column shows the raw (unadjusted) figure; the park "
                   "adjustment is applied only inside the RAR/WAR calculation.")
        with st.expander("Park factors by stadium"):
            st.dataframe(_pf_table, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────
#  PAGE: STARTERS VS BULLPEN
# ─────────────────────────────────────────
elif page == "Starters vs Bullpen":
    st.title("Starters vs Bullpen")
    st.caption("The team's cumulative starter stat line vs. its bullpen stat line, same advanced "
               "metrics used everywhere else in the app (FIP/xFIP/xERA, Stuff+, whiff/chase, "
               "first-pitch-strike%) plus the basic box-score numbers. A pitcher's own pitches "
               "count toward 'Starters' only for games where he threw the first pitch for his "
               "team — any relief outing, even by a regular starter, counts toward Bullpen for "
               "that game.")

    sb_teams = sorted(_team_options(df_all["PitcherTeam"]))
    sb_team = st.selectbox("Team", options=sb_teams,
                           index=sb_teams.index(MY_TEAM) if MY_TEAM in sb_teams else 0,
                           format_func=team_label, key="sb_team")

    # ── Figure out each game's starter for a given team: whoever threw the
    # first pitch for that team that game. Everyone else's work that game is
    # bullpen usage, even if it's normally a starter making a relief outing.
    # Reused both for the single-team breakdown below and the all-teams
    # leaderboards further down. ──
    def _sb_split(team_code):
        tp = df_all[df_all["PitcherTeam"] == team_code].copy()
        if tp.empty or "GameID" not in tp.columns:
            return tp, tp
        order_cols = [c for c in ["Inning", "PAofInning", "PitchofPA"] if c in tp.columns]
        starter_map = {}
        for gid, grp in tp.groupby("GameID"):
            first_row = grp.sort_values(order_cols).iloc[0] if order_cols else grp.iloc[0]
            starter_map[gid] = first_row["Pitcher"]
        tp["_starter_game"] = tp["GameID"].map(starter_map) == tp["Pitcher"]
        return tp[tp["_starter_game"]], tp[~tp["_starter_game"]]

    team_p = df_all[df_all["PitcherTeam"] == sb_team]
    if team_p.empty or "GameID" not in team_p.columns:
        st.info("No pitching data for this team.")
        st.stop()

    starters_df, bullpen_df = _sb_split(sb_team)

    STRIKE_C = {"StrikeCalled", "StrikeSwinging", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"}
    SWING_C  = {"StrikeSwinging", "InPlay", "FoulBallNotFieldable", "FoulBallFieldable", "FoulTip", "FoulBall"}
    FIP_CONSTANT  = 3.10
    LEAGUE_HR_FB  = 0.08
    XERA_CONSTANT = 3.35
    ZH, ZB, ZT = 0.83, 1.755, 3.378

    def _sb_stat_line(sub):
        if sub.empty:
            return None
        bf  = int((sub["PitchofPA"] == 1).sum()) if "PitchofPA" in sub.columns else 0
        k   = int((sub["KorBB"].eq("Strikeout") |
                  ((sub["PitchCall"].isin(["StrikeSwinging", "StrikeCalled"])) &
                   (sub.get("Strikes", pd.Series(dtype=int)) == 2))).sum())
        bb  = int(sub["KorBB"].eq("Walk").sum())
        hbp = int(sub["PitchCall"].eq("HitByPitch").sum())
        hr  = int(sub["PlayResult"].eq("HomeRun").sum())
        h   = int(sub["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum())
        ab  = int(_ab_mask(sub).sum())
        outs = sub["OutsOnPlay"].fillna(0).sum() + k
        ip_num = outs / 3
        ip_disp = f"{int(outs // 3)}.{int(outs % 3)}"
        runs = int(sub["RunsScored"].fillna(0).sum()) if "RunsScored" in sub.columns else 0
        pitchers_here = [p for p in sub["Pitcher"].dropna().unique() if not _is_removed(p)]

        opp_avg = h / ab if ab else 0.0
        whip = (bb + h) / ip_num if ip_num > 0 else 0.0
        k9   = k / ip_num * 9 if ip_num > 0 else 0.0
        bb9  = bb / ip_num * 9 if ip_num > 0 else 0.0
        kpct  = k / max(bf, 1)
        bbpct = bb / max(bf, 1)

        fb_t = sub["TaggedHitType"].eq("FlyBall").sum() if "TaggedHitType" in sub.columns else 0
        fb_a = sub["AutoHitType"].eq("FlyBall").sum() if "AutoHitType" in sub.columns else 0
        fb   = max(fb_t, fb_a)
        xhr  = fb * LEAGUE_HR_FB
        fip  = (13*hr + 3*(bb+hbp) - 2*k) / ip_num + FIP_CONSTANT if ip_num > 0 else None
        xfip = (13*xhr + 3*(bb+hbp) - 2*k) / ip_num + FIP_CONSTANT if ip_num > 0 else None
        xera = (13*xhr + 3*(bb+hbp) - 2*k) / ip_num + XERA_CONSTANT if ip_num > 0 else None

        fp  = sub[sub["PitchofPA"] == 1] if "PitchofPA" in sub.columns else sub.iloc[0:0]
        fps = fp["PitchCall"].isin(STRIKE_C).mean() if len(fp) else None

        sw = sub[sub["PitchCall"].isin(SWING_C)]
        whiff_pct = sub["PitchCall"].eq("StrikeSwinging").sum() / len(sw) if len(sw) > 0 else None

        located = sub[sub["PlateLocSide"].notna() & sub["PlateLocHeight"].notna()]
        oz = located[(located["PlateLocSide"].abs() > ZH) | (~located["PlateLocHeight"].between(ZB, ZT))]
        chase_pct = oz["PitchCall"].isin(SWING_C).mean() if len(oz) > 0 else None

        sp = (stuff_plus_df[stuff_plus_df["Pitcher"].isin(pitchers_here)]
              if not stuff_plus_df.empty else pd.DataFrame())
        stuff_avg = ((sp["StuffPlus"] * sp["Pitches"]).sum() / sp["Pitches"].sum()
                    if len(sp) and sp["Pitches"].sum() > 0 else None)

        return {
            "Pitchers": len(pitchers_here), "Games": sub["GameID"].nunique(), "Pitches": len(sub),
            "IP": ip_disp, "BF": bf, "H": h, "R": runs, "BB": bb, "SO": k, "HR": hr, "HBP": hbp,
            "Opp AVG": f"{opp_avg:.3f}", "WHIP": f"{whip:.2f}",
            "K/9": f"{k9:.1f}", "BB/9": f"{bb9:.1f}",
            "K%": f"{kpct:.0%}", "BB%": f"{bbpct:.0%}", "K-BB%": f"{kpct - bbpct:.0%}",
            "FIP": f"{fip:.2f}" if fip is not None else "—",
            "xFIP": f"{xfip:.2f}" if xfip is not None else "—",
            "xERA": f"{xera:.2f}" if xera is not None else "—",
            "FPS%": f"{fps:.0%}" if fps is not None else "—",
            "Whiff%": f"{whiff_pct:.0%}" if whiff_pct is not None else "—",
            "Chase%": f"{chase_pct:.0%}" if chase_pct is not None else "—",
            "Stuff+": f"{stuff_avg:.0f}" if stuff_avg is not None else "—",
        }

    s_line = _sb_stat_line(starters_df)
    b_line = _sb_stat_line(bullpen_df)

    if s_line is None and b_line is None:
        st.info("Not enough data to build this comparison yet.")
        st.stop()

    metric_order = ["Pitchers", "Games", "Pitches", "IP", "BF", "H", "R", "BB", "SO", "HR", "HBP",
                    "Opp AVG", "WHIP", "K/9", "BB/9", "K%", "BB%", "K-BB%",
                    "FIP", "xFIP", "xERA", "FPS%", "Whiff%", "Chase%", "Stuff+"]
    comp_df = pd.DataFrame([
        {"Metric": m, "Starters": s_line[m] if s_line else "—", "Bullpen": b_line[m] if b_line else "—"}
        for m in metric_order
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.caption("Opp AVG/WHIP/K-9/BB-9/K%/BB%/K-BB% are the basic box-score numbers. FIP/xFIP/xERA "
               "use the same constants as League Rankings, so they're comparable across pages. "
               "Stuff+ is pitch-count-weighted across every pitch type each pitcher in the group "
               "threw (100 = league-average whiff rate for that shape). Whiff% is swings-and-misses "
               "over all swings; Chase% is swings on pitches outside the strike zone.")

    # ── Every pitch belongs to exactly one bucket — verify that here so it's
    # provable in-app, not just asserted. A pitcher can legitimately show up
    # in BOTH pitcher counts below (started some games, relieved in others);
    # that's real usage, not the same appearance being counted twice. ──
    _total_pitches = len(team_p)
    _split_pitches = len(starters_df) + len(bullpen_df)
    starter_names = set(starters_df["Pitcher"].dropna().unique())
    bullpen_names = set(bullpen_df["Pitcher"].dropna().unique())
    both_roles = sorted(starter_names & bullpen_names, key=player_last)
    if _split_pitches == _total_pitches:
        st.caption(f"✓ Every pitch counted exactly once: {_total_pitches:,} team pitches = "
                   f"{len(starters_df):,} starter pitches + {len(bullpen_df):,} bullpen pitches. "
                   f"No pitcher's own pitches are ever in both buckets at once.")
    else:
        st.warning(f"⚠ {_total_pitches:,} team pitches but the split only accounts for "
                  f"{_split_pitches:,} — something's off with the starter/bullpen assignment.")
    if both_roles:
        st.caption("Pitchers below counted in **both** the Starters and Bullpen pitcher totals — "
                   "not double-counted, they made starts in some games and relief appearances in "
                   "others, and each individual game's pitches only ever land in one bucket: " +
                   ", ".join(player_last(p) for p in both_roles) + ".")

    with st.expander("Who's classified as a starter vs. bullpen arm"):
        starter_counts = (starters_df.groupby("Pitcher")["GameID"].nunique()
                          .sort_values(ascending=False))
        bullpen_counts = (bullpen_df.groupby("Pitcher")["GameID"].nunique()
                          .sort_values(ascending=False))
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("**Starts made**")
            if len(starter_counts):
                st.dataframe(pd.DataFrame({"Pitcher": [player_last(p) for p in starter_counts.index],
                                           "Starts": starter_counts.values}),
                            use_container_width=True, hide_index=True)
            else:
                st.caption("None yet.")
        with ec2:
            st.markdown("**Relief appearances**")
            if len(bullpen_counts):
                st.dataframe(pd.DataFrame({"Pitcher": [player_last(p) for p in bullpen_counts.index],
                                           "Relief G": bullpen_counts.values}),
                            use_container_width=True, hide_index=True)
            else:
                st.caption("None yet.")

    # ── League-wide: every team's starters ranked against every other
    # team's starters, and same for bullpens — built from each team's
    # official season pitching stats (real ERA/W-L/SV, which TrackMan can't
    # give us — no earned-vs-unearned split, no win/loss/save logic in pitch
    # data), with each pitcher's whole season assigned to Starter or
    # Reliever by whichever role he made more appearances in. ──
    st.divider()
    st.markdown("## League Rankings — Starters vs Bullpen")
    st.caption("Every team's cumulative starter line, and every team's cumulative bullpen line, "
               "ranked separately — so you can see whose rotation is strong even if their pen is "
               "shaky, or the other way around. Built from official season pitching stats (each "
               "pitcher's whole season goes to whichever role — starting or relieving — he made "
               "more appearances in), not the game-by-game TrackMan split used above. Sorted by "
               "ERA (lower is better).")

    # Demo fallback for Data/official_pitching_season.csv, which hasn't been
    # exported for the league yet. One starter + one reliever per FCBL team
    # (matching whoever actually pitched in the TrackMan games where
    # possible), so the league-wide starter/bullpen rankings below have
    # something to show for every club instead of sitting empty. Delete
    # this block once official_pitching_season.csv is added.
    _DEMO_OFFICIAL_PITCHING = [
        {"Player": "Brooks, Tyler", "Team": "BRK_BAN", "APP": 11, "GS": 11, "W": 6, "L": 2, "SV": 0,
         "IP": 58.0, "H": 48, "R": 26, "ER": 22, "BB": 16, "SO": 64, "HR": 5, "AB": 188},
        {"Player": "Frost, Adam", "Team": "BRK_BAN", "APP": 9, "GS": 5, "W": 3, "L": 2, "SV": 0,
         "IP": 32.1, "H": 30, "R": 17, "ER": 15, "BB": 13, "SO": 29, "HR": 3, "AB": 118},
        {"Player": "Sharpe, Devon", "Team": "BRK_BAN", "APP": 14, "GS": 0, "W": 2, "L": 1, "SV": 3,
         "IP": 19.2, "H": 15, "R": 8, "ER": 7, "BB": 9, "SO": 22, "HR": 1, "AB": 59},
        {"Player": "Ito, Mason", "Team": "BRK_BAN", "APP": 15, "GS": 0, "W": 1, "L": 0, "SV": 1,
         "IP": 17.0, "H": 14, "R": 7, "ER": 6, "BB": 7, "SO": 19, "HR": 1, "AB": 55},
        {"Player": "Delacruz, Ray", "Team": "BRK_BAN", "APP": 13, "GS": 0, "W": 1, "L": 2, "SV": 0,
         "IP": 15.1, "H": 16, "R": 11, "ER": 9, "BB": 8, "SO": 13, "HR": 2, "AB": 63},
        {"Player": "Bennett, Cole", "Team": "BRK_BAN", "APP": 12, "GS": 0, "W": 0, "L": 1, "SV": 2,
         "IP": 13.2, "H": 11, "R": 6, "ER": 5, "BB": 6, "SO": 15, "HR": 1, "AB": 43},
        {"Player": "Delgado, Marcus", "Team": "CON_RIV", "APP": 10, "GS": 10, "W": 5, "L": 3, "SV": 0,
         "IP": 52.0, "H": 47, "R": 26, "ER": 22, "BB": 17, "SO": 50, "HR": 6, "AB": 184},
        {"Player": "Dunmore, Chris", "Team": "CON_RIV", "APP": 13, "GS": 3, "W": 2, "L": 3, "SV": 1,
         "IP": 33.2, "H": 32, "R": 20, "ER": 17, "BB": 15, "SO": 31, "HR": 4, "AB": 125},
        {"Player": "Hollis, Grant", "Team": "CON_RIV", "APP": 16, "GS": 0, "W": 1, "L": 1, "SV": 4,
         "IP": 21.1, "H": 18, "R": 10, "ER": 9, "BB": 10, "SO": 24, "HR": 2, "AB": 71},
        {"Player": "Halstrom, Owen", "Team": "DOV_ANC", "APP": 9, "GS": 9, "W": 3, "L": 5, "SV": 0,
         "IP": 45.2, "H": 46, "R": 29, "ER": 25, "BB": 20, "SO": 38, "HR": 7, "AB": 180},
        {"Player": "Yun, Parker", "Team": "DOV_ANC", "APP": 12, "GS": 0, "W": 1, "L": 2, "SV": 1,
         "IP": 17.0, "H": 17, "R": 11, "ER": 10, "BB": 9, "SO": 15, "HR": 2, "AB": 67},
        {"Player": "Ferris, Nate", "Team": "POR_PRI", "APP": 8, "GS": 8, "W": 2, "L": 5, "SV": 0,
         "IP": 40.1, "H": 43, "R": 30, "ER": 26, "BB": 21, "SO": 33, "HR": 8, "AB": 169},
        {"Player": "Locke, Bryan", "Team": "POR_PRI", "APP": 14, "GS": 0, "W": 0, "L": 3, "SV": 2,
         "IP": 18.2, "H": 19, "R": 13, "ER": 12, "BB": 11, "SO": 17, "HR": 3, "AB": 75},
        {"Player": "Sato, Reggie", "Team": "MAN_MIL", "APP": 10, "GS": 10, "W": 3, "L": 4, "SV": 0,
         "IP": 48.0, "H": 45, "R": 27, "ER": 23, "BB": 19, "SO": 42, "HR": 6, "AB": 176},
        {"Player": "Whipple, Dane", "Team": "MAN_MIL", "APP": 15, "GS": 0, "W": 1, "L": 1, "SV": 3,
         "IP": 19.1, "H": 17, "R": 10, "ER": 9, "BB": 9, "SO": 20, "HR": 2, "AB": 67},
    ]

    @st.cache_data(ttl=300, max_entries=3)
    def _load_official_pitching():
        f = DATA_DIR / "official_pitching_season.csv"
        if not f.exists():
            return pd.DataFrame(_DEMO_OFFICIAL_PITCHING)
        try:
            df = pd.read_csv(f)
            return df if not df.empty else pd.DataFrame(_DEMO_OFFICIAL_PITCHING)
        except Exception:
            return pd.DataFrame(_DEMO_OFFICIAL_PITCHING)

    def _official_ip_outs(ip_str):
        try:
            ip = float(ip_str)
        except (TypeError, ValueError):
            return 0
        whole = int(ip)
        frac = round((ip - whole) * 10)  # baseball notation: .1/.2 = thirds
        return whole * 3 + frac

    official_pitch = _load_official_pitching()

    if official_pitch.empty:
        st.info("No official pitching stats loaded yet.")
    else:
        op = official_pitch.copy()
        _num = lambda s: pd.to_numeric(s, errors="coerce").fillna(0)
        op["_APP"]  = _num(op["APP"])
        op["_GS"]   = _num(op["GS"])
        op["_outs"] = op["IP"].apply(_official_ip_outs)
        op["Role"]  = np.where(op["_GS"] >= op["_APP"] - op["_GS"], "Starter", "Reliever")
        for c in ["W", "L", "SV", "H", "R", "ER", "BB", "SO", "HR", "AB"]:
            op[f"_{c}"] = _num(op[c])

        def _off_league_table(role):
            rows = []
            for t, grp in op[op["Role"] == role].groupby("Team"):
                outs = grp["_outs"].sum()
                ip_num = outs / 3
                if ip_num <= 0:
                    continue
                h, bb, so, hr = grp["_H"].sum(), grp["_BB"].sum(), grp["_SO"].sum(), grp["_HR"].sum()
                er, ab = grp["_ER"].sum(), grp["_AB"].sum()
                rows.append({
                    "Team": team_label(t), "Pitchers": grp["Player"].nunique(),
                    "IP": f"{int(outs // 3)}.{int(outs % 3)}",
                    "W": int(grp["_W"].sum()), "L": int(grp["_L"].sum()), "SV": int(grp["_SV"].sum()),
                    "ERA": round(9 * er / ip_num, 2), "WHIP": round((bb + h) / ip_num, 2),
                    "K/9": round(so / ip_num * 9, 1), "BB/9": round(bb / ip_num * 9, 1),
                    "H": int(h), "BB": int(bb), "SO": int(so), "HR": int(hr),
                    "Opp AVG": round(h / ab, 3) if ab > 0 else None,
                })
            d = pd.DataFrame(rows)
            if d.empty:
                return d
            d = d.sort_values("ERA").reset_index(drop=True)
            d.insert(0, "Rank", range(1, len(d) + 1))
            return d

        st.markdown("### Starters, all teams")
        _starter_league_df = _off_league_table("Starter")
        if _starter_league_df.empty:
            st.info("Not enough starter data across the league yet.")
        else:
            st.dataframe(_starter_league_df, use_container_width=True, hide_index=True)

        st.markdown("### Bullpens, all teams")
        _bullpen_league_df = _off_league_table("Reliever")
        if _bullpen_league_df.empty:
            st.info("Not enough bullpen data across the league yet.")
        else:
            st.dataframe(_bullpen_league_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────
#  PAGE: REPORT GENERATOR
# ─────────────────────────────────────────
elif page == "Report Generator":
    st.title("Printable Scouting Sheet")
    st.caption("A printable one-page scouting sheet for any team, rebuilt from the data every "
               "time it loads. Pitchers on top, hitters below — same layout as the dugout sheet.")

    _rg_teams = sorted((set(df_all["PitcherTeam"].dropna()) | set(df_all["BatterTeam"].dropna())) - EXCLUDED_TEAMS)
    ca, cb = st.columns([2, 1])
    with ca:
        rg_team = st.selectbox("Team", options=_rg_teams, format_func=team_label,
                               index=_rg_teams.index(MY_TEAM) if MY_TEAM in _rg_teams else 0,
                               key="rg_team")
    with cb:
        rg_min = st.number_input("Min pitches to include a player", 1, 200, 10, 5, key="rg_min",
                                 help="Raise this to keep the sheet to a single printed page.")

    pit_df, hit_df = _scout_tables(df_all, rg_team, int(rg_min))

    # Optional player selection. Empty = everyone (default), so the report is
    # unchanged unless you deliberately narrow it. Names are the report's own
    # last-name labels.
    with st.expander("Choose specific players (optional — leave empty for the whole team)"):
        sc1, sc2 = st.columns(2)
        with sc1:
            pit_pick = st.multiselect("Pitchers", options=list(pit_df["Pitcher"]) if len(pit_df) else [],
                                      default=[], key="rg_pit_pick",
                                      help="Empty shows every pitcher above the pitch minimum.")
        with sc2:
            hit_pick = st.multiselect("Hitters", options=list(hit_df["Hitter"]) if len(hit_df) else [],
                                      default=[], key="rg_hit_pick",
                                      help="Empty shows every hitter above the pitch minimum.")
    if pit_pick and len(pit_df):
        pit_df = pit_df[pit_df["Pitcher"].isin(pit_pick)].reset_index(drop=True)
    if hit_pick and len(hit_df):
        hit_df = hit_df[hit_df["Hitter"].isin(hit_pick)].reset_index(drop=True)

    st.markdown("### Pitchers")
    st.caption("**v LHH / v RHH** — opponent batting average by batter hand. "
               "**FPS%** — first-pitch strike rate (a called or swinging strike, foul, or ball in "
               "play on 0-0). **FPS% L / R** — the same split by batter hand.")
    if len(pit_df):
        st.dataframe(pit_df, use_container_width=True, hide_index=True)
    else:
        st.info("No pitchers clear the minimum-pitch filter.")

    st.markdown("### Hitters")
    st.caption("**v LHP / v RHP** — batting average by pitcher hand. **T%** — how often he takes "
               "the pitch in that count. **IZ T%** — how often he takes a pitch that is inside the "
               "strike zone in that count (high means he is watching strikes go by). "
               "**L-AB / R-AB** — hits and at-bats against left- and right-handers.")
    if len(hit_df):
        st.dataframe(hit_df, use_container_width=True, hide_index=True)
    else:
        st.info("No hitters clear the minimum-pitch filter.")

    st.divider()
    if len(pit_df) or len(hit_df):
        try:
            pdf = _build_scout_pdf(team_label(rg_team), pit_df, hit_df)
            fn = "scouting_sheet_" + str(rg_team).replace(" ", "_") + ".pdf"
            st.download_button("⬇ Download printable sheet (PDF, one page)", data=pdf,
                               file_name=fn, mime="application/pdf", key="rg_pdf")
            st.caption("Landscape letter. Font shrinks automatically as players are added so the "
                       "sheet stays on one page; raise the minimum-pitch filter if it ever spills over.")
        except Exception as _e:
            _pdf_unavailable(_e)

    st.caption("Count columns ignore any game whose ball-strike counts never advance, so corrupt "
               "tracking files cannot distort take rates. The # column fills from "
               "Data/jersey_numbers.json when present, and stays blank to write in by hand otherwise.")


elif page == "League Rankings":
    st.title("League Pitching Rankings")
    st.caption("FIP, xFIP, and xERA for every FCBL team — lower is better, sorted by xERA, "
               "park-adjusted using each team's actual mix of home and road ballparks.")
    import plotly.graph_objects as go_lr

    FIP_CONSTANT  = 3.10
    LEAGUE_HR_FB  = 0.08
    XERA_CONSTANT = 3.35

    _pf_hash = (len(df_all), tuple(df_all.columns))
    _game_pf, _pf_table = _compute_park_factors(_pf_hash)

    team_rows = []
    for _team, _grp in df_all.groupby("PitcherTeam"):
        if _team in EXCLUDED_TEAMS:
            continue
        _gc = [c for c in ["GameID","Inning","Top/Bottom","PAofInning"] if c in _grp.columns]
        _last = _grp.groupby(_gc).last().reset_index() if _gc else _grp.copy()
        _k   = (_last["KorBB"].eq("Strikeout") |
                ((_last["PitchCall"].isin(["StrikeSwinging","StrikeCalled"])) &
                 (_last["Strikes"] == 2))).sum()
        _bb  = _last["KorBB"].eq("Walk").sum()
        _hbp = _last["PitchCall"].eq("HitByPitch").sum()
        _hr  = _last["PlayResult"].eq("HomeRun").sum()
        _outs = _grp["OutsOnPlay"].fillna(0).sum() + _k
        _ip   = _outs / 3
        if _ip < 5:
            continue
        _fb_t = _grp["TaggedHitType"].eq("FlyBall").sum()
        _fb_a = _grp["AutoHitType"].eq("FlyBall").sum() if "AutoHitType" in _grp.columns else 0
        _fb   = max(_fb_t, _fb_a)
        _xhr  = _fb * LEAGUE_HR_FB
        _fip  = (13*_hr  + 3*(_bb+_hbp) - 2*_k) / _ip + FIP_CONSTANT
        _xfip = (13*_xhr + 3*(_bb+_hbp) - 2*_k) / _ip + FIP_CONSTANT
        _xera = (13*_xhr + 3*(_bb+_hbp) - 2*_k) / _ip + XERA_CONSTANT
        _k9   = _k / _ip * 9
        _bb9  = _bb / _ip * 9

        # Park-adjust: a pitch-weighted average of the park factor across
        # every game this team's staff actually pitched in (roughly half
        # home, half whichever parks they visited on the road), then divide
        # the runs-based metrics by it — a hitter-friendly park (PF > 1)
        # deflates the raw number back down to a fair comparison.
        _team_pf = _grp["GameID"].map(_game_pf).mean() if _game_pf else 1.0
        _team_pf = _team_pf if pd.notna(_team_pf) and _team_pf > 0 else 1.0
        _fip  /= _team_pf
        _xfip /= _team_pf
        _xera /= _team_pf

        team_rows.append({
            "Team": team_label(_team), "_code": _team,
            "IP": to_ip(_outs), "K": int(_k), "BB": int(_bb), "HR": int(_hr),
            "K/9": round(_k9,1), "BB/9": round(_bb9,1), "PF": round(_team_pf, 3),
            "FIP": round(_fip,2), "xFIP": round(_xfip,2), "xERA": round(_xera,2),
        })

    if not team_rows:
        st.info("Not enough data yet (need 5+ IP per team).")
    else:
        _tdf = pd.DataFrame(team_rows).sort_values("xERA").reset_index(drop=True)
        _tdf.index += 1

        _fig = go_lr.Figure()
        for _met, _col in [("FIP","#3b82f6"),("xFIP","#22c55e"),("xERA","#f59e0b")]:
            _fig.add_trace(go_lr.Bar(
                name=_met, x=_tdf["Team"], y=_tdf[_met],
                marker_color=_col, opacity=0.85,
                text=_tdf[_met].map(lambda v: f"{v:.2f}"),
                textposition="outside",
                textfont=dict(size=10, color="#1e293b"),
            ))
        _fig.add_hline(y=FIP_CONSTANT, line_dash="dash",
            line_color="rgba(15,23,42,0.3)",
            annotation_text="League Avg",
            annotation_font_color="rgba(15,23,42,0.45)")
        _fig.update_layout(
            height=440, barmode="group",
            plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF",
            font=dict(color="#1e293b"),
            xaxis=dict(gridcolor="#E2E8F0", tickangle=-20),
            yaxis=dict(title="ERA Scale (lower = better)",
                       gridcolor="#E2E8F0", zeroline=False,
                       range=[0, _tdf["xERA"].max()+1.5]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b")),
            margin=dict(l=40, r=60, t=30, b=80)
        )
        st.plotly_chart(_fig, use_container_width=True)
        st.divider()

        st.dataframe(
            _tdf[["Team","IP","K","BB","HR","K/9","BB/9","PF","FIP","xFIP","xERA"]],
            use_container_width=True, hide_index=False,
            column_config={
                "FIP":  st.column_config.NumberColumn("FIP",  format="%.2f"),
                "xFIP": st.column_config.NumberColumn("xFIP", format="%.2f"),
                "xERA": st.column_config.NumberColumn("xERA", format="%.2f"),
                "K/9":  st.column_config.NumberColumn("K/9",  format="%.1f"),
                "BB/9": st.column_config.NumberColumn("BB/9", format="%.1f"),
                "PF":   st.column_config.NumberColumn("PF", format="%.3f",
                        help="Pitch-weighted average park factor across the parks this staff "
                             "actually pitched in. Below 1 = pitcher-friendly mix, above 1 = "
                             "hitter-friendly. FIP/xFIP/xERA are already divided by this."),
            }
        )
        st.caption("PF = park factor (1.000 = league-neutral), from combined runs/game at each "
                   "stadium, shrunk toward 1.0 for sample size and capped to ±15% — a single "
                   "college-summer season doesn't support a bigger claim than that. FIP/xFIP/xERA "
                   "above are already park-adjusted (divided by each team's own PF).")
        if not _pf_table.empty:
            with st.expander("Park factors by stadium"):
                st.dataframe(_pf_table, use_container_width=True, hide_index=True)
        st.divider()

        _best  = _tdf.iloc[0]
        _worst = _tdf.iloc[-1]
        _nas   = _tdf[_tdf["_code"] == MY_TEAM]
        _c1, _c2, _c3 = st.columns(3)
        def _card(col, border, label, tname, fip, xfip, xera):
            col.markdown(
                f"<div style='background:#F8FAFC;border:1.5px solid {border};"
                f"border-radius:8px;padding:1rem;'>"
                f"<div style='font-size:0.75rem;color:{border};font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;'>{label}</div>"
                f"<div style='font-size:1.1rem;font-weight:700;color:#1e293b;'>{tname}</div>"
                f"<div style='font-size:0.85rem;color:#475569;'>"
                f"FIP {fip:.2f} · xFIP {xfip:.2f} · xERA {xera:.2f}</div>"
                f"</div>", unsafe_allow_html=True)
        _card(_c1, "#22c55e", "Best xERA",  _best["Team"],  _best["FIP"],  _best["xFIP"],  _best["xERA"])
        _card(_c2, "#ef4444", "Worst xERA", _worst["Team"], _worst["FIP"], _worst["xFIP"], _worst["xERA"])
        if len(_nas) > 0:
            _nr = _nas.iloc[0]
            _rn = _nas.index[0]
            _bc = "#22c55e" if _rn <= 2 else "#f59e0b" if _rn <= 4 else "#ef4444"
            _card(_c3, _bc, f"Brookhaven — Rank #{_rn}", _nr["Team"], _nr["FIP"], _nr["xFIP"], _nr["xERA"])


    st.divider()
    # ── HITTING METRICS TAB (xBA / xwOBA leaderboard) ─────────────────────
    st.markdown("### Hitting Metrics — xBA & xwOBA")
    st.caption("Expected batting average and expected weighted on-base average from the "
               "quality of contact (exit velocity and launch angle) on each ball in play. "
               "Higher is better. A hitter well above his real AVG/wOBA has been unlucky; "
               "well below means he has been getting favourable results.")
    HIT_MIN_PA = st.number_input("Minimum plate appearances to qualify", 1, 200, 15, 5,
                                 key="hitmet_min",
                                 help="Raise this to steady the leaderboard; xBA needs a "
                                      "handful of batted balls before it means much.")
    _hm = df_all
    HITS = ["Single", "Double", "Triple", "HomeRun"]
    hit_rows = []
    for b, grp in _hm.groupby("Batter"):
        if pd.isna(b) or _is_removed(b) or _is_report_hidden(b):
            continue
        pa = int((grp["PitchofPA"] == 1).sum())
        if pa < int(HIT_MIN_PA):
            continue
        ab = int((_ab_mask(grp)).sum())
        h = int(grp["PlayResult"].isin(HITS).sum())
        ba = h / ab if ab else np.nan
        xba, _xslg, xwoba, n_bip = batter_expected_stats(grp)
        hit_rows.append({
            "Hitter": player_last(b),
            "Team": team_label(grp["BatterTeam"].dropna().iloc[0]) if grp["BatterTeam"].notna().any() else "—",
            "PA": pa, "AB": ab, "H": h,
            "AVG": round(ba, 3) if pd.notna(ba) else None,
            "xBA": round(xba, 3) if xba is not None else None,
            "xwOBA": round(xwoba, 3) if xwoba is not None else None,
            "BIP": n_bip,
            "_diff": (ba - xba) if (pd.notna(ba) and xba is not None) else None,
        })
    if not hit_rows:
        st.info("No hitters meet the plate-appearance minimum yet.")
    else:
        hdf = pd.DataFrame(hit_rows)
        hdf["AVG-xBA"] = hdf["_diff"].apply(lambda v: round(v, 3) if pd.notna(v) else None)
        hdf = hdf.drop(columns="_diff")
        sort_by = st.radio("Rank by", ["xwOBA", "xBA", "AVG"], horizontal=True, key="hitmet_sort")
        hdf = hdf.sort_values(sort_by, ascending=False, na_position="last").reset_index(drop=True)
        hdf.index = hdf.index + 1
        st.dataframe(hdf, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Qualified hitters", str(len(hdf)))
        c2.metric("League avg xBA", f"{hdf['xBA'].mean():.3f}" if hdf['xBA'].notna().any() else "—")
        c3.metric("League avg xwOBA", f"{hdf['xwOBA'].mean():.3f}" if hdf['xwOBA'].notna().any() else "—")
        st.caption("AVG-xBA is actual minus expected: a positive number means he is out-hitting "
                   "his contact quality (some luck), a negative number means the reverse. "
                   "xBA and xwOBA use the same models as the xBA Report and Player Report, so "
                   "the numbers agree across the app.")



elif page == "xBA Report":
    st.title("Expected Stats Report — xBA, xSLG, xwOBA")
    st.caption(
        "Expected stats estimate outcomes based on exit velocity and launch angle using the MLB Statcast lookup table, "
        "calibrated to FCBL. They remove the influence of defense, park, and luck — "
        "showing true contact quality regardless of results."
    )
    st.divider()

    def build_xba_table(team_filter, label):
        if team_filter == "our":
            subset = df_all[df_all["BatterTeam"] == MY_TEAM]
        else:
            subset = df_all[df_all["BatterTeam"] != MY_TEAM]

        batters = _player_options_reports(subset["Batter"])
        rows = []
        for batter in batters:
            bp = subset[subset["Batter"] == batter]
            xba, xslg, xwoba, xba_n = batter_expected_stats(bp)
            if xba is None or xba_n < 2:
                continue
            pa = bp[bp["PitchofPA"]==1].shape[0] if "PitchofPA" in bp.columns else len(bp)
            team = bp["BatterTeam"].iloc[0]
            side = bp["BatterSide"].iloc[0] if "BatterSide" in bp.columns else "?"

            # K% and BB%
            SWING_CALLS_XBA = {"StrikeSwinging","InPlay","FoulBallNotFieldable","FoulBallFieldable","FoulTip","FoulBall"}
            k_pct_xba  = bp["KorBB"].eq("Strikeout").sum() / max(pa, 1)
            bb_pct_xba = bp["KorBB"].eq("Walk").sum() / max(pa, 1)
            # BB/K ratio
            k_count  = bp["KorBB"].eq("Strikeout").sum()
            bb_count = bp["KorBB"].eq("Walk").sum()
            bb_k     = round(bb_count / k_count, 2) if k_count > 0 else None

            rows.append({
                "Batter":  batter,
                "Team":    team_label(team),
                "B":       side,
                "PA":      pa,
                "BIP":     xba_n,
                "K%":      k_pct_xba,
                "BB%":     bb_pct_xba,
                "BB/K":    bb_k,
                "xBA":     xba,
                "xSLG":    xslg,
                "xwOBA":   xwoba,
            })
        if not rows:
            st.info("Not enough batted ball data yet (need 2+ fair balls in play per player).")
            with st.expander("Debug — click to inspect"):
                st.write(f"Batters checked: {len(batters)}")
                st.write(f"Team filter: {team_filter}")
                for b in batters[:5]:
                    bp2 = subset[subset["Batter"] == b]
                    fair2 = bp2[
                        bp2["ExitSpeed"].notna() & bp2["Angle"].notna() &
                        bp2["Direction"].notna() &
                        (bp2["Distance"].fillna(0) >= 10) &
                        (bp2["Direction"].abs() <= 45)
                    ]
                    st.write(f"{b}: {len(bp2)} pitches, {len(fair2)} fair balls")
                    if "GameID" in bp2.columns:
                        st.write(bp2.groupby("GameID").size().reset_index(name="Pitches"))
            return
        df = pd.DataFrame(rows).sort_values("xwOBA", ascending=False).reset_index(drop=True)
        df.index += 1

        # Color-coded verdict
        # Summary metrics at top
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Players tracked", len(df))
        m2.metric("Avg xwOBA", f"{df['xwOBA'].mean():.3f}")
        st.divider()

        # Main table
        display = df.copy()
        display["K%"]    = display["K%"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—")
        display["BB%"]   = display["BB%"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—")
        display["BB/K"]  = display["BB/K"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        display["xBA"]   = display["xBA"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        display["xSLG"]  = display["xSLG"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        display["xwOBA"] = display["xwOBA"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        st.dataframe(display[["Batter","Team","B","PA","K%","BB%","xBA","xSLG","xwOBA"]],
                     use_container_width=True, height=420, hide_index=False)

        # Debug expander — helps verify BIP counts
        with st.expander("Debug — inspect BIP for a specific player"):
            debug_batter = st.selectbox("Select player", options=df["Batter"].tolist(), key=f"xba_debug_batter_{team_filter}")
            if debug_batter:
                if team_filter == "our":
                    dbp = df_all[df_all["BatterTeam"] == MY_TEAM]
                else:
                    dbp = df_all[df_all["BatterTeam"] != MY_TEAM]
                dbp = dbp[dbp["Batter"] == debug_batter].copy()
                fair = dbp[
                    dbp["ExitSpeed"].notna() & dbp["Angle"].notna() &
                    dbp["Direction"].notna() &
                    (dbp["Distance"].fillna(0) >= 10) &
                    (dbp["Direction"].abs() <= 45)
                ]
                st.write(f"Total pitches: {len(dbp)} | Fair BIP: {len(fair)}")
                if "GameID" in dbp.columns:
                    st.write("Pitches per game:")
                    st.write(dbp.groupby("GameID").size().reset_index(name="Pitches"))
                    st.write("Fair BIP per game:")
                    st.write(fair.groupby("GameID").size().reset_index(name="BIP"))
                st.write("Fair ball details:")
                cols = [c for c in ["GameID","ExitSpeed","Angle","Direction","Distance","PlayResult"] if c in fair.columns]
                st.dataframe(fair[cols])

        st.divider()

        # Game log expander — click any player to see their AB log
        st.markdown("#### Player Game Log")
        st.caption("Select a hitter to see every at-bat with outcome, exit velocity, and xBA.")

        log_batter = st.selectbox(
            "Select hitter",
            options=[""] + df["Batter"].tolist(),
            format_func=lambda x: player_last(x) if x else "Select a hitter…",
            key=f"xba_gamelog_{team_filter}"
        )

        if log_batter:
            lbp = subset[subset["Batter"] == log_batter].copy()

            # Get last pitch of each PA for outcome
            gc = [c for c in ["GameID","Date","Inning","Top/Bottom","PAofInning"] if c in lbp.columns]
            if gc:
                ab_log = lbp.groupby([c for c in ["GameID","Inning","Top/Bottom","PAofInning"] if c in lbp.columns]).last().reset_index()
            else:
                ab_log = lbp.copy()

            if "Date" in ab_log.columns:
                ab_log["Date"] = pd.to_datetime(ab_log["Date"], errors="coerce")

            # Classify outcome
            def classify_outcome(row):
                pr = row.get("PlayResult","")
                kb = row.get("KorBB","")
                pc = row.get("PitchCall","")
                if pr in ["Single","Double","Triple","HomeRun"]:
                    return pr
                elif kb == "Walk" or pc == "BallCalled" and str(kb) not in ["","Undefined","nan"]:
                    return "Walk"
                elif pc == "HitByPitch":
                    return "HBP"
                elif kb == "Strikeout" or pc in ["StrikeSwinging","StrikeCalled"]:
                    return "Strikeout"
                elif pr in ["Out","FieldersChoice","Error","Sacrifice"]:
                    return pr
                elif kb == "Walk":
                    return "Walk"
                else:
                    return str(pr) if pd.notna(pr) and pr not in ["","Undefined"] else "—"

            ab_log["Outcome"] = ab_log.apply(classify_outcome, axis=1)

            # xBA per AB
            def ab_xba(row):
                if pd.notna(row.get("ExitSpeed")) and pd.notna(row.get("Angle")):
                    if row.get("Distance",0) >= 10 and abs(row.get("Direction",999)) <= 45:
                        return calc_xba(row["ExitSpeed"], row["Angle"])
                return None

            ab_log["xBA"] = ab_log.apply(ab_xba, axis=1)

            # Outcome color
            OUTCOME_COLORS = {
                "Single":"#22c55e","Double":"#3b82f6","Triple":"#8b5cf6",
                "HomeRun":"#f59e0b","Walk":"#06b6d4","HBP":"#06b6d4",
                "Out":"#ef4444","FieldersChoice":"#f97316","Error":"#ec4899",
                "Strikeout":"#ef4444","Sacrifice":"#64748b","—":"#475569"
            }

            # Summary stats
            hits    = ab_log["Outcome"].isin(["Single","Double","Triple","HomeRun"]).sum()
            walks   = ab_log["Outcome"].isin(["Walk","HBP"]).sum()
            ks      = ab_log["Outcome"].eq("Strikeout").sum()
            abs_c   = ab_log["Outcome"].isin(["Single","Double","Triple","HomeRun",
                                               "Out","FieldersChoice","Error","Strikeout"]).sum()
            ba_calc = hits / abs_c if abs_c > 0 else 0
            xba_avg = ab_log["xBA"].mean()

            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("PA",      len(ab_log))
            sc2.metric("AB",      int(abs_c))
            sc3.metric("H",       int(hits))
            sc4.metric("K",       int(ks))
            sc5.metric("xBA",     f"{xba_avg:.3f}" if pd.notna(xba_avg) else "—")

            st.divider()

            # Game-by-game log
            for game_id, game_grp in ab_log.groupby("GameID", sort=False):
                date_str = game_grp["Date"].iloc[0].strftime("%b %d") if "Date" in game_grp.columns and pd.notna(game_grp["Date"].iloc[0]) else game_id
                game_grp = game_grp.sort_values("Inning")

                st.markdown(f"**{date_str}** — {game_id.split('-')[1] if '-' in game_id else game_id}")

                ab_html = "<div style='display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;'>"
                for _, ab in game_grp.iterrows():
                    outcome  = ab["Outcome"]
                    color    = OUTCOME_COLORS.get(outcome, "#64748b")
                    inn      = int(ab["Inning"]) if pd.notna(ab.get("Inning")) else "?"
                    ev_str   = f"{ab['ExitSpeed']:.0f} mph" if pd.notna(ab.get("ExitSpeed")) else ""
                    la_str   = f"{ab['Angle']:.0f}°" if pd.notna(ab.get("Angle")) else ""
                    xba_str  = f"xBA {ab['xBA']:.3f}" if pd.notna(ab.get("xBA")) else ""
                    pitcher  = player_last(ab.get("Pitcher","")) if pd.notna(ab.get("Pitcher","")) else ""
                    hand     = ab.get("PitcherThrows","?")
                    count    = f"{int(ab.get('Balls',0))}-{int(ab.get('Strikes',0))}" if pd.notna(ab.get("Balls")) else ""

                    ab_html += (
                        f"<div style='background:#F8FAFC;border:1.5px solid {color};"
                        f"border-radius:8px;padding:8px 12px;min-width:140px;'>"
                        f"<div style='font-size:0.7rem;color:#64748b;'>Inn {inn} · {count}</div>"
                        f"<div style='font-size:1rem;font-weight:800;color:{color};'>{outcome}</div>"
                        f"<div style='font-size:0.72rem;color:#475569;margin-top:2px;'>"
                        f"{ev_str}{' · ' if ev_str and la_str else ''}{la_str}</div>"
                        f"<div style='font-size:0.7rem;color:#22c55e;'>{xba_str}</div>"
                        f"<div style='font-size:0.68rem;color:#475569;margin-top:2px;'>"
                        f"vs {pitcher} ({hand[0] if hand else '?'})</div>"
                        f"</div>"
                    )
                ab_html += "</div>"
                st.markdown(ab_html, unsafe_allow_html=True)

        st.divider()

        # xBA bar chart — contact quality ranking
        st.markdown("#### Expected Stats — Contact Quality Ranking")
        st.caption("Based on MLB Statcast EV x LA lookup table, calibrated to FCBL. xwOBA is the most complete single metric.")

        import plotly.graph_objects as go_xba
        df_sorted = df.sort_values("xwOBA", ascending=True)
        bar_colors = ["#22c55e" if v >= 0.350 else "#3b82f6" if v >= 0.280 else "#64748b"
                      for v in df_sorted["xwOBA"]]
        fig = go_xba.Figure()
        fig.add_trace(go_xba.Bar(
            x=df_sorted["xwOBA"],
            y=df_sorted["Batter"].map(player_last),
            orientation="h",
            marker_color=bar_colors,
            text=df_sorted["xwOBA"].map(lambda v: f"{v:.3f}"),
            textposition="outside",
            textfont=dict(size=11, color="#1e293b"),
            hovertemplate="<b>%{y}</b><br>xwOBA: %{x:.3f}<extra></extra>",
        ))
        fig.add_vline(x=df["xwOBA"].mean(), line_dash="dash",
                      line_color="rgba(15,23,42,0.35)",
                      annotation_text="Team avg",
                      annotation_font_color="rgba(15,23,42,0.5)")
        fig.update_layout(
            height=max(300, len(df)*40),
            plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF",
            font=dict(color="#1e293b"),
            xaxis=dict(title="xwOBA", gridcolor="#E2E8F0", zeroline=False),
            yaxis=dict(gridcolor="#E2E8F0"),
            margin=dict(l=20, r=60, t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(df) >= 2:
            best_xba  = df.loc[df["xwOBA"].idxmax()]
            worst_xba = df.loc[df["xwOBA"].idxmin()]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                    <div style='background:#F8FAFC;border:1.5px solid #22c55e;border-radius:8px;padding:1rem;'>
                        <div style='font-size:0.75rem;color:#22c55e;font-weight:700;
                        text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;'>
                        Best Contact Quality</div>
                        <div style='font-size:1.1rem;font-weight:700;color:#1e293b;'>{player_last(best_xba['Batter'])}</div>
                        <div style='font-size:0.85rem;color:#475569;'>
                        xBA {best_xba['xBA']:.3f} · xSLG {best_xba['xSLG']:.3f} · xwOBA {best_xba['xwOBA']:.3f} — {int(best_xba['BIP'])} BIP</div>
                    </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                    <div style='background:#F8FAFC;border:1.5px solid #ef4444;border-radius:8px;padding:1rem;'>
                        <div style='font-size:0.75rem;color:#ef4444;font-weight:700;
                        text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;'>
                        Weakest Contact Quality</div>
                        <div style='font-size:1.1rem;font-weight:700;color:#1e293b;'>{player_last(worst_xba['Batter'])}</div>
                        <div style='font-size:0.85rem;color:#475569;'>
                        xBA {worst_xba['xBA']:.3f} · xSLG {worst_xba['xSLG']:.3f} · xwOBA {worst_xba['xwOBA']:.3f} — {int(worst_xba['BIP'])} BIP</div>
                    </div>""", unsafe_allow_html=True)

    st.markdown("### Brookhaven Bandits — Contact Quality")
    build_xba_table("our", MY_TEAM)


# ─────────────────────────────────────────
#  PAGE: TEAM TOTALS
# ─────────────────────────────────────────
elif page == "Team Totals":
    st.title("Season Totals by Opponent")
    st.caption(f"{team_label(MY_TEAM)}'s combined offensive and pitching stat lines, split out by "
               "opponent, plus the team's overall totals — both with and without games against "
               "the Lowell Spinners called out separately.")

    LOWELL_CODE = "LOW_SPI1"  # TEAM_LABELS[LOWELL_CODE] == "Lowell Spinners"

    @st.cache_data(ttl=300, max_entries=3)
    def _load_official_team_gamelog(kind):
        """Official per-game team hitting/pitching log exported from
        thefuturesleague.com (Data/official_team_gamelog_hitting.csv /
        _pitching.csv). Used as the ground truth for Team Totals instead of
        the TrackMan-derived numbers, since it's the site the league (and
        opponents) actually go by. Empty DataFrame if the file isn't there."""
        f = DATA_DIR / f"official_team_gamelog_{kind}.csv"
        if not f.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(f)
        except Exception:
            return pd.DataFrame()
        if kind == "pitching" and not df.empty:
            # The site publishes WHIP but not walks directly — back it out
            # from WHIP * IP - H (IP in baseball notation: .1/.2 = thirds).
            def _outs(ip):
                whole = int(ip)
                frac = round((ip - whole) * 10)  # 0, 1, or 2 -> outs past whole innings
                return whole * 3 + frac
            outs = df["IP"].apply(_outs)
            ip_actual = outs / 3
            df["BB"] = ((df["WHIP"] * ip_actual).round() - df["H"]).clip(lower=0).astype(int)
            df["_outs"] = outs
        return df

    official_hit = _load_official_team_gamelog("hitting")
    official_pit = _load_official_team_gamelog("pitching")

    def _official_bat_line(sub):
        ab = int(sub["AB"].sum())
        h  = int(sub["H"].sum())
        doubles = int(sub["2B"].sum())
        triples = int(sub["3B"].sum())
        hr      = int(sub["HR"].sum())
        singles = h - doubles - triples - hr
        bb  = int(sub["BB"].sum())
        so  = int(sub["K"].sum())
        hbp = int(sub["HBP"].sum())
        sf  = int(sub["SF"].sum())
        sh  = int(sub["SH"].sum())
        rbi = int(sub["RBI"].sum())
        pa  = ab + bb + hbp + sf + sh
        avg = h / ab if ab else 0.0
        obp = (h + bb + hbp) / max(ab + bb + hbp + sf, 1)
        slg = (singles + 2 * doubles + 3 * triples + 4 * hr) / max(ab, 1)
        return {
            "PA": pa, "AB": ab, "H": h, "2B": doubles, "3B": triples, "HR": hr,
            "BB": bb, "SO": so, "HBP": hbp, "RBI": rbi,
            "AVG": f"{avg:.3f}", "OBP": f"{obp:.3f}", "SLG": f"{slg:.3f}",
            "OPS": f"{obp + slg:.3f}",
            "K%": f"{so / max(pa, 1):.0%}", "BB%": f"{bb / max(pa, 1):.0%}",
        }

    def _official_pitch_line(sub):
        outs = int(sub["_outs"].sum())
        ip_num  = outs / 3
        ip_disp = f"{outs // 3}.{outs % 3}"
        h  = int(sub["H"].sum())
        r  = int(sub["R"].sum())
        er = int(sub["ER"].sum())
        bb = int(sub["BB"].sum())
        k  = int(sub["K"].sum())
        hr = int(sub["HR"].sum())
        era  = 9 * er / ip_num if ip_num > 0 else 0.0
        whip = (bb + h) / ip_num if ip_num > 0 else 0.0
        return {
            "IP": ip_disp, "H": h, "R": r, "ER": er, "BB": bb, "SO": k, "HR": hr,
            "ERA": f"{era:.2f}", "WHIP": f"{whip:.2f}",
        }

    # ── TrackMan fallback — only used if the official CSVs aren't present ──
    def _team_bat_line(sub):
        pa = int((sub["PitchofPA"] == 1).sum()) if "PitchofPA" in sub.columns else 0
        ab = int(_ab_mask(sub).sum())
        h  = int(sub["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum())
        doubles = int(sub["PlayResult"].eq("Double").sum())
        triples = int(sub["PlayResult"].eq("Triple").sum())
        hr      = int(sub["PlayResult"].eq("HomeRun").sum())
        singles = h - doubles - triples - hr
        bb  = int(sub["KorBB"].eq("Walk").sum())
        so  = int(sub["KorBB"].eq("Strikeout").sum())
        hbp = int(sub["PitchCall"].eq("HitByPitch").sum())
        sac = int(sub["PlayResult"].eq("Sacrifice").sum())
        avg = h / ab if ab else 0.0
        obp = (h + bb + hbp) / max(pa - sac, 1)
        slg = (singles + 2 * doubles + 3 * triples + 4 * hr) / max(ab, 1)
        return {
            "PA": pa, "AB": ab, "H": h, "2B": doubles, "3B": triples, "HR": hr,
            "BB": bb, "SO": so, "HBP": hbp,
            "AVG": f"{avg:.3f}", "OBP": f"{obp:.3f}", "SLG": f"{slg:.3f}",
            "OPS": f"{obp + slg:.3f}",
            "K%": f"{so / max(pa, 1):.0%}", "BB%": f"{bb / max(pa, 1):.0%}",
        }

    def _team_pitch_line(sub):
        bf = int((sub["PitchofPA"] == 1).sum()) if "PitchofPA" in sub.columns else 0
        k  = int(sub["KorBB"].eq("Strikeout").sum())
        bb = int(sub["KorBB"].eq("Walk").sum())
        h  = int(sub["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"]).sum())
        runs = int(sub["RunsScored"].fillna(0).sum()) if "RunsScored" in sub.columns else 0
        outs = (sub["OutsOnPlay"].fillna(0).sum() + k) if "OutsOnPlay" in sub.columns else k
        ip_num  = outs / 3
        ip_disp = f"{int(outs // 3)}.{int(outs % 3)}"
        ab = int(_ab_mask(sub).sum())
        opp_avg = h / ab if ab else 0.0
        whip = (bb + h) / ip_num if ip_num > 0 else 0.0
        return {
            "BF": bf, "IP": ip_disp, "H": h, "R": runs, "BB": bb, "SO": k,
            "Opp AVG": f"{opp_avg:.3f}", "WHIP": f"{whip:.2f}",
            "K%": f"{k / max(bf, 1):.0%}", "BB%": f"{bb / max(bf, 1):.0%}",
        }

    our_bat = df_all[df_all["BatterTeam"] == MY_TEAM]
    our_pit = df_all[df_all["PitcherTeam"] == MY_TEAM]

    st.markdown("### Offense")
    if not official_hit.empty:
        st.caption("Source: official game log (thefuturesleague.com), through the last game "
                   "loaded into Data/official_team_gamelog_hitting.csv.")
        bat_opponents = sorted(t for t in official_hit["Opponent"].dropna().unique()
                               if t != MY_TEAM and t not in EXCLUDED_TEAMS)
        bat_rows = [{"Opponent": team_label(opp),
                    **_official_bat_line(official_hit[official_hit["Opponent"] == opp])}
                   for opp in bat_opponents]
        bat_rows.append({"Opponent": "Overall", **_official_bat_line(official_hit)})
        bat_rows.append({"Opponent": f"Overall (excl. {team_label(LOWELL_CODE)})",
                         **_official_bat_line(official_hit[official_hit["Opponent"] != LOWELL_CODE])})
        st.dataframe(pd.DataFrame(bat_rows), use_container_width=True, hide_index=True)
    elif our_bat.empty:
        st.info("No offensive data found for this team.")
    else:
        st.caption("Source: TrackMan (no Data/official_team_gamelog_hitting.csv on file).")
        bat_opponents = sorted(t for t in our_bat["PitcherTeam"].dropna().unique()
                               if t != MY_TEAM and t not in EXCLUDED_TEAMS)
        bat_rows = [{"Opponent": team_label(opp),
                    **_team_bat_line(our_bat[our_bat["PitcherTeam"] == opp])}
                   for opp in bat_opponents]
        bat_rows.append({"Opponent": "Overall", **_team_bat_line(our_bat)})
        bat_rows.append({"Opponent": f"Overall (excl. {team_label(LOWELL_CODE)})",
                         **_team_bat_line(our_bat[our_bat["PitcherTeam"] != LOWELL_CODE])})
        st.dataframe(pd.DataFrame(bat_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Pitching")
    if not official_pit.empty:
        st.caption("Source: official game log (thefuturesleague.com). Walks aren't published "
                   "directly — they're backed out from WHIP × IP − H, so BB/WHIP may be "
                   "off by a game or two of rounding.")
        pit_opponents = sorted(t for t in official_pit["Opponent"].dropna().unique()
                               if t != MY_TEAM and t not in EXCLUDED_TEAMS)
        pit_rows = [{"Opponent": team_label(opp),
                    **_official_pitch_line(official_pit[official_pit["Opponent"] == opp])}
                   for opp in pit_opponents]
        pit_rows.append({"Opponent": "Overall", **_official_pitch_line(official_pit)})
        pit_rows.append({"Opponent": f"Overall (excl. {team_label(LOWELL_CODE)})",
                         **_official_pitch_line(official_pit[official_pit["Opponent"] != LOWELL_CODE])})
        st.dataframe(pd.DataFrame(pit_rows), use_container_width=True, hide_index=True)
    elif our_pit.empty:
        st.info("No pitching data found for this team.")
    else:
        st.caption("Source: TrackMan (no Data/official_team_gamelog_pitching.csv on file).")
        pit_opponents = sorted(t for t in our_pit["BatterTeam"].dropna().unique()
                               if t != MY_TEAM and t not in EXCLUDED_TEAMS)
        pit_rows = [{"Opponent": team_label(opp),
                    **_team_pitch_line(our_pit[our_pit["BatterTeam"] == opp])}
                   for opp in pit_opponents]
        pit_rows.append({"Opponent": "Overall", **_team_pitch_line(our_pit)})
        pit_rows.append({"Opponent": f"Overall (excl. {team_label(LOWELL_CODE)})",
                         **_team_pitch_line(our_pit[our_pit["BatterTeam"] != LOWELL_CODE])})
        st.dataframe(pd.DataFrame(pit_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────
#  SEASON REPORT — hitter/pitcher wrap-up reports built for handing to
#  players on their way out at season's end. Everything below is rebuilt
#  from the Data/ folder (and Data/official_stats.csv, once it exists)
#  every time the page loads, so it updates automatically as the last
#  games of the season are entered — nothing here is hand-typed.
# ─────────────────────────────────────────
_SR_RED = "#C8102E"

def _sr_banner(subtitle):
    st.markdown(
        "<div style='text-align:center;padding:14px 0 8px 0;border-bottom:3px solid " + _SR_RED + ";'>"
        "<div style='font-family:Oswald,Inter,sans-serif;font-size:30px;font-weight:700;"
        "color:" + _SR_RED + " !important;letter-spacing:.05em;line-height:1.1;'>"
        "BROOKHAVEN BANDITS SEASON REPORT</div>"
        "<div style='font-family:Inter,sans-serif;font-size:14px;color:#000 !important;"
        "margin-top:4px;'>" + subtitle + "</div></div>",
        unsafe_allow_html=True)

def _sr_section(title):
    st.markdown(
        "<div style='font-family:Oswald,Inter,sans-serif;text-transform:uppercase;"
        "letter-spacing:.06em;font-size:1rem;font-weight:600;color:#000 !important;"
        "margin-top:1.3rem;margin-bottom:.4rem;border-bottom:1px solid #E2E8F0;padding-bottom:3px;'>"
        + title + "</div>", unsafe_allow_html=True)

_SR_SWING_C = {"StrikeSwinging", "InPlay", "FoulBallNotFieldable", "FoulBallFieldable", "FoulTip", "FoulBall"}
_SR_HITS = ["Single", "Double", "Triple", "HomeRun"]


def _official_k_bb(player_name):
    """Season K% and BB% per plate appearance from the official league stats
    (Data/official_stats.csv, built from the league stats PDF), or (None, None)
    when that file isn't on file yet — the caller then falls back to the
    TrackMan-derived rates. PA is rebuilt from AB+BB+HBP+SF+SH when the export
    doesn't carry a PA column of its own.

    Season totals only, so this is valid for the Overall split — never for
    vs RHP / vs LHP, which the official export doesn't break out."""
    ab = get_official_stat(player_name, "ab")
    bb = get_official_stat(player_name, "bb")
    so = get_official_stat(player_name, "so")
    if ab is None or bb is None or so is None:
        return None, None
    pa = get_official_stat(player_name, "pa")
    if pa is None:
        pa = (ab + bb + (get_official_stat(player_name, "hbp") or 0)
              + (get_official_stat(player_name, "sf") or 0)
              + (get_official_stat(player_name, "sh") or 0))
    if not pa or pa <= 0:
        return None, None
    return so / pa, bb / pa


def _count_matrix_html(count_matrix, pitch_types, count_totals, all_counts):
    """Pitch usage % at every ball-strike count, as an HTML table — same
    look as the Pitcher Scouting page's Full Count Matrix."""
    header = ("<tr><th style='background:#F1F5F9;padding:5px 8px;text-align:left;"
              "color:#64748b;font-size:.72rem;'>Pitch</th>")
    for b, s_ in all_counts:
        n = count_totals.get((b, s_), 0)
        header += (f"<th style='background:#F1F5F9;padding:5px 6px;text-align:center;"
                   f"color:#64748b;font-size:.72rem;'>{b}-{s_}<br>"
                   f"<span style='font-size:.6rem;'>n={n}</span></th>")
    header += "</tr>"
    rows_html = ""
    for pt in pitch_types:
        color = PITCH_COLORS.get(pt, "#64748b")
        row = (f"<tr><td style='padding:5px 8px;color:{color};font-weight:600;"
               f"font-size:.78rem;white-space:nowrap;'>{pt}</td>")
        for b, s_ in all_counts:
            pct = count_matrix.get(pt, {}).get((b, s_), 0)
            n = count_totals.get((b, s_), 0)
            if n == 0:
                bg, text, tc = "#F1F5F9", "—", "#475569"
            else:
                alpha = min(pct * 1.5, 1.0)
                hexc = color.lstrip("#")
                r, g, bl = int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16)
                bg = f"rgba({r},{g},{bl},{alpha:.2f})"
                text = f"{pct:.0%}" if pct > 0 else "—"
                tc = "#ffffff" if alpha > 0.4 else "#1e293b"
            row += (f"<td style='padding:5px 6px;text-align:center;background:{bg};"
                    f"font-size:.76rem;font-weight:700;color:{tc};'>{text}</td>")
        row += "</tr>"
        rows_html += row
    return ("<div style='overflow-x:auto;'><table style='border-collapse:collapse;width:100%;"
            "background:#fff;border-radius:8px;overflow:hidden;'><thead>" + header + "</thead>"
            "<tbody>" + rows_html + "</tbody></table></div>")


# ── Hitter season report ──
def _hitter_season_stats(bp):
    """All Season Report numbers for one slice of a hitter's pitches — pass
    an already hand-filtered slice (overall, vs RHP, or vs LHP)."""
    ab = int(_ab_mask(bp).sum())
    h = int(bp["PlayResult"].isin(_SR_HITS).sum())
    pa = int((bp["PitchofPA"] == 1).sum()) if "PitchofPA" in bp.columns else ab
    bb = int(bp["KorBB"].eq("Walk").sum())
    k = int(bp["KorBB"].eq("Strikeout").sum())
    hbp = int(bp["PitchCall"].eq("HitByPitch").sum())
    dbl = int(bp["PlayResult"].eq("Double").sum())
    trp = int(bp["PlayResult"].eq("Triple").sum())
    hr = int(bp["PlayResult"].eq("HomeRun").sum())
    singles = h - dbl - trp - hr
    ba = h / ab if ab else None
    obp = (h + bb + hbp) / pa if pa else None
    slg = (singles + 2*dbl + 3*trp + 4*hr) / ab if ab else None
    ops = (obp + slg) if (obp is not None and slg is not None) else None
    k_pct = k / pa if pa else None
    bb_pct = bb / pa if pa else None

    bip = bp[bp["PitchCall"].eq("InPlay")]
    xba, xslg, xwoba, xba_n = batter_expected_stats(bp)
    avg_ev, max_ev, ev90 = _ev_stats(bip)
    barrel, hard, n_bip = _quality_rates(bip)

    haz = _attack_zone_frame(bp)
    z_sw, o_sw, n_in, n_out = _true_zone_swing(haz)
    whiff_pct = _safe_whiff(bp)

    az_rows = []
    if len(haz) >= 10:
        vc = haz["_az"].value_counts(normalize=True)
        for z in _AZ_ORDER:
            az_rows.append({"Zone": z, "His %": float(vc.get(z, 0.0)) * 100})

    pitch_rows = []
    bp_clean = bp[bp["PitchType"].notna() & (bp["PitchType"] != "None")]
    for pt, sub in bp_clean.groupby("PitchType"):
        sub_ab = int(_ab_mask(sub).sum())
        if sub_ab < 3:
            continue
        sub_h = int(sub["PlayResult"].isin(_SR_HITS).sum())
        sub_whiff = _safe_whiff(sub)
        sub_bip = sub[sub["ExitSpeed"].notna() & sub["Angle"].notna() &
                      sub["PlayResult"].isin(_SR_HITS + ["Out", "Error", "FieldersChoice"])]
        sub_xwoba = (sub_bip.apply(lambda r: calc_xwoba_bip(r["ExitSpeed"], r["Angle"]), axis=1).mean()
                     if len(sub_bip) else None)
        pitch_rows.append({"Pitch": pt, "AB": sub_ab, "H": sub_h, "AVG": sub_h / sub_ab,
                           "Whiff%": sub_whiff, "xwOBA": sub_xwoba, "_n": len(sub)})
    pitch_rows.sort(key=lambda r: -r["_n"])

    return dict(PA=pa, AB=ab, H=h, BB=bb, K=k, HBP=hbp, Doubles=dbl, Triples=trp, HR=hr,
                BA=ba, OBP=obp, SLG=slg, OPS=ops, KPct=k_pct, BBPct=bb_pct,
                xBA=xba, xSLG=xslg, xwOBA=xwoba,
                AvgEV=avg_ev, MaxEV=max_ev, EV90=ev90,
                Barrel=barrel, Hard=hard, NBip=n_bip,
                ZSwing=z_sw, OSwing=o_sw, NIn=n_in, NOut=n_out, Whiff=whiff_pct,
                AttackZones=az_rows, PitchRows=pitch_rows, RawDF=bp, Bip=bip, Haz=haz)


@st.cache_data(ttl=600, max_entries=2)
def _league_hitter_benchmarks(_hash):
    """League-wide (pooled, every hitter/every team) version of every
    _hitter_season_stats number, for 'Lg avg' reference text under each stat."""
    return _hitter_season_stats(df_all)


def _render_hitter_season_report(batter):
    bp_all = df_all[df_all["Batter"] == batter].copy()
    for c in ["ExitSpeed", "Angle", "PlateLocSide", "PlateLocHeight"]:
        bp_all[c] = pd.to_numeric(bp_all[c], errors="coerce")
    side = bp_all["BatterSide"].dropna().iloc[0] if bp_all["BatterSide"].notna().any() else "?"

    _sr_banner(f"{player_last(batter)} &middot; Bats {side}")

    _sr_section("Official Season Line")
    off_row = {}
    for lbl, key in [("G", "g"), ("AB", "ab"), ("H", "h"), ("2B", "doubles"), ("3B", "triples"),
                      ("HR", "hr"), ("RBI", "rbi"), ("BB", "bb"), ("SO", "so"), ("SB", "sb"),
                      ("AVG", "ba"), ("OBP", "obp"), ("SLG", "slg"), ("OPS", "ops")]:
        v = get_official_stat(batter, key)
        if v is not None:
            off_row[lbl] = (f"{v:.3f}" if key in ("ba", "obp", "slg", "ops") else str(int(v)))
    if off_row:
        st.dataframe(pd.DataFrame([off_row]), use_container_width=True, hide_index=True)
    else:
        st.caption("No Data/official_stats.csv loaded yet — add the season stat export and this "
                   "fills in automatically.")

    st.divider()
    tab_overall, tab_rhp, tab_lhp = st.tabs(["Overall", "vs Righties (RHP)", "vs Lefties (LHP)"])
    with tab_overall:
        _render_hitter_split(batter, bp_all, "Overall")
    with tab_rhp:
        _render_hitter_split(batter, bp_all[bp_all["PitcherThrows"] == "Right"], "vs RHP")
    with tab_lhp:
        _render_hitter_split(batter, bp_all[bp_all["PitcherThrows"] == "Left"], "vs LHP")


def _render_hitter_split(batter, bp, split_choice):
    s = _hitter_season_stats(bp)
    lg = _league_hitter_benchmarks((len(df_all), tuple(df_all.columns)))

    def _lg3(key):
        v = lg.get(key)
        return f"Lg: {v:.3f}" if v is not None else None

    def _lg1(key):
        v = lg.get(key)
        return f"Lg: {v:.1f}" if v is not None else None

    def _lgpct2(key):
        v = lg.get(key)
        return f"Lg: {v:.0f}%" if v == v else None

    def _lgpct(key):
        v = lg.get(key)
        return f"Lg: {100*v:.0f}%" if v is not None else None

    _sr_section("Advanced Stats")
    a = st.columns(4)
    a[0].metric("xwOBA", f"{s['xwOBA']:.3f}" if s["xwOBA"] is not None else "—",
               delta=_lg3("xwOBA"), delta_color="off")
    a[1].metric("xBA", f"{s['xBA']:.3f}" if s["xBA"] is not None else "—",
               delta=_lg3("xBA"), delta_color="off")
    a[2].metric("Barrel%", f"{s['Barrel']:.1f}%" if s["Barrel"] == s["Barrel"] else "—",
               delta=_lgpct2("Barrel"), delta_color="off")
    a[3].metric("Hard-Hit%", f"{s['Hard']:.1f}%" if s["Hard"] == s["Hard"] else "—",
               delta=_lgpct2("Hard"), delta_color="off")
    b = st.columns(3)
    b[0].metric("Avg EV", f"{s['AvgEV']:.1f}" if s["AvgEV"] is not None else "—",
               delta=_lg1("AvgEV"), delta_color="off")
    b[1].metric("Max EV", f"{s['MaxEV']:.1f}" if s["MaxEV"] is not None else "—")
    b[2].metric("EV90", f"{s['EV90']:.1f}" if s["EV90"] is not None else "—",
               delta=_lg1("EV90"), delta_color="off",
               help="90th percentile exit velocity on batted balls.")

    _sr_section("Plate Discipline")
    dd = st.columns(4)
    dd[0].metric("Z-Swing%", f"{s['ZSwing']:.0f}%", delta=_lgpct2("ZSwing"), delta_color="off",
                help=f"n={s['NIn']}")
    dd[1].metric("O-Swing%", f"{s['OSwing']:.0f}%", delta=_lgpct2("OSwing"), delta_color="off",
                help=f"n={s['NOut']}")
    dd[2].metric("Whiff%", f"{100*s['Whiff']:.0f}%" if s["Whiff"] is not None else "—",
                delta=_lgpct("Whiff"), delta_color="off")
    k_pct, bb_pct, kbb_src = s["KPct"], s["BBPct"], "TrackMan pitch data"
    if split_choice == "Overall":
        _ok, _obb = _official_k_bb(batter)
        if _ok is not None:
            k_pct, bb_pct, kbb_src = _ok, _obb, "official league season stats"
    dd[3].metric("K% / BB%", f"{100*(k_pct or 0):.0f}% / {100*(bb_pct or 0):.0f}%",
                delta=(f"Lg: {100*lg['KPct']:.0f}% / {100*lg['BBPct']:.0f}%"
                      if lg.get("KPct") is not None and lg.get("BBPct") is not None else None),
                delta_color="off",
                help=f"Per plate appearance, from {kbb_src}.")

    _sr_section("Performance vs Each Pitch Type")
    if s["PitchRows"]:
        pr_df = pd.DataFrame(s["PitchRows"])[["Pitch", "AB", "H", "AVG", "Whiff%", "xwOBA"]].copy()
        pr_df["AVG"] = pr_df["AVG"].map(lambda v: f"{v:.3f}")
        pr_df["Whiff%"] = pr_df["Whiff%"].map(lambda v: f"{100*v:.0f}%" if v is not None else "—")
        pr_df["xwOBA"] = pr_df["xwOBA"].map(lambda v: f"{v:.3f}" if v is not None else "—")
        st.dataframe(pr_df, use_container_width=True, hide_index=True)
    else:
        st.caption("Not enough at-bats against any single pitch type yet (need 3+ AB).")

    _sr_section("Heat Maps")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.caption("Pitch Location (seen)")
        if len(bp) >= 5:
            _render_kde_heatmap(bp, weight_col=None, key_suffix=f"sr_h_loc_{batter}_{split_choice}")
        else:
            st.info("Not enough location data.")
    with h2:
        st.caption("Hard Contact")
        bip_loc = s["Bip"][s["Bip"]["PlateLocSide"].notna() & s["Bip"]["PlateLocHeight"].notna()]
        if len(bip_loc) >= 5:
            _render_kde_heatmap(bip_loc, weight_col="ExitSpeed",
                                key_suffix=f"sr_h_ev_{batter}_{split_choice}")
        else:
            st.info("Not enough contact data.")
    with h3:
        st.caption("Whiffs")
        sw = bp[bp["PitchCall"].isin(_SR_SWING_C) & bp["PlateLocSide"].notna() &
               bp["PlateLocHeight"].notna()].copy()
        if len(sw):
            sw["_whiff_w"] = sw["PitchCall"].eq("StrikeSwinging").astype(float)
        if len(sw) >= 5:
            _render_kde_heatmap(sw, weight_col="_whiff_w", key_suffix=f"sr_h_wh_{batter}_{split_choice}")
        else:
            st.info("Not enough swing data.")

    _sr_section("Swing Decisions")
    if s["AttackZones"]:
        lg_az = _league_attack_zone_rates(df_all, "swing")
        zrows = [{"Zone": r["Zone"], "His Swing %": f"{r['His %']:.0f}%",
                 "League Swing %": f"{lg_az[r['Zone']]:.0f}%"}
                for r in s["AttackZones"]]
        st.dataframe(pd.DataFrame(zrows), use_container_width=True, hide_index=True)
        sw_by = {}
        hz = s["Haz"]
        for z in _AZ_ORDER:
            zz = hz[hz["_az"] == z]
            sw_by[z] = float(zz["_swing"].mean() * 100) if len(zz) else 0.0
        components.html(_attack_zone_svg(sw_by, "Swing% by zone", is_rate=True), height=380)
    else:
        st.caption("Not enough located pitches yet for a swing-decision breakdown.")


def _build_hitter_season_pdf(batter_name, side_lbl, off_row, splits, logo_path="assets/nashua_logo.png",
                             full_name=None):
    """splits: {"Overall": stats, "vs RHP": stats, "vs LHP": stats}, each from
    _hitter_season_stats(). Mirrors the on-screen Season Report page: a
    monochrome black/white/gray layout (no team color) and one full page
    per split (Overall, vs RHP, vs LHP) with the same stat cards, tables,
    and heat maps shown on-screen — not a condensed side-by-side comparison
    table.

    full_name is the TrackMan "Last, First" name, used to look the hitter up in
    the official league stats for K%/BB%; batter_name is display-only (last name),
    which would collide between two players sharing a surname."""
    import io, os
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
                                    PageBreak, KeepTogether)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    # Monochrome, print-shop palette — no team color, reads like an official
    # scouting/stat sheet rather than a branded flyer.
    RED = colors.HexColor("#1A1A1A")     # kept as RED for call-site compat; now near-black
    INK = colors.HexColor("#000000")
    GRAY = colors.HexColor("#4B4B4B")
    PANEL = colors.HexColor("#F4F4F4")
    STRIPE = colors.HexColor("#F0F0F0")
    RULE = colors.HexColor("#B9B9B9")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.55*inch, bottomMargin=0.55*inch)
    styles = getSampleStyleSheet()
    story = []

    def _rule(height=2.0):
        t = Table([[""]], colWidths=[7.4*inch], rowHeights=[height])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), RED),
                              ("TOPPADDING", (0, 0), (-1, -1), 0),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        return t

    title = ParagraphStyle("s_t", parent=styles["Title"], textColor=INK, fontSize=22, spaceAfter=0, alignment=0)
    sub = ParagraphStyle("s_s", parent=styles["Normal"], textColor=GRAY, fontSize=12, spaceAfter=0)
    title_block = [Paragraph("Brookhaven Bandits Season Report", title),
                   Paragraph(f"{batter_name} (Bats {side_lbl})", sub)]
    if logo_path and os.path.exists(logo_path):
        try:
            head = Table([[Image(logo_path, width=0.7*inch, height=0.7*inch), title_block]],
                        colWidths=[0.85*inch, 6.55*inch])
        except Exception:
            head = Table([[title_block]], colWidths=[7.4*inch])
    else:
        head = Table([[title_block]], colWidths=[7.4*inch])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (0, 0), 0)]))
    story += [head, Spacer(1, 8), _rule(1.6), Spacer(1, 12)]

    h = ParagraphStyle("s_h", parent=styles["Heading2"], textColor=INK, fontSize=14, spaceBefore=12, spaceAfter=4)
    h2 = ParagraphStyle("s_h2", parent=styles["Heading1"], textColor=INK, fontSize=18, spaceBefore=2, spaceAfter=8)
    body = ParagraphStyle("s_b", parent=styles["Normal"], textColor=INK, fontSize=9.5, spaceAfter=2)

    def _section(t):
        story.append(Paragraph(t, h)); story.append(_rule(1.0)); story.append(Spacer(1, 6))

    def _tbl(data, col_widths=None):
        t = Table(data, hAlign="LEFT", colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("TEXTCOLOR", (0, 1), (-1, -1), INK), ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("GRID", (0, 0), (-1, -1), 0.6, RULE), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, STRIPE]),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TOPPADDING", (0, 0), (-1, -1), 5.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7)]))
        return t

    def _card(label, value, sub_txt, width):
        lbl_s = ParagraphStyle("c_l", fontName="Helvetica-Bold", fontSize=7, textColor=GRAY, leading=8.5)
        val_s = ParagraphStyle("c_v", fontName="Helvetica-Bold", fontSize=16, textColor=INK, leading=18)
        sub_s = ParagraphStyle("c_s", fontName="Helvetica", fontSize=7, textColor=GRAY, leading=8.5)
        rows = [[Paragraph(label.upper(), lbl_s)], [Paragraph(str(value), val_s)],
                [Paragraph(sub_txt if sub_txt else "&nbsp;", sub_s)]]
        t = Table(rows, colWidths=[width])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PANEL), ("LINEBEFORE", (0, 0), (0, -1), 2.2, INK),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
        return t

    def _card_row(specs):
        n = len(specs)
        gap = 0.08 * inch
        card_w = (7.4 * inch - gap * (n - 1)) / n if n else 7.4 * inch
        row_data, col_widths = [], []
        for i, (label, value, sub_txt) in enumerate(specs):
            row_data.append(_card(label, value, sub_txt, card_w))
            col_widths.append(card_w)
            if i < n - 1:
                row_data.append(""); col_widths.append(gap)
        t = Table([row_data], colWidths=col_widths)
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                              ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 12)]))
        return t

    def _f3(v): return f"{v:.3f}" if v is not None else "—"
    def _f1(v): return f"{v:.1f}" if v is not None else "—"
    def _fpct(v): return f"{100*v:.0f}%" if v is not None else "—"
    def _fpct2(v): return f"{v:.0f}%" if v == v else "—"

    _section("Official Season Line")
    if off_row:
        story.append(_tbl([list(off_row.keys()), [str(v) for v in off_row.values()]]))
    else:
        story.append(Paragraph("No Data/official_stats.csv on file yet.", body))
    story.append(Spacer(1, 10))

    lg = _league_hitter_benchmarks((len(df_all), tuple(df_all.columns)))

    def _lg3(key):
        v = lg.get(key); return f"Lg: {v:.3f}" if v is not None else None
    def _lg1(key):
        v = lg.get(key); return f"Lg: {v:.1f}" if v is not None else None
    def _lgpct2(key):
        v = lg.get(key); return f"Lg: {v:.0f}%" if v == v else None
    def _lgpct(key):
        v = lg.get(key); return f"Lg: {100*v:.0f}%" if v is not None else None

    for i, (sp_key, sp_label) in enumerate(
            [("Overall", "Overall"), ("vs RHP", "vs Righties (RHP)"), ("vs LHP", "vs Lefties (LHP)")]):
        s = splits[sp_key]
        if i > 0:
            story.append(PageBreak())
        story.append(Paragraph(sp_label, h2)); story.append(_rule(1.6)); story.append(Spacer(1, 8))

        story.append(KeepTogether([
            Paragraph("Advanced Stats", h),
            _card_row([
                ("xwOBA", _f3(s["xwOBA"]), _lg3("xwOBA")), ("xBA", _f3(s["xBA"]), _lg3("xBA")),
                ("Barrel%", _fpct2(s["Barrel"]), _lgpct2("Barrel")),
                ("Hard-Hit%", _fpct2(s["Hard"]), _lgpct2("Hard")),
            ]),
            _card_row([
                ("Avg EV", _f1(s["AvgEV"]), _lg1("AvgEV")), ("Max EV", _f1(s["MaxEV"]), None),
                ("EV90", _f1(s["EV90"]), _lg1("EV90")),
            ]),
        ]))

        kbb_sub = (f"Lg: {100*lg['KPct']:.0f}% / {100*lg['BBPct']:.0f}%"
                  if lg.get("KPct") is not None and lg.get("BBPct") is not None else None)
        k_pct, bb_pct = s["KPct"], s["BBPct"]
        if sp_key == "Overall":
            _ok, _obb = _official_k_bb(full_name or batter_name)
            if _ok is not None:
                k_pct, bb_pct = _ok, _obb
        story.append(KeepTogether([
            Paragraph("Plate Discipline", h),
            _card_row([
                ("Z-Swing%", f"{s['ZSwing']:.0f}%", _lgpct2("ZSwing")),
                ("O-Swing%", f"{s['OSwing']:.0f}%", _lgpct2("OSwing")),
                ("Whiff%", _fpct(s["Whiff"]), _lgpct("Whiff")),
                ("K% / BB%", f"{100*(k_pct or 0):.0f}% / {100*(bb_pct or 0):.0f}%", kbb_sub),
            ]),
        ]))

        pt_content = [Paragraph("Performance vs Each Pitch Type", h), _rule(1.1), Spacer(1, 5)]
        if s["PitchRows"]:
            pt_rows = [["Pitch", "AB", "H", "AVG", "Whiff%", "xwOBA"]]
            for r in s["PitchRows"]:
                pt_rows.append([r["Pitch"], str(r["AB"]), str(r["H"]), f"{r['H']/r['AB']:.3f}",
                                _fpct(r["Whiff%"]), _f3(r["xwOBA"])])
            pt_content.append(_tbl(pt_rows))
        else:
            pt_content.append(Paragraph("Not enough at-bats against any single pitch type.", body))
        story.append(KeepTogether(pt_content))

        hm_content = [Paragraph("Heat Maps", h), _rule(1.1), Spacer(1, 5)]
        bip = s["Bip"]
        sw = s["RawDF"][s["RawDF"]["PitchCall"].isin(_SR_SWING_C)].copy()
        if len(sw):
            sw["_whiff_w"] = sw["PitchCall"].eq("StrikeSwinging").astype(float)
        panels = [(s["RawDF"], None, "Pitch Location"), (bip, "ExitSpeed", "Hard Contact"), (sw, "_whiff_w", "Whiffs")]
        imgs = [_kde_heatmap_png(d, weight_col=w) if len(d) >= 5 else None for d, w, _ in panels]
        if any(imgs):
            cap_row = [Paragraph(f"<b>{lbl}</b>", body) for _, _, lbl in panels]
            img_row = [Image(io.BytesIO(png), width=2.1*inch, height=2.35*inch) if png
                      else Paragraph("Not enough data", body) for png in imgs]
            t = Table([cap_row, img_row], colWidths=[2.2*inch]*3)
            t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            hm_content.append(t)
        else:
            hm_content.append(Paragraph("Not enough location data for heat maps.", body))
        story.append(KeepTogether(hm_content))

        sd_content = [Paragraph("Swing Decisions", h), _rule(1.1), Spacer(1, 5)]
        if s["AttackZones"]:
            lg_az = _league_attack_zone_rates(df_all, "swing")
            az_rows = [["Zone", "His Swing %", "League Swing %"]]
            for r in s["AttackZones"]:
                az_rows.append([r["Zone"], f"{r['His %']:.0f}%", f"{lg_az[r['Zone']]:.0f}%"])
            sd_content.append(_tbl(az_rows))
        else:
            sd_content.append(Paragraph("Not enough located pitches for a swing-decision breakdown.", body))
        story.append(KeepTogether(sd_content))

    doc.build(story)
    return buf.getvalue()


# ── Pitcher season report ──
def _pitcher_season_stats(pitcher_name, pp):
    pp = pp.copy()
    for c in ["RelSpeed", "SpinRate", "InducedVertBreak", "HorzBreak",
             "PlateLocSide", "PlateLocHeight", "Balls", "Strikes"]:
        if c in pp.columns:
            pp[c] = pd.to_numeric(pp[c], errors="coerce")
    pp_clean = pp[pp["PitchType"].notna() & (pp["PitchType"] != "None")]
    total = len(pp_clean)

    agg_dict = {"Count": ("PitchType", "count"), "AvgVelo": ("RelSpeed", "mean")}
    if "SpinRate" in pp_clean.columns: agg_dict["AvgSpin"] = ("SpinRate", "mean")
    if "InducedVertBreak" in pp_clean.columns: agg_dict["AvgIVB"] = ("InducedVertBreak", "mean")
    if "HorzBreak" in pp_clean.columns: agg_dict["AvgHB"] = ("HorzBreak", "mean")
    mix = pp_clean.groupby("PitchType").agg(**agg_dict).reset_index() if total else \
        pd.DataFrame(columns=["PitchType", "Count", "AvgVelo"])
    if len(mix):
        mix["Pct"] = (mix["Count"] / max(total, 1) * 100).round(1)
        mix = mix.sort_values("Count", ascending=False).reset_index(drop=True)
        stuff = (stuff_plus_df[stuff_plus_df["Pitcher"] == pitcher_name][["PitchType", "StuffPlus"]]
                if not stuff_plus_df.empty else pd.DataFrame(columns=["PitchType", "StuffPlus"]))
        mix = mix.merge(stuff, on="PitchType", how="left")

    pitch_types = [p for p in mix["PitchType"].tolist() if p not in ("Undefined", "Other")] if len(mix) else []

    az_rows = []
    if "PlateLocSide" in pp_clean.columns:
        _pz = pp_clean.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if len(_pz) >= 10:
            _pz = _pz.copy()
            _pz["_z"] = _pz.apply(lambda r: attack_zone(r["PlateLocSide"], r["PlateLocHeight"]), axis=1)
            _lg = _league_attack_zone_rates(df_all, "pitch")
            _pv = _pz["_z"].value_counts(normalize=True) * 100
            for z in _AZ_ORDER:
                az_rows.append({"Zone": z, "His %": float(_pv.get(z, 0.0)), "League %": _lg[z]})

    all_counts = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1),
                 (0, 2), (1, 2), (2, 2), (3, 2)]
    count_matrix, count_totals = {}, {}
    if "Balls" in pp_clean.columns and "Strikes" in pp_clean.columns:
        for b, s_ in all_counts:
            cp = pp_clean[(pp_clean["Balls"] == b) & (pp_clean["Strikes"] == s_)]
            count_totals[(b, s_)] = len(cp)
            for pt in pitch_types:
                pct = (cp["PitchType"] == pt).sum() / len(cp) if len(cp) else 0
                count_matrix.setdefault(pt, {})[(b, s_)] = pct

    fip_m = _season_pitcher_fip_metrics(pp)
    xera = calc_xera_estimate(pp, df_all)

    return dict(Mix=mix, PitchTypes=pitch_types, AttackZones=az_rows,
                CountMatrix=count_matrix, CountTotals=count_totals, AllCounts=all_counts,
                FipMetrics=fip_m, xERA=xera, Total=total, Pitches=pp, PitchesClean=pp_clean)


def _movement_vs_league_png(pitcher_name, pp_overall, width_in=4.6, height_in=4.0):
    """Matplotlib twin of the on-screen movement-vs-league scatter, for PDF embedding."""
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    try:
        import matchup_model as mm
    except Exception:
        return None

    prof = mm.pitcher_arsenal_profile(pp_overall, min_n=10)
    if not prof:
        return None
    lg = df_all.copy()
    for c in ["InducedVertBreak", "HorzBreak"]:
        lg[c] = pd.to_numeric(lg[c], errors="coerce")
    lg_avg = (lg[lg["PitchType"].isin(prof.keys())]
             .groupby(["Pitcher", "PitchType"])
             .agg(IVB=("InducedVertBreak", "mean"), HB=("HorzBreak", "mean"), N=("PitchType", "count"))
             .reset_index())
    lg_avg = lg_avg[(lg_avg["N"] >= 10) & (lg_avg["Pitcher"] != pitcher_name)]

    fig, ax = _plt.subplots(figsize=(width_in, height_in), dpi=150)
    for pt in prof:
        color = PITCH_COLORS.get(pt, "#64748b")
        sub = lg_avg[lg_avg["PitchType"] == pt]
        if len(sub):
            ax.scatter(sub["HB"], sub["IVB"], color=color, s=14, alpha=0.18, linewidths=0)
    for pt, d in prof.items():
        color = PITCH_COLORS.get(pt, "#64748b")
        ax.scatter([d["hb"]], [d["ivb"]], color=color, s=140, marker="*",
                  edgecolors="#1e293b", linewidths=1.2, zorder=5)
        ax.annotate(pt, (d["hb"], d["ivb"]), fontsize=7, color="#1e293b",
                   xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="#cbd5e1", linewidth=0.8)
    ax.axvline(0, color="#cbd5e1", linewidth=0.8)
    ax.set_xlim(-26, 26); ax.set_ylim(-24, 32)
    ax.set_xlabel("Horizontal Break (in)", fontsize=8)
    ax.set_ylabel("Induced Vert Break (in)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.4)
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    _plt.close(fig)
    return buf.getvalue()


def _render_movement_vs_league(pitcher_name, pp_overall):
    try:
        import matchup_model as mm
    except Exception as e:
        st.caption(f"Movement-vs-league plot unavailable: {type(e).__name__} — {e}")
        return
    prof = mm.pitcher_arsenal_profile(pp_overall, min_n=10)
    if not prof:
        st.caption("Not enough pitches of any single type (10+) to plot movement.")
        return
    lg = df_all.copy()
    for c in ["InducedVertBreak", "HorzBreak"]:
        lg[c] = pd.to_numeric(lg[c], errors="coerce")
    lg_avg = (lg[lg["PitchType"].isin(prof.keys())]
             .groupby(["Pitcher", "PitchType"])
             .agg(IVB=("InducedVertBreak", "mean"), HB=("HorzBreak", "mean"), N=("PitchType", "count"))
             .reset_index())
    lg_avg = lg_avg[(lg_avg["N"] >= 10) & (lg_avg["Pitcher"] != pitcher_name)]

    fig = go.Figure()
    for pt in prof:
        color = PITCH_COLORS.get(pt, "#64748b")
        sub = lg_avg[lg_avg["PitchType"] == pt]
        if len(sub):
            fig.add_trace(go.Scatter(x=sub["HB"], y=sub["IVB"], mode="markers", showlegend=False,
                                     marker=dict(color=color, size=6, opacity=0.18), hoverinfo="skip"))
    for pt, d in prof.items():
        color = PITCH_COLORS.get(pt, "#64748b")
        fig.add_trace(go.Scatter(x=[d["hb"]], y=[d["ivb"]], mode="markers+text", name=pt,
                                 text=[pt], textposition="top center",
                                 marker=dict(color=color, size=18, symbol="star",
                                            line=dict(color="#1e293b", width=1.5))))
    fig.add_hline(y=0, line_color="#cbd5e1"); fig.add_vline(x=0, line_color="#cbd5e1")
    fig.update_layout(height=460, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                      font=dict(color="#1e293b"), showlegend=False,
                      xaxis=dict(title="Horizontal Break (in)", range=[-26, 26], gridcolor="#E2E8F0",
                                zeroline=False, scaleanchor="y", scaleratio=1),
                      yaxis=dict(title="Induced Vert Break (in)", range=[-24, 32], gridcolor="#E2E8F0",
                                zeroline=False))
    st.plotly_chart(fig, use_container_width=True, key=f"sr_p_move_{pitcher_name}")
    st.caption("Large stars = this pitcher's average movement per pitch type. Faded dots = every "
              "other pitcher in the league averaging 10+ of that same pitch type this season.")


@st.cache_data(ttl=600, max_entries=2)
def _league_pitcher_benchmarks(_hash):
    """League-wide (pooled, every pitcher/every team) FIP/xFIP for 'Lg avg' reference text."""
    return _season_pitcher_fip_metrics(df_all)


def _render_pitcher_season_report(pitcher):
    pp_all = df_all[df_all["Pitcher"] == pitcher].copy()
    throws = pp_all["PitcherThrows"].dropna().iloc[0] if pp_all["PitcherThrows"].notna().any() else "?"
    throws_lbl = {"Right": "RHP", "Left": "LHP"}.get(throws, throws)

    _sr_banner(f"{player_last(pitcher)} &middot; {throws_lbl}")

    _sr_section("Official Season Line")
    off_row = {}
    for lbl, key in [("G", "app"), ("GS", "gs"), ("W", "w"), ("L", "l"), ("SV", "sv"), ("IP", "ip"),
                      ("H", "h"), ("ER", "er"), ("BB", "bb"), ("SO", "so"), ("ERA", "era"), ("WHIP", "whip")]:
        v = get_official_stat(pitcher, key)
        if v is not None:
            off_row[lbl] = (f"{v:.2f}" if key in ("era", "whip", "ip") else str(int(v)))
    if off_row:
        st.dataframe(pd.DataFrame([off_row]), use_container_width=True, hide_index=True)
    else:
        st.caption("No Data/official_stats.csv loaded yet — add the season stat export and this "
                   "fills in automatically.")

    st.divider()
    tab_overall, tab_rhh, tab_lhh = st.tabs(["Overall", "vs Righties (RHH)", "vs Lefties (LHH)"])
    with tab_overall:
        _render_pitcher_split(pitcher, pp_all, "Overall")
        _sr_section("Movement vs League")
        _render_movement_vs_league(pitcher, pp_all)
        _sr_section("Tunneling")
        prof = None
        try:
            import matchup_model as mm
            prof = mm.pitcher_arsenal_profile(pp_all, min_n=15)
        except Exception as e:
            st.caption(f"Tunneling unavailable: {type(e).__name__} — {e}")
        if prof is not None:
            if len(prof) >= 2:
                trows = mm.tunnel_pairs(prof)
                tdf = pd.DataFrame(trows).sort_values("score", ascending=False)
                tdf["release_gap"] = tdf["release_gap"].round(1)
                tdf["move_sep"] = tdf["move_sep"].round(1)
                tdf["velo_gap"] = tdf["velo_gap"].round(1)
                tdf = tdf.rename(columns={"a": "Pitch A", "b": "Pitch B", "release_gap": "Release Gap (in)",
                                          "move_sep": "Movement Sep (in)", "velo_gap": "Velo Gap", "grade": "Grade"})
                st.dataframe(tdf[["Pitch A", "Pitch B", "Release Gap (in)", "Movement Sep (in)", "Velo Gap", "Grade"]],
                            use_container_width=True, hide_index=True)
            else:
                st.caption("Needs 2+ pitch types with 15+ thrown to grade tunneling.")
        st.caption("Movement and tunneling reflect his whole-season arsenal — they aren't split by "
                  "batter hand since a pitcher's own release/shape doesn't change by who's up.")
    with tab_rhh:
        _render_pitcher_split(pitcher, pp_all[pp_all["BatterSide"] == "Right"], "vs RHH")
    with tab_lhh:
        _render_pitcher_split(pitcher, pp_all[pp_all["BatterSide"] == "Left"], "vs LHH")


def _render_pitcher_split(pitcher, pp, split_choice):
    s = _pitcher_season_stats(pitcher, pp)

    _sr_section("Stuff+ by Pitch")
    if len(s["Mix"]):
        cols = st.columns(len(s["Mix"]))
        for i, row in s["Mix"].iterrows():
            with cols[i]:
                sp = row.get("StuffPlus")
                color = PITCH_COLORS.get(row["PitchType"], "#64748b")
                val = f"{sp:.0f}" if pd.notna(sp) else "—"
                st.markdown(
                    f"<div style='text-align:center;border:1.5px solid {color};border-radius:8px;padding:8px;'>"
                    f"<div style='color:{color};font-weight:700;font-size:.72rem;text-transform:uppercase;'>"
                    f"{row['PitchType']}</div>"
                    f"<div style='font-size:1.5rem;font-weight:800;color:#000 !important;'>{val}</div>"
                    f"<div style='font-size:.68rem;color:#64748b;'>Stuff+</div></div>",
                    unsafe_allow_html=True)
    else:
        st.caption("No pitch-type data.")

    _sr_section("Overall Pitch Mix" + ("" if split_choice == "Overall" else f" ({split_choice})"))
    if len(s["Mix"]):
        disp = s["Mix"].copy()
        disp["Usage%"] = disp["Pct"].map(lambda v: f"{v:.0f}%")
        disp["Velo"] = disp["AvgVelo"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
        cols_show = ["PitchType", "Usage%", "Count", "Velo"]
        if "AvgIVB" in disp.columns:
            disp["IVB"] = disp["AvgIVB"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
            cols_show.append("IVB")
        if "AvgHB" in disp.columns:
            disp["HB"] = disp["AvgHB"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
            cols_show.append("HB")
        st.dataframe(disp[cols_show], use_container_width=True, hide_index=True)
    else:
        st.caption("No pitch-type data.")

    _sr_section("Location Numbers — Attack Zones")
    if s["AttackZones"]:
        zdf = pd.DataFrame([{"Zone": r["Zone"], "His %": f"{r['His %']:.0f}%",
                            "League %": f"{r['League %']:.0f}%"} for r in s["AttackZones"]])
        st.dataframe(zdf, use_container_width=True, hide_index=True)
    else:
        st.caption("Not enough located pitches for a zone profile.")

    _sr_section("Full Count Matrix")
    if s["CountMatrix"]:
        st.markdown(_count_matrix_html(s["CountMatrix"], s["PitchTypes"], s["CountTotals"], s["AllCounts"]),
                   unsafe_allow_html=True)
    else:
        st.caption("Count data not available.")

    _sr_section("Location Heat Maps")
    hh1, hh2, hh3 = st.columns(3)
    pz = (pp[pp["PlateLocSide"].notna() & pp["PlateLocHeight"].notna()]
         if "PlateLocSide" in pp.columns else pp.iloc[0:0])
    with hh1:
        st.caption("Pitch Location")
        if len(pz) >= 5:
            _render_kde_heatmap(pz, weight_col=None, key_suffix=f"sr_p_loc_{pitcher}_{split_choice}")
        else:
            st.info("Not enough location data.")
    with hh2:
        st.caption("Hard Contact Allowed")
        bip = pz[pz["ExitSpeed"].notna()] if len(pz) else pz
        if len(bip) >= 5:
            _render_kde_heatmap(bip, weight_col="ExitSpeed", key_suffix=f"sr_p_ev_{pitcher}_{split_choice}")
        else:
            st.info("Not enough contact data.")
    with hh3:
        st.caption("Whiff Zones")
        sw = pz[pz["PitchCall"].isin(_SR_SWING_C)].copy() if len(pz) else pz
        if len(sw):
            sw["_whiff_w"] = sw["PitchCall"].eq("StrikeSwinging").astype(float)
        if len(sw) >= 5:
            _render_kde_heatmap(sw, weight_col="_whiff_w", key_suffix=f"sr_p_wh_{pitcher}_{split_choice}")
        else:
            st.info("Not enough swing data.")

    _sr_section("FIP / xFIP / xERA")
    fm = s["FipMetrics"]
    lg_fm = _league_pitcher_benchmarks((len(df_all), tuple(df_all.columns)))
    fc = st.columns(4)
    fc[0].metric("IP", fm["IP"])
    fc[1].metric("FIP", f"{fm['FIP']:.2f}" if fm["FIP"] is not None else "—",
               delta=(f"Lg: {lg_fm['FIP']:.2f}" if lg_fm["FIP"] is not None else None), delta_color="off")
    fc[2].metric("xFIP", f"{fm['xFIP']:.2f}" if fm["xFIP"] is not None else "—",
               delta=(f"Lg: {lg_fm['xFIP']:.2f}" if lg_fm["xFIP"] is not None else None), delta_color="off")
    fc[3].metric("xERA*", f"{s['xERA']:.2f}" if s["xERA"] is not None else "—",
               delta=f"Lg: {_SEASON_FIP_CONSTANT:.2f}", delta_color="off")
    st.caption("*xERA is a DiamondIntel estimate built from tracked exit velo/launch angle allowed, "
              "rescaled to a runs/9 basis — not Statcast's proprietary metric.")


def _build_pitcher_season_pdf(pitcher_name, throws_lbl, off_row, splits, overall_pp,
                              logo_path="assets/nashua_logo.png"):
    """splits: {"Overall": stats, "vs RHH": stats, "vs LHH": stats}, each from
    _pitcher_season_stats(). overall_pp: full-season unfiltered pitches, for
    tunneling + movement-vs-league (neither is meaningfully hand-specific).
    Mirrors the on-screen Season Report page: monochrome black/white/gray
    layout (no team color), one full page per split with the same stat
    cards and tables shown on-screen, not a condensed comparison table."""
    import io, os
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
                                    PageBreak, KeepTogether)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    # Monochrome, print-shop palette — no team color, reads like an official
    # scouting/stat sheet rather than a branded flyer.
    RED = colors.HexColor("#1A1A1A")     # kept as RED for call-site compat; now near-black
    INK = colors.HexColor("#000000")
    GRAY = colors.HexColor("#4B4B4B")
    PANEL = colors.HexColor("#F4F4F4")
    STRIPE = colors.HexColor("#F0F0F0")
    RULE = colors.HexColor("#B9B9B9")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.55*inch, bottomMargin=0.55*inch)
    styles = getSampleStyleSheet()
    story = []

    def _rule(height=2.0):
        t = Table([[""]], colWidths=[7.4*inch], rowHeights=[height])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), RED),
                              ("TOPPADDING", (0, 0), (-1, -1), 0),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        return t

    title = ParagraphStyle("sp_t", parent=styles["Title"], textColor=INK, fontSize=22, spaceAfter=0, alignment=0)
    sub = ParagraphStyle("sp_s", parent=styles["Normal"], textColor=GRAY, fontSize=12, spaceAfter=0)
    title_block = [Paragraph("Brookhaven Bandits Season Report", title),
                   Paragraph(f"{pitcher_name} ({throws_lbl})", sub)]
    if logo_path and os.path.exists(logo_path):
        try:
            head = Table([[Image(logo_path, width=0.7*inch, height=0.7*inch), title_block]],
                        colWidths=[0.85*inch, 6.55*inch])
        except Exception:
            head = Table([[title_block]], colWidths=[7.4*inch])
    else:
        head = Table([[title_block]], colWidths=[7.4*inch])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (0, 0), 0)]))
    story += [head, Spacer(1, 8), _rule(1.6), Spacer(1, 12)]

    h = ParagraphStyle("sp_h", parent=styles["Heading2"], textColor=INK, fontSize=14, spaceBefore=12, spaceAfter=4)
    h2 = ParagraphStyle("sp_h2", parent=styles["Heading1"], textColor=INK, fontSize=18, spaceBefore=2, spaceAfter=8)
    body = ParagraphStyle("sp_b", parent=styles["Normal"], textColor=INK, fontSize=9.5, spaceAfter=2)
    foot = ParagraphStyle("sp_f", parent=styles["Normal"], textColor=GRAY, fontSize=8)

    def _section(t):
        story.append(Paragraph(t, h)); story.append(_rule(1.0)); story.append(Spacer(1, 6))

    def _tbl(data, col_widths=None, dense=False):
        """dense=True is for the 13-column Full Count Matrix — tighter font/
        padding so a dozen narrow columns still fit the page width without
        every header cell wrapping."""
        fs, pad_h, pad_v = (7.5, 3, 4) if dense else (9.5, 6, 5.5)
        t = Table(data, hAlign="LEFT", colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("TEXTCOLOR", (0, 1), (-1, -1), INK), ("FONTSIZE", (0, 0), (-1, -1), fs),
            ("GRID", (0, 0), (-1, -1), 0.6, RULE), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, STRIPE]),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TOPPADDING", (0, 0), (-1, -1), pad_v), ("BOTTOMPADDING", (0, 0), (-1, -1), pad_v),
            ("LEFTPADDING", (0, 0), (-1, -1), pad_h), ("RIGHTPADDING", (0, 0), (-1, -1), pad_h)]))
        return t

    def _card(label, value, sub_txt, width):
        lbl_s = ParagraphStyle("pc_l", fontName="Helvetica-Bold", fontSize=7, textColor=GRAY, leading=8.5)
        val_s = ParagraphStyle("pc_v", fontName="Helvetica-Bold", fontSize=16, textColor=INK, leading=18)
        sub_s = ParagraphStyle("pc_s", fontName="Helvetica", fontSize=7, textColor=GRAY, leading=8.5)
        rows = [[Paragraph(label.upper(), lbl_s)], [Paragraph(str(value), val_s)],
                [Paragraph(sub_txt if sub_txt else "&nbsp;", sub_s)]]
        t = Table(rows, colWidths=[width])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PANEL), ("LINEBEFORE", (0, 0), (0, -1), 2.2, INK),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
        return t

    def _card_row(specs):
        n = len(specs)
        gap = 0.08 * inch
        card_w = (7.4 * inch - gap * (n - 1)) / n if n else 7.4 * inch
        row_data, col_widths = [], []
        for i, (label, value, sub_txt) in enumerate(specs):
            row_data.append(_card(label, value, sub_txt, card_w))
            col_widths.append(card_w)
            if i < n - 1:
                row_data.append(""); col_widths.append(gap)
        t = Table([row_data], colWidths=col_widths)
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                              ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 12)]))
        return t

    _section("Official Season Line")
    if off_row:
        story.append(_tbl([list(off_row.keys()), [str(v) for v in off_row.values()]]))
    else:
        story.append(Paragraph("No Data/official_stats.csv on file yet.", body))
    story.append(Spacer(1, 10))

    lg_fm = _league_pitcher_benchmarks((len(df_all), tuple(df_all.columns)))

    for i, (sp_key, sp_label) in enumerate(
            [("Overall", "Overall"), ("vs RHH", "vs Righties (RHH)"), ("vs LHH", "vs Lefties (LHH)")]):
        s = splits[sp_key]
        if i > 0:
            story.append(PageBreak())
        story.append(Paragraph(sp_label, h2)); story.append(_rule(1.6)); story.append(Spacer(1, 8))

        sp_content = [Paragraph("Stuff+ by Pitch", h)]
        if len(s["Mix"]):
            specs = []
            for _, r in s["Mix"].iterrows():
                sp_val = r.get("StuffPlus")
                specs.append((r["PitchType"], f"{sp_val:.0f}" if pd.notna(sp_val) else "—", None))
            sp_content.append(_card_row(specs))
        else:
            sp_content.append(Paragraph("No pitch-type data.", body))
        story.append(KeepTogether(sp_content))

        pm_content = [Paragraph("Pitch Mix", h), _rule(1.1), Spacer(1, 5)]
        if len(s["Mix"]):
            disp = s["Mix"]
            rows = [["Pitch", "Usage%", "Pitches", "Velo", "IVB", "HB"]]
            for _, r in disp.iterrows():
                rows.append([r["PitchType"], f"{r['Pct']:.0f}%", str(int(r["Count"])),
                            f"{r['AvgVelo']:.1f}" if pd.notna(r["AvgVelo"]) else "—",
                            f"{r['AvgIVB']:.1f}" if "AvgIVB" in disp.columns and pd.notna(r.get("AvgIVB")) else "—",
                            f"{r['AvgHB']:.1f}" if "AvgHB" in disp.columns and pd.notna(r.get("AvgHB")) else "—"])
            pm_content.append(_tbl(rows))
        else:
            pm_content.append(Paragraph("No pitch-type data.", body))
        story.append(KeepTogether(pm_content))

        az_content = [Paragraph("Location Numbers — Attack Zones", h), _rule(1.1), Spacer(1, 5)]
        if s["AttackZones"]:
            az_rows = [["Zone", "His %", "League %"]]
            for r in s["AttackZones"]:
                az_rows.append([r["Zone"], f"{r['His %']:.0f}%", f"{r['League %']:.0f}%"])
            az_content.append(_tbl(az_rows))
        else:
            az_content.append(Paragraph("Not enough located pitches.", body))
        story.append(KeepTogether(az_content))

        cm_content = [Paragraph("Full Count Matrix", h), _rule(1.1), Spacer(1, 5)]
        all_pt = s["PitchTypes"]
        if s["CountMatrix"] and all_pt:
            header_row = ["Pitch"] + [f"{b}-{s_}" for b, s_ in s["AllCounts"]]
            rows = [header_row]
            for pt in all_pt:
                row = [pt]
                for bs in s["AllCounts"]:
                    n = s["CountTotals"].get(bs, 0)
                    pct = s["CountMatrix"].get(pt, {}).get(bs, 0)
                    row.append("—" if n == 0 else (f"{pct:.0%}" if pct > 0 else "0%"))
                rows.append(row)
            cm_content.append(_tbl(rows, col_widths=[0.6*inch] + [0.53*inch]*12, dense=True))
        else:
            cm_content.append(Paragraph("Count data not available.", body))
        story.append(KeepTogether(cm_content))

        lhm_content = [Paragraph("Location Heat Maps", h), _rule(1.1), Spacer(1, 5)]
        pz = (s["PitchesClean"][s["PitchesClean"]["PlateLocSide"].notna() &
             s["PitchesClean"]["PlateLocHeight"].notna()] if "PlateLocSide" in s["PitchesClean"].columns
             else s["PitchesClean"].iloc[0:0])
        bip = pz[pz["ExitSpeed"].notna()] if len(pz) else pz
        sw = pz[pz["PitchCall"].isin(_SR_SWING_C)].copy() if len(pz) else pz
        if len(sw):
            sw["_whiff_w"] = sw["PitchCall"].eq("StrikeSwinging").astype(float)
        panels = [(pz, None, "Pitch Location"), (bip, "ExitSpeed", "Hard Contact"), (sw, "_whiff_w", "Whiffs")]
        imgs = [_kde_heatmap_png(d, weight_col=w) if len(d) >= 5 else None for d, w, _ in panels]
        if any(imgs):
            cap_row = [Paragraph(f"<b>{lbl}</b>", body) for _, _, lbl in panels]
            img_row = [Image(io.BytesIO(png), width=2.1*inch, height=2.35*inch) if png
                      else Paragraph("Not enough data", body) for png in imgs]
            t = Table([cap_row, img_row], colWidths=[2.2*inch]*3)
            t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            lhm_content.append(t)
        else:
            lhm_content.append(Paragraph("Not enough location data for heat maps.", body))
        story.append(KeepTogether(lhm_content))

        if sp_key == "Overall":
            mv_content = [Paragraph("Movement vs League", h), _rule(1.1), Spacer(1, 5)]
            mv_png = _movement_vs_league_png(pitcher_name, overall_pp)
            if mv_png:
                mv_content.append(Image(io.BytesIO(mv_png), width=4.2*inch, height=3.7*inch))
            else:
                mv_content.append(Paragraph("Not enough pitches of any single type to plot movement.", body))
            story.append(KeepTogether(mv_content))

            tn_content = [Paragraph("Tunneling", h), _rule(1.1), Spacer(1, 5)]
            prof = None
            try:
                import matchup_model as mm
                prof = mm.pitcher_arsenal_profile(overall_pp, min_n=15)
            except Exception as e:
                tn_content.append(Paragraph(f"Tunneling unavailable: {type(e).__name__} — {e}", body))
            if prof is not None:
                if len(prof) >= 2:
                    trows = mm.tunnel_pairs(prof)
                    trows.sort(key=lambda r: -r["score"])
                    rows = [["Pitch A", "Pitch B", "Release Gap (in)", "Move Sep (in)", "Velo Gap", "Grade"]]
                    for r in trows:
                        rows.append([r["a"], r["b"], f"{r['release_gap']:.1f}", f"{r['move_sep']:.1f}",
                                    f"{r['velo_gap']:.1f}", r["grade"]])
                    tn_content.append(_tbl(rows))
                else:
                    tn_content.append(Paragraph("Needs 2+ pitch types with 15+ thrown to grade tunneling.", body))
            tn_content.append(Paragraph("Movement and tunneling reflect his whole-season arsenal — not split "
                                        "by batter hand.", foot))
            story.append(KeepTogether(tn_content))

        fip_content = [Paragraph("FIP / xFIP / xERA", h), _rule(1.1), Spacer(1, 5)]
        fm = s["FipMetrics"]
        fip_content.append(_card_row([
            ("IP", fm["IP"], None),
            ("FIP", f"{fm['FIP']:.2f}" if fm["FIP"] is not None else "—",
             f"Lg: {lg_fm['FIP']:.2f}" if lg_fm["FIP"] is not None else None),
            ("xFIP", f"{fm['xFIP']:.2f}" if fm["xFIP"] is not None else "—",
             f"Lg: {lg_fm['xFIP']:.2f}" if lg_fm["xFIP"] is not None else None),
            ("xERA*", f"{s['xERA']:.2f}" if s["xERA"] is not None else "—", f"Lg: {_SEASON_FIP_CONSTANT:.2f}"),
        ]))
        if sp_key == "Overall":
            fip_content.append(Spacer(1, 4))
            fip_content.append(Paragraph("*xERA is a DiamondIntel estimate built from tracked exit velo/launch "
                                         "angle allowed, rescaled to a runs/9 basis &mdash; not Statcast's "
                                         "proprietary metric.", foot))
        story.append(KeepTogether(fip_content))

    doc.build(story)
    return buf.getvalue()


if page == "Season Report":
    st.title("End-of-Season Player Report")
    st.caption("Post-season wrap-up reports for players on their way out — rebuilt from the data "
              "every time this loads, so the last games of the season are included automatically. "
              "Add Data/official_stats.csv (built from the league stats PDF, once the season is "
              "final) to fill in the official season line at the top of each report — it also "
              "takes over K% and BB% on the Overall tab.")

    sr_type = st.radio("Report type", ["Hitter", "Pitcher"], horizontal=True, key="sr_type")

    if sr_type == "Hitter":
        sr_names = sorted([b for b in df_all[df_all["BatterTeam"] == MY_TEAM]["Batter"].dropna().unique()
                          if not _is_removed(b)])
        if not sr_names:
            st.info("No Bandits hitters found.")
        else:
            sr_batter = st.selectbox("Player", options=sr_names, format_func=player_last, key="sr_batter")
            _render_hitter_season_report(sr_batter)

            st.divider()
            st.markdown("### Download")
            try:
                bp_all = df_all[df_all["Batter"] == sr_batter].copy()
                for c in ["ExitSpeed", "Angle", "PlateLocSide", "PlateLocHeight"]:
                    bp_all[c] = pd.to_numeric(bp_all[c], errors="coerce")
                side = bp_all["BatterSide"].dropna().iloc[0] if bp_all["BatterSide"].notna().any() else "?"
                h_splits = {
                    "Overall": _hitter_season_stats(bp_all),
                    "vs RHP": _hitter_season_stats(bp_all[bp_all["PitcherThrows"] == "Right"]),
                    "vs LHP": _hitter_season_stats(bp_all[bp_all["PitcherThrows"] == "Left"]),
                }
                off_row = {}
                for lbl, key in [("G", "g"), ("AB", "ab"), ("H", "h"), ("2B", "doubles"), ("3B", "triples"),
                                 ("HR", "hr"), ("RBI", "rbi"), ("BB", "bb"), ("SO", "so"), ("SB", "sb"),
                                 ("AVG", "ba"), ("OBP", "obp"), ("SLG", "slg"), ("OPS", "ops")]:
                    v = get_official_stat(sr_batter, key)
                    if v is not None:
                        off_row[lbl] = (f"{v:.3f}" if key in ("ba", "obp", "slg", "ops") else str(int(v)))
                pdf = _build_hitter_season_pdf(player_last(sr_batter), side, off_row, h_splits,
                                               full_name=sr_batter)
                st.download_button("⬇ Download season report (PDF)", data=pdf,
                                   file_name=f"{sr_batter.replace(', ', '_').replace(' ', '_')}_season_report.pdf",
                                   mime="application/pdf", key="sr_h_pdf")
            except Exception as _e:
                _pdf_unavailable(_e)

    else:
        sr_names = sorted([p for p in df_all[df_all["PitcherTeam"] == MY_TEAM]["Pitcher"].dropna().unique()
                          if not _is_removed(p)])
        if not sr_names:
            st.info("No Bandits pitchers found.")
        else:
            sr_pitcher = st.selectbox("Player", options=sr_names, format_func=player_last, key="sr_pitcher")
            _render_pitcher_season_report(sr_pitcher)

            st.divider()
            st.markdown("### Download")
            try:
                pp_all = df_all[df_all["Pitcher"] == sr_pitcher].copy()
                throws = pp_all["PitcherThrows"].dropna().iloc[0] if pp_all["PitcherThrows"].notna().any() else "?"
                throws_lbl = {"Right": "RHP", "Left": "LHP"}.get(throws, throws)
                p_splits = {
                    "Overall": _pitcher_season_stats(sr_pitcher, pp_all),
                    "vs RHH": _pitcher_season_stats(sr_pitcher, pp_all[pp_all["BatterSide"] == "Right"]),
                    "vs LHH": _pitcher_season_stats(sr_pitcher, pp_all[pp_all["BatterSide"] == "Left"]),
                }
                off_row = {}
                for lbl, key in [("G", "app"), ("GS", "gs"), ("W", "w"), ("L", "l"), ("SV", "sv"), ("IP", "ip"),
                                 ("H", "h"), ("ER", "er"), ("BB", "bb"), ("SO", "so"), ("ERA", "era"), ("WHIP", "whip")]:
                    v = get_official_stat(sr_pitcher, key)
                    if v is not None:
                        off_row[lbl] = (f"{v:.2f}" if key in ("era", "whip", "ip") else str(int(v)))
                pdf = _build_pitcher_season_pdf(player_last(sr_pitcher), throws_lbl, off_row, p_splits, pp_all)
                st.download_button("⬇ Download season report (PDF)", data=pdf,
                                   file_name=f"{sr_pitcher.replace(', ', '_').replace(' ', '_')}_season_report.pdf",
                                   mime="application/pdf", key="sr_p_pdf")
            except Exception as _e:
                _pdf_unavailable(_e)

    st.divider()
    st.markdown("### Download Whole Team")
    st.caption("Builds every hitter's and pitcher's season report PDF and bundles them into one ZIP "
              "— for handing out at the end of the year. Raise the minimum below to skip players "
              "with only a handful of pitches on record.")
    sr_min = st.number_input("Minimum pitches to include a player", 1, 500, 20, 5, key="sr_team_min")
    if st.button("Build whole-team ZIP", key="sr_build_zip"):
        import zipfile, io as _zip_io
        zip_buf = _zip_io.BytesIO()
        n_built = 0
        with st.spinner("Building every player's report — this can take a minute for a full roster…"):
            with zipfile.ZipFile(zip_buf, "w") as zf:
                hitters = [b for b in df_all[df_all["BatterTeam"] == MY_TEAM]["Batter"].dropna().unique()
                          if not _is_removed(b)]
                for b in hitters:
                    bp_all = df_all[df_all["Batter"] == b].copy()
                    if len(bp_all) < sr_min:
                        continue
                    for c in ["ExitSpeed", "Angle", "PlateLocSide", "PlateLocHeight"]:
                        bp_all[c] = pd.to_numeric(bp_all[c], errors="coerce")
                    side = bp_all["BatterSide"].dropna().iloc[0] if bp_all["BatterSide"].notna().any() else "?"
                    try:
                        h_splits = {
                            "Overall": _hitter_season_stats(bp_all),
                            "vs RHP": _hitter_season_stats(bp_all[bp_all["PitcherThrows"] == "Right"]),
                            "vs LHP": _hitter_season_stats(bp_all[bp_all["PitcherThrows"] == "Left"]),
                        }
                        off_row = {}
                        for lbl, key in [("G", "g"), ("AB", "ab"), ("H", "h"), ("2B", "doubles"), ("3B", "triples"),
                                         ("HR", "hr"), ("RBI", "rbi"), ("BB", "bb"), ("SO", "so"), ("SB", "sb"),
                                         ("AVG", "ba"), ("OBP", "obp"), ("SLG", "slg"), ("OPS", "ops")]:
                            v = get_official_stat(b, key)
                            if v is not None:
                                off_row[lbl] = (f"{v:.3f}" if key in ("ba", "obp", "slg", "ops") else str(int(v)))
                        pdf = _build_hitter_season_pdf(player_last(b), side, off_row, h_splits,
                                                       full_name=b)
                        b_slug = b.replace(", ", "_").replace(" ", "_")
                        zf.writestr(f"hitters/{b_slug}_season_report.pdf", pdf)
                        n_built += 1
                    except Exception:
                        continue

                pitchers = [p for p in df_all[df_all["PitcherTeam"] == MY_TEAM]["Pitcher"].dropna().unique()
                           if not _is_removed(p)]
                for p in pitchers:
                    pp_all = df_all[df_all["Pitcher"] == p].copy()
                    if len(pp_all) < sr_min:
                        continue
                    throws = pp_all["PitcherThrows"].dropna().iloc[0] if pp_all["PitcherThrows"].notna().any() else "?"
                    throws_lbl = {"Right": "RHP", "Left": "LHP"}.get(throws, throws)
                    try:
                        p_splits = {
                            "Overall": _pitcher_season_stats(p, pp_all),
                            "vs RHH": _pitcher_season_stats(p, pp_all[pp_all["BatterSide"] == "Right"]),
                            "vs LHH": _pitcher_season_stats(p, pp_all[pp_all["BatterSide"] == "Left"]),
                        }
                        off_row = {}
                        for lbl, key in [("G", "app"), ("GS", "gs"), ("W", "w"), ("L", "l"), ("SV", "sv"), ("IP", "ip"),
                                         ("H", "h"), ("ER", "er"), ("BB", "bb"), ("SO", "so"), ("ERA", "era"), ("WHIP", "whip")]:
                            v = get_official_stat(p, key)
                            if v is not None:
                                off_row[lbl] = (f"{v:.2f}" if key in ("era", "whip", "ip") else str(int(v)))
                        pdf = _build_pitcher_season_pdf(player_last(p), throws_lbl, off_row, p_splits, pp_all)
                        p_slug = p.replace(", ", "_").replace(" ", "_")
                        zf.writestr(f"pitchers/{p_slug}_season_report.pdf", pdf)
                        n_built += 1
                    except Exception:
                        continue
        if n_built:
            st.download_button(f"⬇ Download {n_built} season reports (ZIP)", data=zip_buf.getvalue(),
                               file_name="nashua_silver_knights_season_reports.zip",
                               mime="application/zip", key="sr_team_zip_dl")
        else:
            st.warning("No players cleared the minimum-pitch threshold.")

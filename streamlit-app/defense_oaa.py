"""
Defensive metrics page for DiamondIntel.
OAA (Outs Above Average) for outfielders and OAA (Outs Above Average) for infielders.
Uses positioning CSVs + game CSVs from the Data/ folder.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ─── physics priors ───────────────────────────────────────────────────────────
def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def of_prior(dist, time, reaction=0.5, speed=22.0, scale=10.0):
    return _sig((speed * np.maximum(time - reaction, 0) - dist) / scale)

def if_prior(gap, time, reaction=0.3, speed=20.0, scale=6.0):
    return _sig((speed * np.maximum(time - reaction, 0) - gap) / scale)

# ─── prior + correction logistic ──────────────────────────────────────────────
class PriorCorrectedLogit:
    def __init__(self, lam=1.0, n_iter=50, min_events=150):
        self.lam, self.n_iter, self.min_events = lam, n_iter, min_events
        self.fitted = False

    def fit(self, X, y, prior_p):
        X = np.asarray(X, float); y = np.asarray(y, float)
        prior_p = np.clip(np.asarray(prior_p, float), 1e-4, 1 - 1e-4)
        offset = np.log(prior_p / (1 - prior_p))
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        Xs = (X - self.mu) / self.sd
        n, k = Xs.shape
        A = np.hstack([np.ones((n, 1)), Xs])
        beta = np.zeros(k + 1)
        R = np.eye(k + 1) * self.lam
        if n < self.min_events or len(np.unique(y)) < 2:
            self.beta = beta; self.fitted = False; return self
        for _ in range(self.n_iter):
            p = _sig(offset + A @ beta)
            W = np.clip(p * (1 - p), 1e-6, None)
            grad = A.T @ (y - p) - R @ beta
            H = (A * W[:, None]).T @ A + R
            try:
                beta = beta + np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
        self.beta = beta; self.fitted = True
        return self

    def predict(self, X, prior_p):
        prior_p = np.clip(np.asarray(prior_p, float), 1e-4, 1 - 1e-4)
        offset = np.log(prior_p / (1 - prior_p))
        if not self.fitted:
            return _sig(offset)
        Xs = (np.asarray(X, float) - self.mu) / self.sd
        A = np.hstack([np.ones((len(Xs), 1)), Xs])
        return _sig(offset + A @ self.beta)

def _auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(p); ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(p) + 1)
    return (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

# ─── constants ────────────────────────────────────────────────────────────────
ALL_POS  = ["1B", "2B", "3B", "SS", "LF", "CF", "RF"]
OUTFIELD = ["LF", "CF", "RF"]
INFIELD  = ["1B", "2B", "3B", "SS"]
AIR      = ["FlyBall", "LineDrive", "Popup"]

# ─── data helpers ─────────────────────────────────────────────────────────────
def _is_positioning(df):
    return "SS_PositionAtReleaseX" in df.columns

def _is_game(df):
    return {"AutoHitType", "HitTrajectoryXc1", "PitchUID"} <= set(df.columns)

@st.cache_data(ttl=300, show_spinner=False)
def load_defensive_data(data_dir):
    from pathlib import Path
    csvs = list(Path(data_dir).glob("*.csv"))
    games, poss = [], []
    for f in csvs:
        try:
            header = pd.read_csv(f, nrows=0)
            if _is_positioning(header):
                poss.append(pd.read_csv(f))
            elif _is_game(header):
                games.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    if not games:
        return None, None, "No game CSVs found."
    if not poss:
        return None, None, "No positioning CSVs found in Data/ folder. Add them to GitHub and refresh."
    g = pd.concat(games, ignore_index=True).drop_duplicates("PitchUID")
    p = pd.concat(poss,  ignore_index=True).drop_duplicates("PitchUID")
    return g, p, None

def fielder_long(p):
    rows = []
    for pos in ALL_POS:
        x_col = f"{pos}_PositionAtReleaseX"
        z_col = f"{pos}_PositionAtReleaseZ"
        n_col = f"{pos}_Name"
        if not all(c in p.columns for c in [x_col, z_col, n_col]):
            continue
        s = p[["PitchUID", x_col, z_col, n_col]].copy()
        s.columns = ["PitchUID", "f_depth", "f_lat", "player"]
        s["position"] = pos
        rows.append(s)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).dropna(subset=["f_depth", "f_lat"])

def outfield_plays(g, fp):
    air = g[
        (g["PitchCall"] == "InPlay") &
        (g["AutoHitType"].isin(AIR)) &
        g["Distance"].notna() &
        g["HangTime"].notna() &
        (g["PlayResult"] != "HomeRun")
    ].copy()
    if air.empty:
        return pd.DataFrame()
    bearing_col = "Bearing" if "Bearing" in air.columns else "Direction"
    air["hit_x"] = air["Distance"] * np.cos(np.radians(air[bearing_col]))
    air["hit_z"] = -air["Distance"] * np.sin(np.radians(air[bearing_col]))
    m = air.merge(fp[fp["position"].isin(OUTFIELD)], on="PitchUID")
    if m.empty:
        return pd.DataFrame()
    m["dist"] = np.hypot(m["f_depth"] - m["hit_x"], m["f_lat"] - m["hit_z"])
    resp = m.loc[m.groupby("PitchUID")["dist"].idxmin()].copy()
    resp["depth_back"] = resp["hit_x"] - resp["f_depth"]
    resp["out_made"]   = (resp["PlayResult"] == "Out").astype(int)
    resp["prior"]      = of_prior(resp["dist"], resp["HangTime"])
    resp["feat1"] = resp["dist"]
    resp["feat2"] = resp["HangTime"]
    resp["feat3"] = resp["depth_back"]
    return resp

def infield_plays(g, fp):
    gb = g[
        (g["PitchCall"] == "InPlay") &
        (g["AutoHitType"] == "GroundBall") &
        g["HitTrajectoryXc1"].notna() &
        (g["HitTrajectoryXc1"] > 5)
    ].copy()
    if gb.empty:
        return pd.DataFrame()
    ifp = fp[fp["position"].isin(INFIELD)]
    m = gb.merge(ifp, on="PitchUID")
    if m.empty:
        return pd.DataFrame()
    m["t_reach"] = (m["f_depth"] - m["HitTrajectoryXc0"]) / m["HitTrajectoryXc1"]
    m = m[m["t_reach"] > 0].copy()
    m["ball_lat"] = -(m["HitTrajectoryZc0"] + m["HitTrajectoryZc1"] * m["t_reach"])
    m["gap"]      = (m["ball_lat"] - m["f_lat"]).abs()
    m["prior"]    = if_prior(m["gap"], m["t_reach"])
    resp = m.loc[m.groupby("PitchUID")["prior"].idxmax()].copy()
    resp["out_made"] = resp["PlayResult"].isin(["Out", "FieldersChoice"]).astype(int)
    resp["feat1"] = resp["gap"]
    resp["feat2"] = resp["t_reach"]
    resp["feat3"] = resp["ExitSpeed"].fillna(80)
    return resp

def score_plays(plays, metric):
    if plays.empty:
        return pd.DataFrame(), False, float("nan")
    X = plays[["feat1","feat2","feat3"]].values
    y = plays["out_made"].values
    model = PriorCorrectedLogit().fit(X, y, plays["prior"].values)
    plays = plays.copy()
    plays["exp_out"] = model.predict(X, plays["prior"].values)
    plays[metric]    = plays["out_made"] - plays["exp_out"]
    a = _auc(y, plays["exp_out"].values)
    return plays, model.fitted, a

# ─── main render ──────────────────────────────────────────────────────────────
def render(data_dir, player_last, MY_TEAM, team_label=None, TEAM_LABELS=None):
    if team_label is None: team_label = lambda x: x
    if TEAM_LABELS is None: TEAM_LABELS = {}
    st.title("Defensive Metrics — OAA")
    st.caption(
        "**OAA** (Outs Above Average) measures how many outs a fielder made "
        "vs how many the model expected based on distance, hang time, and starting position. "
        "Applied to both outfielders (air balls) and infielders (ground balls). "
        "Requires positioning CSVs in the Data/ folder."
    )
    st.divider()

    with st.spinner("Loading defensive data…"):
        g, p, err = load_defensive_data(data_dir)

    if err:
        st.warning(err)
        return

    fp = fielder_long(p)
    if fp.empty:
        st.warning("Positioning data found but couldn't parse fielder positions. "
                   "Check that positioning CSVs have columns like SS_PositionAtReleaseX.")
        return

    st.success(f"Loaded {g['PitchUID'].nunique():,} pitches · "
               f"{p['PitchUID'].nunique():,} positioning records")
    st.divider()

    of_raw = outfield_plays(g, fp)
    if_raw = infield_plays(g, fp)

    of_plays, of_fitted, of_auc = score_plays(of_raw, "oaa")
    if_plays, if_fitted, if_auc = score_plays(if_raw, "oaa")

    tab_of, tab_if, tab_detail = st.tabs(["Outfield OAA", "Infield OAA", "Play Log"])

    # ── OUTFIELD OAA ──────────────────────────────────────────────────────────
    with tab_of:
        st.markdown("### Outfield — Outs Above Average (OAA)")
        mode = "Data-fit model" if of_fitted else "Prior-only (cold start — need 150+ air ball chances)"
        st.caption(f"{mode} · AUC: {of_auc:.3f}" if not np.isnan(of_auc) else mode)

        if of_plays.empty:
            st.info("No outfield air ball opportunities found.")
        else:
            # Add team column if available
            of_plays["_team"] = of_plays.get("PitcherTeam", of_plays.get("BatterTeam", "Unknown"))
            # Fielding team = pitching team
            team_map = of_plays.groupby("player")["_team"].agg(lambda x: x.mode()[0] if len(x)>0 else "—").to_dict()

            lb = (of_plays.groupby(["player","position"])
                  .agg(Chances=("oaa","count"),
                       Outs=("out_made","sum"),
                       Expected=("exp_out","sum"),
                       OAA=("oaa","sum"))
                  .round(2).sort_values("OAA", ascending=False).reset_index())
            lb["Team"] = lb["player"].map(team_map).map(lambda t: team_label(t) if t in TEAM_LABELS else t)

            # Highlight NAS_SIL players
            nas_players = set()
            if "BatterTeam" in of_plays.columns:
                # Fielders are the defensive team — opposite of batter team
                pass
            if "PitcherTeam" in of_plays.columns:
                nas_of = of_plays[of_plays["PitcherTeam"] == MY_TEAM]["player"].dropna().unique()
                nas_players = set(nas_of)

            # Team filter
            all_teams_of = ["All Teams"] + sorted(lb["Team"].dropna().unique().tolist())
            team_f_of = st.selectbox("Filter by team", all_teams_of, key="def_of_team",
                format_func=lambda t: t)
            if team_f_of != "All Teams":
                lb = lb[lb["Team"] == team_f_of]

            m1, m2, m3 = st.columns(3)
            m1.metric("Air Ball Chances", len(of_plays))
            m2.metric("Fielders Tracked", lb["player"].nunique())
            m3.metric("League Avg OAA", "0.00")
            st.divider()

            # Bar chart
            fig = go.Figure()
            colors = []
            for _, row in lb.iterrows():
                if row["player"] in nas_players:
                    colors.append("#f59e0b")
                elif row["OAA"] > 0:
                    colors.append("#22c55e")
                else:
                    colors.append("#ef4444")

            fig.add_trace(go.Bar(
                x=lb["player"].map(lambda n: player_last(n) if pd.notna(n) else n),
                y=lb["OAA"],
                marker_color=colors,
                text=lb["OAA"].map(lambda v: f"{v:+.2f}"),
                textposition="outside",
                textfont=dict(size=10, color="#e2e8f0"),
                hovertemplate="<b>%{x}</b><br>OAA: %{y:+.2f}<extra></extra>",
            ))
            fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
            fig.update_layout(
                height=380, plot_bgcolor="#111827", paper_bgcolor="#0a0e1a",
                font=dict(color="#e2e8f0"),
                xaxis=dict(gridcolor="#1e2d45", tickangle=-30),
                yaxis=dict(title="OAA", gridcolor="#1e2d45", zeroline=False),
                margin=dict(l=40, r=40, t=20, b=80)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Full table
            lb["Expected"] = lb["Expected"].map(lambda v: f"{v:.1f}")
            lb["OAA"]      = lb["OAA"].map(lambda v: f"{v:+.2f}")
            st.dataframe(lb[["player","Team","position","Chances","Outs","Expected","OAA"]],
                         use_container_width=True, hide_index=True,
                         column_config={"player": "Fielder", "Team": "Team", "position": "Pos"})

            if not of_fitted:
                st.warning("Model in cold-start mode — showing physics prior only. "
                           "Scores will sharpen after 150+ air ball chances accumulate.")

    # ── INFIELD RAE ───────────────────────────────────────────────────────────
    with tab_if:
        st.markdown("### Infield — Outs Above Average (OAA)")
        mode = "Data-fit model" if if_fitted else "Prior-only (cold start — need 150+ ground ball chances)"
        st.caption(f"{mode} · AUC: {if_auc:.3f}" if not np.isnan(if_auc) else mode)

        if if_plays.empty:
            st.info("No infield ground ball opportunities found.")
        else:
            if_plays["_team"] = if_plays.get("PitcherTeam", if_plays.get("BatterTeam", "Unknown"))
            team_map_if = if_plays.groupby("player")["_team"].agg(lambda x: x.mode()[0] if len(x)>0 else "—").to_dict()

            lb = (if_plays.groupby(["player","position"])
                  .agg(Chances=("oaa","count"),
                       Outs=("out_made","sum"),
                       Expected=("exp_out","sum"),
                       OAA=("oaa","sum"))
                  .round(2).sort_values("OAA", ascending=False).reset_index())
            lb["Team"] = lb["player"].map(team_map_if).map(lambda t: team_label(t) if t in TEAM_LABELS else t)

            all_teams_if = ["All Teams"] + sorted(lb["Team"].dropna().unique().tolist())
            team_f_if = st.selectbox("Filter by team", all_teams_if, key="def_if_team",
                format_func=lambda t: t)
            if team_f_if != "All Teams":
                lb = lb[lb["Team"] == team_f_if]

            m1, m2, m3 = st.columns(3)
            m1.metric("Ground Ball Chances", len(if_plays))
            m2.metric("Fielders Tracked", lb["player"].nunique())
            m3.metric("League Avg OAA", "0.00")
            st.divider()

            fig2 = go.Figure()
            bar_colors = ["#22c55e" if v > 0 else "#ef4444" for v in lb["OAA"]]
            fig2.add_trace(go.Bar(
                x=lb["player"].map(lambda n: player_last(n) if pd.notna(n) else n),
                y=lb["OAA"],
                marker_color=bar_colors,
                text=lb["OAA"].map(lambda v: f"{v:+.2f}"),
                textposition="outside",
                textfont=dict(size=10, color="#e2e8f0"),
                hovertemplate="<b>%{x}</b><br>OAA: %{y:+.2f}<extra></extra>",
            ))
            fig2.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
            fig2.update_layout(
                height=380, plot_bgcolor="#111827", paper_bgcolor="#0a0e1a",
                font=dict(color="#e2e8f0"),
                xaxis=dict(gridcolor="#1e2d45", tickangle=-30),
                yaxis=dict(title="OAA", gridcolor="#1e2d45", zeroline=False),
                margin=dict(l=40, r=40, t=20, b=80)
            )
            st.plotly_chart(fig2, use_container_width=True)

            lb["Expected"] = lb["Expected"].map(lambda v: f"{v:.1f}")
            lb["OAA"]      = lb["OAA"].map(lambda v: f"{v:+.2f}")
            st.dataframe(lb[["player","Team","position","Chances","Outs","Expected","OAA"]],
                         use_container_width=True, hide_index=True,
                         column_config={"player": "Fielder", "Team": "Team", "position": "Pos"})

            if not if_fitted:
                st.warning("Model in cold-start mode. Infield OAA is the more "
                           "provisional metric — read as directional until 150+ chances.")

    # ── PLAY LOG ──────────────────────────────────────────────────────────────
    with tab_detail:
        st.markdown("### Individual Play Log")
        st.caption("Every scored defensive play. Filter by fielder to review specific opportunities.")

        all_plays = []
        if not of_plays.empty:
            of_log = of_plays[["player","position","AutoHitType","Distance",
                                "HangTime","out_made","exp_out","oaa"]].copy()
            of_log["metric"] = "OAA"
            of_log.columns = ["Fielder","Pos","Hit Type","Distance","Hang Time",
                               "Out Made","Expected","Score","Metric"]
            all_plays.append(of_log)
        if not if_plays.empty:
            if_log = if_plays[["player","position","AutoHitType","gap",
                                "t_reach","out_made","exp_out","oaa"]].copy()
            if_log["metric"] = "OAA"
            if_log.columns = ["Fielder","Pos","Hit Type","Gap","Time to Reach",
                               "Out Made","Expected","Score","Metric"]
            all_plays.append(if_log)

        if all_plays:
            combined_log = pd.concat(all_plays, ignore_index=True)
            fielders = ["All"] + sorted(combined_log["Fielder"].dropna().unique().tolist())
            sel = st.selectbox("Filter by fielder",
                options=fielders,
                format_func=lambda x: player_last(x) if x != "All" else "All Fielders",
                key="def_fielder")
            view = combined_log if sel == "All" else combined_log[combined_log["Fielder"] == sel]
            view = view.copy()
            view["Fielder"]  = view["Fielder"].map(lambda n: player_last(n) if pd.notna(n) else n)
            view["Expected"] = view["Expected"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
            view["Score"]    = view["Score"].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "—")
            st.dataframe(view, use_container_width=True, height=500, hide_index=True)
        else:
            st.info("No play data available yet.")

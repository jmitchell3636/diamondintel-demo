"""
returner_board_page.py -- FCBL returner board, wired to this team's real
TrackMan captures in Data/.

Two tabs, two persisted watchlists:

  * Pitchers -- every pitcher who has thrown a pitch in a Nashua game, ranked
    for bring-back priority using fcbl/ + mlbproj/ (see docs/README_FCBL.md:
    priority = UCB(run value) x P(available)). Watchlist: big_board.csv.
  * Hitters -- every hitter who has batted in a Nashua game. fcbl/ has no
    hitter reliability/shrinkage model yet (a documented gap in
    docs/README_FCBL.md -- "No hitters. Same structure applies; the
    components and reliability constants differ."), so this is deliberately
    simpler: raw wOBA (same weights as app.py's Player WAR page) x the same
    availability/eligibility model pitchers use, no shrinkage or posterior
    uncertainty. Watchlist: big_board_hitters.csv.

Two caveats specific to running the pitcher board on live team data rather
than the synthetic demo in run_board.py (the hitter side inherits the same
two, for the same reasons):

  * Coverage is Nashua-centric -- an opposing player's line here is only his
    appearances against Nashua, not his full FCBL season. Samples run
    thinner than the demo, so heavy shrinkage will dominate the low-IP end
    for pitchers. That is the correct response to thin data, not a bug.
  * Without a roster CSV (player_id, class_year, division[, age/grad_year]),
    every player defaults to SO / D1, which flattens age-driven draft
    eligibility and the future-value premium. Upload one to sharpen either
    board -- one roster file covers both tabs.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import rankdata

from mlbproj.core import calibrate
from fcbl.loader import load_trackman, attach_roster, validate
from fcbl.synth_fcbl import season_totals
from fcbl.reliability import reliability_table, moment_m, NUM, DEN
from fcbl.value import StuffPrior, shrink, run_value
from fcbl.board import build_board, availability

COMPS = ["k_rate", "bb_rate", "hr_fb", "gb_rate", "babip"]

PITCHER_BIG_BOARD_PATH = Path(__file__).parent / "big_board.csv"
PITCHER_BIG_BOARD_COLS = ["player_id", "team", "role", "division", "class_year",
                           "ip", "raa9_ucb", "p_available", "priority_score",
                           "tier", "added_on", "notes"]

HITTER_BIG_BOARD_PATH = Path(__file__).parent / "big_board_hitters.csv"
HITTER_BIG_BOARD_COLS = ["player_id", "team", "division", "class_year", "pa",
                          "woba", "p_available", "priority_score", "tier",
                          "added_on", "notes"]

PITCHER_DISPLAY_COLS = ["player_id", "team", "school", "role", "division",
                         "class_year", "real_class", "ip", "velo", "raa9",
                         "raa9_sd", "raa9_ucb", "p_available", "priority_score",
                         "tier"]

HITTER_DISPLAY_COLS = ["player_id", "team", "school", "division", "class_year",
                        "real_class", "bats", "pa", "avg", "woba", "bb", "k",
                        "hr", "p_available", "priority_score", "tier"]

PITCHER_TIER_CAPTION = (
    "Tiers: 1 priority re-recruit, 2 long shot (great arm, hard to get), "
    "3 high-variance flyer (thin sample, real upside), 4 solid depth, 5 pass.")

HITTER_TIER_CAPTION = (
    "Tiers are simpler here than the pitcher board -- no shrinkage/"
    "reliability model for hitters yet, so this is a plain quantile split "
    "of wOBA x Chance We Get Him Back: 1 priority re-recruit (top 20%), "
    "2 solid target (top 50%), 3 depth / pass.")

# Same linear weights as app.py's Player WAR page (WAR_WOBA_W), duplicated
# here so this module has no import-time dependency on app.py -- keep them
# in sync if the WAR page's weights ever change, so wOBA agrees app-wide.
HITTER_WOBA_W = {"BB": 0.69, "HBP": 0.72, "1B": 0.89, "2B": 1.27, "3B": 1.62, "HR": 2.10}

# Plain-English column headers for coaches -- internal names stay as the
# fcbl/mlbproj field names (or app.py's raw stat names) everywhere else.
COLUMN_LABELS = {
    "player_id": "Player",
    "team": "Team",
    "school": "School",
    "role": "Role",
    "division": "Division",
    "class_year": "Class",
    "real_class": "Verified Class?",
    "ip": "Innings Pitched",
    "velo": "Velocity (mph)",
    "raa9": "Run Value (per 9 IP)",
    "raa9_sd": "Uncertainty (+/-)",
    "raa9_ucb": "Upside Score",
    "p_available": "Chance We Get Him Back",
    "priority_score": "Priority Score",
    "tier": "Tier",
    "added_on": "Added On",
    "notes": "Notes",
    "component": "Stat",
    "league_mean": "League Average",
    "m": "Shrinkage Weight",
    "sd_true": "True-Talent Spread",
    "median_denominator": "Median Sample Size",
    "reliability_at_median": "Reliability",
    "split_half_r": "Split-Half Correlation",
    "spearman_brown_r": "Split-Half Correlation (Corrected)",
    "n_players": "Players Used",
    "check": "Check",
    "status": "Status",
    "detail": "Detail",
    "pa": "PA",
    "avg": "AVG",
    "woba": "wOBA",
    "bb": "BB",
    "k": "K",
    "hr": "HR",
    "bats": "Bats",
}


def _display(df):
    """Swap in plain-English headers for display (values untouched -- see
    _number_format for how every number is forced to exactly 2 decimals,
    and _mlb_style for the columns formatted as text instead)."""
    return df.rename(columns=COLUMN_LABELS)


def _fmt_rate3(x):
    """.xxx, no leading zero -- how MLB shows AVG/OBP/SLG/wOBA. Never apply
    this to a dataframe that round-trips back through st.data_editor for
    persistence (see _render_big_board_section's save logic), since it
    turns a numeric column into text."""
    if pd.isna(x):
        return ""
    sign = "-" if x < 0 else ""
    return sign + f"{abs(x):.3f}"[1:]


def _fmt_ip(x):
    """MLB innings-pitched notation: thirds, not decimal. 11.2 means 11 and
    2/3 innings (2 outs into the 12th), not 11.2 innings."""
    if pd.isna(x):
        return ""
    outs = int(round(x * 3))
    whole, third = divmod(outs, 3)
    return f"{whole}.{third}"


def _fmt_pct0(x):
    """Whole-number percent, matching how rate stats read elsewhere in this
    app (e.g. the Player Report page's BB%/K%)."""
    if pd.isna(x):
        return ""
    return f"{100 * x:.0f}%"


def _mlb_style(df, team_label=None):
    """Reformat rate stats the way MLB box scores / leaderboards show them,
    on a copy: AVG and wOBA as .xxx with no leading zero, IP in thirds
    rather than true decimal, and the return-probability column as a
    percent. Also swaps the raw team code (e.g. NAS_SIL) for its readable
    location/name (e.g. Nashua Silver Knights) when a team_label lookup is
    passed in. Read-only display transform -- never feed the result back
    into st.data_editor for saving (see _render_big_board_section), and
    never use it for row lookups -- the raw `team` code is still what
    goto callbacks and merges key off of."""
    out = df.copy()
    for col in ("avg", "woba"):
        if col in out.columns:
            out[col] = out[col].map(_fmt_rate3)
    if "ip" in out.columns:
        out["ip"] = out["ip"].map(_fmt_ip)
    if "p_available" in out.columns:
        out["p_available"] = out["p_available"].map(_fmt_pct0)
    if team_label is not None and "team" in out.columns:
        out["team"] = out["team"].map(lambda t: team_label(t) if pd.notna(t) else t)
    return out


# Whole-count columns (post-_display rename) that read worse with decimals
# tacked on -- "3 PA" not "3.00 PA".
INTEGER_DISPLAY_COLS = {"PA", "BB", "K", "HR", "Players Used", "Median Sample Size"}
# Statcast shows velocity to one decimal, not two.
ONE_DECIMAL_DISPLAY_COLS = {"Velocity (mph)"}


def _number_format(df):
    """NumberColumn format, not .round() -- so e.g. 1.10 doesn't display as
    '1.1', and sorting a displayed column still sorts on the real number
    rather than a rounded/stringified one. Whole counts show with no
    decimals, velocity gets one, everything else still-numeric gets two.
    (AVG/wOBA/IP/return-probability are handled by _mlb_style instead --
    they're text by the time this runs, so select_dtypes skips them.)"""
    def _fmt(c):
        if c in INTEGER_DISPLAY_COLS:
            return "%d"
        if c in ONE_DECIMAL_DISPLAY_COLS:
            return "%.1f"
        return "%.2f"
    return {c: st.column_config.NumberColumn(format=_fmt(c))
            for c in df.select_dtypes(include="number").columns}


@st.cache_data(ttl=600, show_spinner="Aggregating TrackMan pitch data...")
def _load_halves(data_dir):
    return load_trackman(str(data_dir))


def _apply_roster(df, roster):
    """attach_roster's default-fill logic, but usable with no roster at all.
    Works for either pitchers or hitters -- it's just a left-join on
    player_id, position-agnostic."""
    if roster is not None:
        return attach_roster(df, roster)
    out = df.copy()
    out["school_type"] = "4YR"
    out["class_year"] = "SO"
    out["division"] = "D1"
    return out


def _roster_ids(roster):
    """player_ids a roster source actually covers, mirroring attach_roster's
    own id-column normalization so this matches what the join really does."""
    if roster is None:
        return set()
    if hasattr(roster, "seek"):
        roster.seek(0)
    r = pd.read_csv(roster)
    if hasattr(roster, "seek"):
        roster.seek(0)
    if "player_id" not in r.columns:
        for cand in ("Pitcher", "name", "Name", "player"):
            if cand in r.columns:
                r = r.rename(columns={cand: "player_id"})
                break
    return set(r["player_id"].dropna()) if "player_id" in r.columns else set()


def _hitter_season(df_all, excluded_teams=frozenset()):
    """One row per hitter x team, full season, straight from the same raw
    pitch-level data the rest of the app uses (not fcbl/, which is
    pitcher-only). wOBA uses the same weights as app.py's Player WAR page."""
    d = df_all[df_all["Batter"].notna()]
    if excluded_teams:
        d = d[~d["BatterTeam"].isin(excluded_teams)]

    rows = []
    for (batter, team), grp in d.groupby(["Batter", "BatterTeam"]):
        pa = int((grp["PitchofPA"] == 1).sum()) if "PitchofPA" in grp.columns else len(grp)
        if pa <= 0:
            continue
        singles = int(grp["PlayResult"].eq("Single").sum())
        doubles = int(grp["PlayResult"].eq("Double").sum())
        triples = int(grp["PlayResult"].eq("Triple").sum())
        hr      = int(grp["PlayResult"].eq("HomeRun").sum())
        bb      = int(grp["KorBB"].eq("Walk").sum())
        k       = int(grp["KorBB"].eq("Strikeout").sum())
        hbp     = int(grp["PitchCall"].eq("HitByPitch").sum())
        hits = singles + doubles + triples + hr
        ab = max(pa - bb - hbp, 0)
        num = (HITTER_WOBA_W["BB"] * bb + HITTER_WOBA_W["HBP"] * hbp +
               HITTER_WOBA_W["1B"] * singles + HITTER_WOBA_W["2B"] * doubles +
               HITTER_WOBA_W["3B"] * triples + HITTER_WOBA_W["HR"] * hr)
        bats = (grp["BatterSide"].dropna().iloc[0]
               if "BatterSide" in grp.columns and grp["BatterSide"].notna().any()
               else "?")
        rows.append(dict(
            player_id=batter, team=team, pa=pa, ab=ab,
            avg=(hits / ab) if ab else 0.0, bb=bb, k=k, hr=hr,
            woba=num / pa, bats=bats))
    return pd.DataFrame(rows)


def _load_big_board(path, cols):
    if path.exists():
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        for col in cols:
            if col not in df.columns:
                df[col] = ""
        df["notes"] = df["notes"].fillna("")
        return df[cols]
    return pd.DataFrame(columns=cols)


def _save_big_board(df, path):
    df.to_csv(path, index=False)


def _add_to_big_board(rows, path, cols):
    existing = _load_big_board(path, cols)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    old_notes = existing.set_index("player_id")["notes"].to_dict()
    old_added = existing.set_index("player_id")["added_on"].to_dict()

    rows = rows.copy()
    num_cols = rows.select_dtypes(include="number").columns
    rows[num_cols] = rows[num_cols].round(2)
    rows["notes"] = rows["player_id"].map(old_notes).fillna("")
    rows["added_on"] = rows["player_id"].map(old_added).fillna(today)

    kept = existing[~existing.player_id.isin(set(rows.player_id))]
    combined = pd.concat([kept, rows[cols]], ignore_index=True)
    _save_big_board(combined, path)


def _remove_from_big_board(player_ids, path, cols):
    existing = _load_big_board(path, cols)
    _save_big_board(existing[~existing.player_id.isin(player_ids)], path)


def _render_big_board_section(path, cols, goto_callback, page_label, key_prefix,
                              team_label=None):
    """Shared Big Board watchlist UI -- overview grid with editable notes,
    per-row click-through to `page_label`, remove, CSV download. Used by
    both the pitcher and hitter tabs, each with their own persisted CSV."""
    st.divider()
    st.subheader("Big Board -- players we want back")
    big_board = _load_big_board(path, cols)
    if big_board.empty:
        st.caption("Nobody on the Big Board yet -- select rows above and add them.")
        return

    _tl = team_label if team_label is not None else (lambda t: t)
    show_bb = _display(_mlb_style(big_board, team_label=team_label))
    notes_label = COLUMN_LABELS["notes"]
    edited = st.data_editor(
        show_bb, use_container_width=True, hide_index=True,
        column_config=_number_format(show_bb),
        disabled=[c for c in show_bb.columns if c != notes_label],
        key=f"{key_prefix}_bb_editor")
    # Every column except notes is disabled above, so rebuild from the
    # original (unformatted) big_board and only pull the edited notes back
    # out -- never trust `edited`'s other columns. _mlb_style turned ip/
    # woba/avg/p_available into display strings for this grid; saving
    # `edited` as-is would permanently bake those formatted strings into
    # the CSV instead of the real numbers.
    new_notes = edited[notes_label].tolist()
    if new_notes != big_board["notes"].tolist():
        updated = big_board.copy()
        updated["notes"] = new_notes
        _save_big_board(updated, path)

    st.markdown(f"**Click a name to see their {page_label} page:**")
    for row in big_board.itertuples():
        pcol, tcol, ccol, tiercol = st.columns([2, 1, 1, 2])
        pcol.button(row.player_id, key=f"{key_prefix}_goto_{row.player_id}",
                   use_container_width=True, on_click=goto_callback,
                   args=(row.player_id, row.team))
        tcol.write(_tl(row.team))
        ccol.write(getattr(row, "class_year", ""))
        tiercol.write(row.tier)

    label_map = {row.player_id: f"{row.player_id} ({_tl(row.team)}, tier {row.tier})"
                 for row in big_board.itertuples()}
    remove_ids = st.multiselect(
        "Remove from Big Board", options=big_board["player_id"].tolist(),
        format_func=lambda pid: label_map.get(pid, str(pid)),
        key=f"{key_prefix}_remove")
    rcol, dcol = st.columns([1, 3])
    if rcol.button("Remove selected", disabled=not remove_ids,
                   key=f"{key_prefix}_remove_btn"):
        _remove_from_big_board(remove_ids, path, cols)
        st.rerun()
    dcol.download_button(
        "Download Big Board CSV", big_board.to_csv(index=False),
        file_name=path.name, mime="text/csv", key=f"{key_prefix}_download_btn")


def _render_pitchers(DATA_DIR, EXCLUDED_TEAMS, roster, matched_ids, goto_pitcher,
                     last_player, team_label=None):
    c1, c2, c3 = st.columns(3)
    objective = c1.radio(
        "Objective", ["win_now", "pro_value"], horizontal=True,
        format_func=lambda o: "Win Now" if o == "win_now" else "Pro Value",
        key="rb_objective",
        help="Win Now: only ability and availability count. Pro Value: adds "
             "an age premium for developing younger arms.")
    kappa = c2.slider(
        "kappa (optimism)", 0.0, 2.0, 0.8, 0.1, key="rb_kappa",
        help="UCB weight on posterior uncertainty. Raise if you have slots "
             "to gamble on upside, lower it if you need floor.")
    min_bf = c3.number_input(
        "Min BF for confident tier", min_value=10, max_value=300, value=90,
        step=10, key="rb_min_bf")

    try:
        halves = _load_halves(DATA_DIR)
    except FileNotFoundError:
        st.error(f"No TrackMan CSVs found in `{DATA_DIR}`.")
        return
    except KeyError as e:
        st.error(str(e))
        return

    if EXCLUDED_TEAMS:
        halves = halves[~halves["team"].isin(EXCLUDED_TEAMS)]
    halves = _apply_roster(halves, roster)

    with st.expander("Data validation -- run before trusting anything below"):
        val_report = validate(halves)
        val_show = _display(val_report)
        st.dataframe(val_show, use_container_width=True, hide_index=True,
                    column_config=_number_format(val_show))
        bad = val_report[val_report.status != "PASS"]
        if not bad.empty:
            st.warning(f"{len(bad)} check(s) need review.")

    season = season_totals(halves)
    if season.empty:
        st.warning("No pitcher-season rows after aggregation.")
        return
    season["real_class"] = season["player_id"].isin(matched_ids)

    # Pitchers with no tagged-fastball pitches have NaN velo/ivb/hb/spin/
    # extension. StuffPrior predicts row-wise, so a handful of NaN rows only
    # NaN out those players' own stuff scores -- but scipy.stats.rankdata
    # propagates any NaN across the WHOLE ranking, which would silently blank
    # out p_available/priority_score for every pitcher on the board. Impute
    # to the league median (== "no information, regress to average"), the
    # same philosophy fcbl/reliability.py already uses for thin denominators.
    stuff_cols = [c for c in ("velo", "ivb", "hb", "spin", "extension")
                  if c in season.columns]
    missing_stuff = season[stuff_cols].isna().any(axis=1).sum() if stuff_cols else 0
    for col in stuff_cols:
        if season[col].isna().any():
            season[col] = season[col].fillna(season[col].median())
    if missing_stuff:
        st.caption(
            f"{missing_stuff} pitcher(s) had no tagged fastballs and were "
            "given league-median stuff for scoring purposes.")

    env = calibrate()
    sp = StuffPrior().fit(season)
    prior = sp.predict(season)
    m_by = {c: moment_m(season[NUM[c]], season[DEN[c]])[1] for c in COMPS}
    fallback = {c: np.full(len(season), moment_m(season[NUM[c]], season[DEN[c]])[0])
                for c in COMPS if c not in prior}
    est, sd = shrink(season, {**prior, **fallback}, m_by, COMPS)

    val = run_value(season, est, sd, env, n_draws=1200, seed=3)

    stuff_score = (prior["k_rate"] * 2.2 - prior["bb_rate"] * 1.0
                   - prior["hr_fb"] * 1.4)
    finite = np.isfinite(stuff_score)
    if not finite.all():
        # rankdata poisons every rank, not just the missing ones, if any NaN
        # survives to here -- last-resort backstop behind the impute above.
        stuff_score = np.where(finite, stuff_score, np.nanmedian(stuff_score))
    stuff_pct = rankdata(stuff_score) / len(stuff_score)

    board = build_board(season, val, stuff_pct, kappa=kappa,
                        min_bf_for_confidence=min_bf, objective=objective)
    # build_board keeps a fixed internal column set and drops real_class --
    # merge it back on. Safe as a many-to-one merge: real_class is a pure
    # per-player lookup (same value regardless of which team-row it's on),
    # not something that needs positional row alignment like val/stuff_pct.
    board = board.merge(season[["player_id", "real_class"]].drop_duplicates("player_id"),
                        on="player_id", how="left")

    with st.expander("League reliability -- derived from this data, not borrowed from MLB"):
        rt = reliability_table(halves, season)
        rt_show = _display(rt)
        st.dataframe(rt_show, use_container_width=True, hide_index=True,
                    column_config=_number_format(rt_show))
        st.caption(
            "m = league-average-equivalent trials mixed into the shrinkage. "
            "Small m means outcome stats here are already reliable; large m "
            "means lean on the TrackMan-implied stuff prior instead.")

    n_verified = int(board["real_class"].sum())
    if objective == "pro_value" and n_verified < len(board):
        st.caption(
            f"Only {n_verified} of {len(board)} pitchers have a verified class "
            "year (see 'Verified Class?' column) -- everyone else defaults to "
            "Sophomore and gets the same age premium, so Win Now and Pro Value "
            "will rank them identically relative to each other. Upload more "
            "rosters to sharpen this.")

    st.subheader(f"Board -- {len(board)} pitchers")
    cols = [c for c in PITCHER_DISPLAY_COLS if c in board.columns]
    show = board[cols].reset_index(drop=True)

    _tl = team_label if team_label is not None else (lambda t: t)
    if last_player is not None:
        is_last = show["player_id"] == last_player
        if is_last.any():
            r = show[is_last].iloc[0]
            st.info(f"Back from **{last_player}** -- {_tl(r.team)}, {r.tier}. "
                    "Pinned to the top of the board below.")
            # Pin their row first so returning from Pitcher Scouting doesn't
            # strand the coach 200 rows deep with no scroll-to API.
            show = pd.concat([show[is_last], show[~is_last]], ignore_index=True)
        else:
            st.info(f"Back from **{last_player}**.")

    big_board = _load_big_board(PITCHER_BIG_BOARD_PATH, PITCHER_BIG_BOARD_COLS)
    watched = set(big_board.player_id) if not big_board.empty else set()
    show.insert(0, "On Big Board", show["player_id"].isin(watched))

    # _mlb_style before _display: it matches on internal column names (ip),
    # which _display then renames to the pretty header (Innings Pitched).
    board_show = _display(_mlb_style(show, team_label=team_label))
    event = st.dataframe(
        board_show, use_container_width=True, hide_index=True,
        column_config=_number_format(board_show),
        on_select="rerun", selection_mode="multi-row", key="rb_board_table")

    sel_rows = list(event.selection.rows) if event and event.selection else []
    bcol, vcol, _ = st.columns([1.4, 1.4, 2])
    # on_click, not `if vcol.button(...): goto_pitcher(...)` -- a callback
    # runs before the script body re-executes, which is the only point at
    # which nav_cat/nav_page/ps_team/ps_pitcher can legally be set. Setting
    # them from inline code after a plain button check runs AFTER the
    # sidebar's nav_cat/nav_page have already been instantiated for that
    # run, which Streamlit rejects (StreamlitAPIException).
    sel_row = show.iloc[sel_rows[0]] if len(sel_rows) == 1 else None
    vcol.button(
        "View Pitcher Scouting for selected", disabled=sel_row is None,
        key="rb_view_btn", help="Select exactly one row above.",
        on_click=goto_pitcher,
        args=(sel_row["player_id"], sel_row["team"]) if sel_row is not None else (None, None))
    if bcol.button(f"Add {len(sel_rows)} selected to Big Board",
                  disabled=not sel_rows, key="rb_add_btn"):
        # Index into `show`, not `board` -- the last-viewed-player pin above
        # can reorder `show` relative to `board`, and sel_rows are positions
        # in whatever was actually rendered (show, via board_show).
        to_add = show.iloc[sel_rows][
            ["player_id", "team", "role", "division", "class_year", "ip",
             "raa9_ucb", "p_available", "priority_score", "tier"]]
        _add_to_big_board(to_add, PITCHER_BIG_BOARD_PATH, PITCHER_BIG_BOARD_COLS)
        st.success(f"Added {len(sel_rows)} player(s) to the Big Board.")
        st.rerun()

    st.caption(PITCHER_TIER_CAPTION)

    _render_big_board_section(PITCHER_BIG_BOARD_PATH, PITCHER_BIG_BOARD_COLS,
                              goto_pitcher, "Pitcher Scouting", "rb",
                              team_label=team_label)


def _render_hitters(df_all, EXCLUDED_TEAMS, roster, matched_ids, goto_hitter,
                    last_player, team_label=None):
    st.caption(
        "Hitter version of the board. fcbl/ has no hitter reliability or "
        "shrinkage model yet (a documented gap -- see docs/README_FCBL.md), "
        "so this ranks on raw wOBA x Chance We Get Him Back rather than a "
        "shrunk, uncertainty-aware run value like the pitcher board."
    )
    if df_all is None or df_all.empty:
        st.warning("No pitch-level data available to build hitter stats.")
        return

    min_pa = st.number_input(
        "Min PA to include", min_value=1, max_value=100, value=8, step=1,
        key="rb_h_min_pa",
        help="Coverage is Nashua-centric, same as the pitcher board -- an "
             "opposing hitter's PA here is only what he saw against us.")

    season_h = _hitter_season(df_all, EXCLUDED_TEAMS)
    if season_h.empty:
        st.warning("No hitter rows found.")
        return
    season_h = season_h[season_h.pa >= min_pa].reset_index(drop=True)
    if season_h.empty:
        st.warning("No hitters meet the minimum PA.")
        return

    season_h = _apply_roster(season_h, roster)
    season_h["real_class"] = season_h["player_id"].isin(matched_ids)

    woba_pct = rankdata(season_h.woba) / len(season_h)
    av = availability(season_h, woba_pct)
    season_h = pd.concat([season_h.reset_index(drop=True), av.reset_index(drop=True)], axis=1)
    season_h["priority_score"] = season_h.woba * season_h.p_available

    q80, q50 = season_h.priority_score.quantile([0.8, 0.5])
    season_h["tier"] = np.select(
        [season_h.priority_score >= q80, season_h.priority_score >= q50],
        ["1. Priority re-recruit", "2. Solid target"],
        default="3. Depth / pass")
    season_h = season_h.sort_values("priority_score", ascending=False).reset_index(drop=True)

    n_verified = int(season_h["real_class"].sum())
    if n_verified < len(season_h):
        st.caption(
            f"{n_verified} of {len(season_h)} hitters have a verified class "
            "year -- everyone else defaults to Sophomore / D1, which "
            "flattens draft/poach risk. Upload a roster above to sharpen this.")

    st.subheader(f"Board -- {len(season_h)} hitters")
    cols = [c for c in HITTER_DISPLAY_COLS if c in season_h.columns]
    show = season_h[cols].reset_index(drop=True)

    _tl = team_label if team_label is not None else (lambda t: t)
    if last_player is not None:
        is_last = show["player_id"] == last_player
        if is_last.any():
            r = show[is_last].iloc[0]
            st.info(f"Back from **{last_player}** -- {_tl(r.team)}, {r.tier}. "
                    "Pinned to the top of the board below.")
            show = pd.concat([show[is_last], show[~is_last]], ignore_index=True)
        else:
            st.info(f"Back from **{last_player}**.")

    big_board = _load_big_board(HITTER_BIG_BOARD_PATH, HITTER_BIG_BOARD_COLS)
    watched = set(big_board.player_id) if not big_board.empty else set()
    show.insert(0, "On Big Board", show["player_id"].isin(watched))

    board_show = _display(_mlb_style(show, team_label=team_label))
    event = st.dataframe(
        board_show, use_container_width=True, hide_index=True,
        column_config=_number_format(board_show),
        on_select="rerun", selection_mode="multi-row", key="rb_h_board_table")

    sel_rows = list(event.selection.rows) if event and event.selection else []
    bcol, vcol, _ = st.columns([1.4, 1.4, 2])
    sel_row = show.iloc[sel_rows[0]] if len(sel_rows) == 1 else None
    vcol.button(
        "View Batter Analysis for selected", disabled=sel_row is None,
        key="rb_h_view_btn", help="Select exactly one row above.",
        on_click=goto_hitter,
        args=(sel_row["player_id"], sel_row["team"]) if sel_row is not None else (None, None))
    if bcol.button(f"Add {len(sel_rows)} selected to Big Board",
                  disabled=not sel_rows, key="rb_h_add_btn"):
        to_add = show.iloc[sel_rows][
            ["player_id", "team", "division", "class_year", "pa", "woba",
             "p_available", "priority_score", "tier"]]
        _add_to_big_board(to_add, HITTER_BIG_BOARD_PATH, HITTER_BIG_BOARD_COLS)
        st.success(f"Added {len(sel_rows)} player(s) to the Big Board.")
        st.rerun()

    st.caption(HITTER_TIER_CAPTION)

    _render_big_board_section(HITTER_BIG_BOARD_PATH, HITTER_BIG_BOARD_COLS,
                              goto_hitter, "Batter Analysis", "rb_h",
                              team_label=team_label)


def render(DATA_DIR, EXCLUDED_TEAMS=frozenset(), goto_pitcher=None,
          goto_hitter=None, df_all=None, team_label=None):
    # Popped once here (rather than where it's shown) so the flag can't get
    # stuck if an early return fires below -- one-shot regardless of path.
    last_player = st.session_state.pop("rb_last_player", None)
    last_kind = st.session_state.pop("rb_last_kind", None)

    st.title("Returner Board")
    st.caption(
        "Cross-league bring-back priority. Built from every pitch in `Data/` "
        "involving Nashua, so an opposing player's line is only what he did "
        "against Nashua, not his full FCBL season -- expect thin samples."
    )

    with st.expander("Roster CSV (optional -- sharpens class / division / age)"):
        st.caption(
            "Needs `player_id` matching the TrackMan `Pitcher`/`Batter` string "
            "exactly, plus `class_year` and `division`. Add `age` or "
            "`grad_year` if you have them -- without one, every player "
            "defaults to SO / D1. Covers both tabs below."
        )
        uploaded_roster = st.file_uploader("roster.csv", type="csv",
                                           key="rb_roster")

    roster_path = Path(DATA_DIR) / "roster.csv"
    roster = uploaded_roster if uploaded_roster is not None else (
        roster_path if roster_path.exists() else None)
    if roster is None:
        st.info(
            "No roster.csv found or uploaded -- every player is defaulted to "
            "SO / D1. Upload one above for real class year, division and age.")
    matched_ids = _roster_ids(roster)

    tab_p, tab_h = st.tabs(["Pitchers", "Hitters"])
    with tab_p:
        _render_pitchers(DATA_DIR, EXCLUDED_TEAMS, roster, matched_ids,
                         goto_pitcher, last_player if last_kind == "pitcher" else None,
                         team_label=team_label)
    with tab_h:
        _render_hitters(df_all, EXCLUDED_TEAMS, roster, matched_ids,
                        goto_hitter, last_player if last_kind == "hitter" else None,
                        team_label=team_label)

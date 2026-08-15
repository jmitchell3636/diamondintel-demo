"""
loader.py -- turn raw pitch-level TrackMan CSVs into the pitcher-half-season
table the board expects.

Written against the standard TrackMan V3 export. If your column names differ,
edit COLS below -- that dict is the only place names are hardcoded.

Usage
-----
    from fcbl.loader import load_trackman, attach_roster, validate

    df = load_trackman("Data/")                  # folder of CSVs, or one file
    df = attach_roster(df, "Data/roster.csv")    # division/class/school_type
    print(validate(df))
    season = season_totals(df)

The aggregation is plate-appearance based, not pitch based: BF counts PA
outcomes, so a 7-pitch walk is one batter faced, not seven.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# --- column name mapping. Edit here if your export differs. ---------------
COLS = dict(
    pitcher="Pitcher",
    pitcher_id="PitcherId",
    pitcher_team="PitcherTeam",
    throws="PitcherThrows",
    date="Date",
    inning="Inning",
    pa_of_inning="PAofInning",
    top_bottom="Top/Bottom",
    batter="Batter",
    pitch_call="PitchCall",
    kor_bb="KorBB",
    hit_type="TaggedHitType",
    play_result="PlayResult",
    outs_on_play="OutsOnPlay",
    pitch_type="TaggedPitchType",
    auto_pitch_type="AutoPitchType",
    rel_speed="RelSpeed",
    ivb="InducedVertBreak",
    hb="HorzBreak",
    spin="SpinRate",
    extension="Extension",
)

FASTBALLS = {"Fastball", "FourSeamFastBall", "Four-Seam", "TwoSeamFastBall",
             "Sinker", "FF", "SI", "FT"}
BIP_TYPES = {"GroundBall", "LineDrive", "FlyBall", "Popup"}
HIT_RESULTS = {"Single", "Double", "Triple", "HomeRun"}


def _read_any(path):
    p = Path(path)
    files = sorted(p.glob("*.csv")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no CSVs found at {path}")
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    return pd.concat(frames, ignore_index=True)


def load_trackman(path, half_split_date=None, min_bf=1):
    """Aggregate pitch-level TrackMan into pitcher x half-season rows."""
    raw = _read_any(path)
    c = COLS

    missing = [v for k, v in c.items()
               if v not in raw.columns and k in
               ("pitcher", "date", "pitch_call", "kor_bb", "play_result")]
    if missing:
        raise KeyError(f"missing required columns: {missing}. Edit COLS in "
                       "fcbl/loader.py to match your export.")

    d = raw.copy()
    d[c["date"]] = pd.to_datetime(d[c["date"]], errors="coerce")
    d = d[d[c["pitcher"]].notna() & d[c["date"]].notna()]

    # --- half-season split by CALENDAR DATE, not game count, so uneven park
    # coverage does not produce an unbalanced split ---
    if half_split_date is None:
        half_split_date = d[c["date"]].quantile(0.5)
    else:
        half_split_date = pd.to_datetime(half_split_date)
    d["half"] = np.where(d[c["date"]] <= half_split_date, "H1", "H2")

    # --- identify plate-appearance-ending pitches ---
    pc = d[c["pitch_call"]].astype(str)
    kb = d[c["kor_bb"]].astype(str)
    pr = d[c["play_result"]].astype(str)
    ht = d.get(c["hit_type"], pd.Series("Undefined", index=d.index)).astype(str)

    d["_k"] = (kb == "Strikeout").astype(int)
    d["_bb"] = (kb == "Walk").astype(int)
    d["_hbp"] = (pc == "HitByPitch").astype(int)
    d["_inplay"] = (pc == "InPlay").astype(int)
    d["_pa_end"] = ((d._k | d._bb | d._hbp | d._inplay) > 0).astype(int)

    d["_gb"] = ((ht == "GroundBall") & (d._inplay == 1)).astype(int)
    d["_ld"] = ((ht == "LineDrive") & (d._inplay == 1)).astype(int)
    d["_fb"] = (ht.isin(["FlyBall", "Popup"]) & (d._inplay == 1)).astype(int)
    d["_hr"] = (pr == "HomeRun").astype(int)
    d["_hit"] = pr.isin(HIT_RESULTS).astype(int)
    d["_bip"] = ((d._gb + d._ld + d._fb) > 0).astype(int)
    # hits on balls in play, excluding home runs -> BABIP numerator
    d["_hit_bip"] = ((d._hit == 1) & (d._hr == 0) & (d._bip == 1)).astype(int)
    d["_bip_nohr"] = ((d._bip == 1) & (d._hr == 0)).astype(int)

    oop = pd.to_numeric(d.get(c["outs_on_play"], 0), errors="coerce").fillna(0)
    # Some TrackMan exports already include the strikeout in OutsOnPlay and
    # some do not. Adding K unconditionally double-counts and inflates IP,
    # which then corrupts every per-inning rate downstream. Detect which
    # convention this export uses instead of assuming.
    k_rows = d._k == 1
    k_already_counted = bool(k_rows.any() and (oop[k_rows] > 0).mean() > 0.5)
    d["_outs"] = oop if k_already_counted else (d._k + oop)
    d.attrs["k_in_outs_on_play"] = k_already_counted

    grp = [c["pitcher"], "half"]
    if c["pitcher_team"] in d.columns:
        grp.insert(1, c["pitcher_team"])

    agg = d.groupby(grp, as_index=False).agg(
        bf=("_pa_end", "sum"), k=("_k", "sum"), bb=("_bb", "sum"),
        hbp=("_hbp", "sum"), gb=("_gb", "sum"), ld=("_ld", "sum"),
        fb=("_fb", "sum"), hr=("_hr", "sum"),
        bip=("_bip", "sum"), bip_nohr=("_bip_nohr", "sum"),
        hits=("_hit_bip", "sum"), outs=("_outs", "sum"),
        games=(c["date"], "nunique"))
    agg["ip"] = agg.outs / 3.0

    # --- stuff features, measured on fastballs only ---
    # TaggedPitchType is a human tag and is "Undefined" wherever nobody tagged
    # the pitch -- which is most rows in this export. Fall back to the
    # machine-classified AutoPitchType on exactly those rows rather than
    # abandoning TaggedPitchType file-wide, or files with mostly-untagged
    # pitches lose nearly all their fastballs (and velo) even though
    # AutoPitchType classified them fine.
    has_tagged = c["pitch_type"] in d.columns
    has_auto = c["auto_pitch_type"] in d.columns
    if has_tagged:
        pt = d[c["pitch_type"]].astype(str)
        if has_auto:
            auto = d[c["auto_pitch_type"]].astype(str)
            pt = pt.where(~pt.isin(["Undefined", "nan", "None", ""]), auto)
    elif has_auto:
        pt = d[c["auto_pitch_type"]].astype(str)
    else:
        pt = None
    fb_rows = d[pt.isin(FASTBALLS)] if pt is not None else d
    stuff_cols = {k: c[k] for k in ("rel_speed", "ivb", "hb", "spin", "extension")
                  if c[k] in d.columns}
    if stuff_cols:
        st = fb_rows.groupby(grp, as_index=False).agg(
            **{k: (v, "mean") for k, v in stuff_cols.items()})
        st = st.rename(columns=dict(rel_speed="velo", ivb="ivb", hb="hb",
                                    spin="spin", extension="extension"))
        agg = agg.merge(st, on=grp, how="left")

    agg = agg.rename(columns={c["pitcher"]: "player_id",
                              c["pitcher_team"]: "team"})
    if "team" not in agg.columns:
        agg["team"] = "UNK"

    # role: starters are pitchers who typically open a game
    if c["inning"] in d.columns:
        first = d[d[c["inning"]] == 1].groupby(c["pitcher"])[c["date"]].nunique()
        tot = d.groupby(c["pitcher"])[c["date"]].nunique()
        sp_share = (first / tot).fillna(0)
        agg["role"] = np.where(
            agg.player_id.map(sp_share).fillna(0) >= 0.5, "SP", "RP")
    else:
        agg["role"] = "RP"

    return agg[agg.bf >= min_bf].reset_index(drop=True)


def attach_roster(df, roster_path):
    """Join division / class_year / school_type from a roster CSV.

    Roster needs at minimum: player_id (matching the TrackMan `Pitcher` string),
    class_year, division. Optional but better: age or grad_year, school_type.
    """
    r = pd.read_csv(roster_path)
    if "player_id" not in r.columns:
        for cand in ("Pitcher", "name", "Name", "player"):
            if cand in r.columns:
                r = r.rename(columns={cand: "player_id"})
                break
    out = df.merge(r, on="player_id", how="left")

    if "school_type" not in out.columns:
        out["school_type"] = np.where(
            out.get("division", pd.Series("", index=out.index)).astype(str)
            .str.upper().str.contains("JUCO|NJCAA"), "JUCO", "4YR")
    for col, default in (("class_year", "SO"), ("division", "D1")):
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default)
    return out


def validate(df):
    """Sanity report. Run this before trusting anything downstream."""
    rows = []

    def chk(name, ok, detail=""):
        rows.append(dict(check=name, status="PASS" if ok else "REVIEW",
                         detail=detail))

    n = len(df)
    chk("rows present", n > 0, f"{n} pitcher-half rows")
    both = df.groupby("player_id").half.nunique()
    chk("players with both halves", (both == 2).sum() > 20,
        f"{int((both == 2).sum())} of {both.size} -- reliability needs both")

    bad_bf = df[(df.k + df.bb + df.hbp) > df.bf]
    chk("BF >= K+BB+HBP", len(bad_bf) == 0, f"{len(bad_bf)} rows violate")

    bad_bip = df[(df.gb + df.ld + df.fb) != df.bip]
    chk("GB+LD+FB == BIP", len(bad_bip) == 0, f"{len(bad_bip)} rows violate")

    if "velo" in df.columns:
        v = df.velo.dropna()
        chk("velocity plausible", len(v) > 0 and 70 < v.mean() < 100,
            f"mean {v.mean():.1f} mph, {df.velo.isna().sum()} missing")

    kr = (df.k.sum() / max(df.bf.sum(), 1))
    chk("league K% plausible", 0.12 < kr < 0.35, f"{kr:.3f}")
    bb = (df.bb.sum() / max(df.bf.sum(), 1))
    chk("league BB% plausible", 0.04 < bb < 0.20, f"{bb:.3f}")
    ipbf = df.bf.sum() / max(df.ip.sum(), 1)
    chk("BF per IP plausible", 3.6 < ipbf < 5.2, f"{ipbf:.2f}")

    return pd.DataFrame(rows)

"""
synth_fcbl.py -- synthetic FCBL-shaped data for developing the returner board.

Shaped to match the real problem, not MLB:
  * ~56 game season, so pitchers accumulate 40-60 IP (SP) or 15-25 IP (RP)
  * talent spread is MUCH wider than MLB -- D1, D2 and D3 arms in one league
  * TrackMan features available per pitcher
  * split into first / second half so reliability can be measured within season

Replace with your real TrackMan aggregation. Required columns are listed in
README_FCBL.md.
"""
import numpy as np
import pandas as pd
from scipy.special import logit, expit

DIVISIONS = ["D1", "D2", "D3", "NAIA", "JUCO"]
DIV_P = [0.46, 0.16, 0.22, 0.09, 0.07]
DIV_SHIFT = {"D1": 0.34, "D2": 0.05, "D3": -0.22, "NAIA": -0.14, "JUCO": -0.02}
CLASSES = ["FR", "SO", "JR", "SR"]
CLASS_P = [0.24, 0.30, 0.30, 0.16]

# Age is NOT determined by class year. Redshirts, JUCO transfers, gap years and
# reclassified high schoolers mean a "sophomore" spans roughly 19-22. This is
# exactly why class year is a poor proxy for age and should not be weighted as
# though it were one.
CLASS_AGE_MU = {"FR": 19.3, "SO": 20.2, "JR": 21.2, "SR": 22.3}
CLASS_AGE_SD = {"FR": 0.55, "SO": 0.70, "JR": 0.80, "SR": 0.85}
TEAMS = ["NAS", "WOR", "NSH", "BRO", "PIT", "NOR", "VER", "WES"]


def generate_fcbl(n_pitchers=170, seed=42):
    rng = np.random.default_rng(seed)

    div = rng.choice(DIVISIONS, n_pitchers, p=DIV_P)
    cls = rng.choice(CLASSES, n_pitchers, p=CLASS_P)
    role = rng.choice(["SP", "RP"], n_pitchers, p=[0.42, 0.58])
    team = rng.choice(TEAMS, n_pitchers)

    # school type drives draft eligibility: JUCO players are eligible every year
    school_type = np.where(div == "JUCO", "JUCO", "4YR")

    # age as of June 1, in years. Deliberately overlapping across classes.
    age = np.array([rng.normal(CLASS_AGE_MU[c], CLASS_AGE_SD[c]) for c in cls])
    age = np.clip(age + (school_type == "JUCO") * 0.25, 18.2, 24.5)

    # --- latent stuff, which drives both TrackMan and outcomes ---
    # Development is driven by AGE, not class. Ages 19-22 are the steepest
    # velocity-gain years of a pitcher's life, so the age slope here is positive
    # and large -- the opposite sign of an MLB aging curve.
    stuff_latent = (rng.normal(0, 1.0, n_pitchers)
                    + np.array([DIV_SHIFT[d] for d in div]) * 1.5
                    + 0.26 * (age - 20.5))

    velo = 88.5 + 2.6 * stuff_latent + (role == "RP") * 0.9 + rng.normal(0, 0.9, n_pitchers)
    ivb = 12.0 + 2.6 * stuff_latent + rng.normal(0, 2.6, n_pitchers)
    hb = rng.normal(8.0, 5.0, n_pitchers)
    spin = 2150 + 150 * stuff_latent + rng.normal(0, 190, n_pitchers)
    ext = 5.9 + 0.14 * stuff_latent + rng.normal(0, 0.32, n_pitchers)

    # --- true talent: wide spread, driven by stuff plus a residual ---
    k_true = expit(logit(0.215) + 0.62 * stuff_latent + rng.normal(0, 0.42, n_pitchers))
    bb_true = expit(logit(0.105) - 0.20 * stuff_latent + rng.normal(0, 0.40, n_pitchers))
    gb_true = expit(logit(0.430) - 0.10 * stuff_latent + rng.normal(0, 0.30, n_pitchers))
    hbp_true = np.full(n_pitchers, 0.014)
    ld_true = np.full(n_pitchers, 0.235)
    hrfb_true = expit(logit(0.085) - 0.22 * stuff_latent + rng.normal(0, 0.22, n_pitchers))
    babip_true = expit(logit(0.305) - 0.10 * stuff_latent + rng.normal(0, 0.10, n_pitchers))

    ip_total = np.where(role == "SP",
                        np.clip(rng.normal(46, 14, n_pitchers), 6, 78),
                        np.clip(rng.normal(20, 8, n_pitchers), 3, 42))

    rows = []
    for i in range(n_pitchers):
        # split the season in half so within-season reliability is measurable
        frac = rng.beta(6, 6)
        for half, ip in (("H1", ip_total[i] * frac), ("H2", ip_total[i] * (1 - frac))):
            bf = max(int(round(ip * 4.4)), 1)
            k = rng.binomial(bf, k_true[i])
            bb = rng.binomial(max(bf - k, 0), min(bb_true[i] / max(1 - k_true[i], 1e-6), 1))
            hbp = rng.binomial(max(bf - k - bb, 0), hbp_true[i])
            bip = max(bf - k - bb - hbp, 1)
            gb = rng.binomial(bip, gb_true[i])
            ld = rng.binomial(max(bip - gb, 0), min(ld_true[i] / max(1 - gb_true[i], 1e-6), 1))
            fb = max(bip - gb - ld, 0)
            hr = rng.binomial(fb, hrfb_true[i]) if fb > 0 else 0
            bip_nohr = max(bip - hr, 1)
            hits = rng.binomial(bip_nohr, babip_true[i])
            rows.append(dict(
                player_id=i, half=half, team=team[i], role=role[i],
                division=div[i], class_year=cls[i],
                age=round(age[i], 2), school_type=school_type[i],
                velo=round(velo[i], 1), ivb=round(ivb[i], 1), hb=round(hb[i], 1),
                spin=round(spin[i]), extension=round(ext[i], 2),
                ip=round(ip, 1), bf=bf, k=k, bb=bb, hbp=hbp, gb=gb, ld=ld,
                fb=fb, hr=hr, bip=bip, bip_nohr=bip_nohr, hits=hits))

    df = pd.DataFrame(rows)
    truth = pd.DataFrame(dict(
        player_id=np.arange(n_pitchers), stuff_latent=stuff_latent, age=age,
        true_k=k_true, true_bb=bb_true, true_hrfb=hrfb_true,
        true_babip=babip_true, true_gb=gb_true))
    return df, truth


def season_totals(df):
    """Collapse halves into full-season lines."""
    # Only group on identity columns that actually exist -- real TrackMan
    # output will not have `age` unless a roster with DOB has been joined.
    id_cols = ["player_id", "team", "role", "division", "class_year", "age",
               "school_type", "grad_year", "school"]
    keys = [c for c in id_cols if c in df.columns]
    sum_cols = [c for c in ["ip", "bf", "k", "bb", "hbp", "gb", "ld", "fb",
                            "hr", "bip", "bip_nohr", "hits"] if c in df.columns]
    mean_cols = [c for c in ["velo", "ivb", "hb", "spin", "extension"]
                 if c in df.columns]
    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "mean" for c in mean_cols})
    out = df.groupby(keys, as_index=False, dropna=False).agg(agg)
    return out

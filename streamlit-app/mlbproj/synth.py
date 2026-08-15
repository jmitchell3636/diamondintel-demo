"""
synth.py -- generate a synthetic league where ground truth is known.

Why bother: on real data you never observe true talent, so you cannot tell
whether your shrinkage constant is right or your aging curve is contaminated by
survivorship. Here you can. Build the estimators against this, then swap the
data layer for pybaseball.

Planted structure (what the estimators must recover):
  * per-component talent variance  -> tests the empirical-Bayes constants
  * per-component aging curves     -> tests the aging fit
  * a selection-biased panel       -> tests survivorship handling
  * a small talent random walk     -> punishes over-shrinkage
"""
import numpy as np
import pandas as pd
from scipy.special import logit, expit

from .core import LG, RATE_COMPONENTS

# True talent spread, on the logit scale, at peak age.
TALENT_SD = {
    "k_rate": 0.360, "bb_rate": 0.300, "hbp_rate": 0.220,
    "gb_rate": 0.300, "ld_rate": 0.120, "hr_fb": 0.180, "babip": 0.055,
}

# Year-over-year random walk in true talent. Small, but it is why real
# projections regress more than a fixed-talent model says they should.
TALENT_WALK = {
    "k_rate": 0.075, "bb_rate": 0.085, "hbp_rate": 0.090,
    "gb_rate": 0.055, "ld_rate": 0.050, "hr_fb": 0.110, "babip": 0.030,
}

# True aging: logit-scale delta per year relative to age 27, as a quadratic.
# (linear, quadratic) coefficients in (age - 27).
TRUE_AGING = {
    "k_rate":  (-0.0180, -0.00090),   # peaks early, decays and accelerates
    "bb_rate": (-0.0140,  0.00120),   # command improves into late 20s
    "hbp_rate":(0.0000,   0.00000),
    "gb_rate": (0.0060,   0.00000),   # mild groundball drift with age
    "ld_rate": (0.0020,   0.00000),
    "hr_fb":   (0.0090,   0.00035),
    "babip":   (0.0015,   0.00000),
}

VELO_PEAK_AGE = 24.0
VELO_DECLINE = 0.22      # mph per year after peak
VELO_NOISE = 0.35


def _draw_talent(rng, n, role):
    tal = {}
    for c in RATE_COMPONENTS:
        mu = logit(getattr(LG, c))
        if role == "RP":
            # relievers miss more bats, walk more, in one short burst
            if c == "k_rate":
                mu += 0.28
            if c == "bb_rate":
                mu += 0.10
        tal[c] = mu + rng.normal(0, TALENT_SD[c], n)
    return tal


def _age_delta(c, age):
    a, b = TRUE_AGING[c]
    d = age - 27.0
    return a * d + b * d * d


def _survival_prob(age, prev_ip, velo_delta, role):
    """True hazard: probability of throwing meaningful innings next season."""
    z = (2.15
         - 0.085 * np.maximum(age - 28.0, 0.0) ** 1.35
         + 0.0045 * np.minimum(prev_ip, 200.0)
         + 0.55 * velo_delta                       # losing velo is the tell
         + (0.15 if role == "SP" else 0.0))
    return expit(z)


def generate_league(n_players=900, seasons=(2011, 2024), seed=7):
    """Return (observed_panel, truth_panel) as tidy DataFrames."""
    rng = np.random.default_rng(seed)
    y0, y1 = seasons
    obs_rows, truth_rows = [], []

    roles = rng.choice(["SP", "RP"], size=n_players, p=[0.55, 0.45])
    debut_age = np.clip(rng.normal(24.5, 1.9, n_players), 20, 31)
    debut_year = rng.integers(y0 - 3, y1 - 1, n_players)
    talent0 = {}
    for r in ("SP", "RP"):
        idx = np.where(roles == r)[0]
        t = _draw_talent(rng, len(idx), r)
        for c in RATE_COMPONENTS:
            talent0.setdefault(c, np.zeros(n_players))[idx] = t[c]

    # baseline durability: some arms are simply more available than others
    frailty = rng.normal(0, 0.45, n_players)
    velo0 = rng.normal(94.6 if True else 0, 1.9, n_players) + \
        0.9 * (roles == "RP") + 1.1 * (talent0["k_rate"] - logit(LG.k_rate))

    for i in range(n_players):
        role = roles[i]
        age = debut_age[i]
        cur = {c: talent0[c][i] for c in RATE_COMPONENTS}
        velo = velo0[i]
        prev_ip = 60.0 if role == "SP" else 30.0
        alive = True
        for year in range(int(debut_year[i]), y1 + 1):
            if year < y0 - 3 or not alive:
                break
            # --- talent evolves: random walk, on top of the aging curve ---
            for c in RATE_COMPONENTS:
                cur[c] += rng.normal(0, TALENT_WALK[c])

            velo_prev = velo
            velo = (velo - VELO_DECLINE * max(age - VELO_PEAK_AGE, 0) / 3.0
                    + rng.normal(0, VELO_NOISE))
            velo_delta = velo - velo_prev

            p = _survival_prob(age, prev_ip, velo_delta, role) 
            p = np.clip(p + 0.06 * frailty[i], 0.02, 0.995)
            if rng.random() > p:
                alive = False
                if year >= y0:
                    truth_rows.append(dict(player_id=i, season=year, age=age,
                                           role=role, velo=velo, appeared=0,
                                           ip=0.0,
                                           **{f"true_{c}": expit(cur[c] + _age_delta(c, age))
                                              for c in RATE_COMPONENTS}))
                break

            # --- playing time, conditional on appearing ---
            if role == "SP":
                mu = np.log(165.0) + 0.35 * frailty[i] - 0.02 * max(age - 30, 0)
                ip = float(np.clip(rng.lognormal(mu, 0.42), 8, 235))
            else:
                mu = np.log(58.0) + 0.30 * frailty[i]
                ip = float(np.clip(rng.lognormal(mu, 0.45), 4, 100))

            true_rates = {c: expit(cur[c] + _age_delta(c, age))
                          for c in RATE_COMPONENTS}

            # --- observe noisy counting stats via binomial sampling ---
            bf = int(round(ip * 4.25))
            k = rng.binomial(bf, true_rates["k_rate"])
            bb = rng.binomial(bf - k, true_rates["bb_rate"] /
                              max(1 - true_rates["k_rate"], 1e-6))
            hbp = rng.binomial(max(bf - k - bb, 0),
                               min(true_rates["hbp_rate"] * 1.05, 0.5))
            bip = max(bf - k - bb - hbp, 1)
            gb = rng.binomial(bip, true_rates["gb_rate"])
            ld = rng.binomial(bip - gb, min(true_rates["ld_rate"] /
                                            max(1 - true_rates["gb_rate"], 1e-6), 1.0))
            fb = bip - gb - ld
            hr = rng.binomial(fb, true_rates["hr_fb"]) if fb > 0 else 0
            bip_nohr = max(bip - hr, 1)
            hits = rng.binomial(bip_nohr, true_rates["babip"])

            if year >= y0:
                obs_rows.append(dict(
                    player_id=i, season=year, age=round(age, 1), role=role,
                    velo=round(velo, 2), ip=round(ip, 1), bf=bf,
                    k=k, bb=bb, hbp=hbp, gb=gb, ld=ld, fb=fb, hr=hr,
                    bip=bip, bip_nohr=bip_nohr, hits=hits))
                truth_rows.append(dict(player_id=i, season=year, age=age,
                                       role=role, velo=velo, appeared=1, ip=ip,
                                       **{f"true_{c}": true_rates[c]
                                          for c in RATE_COMPONENTS}))
            prev_ip = ip
            age += 1.0

    obs = pd.DataFrame(obs_rows).sort_values(["player_id", "season"])
    truth = pd.DataFrame(truth_rows).sort_values(["player_id", "season"])
    return obs.reset_index(drop=True), truth.reset_index(drop=True)


def observed_rates(df):
    """Attach observed rate columns with the correct denominators."""
    out = df.copy()
    out["obs_k_rate"] = out.k / out.bf
    out["obs_bb_rate"] = out.bb / out.bf
    out["obs_hbp_rate"] = out.hbp / out.bf
    out["obs_gb_rate"] = out.gb / out.bip
    out["obs_ld_rate"] = out.ld / out.bip
    out["obs_hr_fb"] = np.where(out.fb > 0, out.hr / out.fb.replace(0, np.nan), np.nan)
    out["obs_babip"] = out.hits / out.bip_nohr
    out["denom_k_rate"] = out.bf
    out["denom_bb_rate"] = out.bf
    out["denom_hbp_rate"] = out.bf
    out["denom_gb_rate"] = out.bip
    out["denom_ld_rate"] = out.bip
    out["denom_hr_fb"] = out.fb
    out["denom_babip"] = out.bip_nohr
    return out

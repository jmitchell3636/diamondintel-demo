"""
core.py -- run environment, WAR conversion, and the market $/WAR curve.

Design notes
------------
* Runs are estimated with BaseRuns, not FIP. BaseRuns is multiplicative, so it
  handles the interaction between baserunners and extra-base power correctly and
  stays sane in the tails that a Monte Carlo engine will inevitably draw.
* Everything is expressed as *rates* so the projection layer never has to think
  about playing time; innings are applied once, at the WAR step.
"""
from dataclasses import dataclass, replace
import numpy as np

# ---------------------------------------------------------------------------
# League environment. Swap these for the actual season you're modelling.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LeagueEnv:
    k_rate:   float = 0.225   # per batter faced
    bb_rate:  float = 0.082   # per BF
    hbp_rate: float = 0.011   # per BF
    gb_rate:  float = 0.435   # per ball in play
    ld_rate:  float = 0.245   # per BIP
    fb_rate:  float = 0.320   # per BIP (includes popups)
    hr_fb:    float = 0.125   # per fly ball
    babip:    float = 0.292   # on non-HR balls in play

    ra9:      float = 4.45    # league runs allowed per 9
    rpw:      float = 9.7     # runs per win
    rep_ra9_sp: float = 5.55  # replacement level, starters
    rep_ra9_rp: float = 5.05  # replacement level, relievers

    # Batted-ball-type hit rates on non-HR BIP, used to split BABIP by type.
    gb_hit: float = 0.240
    ld_hit: float = 0.680
    fb_hit: float = 0.130

    # Extra-base distribution of hits, by batted ball type: (1B, 2B, 3B)
    gb_xb: tuple = (0.955, 0.040, 0.005)
    ld_xb: tuple = (0.740, 0.235, 0.025)
    fb_xb: tuple = (0.400, 0.560, 0.040)

    baseruns_b_mult: float = 1.02
    ra9_calibration: float = 1.0  # set by calibrate()


LG = LeagueEnv()

# Component names used everywhere downstream. Each maps to (numerator, denom).
RATE_COMPONENTS = ["k_rate", "bb_rate", "hbp_rate", "gb_rate", "ld_rate",
                   "hr_fb", "babip"]

# Which denominator each rate is measured against.
DENOM = {
    "k_rate": "bf", "bb_rate": "bf", "hbp_rate": "bf",
    "gb_rate": "bip", "ld_rate": "bip",
    "hr_fb": "fb", "babip": "bip_nohr",
}


def _asarray(x):
    return np.atleast_1d(np.asarray(x, dtype=float))


def outcomes_from_rates(rates, bf, env: LeagueEnv = LG):
    """Expand component rates into a full outcome line, per batter faced.

    Returns a dict of counts (arrays) given `bf` batters faced. This is the
    bridge between 'what the projection model predicts' and 'what the run
    estimator needs'.
    """
    bf = _asarray(bf)
    k   = bf * _asarray(rates["k_rate"])
    bb  = bf * _asarray(rates["bb_rate"])
    hbp = bf * _asarray(rates["hbp_rate"])

    bip = np.maximum(bf - k - bb - hbp, 1e-9)

    gb_r = _asarray(rates["gb_rate"])
    ld_r = _asarray(rates["ld_rate"])
    fb_r = np.maximum(1.0 - gb_r - ld_r, 1e-9)   # FB is the residual

    gb, ld, fb = bip * gb_r, bip * ld_r, bip * fb_r
    hr = fb * _asarray(rates["hr_fb"])
    fb_nohr = np.maximum(fb - hr, 0.0)

    # Distribute the pitcher's BABIP across batted-ball types, preserving the
    # league-relative shape. Scale factor s solves:
    #   babip * bip_nohr = s * (gb*gb_hit + ld*ld_hit + fb_nohr*fb_hit)
    bip_nohr = np.maximum(gb + ld + fb_nohr, 1e-9)
    base_hits = gb * env.gb_hit + ld * env.ld_hit + fb_nohr * env.fb_hit
    s = (_asarray(rates["babip"]) * bip_nohr) / np.maximum(base_hits, 1e-9)

    gb_h = np.minimum(gb * env.gb_hit * s, gb)
    ld_h = np.minimum(ld * env.ld_hit * s, ld)
    fb_h = np.minimum(fb_nohr * env.fb_hit * s, fb_nohr)

    singles = gb_h * env.gb_xb[0] + ld_h * env.ld_xb[0] + fb_h * env.fb_xb[0]
    doubles = gb_h * env.gb_xb[1] + ld_h * env.ld_xb[1] + fb_h * env.fb_xb[1]
    triples = gb_h * env.gb_xb[2] + ld_h * env.ld_xb[2] + fb_h * env.fb_xb[2]

    h = singles + doubles + triples + hr
    outs = np.maximum(bf - h - bb - hbp, 1e-9)

    return dict(bf=bf, k=k, bb=bb, hbp=hbp, h=h, hr=hr,
                singles=singles, doubles=doubles, triples=triples,
                gb=gb, ld=ld, fb=fb, outs=outs)


def baseruns_ra9(rates, env: LeagueEnv = LG, bf=1000.0):
    """Runs allowed per 9 innings implied by a set of component rates."""
    o = outcomes_from_rates(rates, bf, env)
    tb = o["singles"] + 2 * o["doubles"] + 3 * o["triples"] + 4 * o["hr"]

    A = o["h"] + o["bb"] + o["hbp"] - o["hr"]
    B = (1.4 * tb - 0.6 * o["h"] - 3 * o["hr"]
         + 0.1 * (o["bb"] + o["hbp"])) * env.baseruns_b_mult
    C = o["outs"]
    D = o["hr"]

    runs = A * B / np.maximum(B + C, 1e-9) + D
    ra9 = runs * 27.0 / C
    return ra9 * env.ra9_calibration


def calibrate(env: LeagueEnv = LG) -> LeagueEnv:
    """Force league-average component rates to reproduce league-average RA9."""
    lg_rates = {c: getattr(env, c) for c in RATE_COMPONENTS}
    raw = float(baseruns_ra9(lg_rates, replace(env, ra9_calibration=1.0))[0])
    return replace(env, ra9_calibration=env.ra9 / raw)


def war_from_ra9(ra9, ip, role="SP", leverage=1.0, env: LeagueEnv = LG):
    """Convert a rate + playing time into wins above replacement.

    Relievers get a leverage multiplier of (1 + gmLI)/2, the standard
    half-credit treatment: a closer's innings are worth more than a mop-up
    man's, but not proportionally more, because leverage is partly the
    manager's choice rather than the pitcher's contribution.
    """
    ra9 = _asarray(ra9)
    ip = _asarray(ip)
    rep = env.rep_ra9_sp if role == "SP" else env.rep_ra9_rp
    raa = (rep - ra9) / 9.0 * ip
    war = raa / env.rpw
    if role == "RP":
        war = war * (1.0 + leverage) / 2.0
    return war


# ---------------------------------------------------------------------------
# Market value: dollars per win is NOT a constant.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketCurve:
    """$ = base * WAR^gamma, with gamma > 1 for star scarcity.

    Roster spots are fixed at 26, so a 6-win pitcher is worth more than two
    3-win pitchers -- the second roster spot has an opportunity cost. Fit
    `base` and `gamma` by regressing guaranteed AAV against *projected* WAR at
    time of signing across historical free agent contracts.
    """
    base: float = 8.6e6     # $ per win at 1 WAR
    gamma: float = 1.18     # convexity
    inflation_mu: float = 0.025
    inflation_sd: float = 0.030
    war_floor: float = 0.0

    def dollars(self, war, years_out=0, inflation=None):
        war = np.maximum(_asarray(war), self.war_floor)
        infl = self.inflation_mu if inflation is None else inflation
        return self.base * np.power(war, self.gamma) * (1.0 + infl) ** years_out


def npv(cashflows, discount_rate=0.06):
    """Present value of a per-year cashflow array (year 0 = today)."""
    cf = _asarray(cashflows)
    t = np.arange(cf.shape[-1])
    return np.sum(cf / (1.0 + discount_rate) ** t, axis=-1)

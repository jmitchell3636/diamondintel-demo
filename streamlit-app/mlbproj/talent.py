"""
talent.py -- empirical-Bayes true-talent estimation for each component rate.

The regression constant m is estimated by method of moments:

    Var(observed) = Var(true) + E[ p(1-p) / n ]
    m = p_lg * (1 - p_lg) / Var(true)

m has a direct interpretation: it is the number of "league average" trials you
mix in, and it is numerically the same thing people call the stabilization
point. Estimating it from your own data beats quoting published constants,
which drift with the run environment.
"""
import numpy as np
import pandas as pd

from .core import RATE_COMPONENTS

# Recency weights applied to raw counts before shrinking (most recent first).
DEFAULT_YEAR_WEIGHTS = (5.0, 4.0, 3.0, 1.5)


def estimate_regression_constant(x, n, min_n=25, trim=0.005):
    """Method-of-moments empirical Bayes constant for a binomial rate."""
    x = np.asarray(x, float)
    n = np.asarray(n, float)
    ok = (n >= min_n) & np.isfinite(x) & np.isfinite(n)
    x, n = x[ok], n[ok]
    if len(x) < 30:
        return np.nan, np.nan

    p_lg = x.sum() / n.sum()
    p_hat = x / n

    # Weight by n so that big samples inform the mean more, but the variance
    # decomposition still needs the per-observation sampling variance.
    w = n / n.sum()
    var_obs = np.sum(w * (p_hat - p_lg) ** 2)
    var_samp = np.sum(w * p_lg * (1 - p_lg) / n)
    var_true = max(var_obs - var_samp, 1e-8)

    m = p_lg * (1 - p_lg) / var_true
    return float(p_lg), float(m)


class TalentModel:
    """Fits per-component league means and regression constants."""

    def __init__(self, components=RATE_COMPONENTS,
                 year_weights=DEFAULT_YEAR_WEIGHTS):
        self.components = list(components)
        self.year_weights = tuple(year_weights)
        self.p_lg, self.m = {}, {}

    def fit(self, df):
        num = dict(k_rate="k", bb_rate="bb", hbp_rate="hbp",
                   gb_rate="gb", ld_rate="ld", hr_fb="hr", babip="hits")
        den = dict(k_rate="bf", bb_rate="bf", hbp_rate="bf",
                   gb_rate="bip", ld_rate="bip", hr_fb="fb", babip="bip_nohr")
        for c in self.components:
            p, m = estimate_regression_constant(df[num[c]], df[den[c]])
            self.p_lg[c], self.m[c] = p, m
        return self

    def summary(self):
        return pd.DataFrame({
            "component": self.components,
            "league_mean": [self.p_lg[c] for c in self.components],
            "regression_constant_m": [self.m[c] for c in self.components],
        })

    # ------------------------------------------------------------------
    def estimate(self, history, aging=None, target_age=None):
        """True-talent estimate from a player's multi-season history.

        `history` is a DataFrame for ONE player, sorted most-recent-first,
        containing raw counts and `age`. If an AgingModel is supplied, each
        past season is first age-adjusted to the target age before pooling --
        otherwise a 33-year-old's age-30 season gets credited at face value.
        """
        num = dict(k_rate="k", bb_rate="bb", hbp_rate="hbp",
                   gb_rate="gb", ld_rate="ld", hr_fb="hr", babip="hits")
        den = dict(k_rate="bf", bb_rate="bf", hbp_rate="bf",
                   gb_rate="bip", ld_rate="bip", hr_fb="fb", babip="bip_nohr")

        out, reliability = {}, {}
        h = history.head(len(self.year_weights))
        w = np.array(self.year_weights[:len(h)], float)

        for c in self.components:
            x = h[num[c]].to_numpy(float)
            n = h[den[c]].to_numpy(float)
            n = np.where(np.isfinite(n) & (n > 0), n, 0.0)
            x = np.where(np.isfinite(x), x, 0.0)

            if aging is not None and target_age is not None and n.sum() > 0:
                # shift each season's observed rate to the target age, then
                # convert back to an equivalent "count" at that age
                ages = h["age"].to_numpy(float)
                p_obs = np.divide(x, n, out=np.full_like(x, self.p_lg[c]),
                                  where=n > 0)
                p_adj = aging.adjust(c, p_obs, ages, target_age)
                x = p_adj * n

            xw, nw = float((x * w).sum()), float((n * w).sum())
            m = self.m[c]
            out[c] = (xw + m * self.p_lg[c]) / (nw + m)
            reliability[c] = nw / (nw + m)   # 0 = pure league avg, 1 = pure obs

        return out, reliability


def marcel_baseline(history, p_lg, weights=(5.0, 4.0, 3.0), regress_pa=100.0):
    """Classic Marcel: weighted average of raw rates, regressed to the mean.

    Included purely as the benchmark every projection system must beat.
    """
    h = history.head(len(weights))
    w = np.array(weights[:len(h)], float)
    out = {}
    num = dict(k_rate="k", bb_rate="bb", hbp_rate="hbp",
               gb_rate="gb", ld_rate="ld", hr_fb="hr", babip="hits")
    den = dict(k_rate="bf", bb_rate="bf", hbp_rate="bf",
               gb_rate="bip", ld_rate="bip", hr_fb="fb", babip="bip_nohr")
    for c in p_lg:
        x = np.nan_to_num(h[num[c]].to_numpy(float))
        n = np.nan_to_num(h[den[c]].to_numpy(float))
        xw, nw = float((x * w).sum()), float((n * w).sum())
        out[c] = (xw + regress_pa * p_lg[c]) / (nw + regress_pa)
    return out

"""
value.py -- true-talent estimation and run value for FCBL arms.

The key departure from the MLB build: with ~45 IP, shrinking toward the LEAGUE
MEAN throws away everything TrackMan already told you. Instead we shrink toward
a PITCHER-SPECIFIC prior predicted from stuff. Velocity and movement are
measured on every pitch, so the prior is well determined even when the outcome
sample is not.

Ranking uses an upper confidence bound rather than the posterior mean. A summer
roster spot is a short-duration option: the downside of a wrong bring-back is
one summer of one slot, the upside is a returning front-line arm. Ranking by
posterior mean systematically buries small-sample high-variance arms, which are
exactly the ones whose upside you are buying.
"""
import numpy as np
import pandas as pd
from scipy.special import logit, expit

from .reliability import NUM, DEN, moment_m

STUFF_FEATURES = ["velo", "ivb", "hb_abs", "spin", "extension", "age"]

# Age enters the STUFF prior only. Ages 19-22 are the steepest velocity-gain
# years in a pitcher's life, and that relationship is far less contaminated by
# league-selection than outcome stats are: a 19-year-old throwing 94 is throwing
# 94 regardless of how selectively he was chosen. Age is deliberately NOT a
# covariate in the outcome model, where a one-season cross-section cannot
# separate development from selection.


def _design(df):
    X = pd.DataFrame(index=df.index)
    X["velo"] = df.velo
    X["ivb"] = df.ivb
    X["hb_abs"] = df.hb.abs()
    X["spin"] = df.spin / 100.0
    X["extension"] = df.extension
    from .age import resolve_age
    X["age"] = resolve_age(df)[0]
    X["is_rp"] = (df.role == "RP").astype(float)
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    Z.insert(0, "const", 1.0)
    return Z


class StuffPrior:
    """Ridge model predicting each component's logit rate from TrackMan."""

    def __init__(self, components=("k_rate", "bb_rate", "hr_fb", "gb_rate"),
                 ridge=6.0):
        self.components = list(components)
        self.ridge = ridge
        self.beta, self.p_lg, self.cols = {}, {}, None
        self.resid_sd = {}

    def fit(self, df, min_bf=15):
        Z = _design(df)
        self.cols = list(Z.columns)
        for c in self.components:
            n = df[DEN[c]].to_numpy(float)
            x = df[NUM[c]].to_numpy(float)
            p_lg, m, _ = moment_m(x, n)
            self.p_lg[c] = p_lg
            # rows with missing stuff features (e.g. no fastball ever tagged)
            # must be excluded from the fit, not just from prediction -- a
            # single NaN row in the ridge solve NaNs out beta for every
            # player, not just the one missing data.
            ok = (n >= min_bf) & Z.notna().all(axis=1).to_numpy()
            if ok.sum() < 15:
                self.beta[c] = np.zeros(Z.shape[1])
                self.resid_sd[c] = 0.0
                continue
            # lightly shrunk rate as the regression target, so tiny samples
            # do not dominate the fit through the logit
            p = (x[ok] + m * p_lg) / (n[ok] + m)
            y = logit(np.clip(p, 1e-4, 1 - 1e-4))
            A = Z.to_numpy()[ok]
            w = np.sqrt(n[ok])
            Aw, yw = A * w[:, None], y * w
            P = self.ridge * np.eye(A.shape[1])
            P[0, 0] = 0.0                      # never penalise the intercept
            b = np.linalg.solve(Aw.T @ Aw + P, Aw.T @ yw)
            self.beta[c] = b
            self.resid_sd[c] = float(np.std(y - A @ b))
        return self

    def predict(self, df):
        Z = _design(df).reindex(columns=self.cols, fill_value=0.0).to_numpy()
        return {c: expit(Z @ self.beta[c]) for c in self.components}


def shrink(df, prior_rates, m_by_comp, components):
    """Beta-binomial shrinkage toward the stuff-implied prior."""
    est, sd = {}, {}
    for c in components:
        n = df[DEN[c]].to_numpy(float)
        x = df[NUM[c]].to_numpy(float)
        m = m_by_comp[c]
        p0 = prior_rates[c]
        p = (x + m * p0) / (n + m)
        est[c] = p
        sd[c] = np.sqrt(np.maximum(p * (1 - p) / (n + m), 1e-12))
    return est, sd


def run_value(df, est, sd, env, n_draws=2000, seed=1, ip_next=None):
    """Runs above league average per 9, with a posterior distribution.

    Draws component vectors from their posteriors, pushes each through
    BaseRuns, and returns the resulting RA9 distribution. This is the same
    machinery as the MLB build, just with a much shorter innings base.
    """
    from mlbproj.core import baseruns_ra9

    rng = np.random.default_rng(seed)
    n_p = len(df)
    comps = ["k_rate", "bb_rate", "hbp_rate", "gb_rate", "ld_rate",
             "hr_fb", "babip"]
    draws = np.zeros((n_draws, n_p))
    for d in range(n_draws):
        rates = {}
        for c in comps:
            if c in est:
                p = np.clip(est[c] + rng.normal(0, sd[c]), 1e-4, 1 - 1e-4)
            else:
                p = np.full(n_p, getattr(env, c))
            rates[c] = p
        draws[d] = baseruns_ra9(rates, env, bf=np.full(n_p, 1000.0))

    ra9_mean = draws.mean(axis=0)
    ra9_sd = draws.std(axis=0)
    # nanmedian: players missing stuff features have NaN draws, and a single
    # NaN in a plain np.median poisons the league baseline (and therefore
    # every player's raa9), not just that player's own row.
    lg = float(np.nanmedian(ra9_mean))
    raa9 = lg - ra9_mean                      # positive = better than average
    raa9_sd = ra9_sd
    return pd.DataFrame(dict(
        player_id=df.player_id.to_numpy(),
        ra9=ra9_mean, ra9_sd=ra9_sd,
        raa9=raa9, raa9_sd=raa9_sd))


def ucb(mean, sd, kappa=0.8):
    """Optimism under uncertainty. kappa=0 ranks by posterior mean.

    kappa ~0.8 is a reasonable default for a decision where the downside is a
    single roster slot for a single summer. Raise it if you have slots to
    gamble with, lower it if the roster is tight and you need floor.
    """
    return np.asarray(mean, float) + kappa * np.asarray(sd, float)

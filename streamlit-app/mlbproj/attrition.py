"""
attrition.py -- who is still pitching, and how much.

Two separate models, because they answer different questions:

  1. Hazard:  P(appears in season t+1 | state at t)   -- logistic
  2. Volume:  log(IP) | appeared                       -- Gaussian regression

Keeping them separate matters because the IP distribution is bimodal (starters
vs relievers) and zero-inflated (injuries, demotions). A single model of "IP
including zeros" would smear a 0-IP season and a 90-IP season into a fictitious
45-IP expectation that describes nobody.

Velocity change is deliberately included: it is the single best early warning
of decline, and it moves a year before the rate stats do.
"""
import numpy as np
import pandas as pd


def irls_logistic(X, y, w=None, ridge=1e-4, iters=60, tol=1e-9):
    """Newton / iteratively reweighted least squares with a ridge penalty."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    beta = np.zeros(p)
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        s = np.maximum(mu * (1 - mu), 1e-8) * w
        z = eta + (y - mu) / np.maximum(mu * (1 - mu), 1e-8)
        XtS = X.T * s
        H = XtS @ X + ridge * np.eye(p)
        g = XtS @ z
        new = np.linalg.solve(H, g)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return beta


def _hazard_features(age, prev_ip, velo_delta, role_is_sp, prev_ip2=None):
    age = np.asarray(age, float)
    prev_ip = np.asarray(prev_ip, float)
    velo_delta = np.asarray(velo_delta, float)
    prev_ip2 = np.zeros_like(prev_ip) if prev_ip2 is None else np.asarray(prev_ip2, float)
    age, prev_ip, velo_delta, prev_ip2, role_is_sp = np.broadcast_arrays(
        age, prev_ip, velo_delta, prev_ip2, np.asarray(role_is_sp, float))
    return np.column_stack([
        np.ones_like(age),
        (age - 28.0),
        np.maximum(age - 30.0, 0.0) ** 2,
        np.log1p(prev_ip),
        np.log1p(prev_ip2),
        velo_delta,
        role_is_sp,
    ])


class AttritionModel:
    def __init__(self):
        self.hazard_beta = None
        self.ip_beta = {}
        self.ip_sigma = {}

    def build_panel(self, obs):
        """Turn a tidy season panel into (state at t -> outcome at t+1) rows."""
        d = obs.sort_values(["player_id", "season"]).copy()
        g = d.groupby("player_id")
        d["velo_prev"] = g["velo"].shift(1)
        d["velo_delta"] = (d["velo"] - d["velo_prev"]).fillna(0.0)
        d["ip_prev"] = g["ip"].shift(1).fillna(0.0)
        d["next_season"] = g["season"].shift(-1)
        d["next_ip"] = g["ip"].shift(-1)
        # appeared next year == there is a row for season+1
        d["appeared_next"] = ((d["next_season"] == d["season"] + 1)
                              & d["next_ip"].notna()).astype(int)
        d["is_sp"] = (d["role"] == "SP").astype(float)
        return d

    def fit(self, obs, last_season=None):
        d = self.build_panel(obs)
        # a player's final observed season is only informative if we know the
        # panel continues past it; otherwise it is right-censored
        if last_season is None:
            last_season = d.season.max()
        fit_d = d[d.season < last_season]

        X = _hazard_features(fit_d.age, fit_d.ip, fit_d.velo_delta,
                             fit_d.is_sp, fit_d.ip_prev)
        y = fit_d.appeared_next.to_numpy(float)
        self.hazard_beta = irls_logistic(X, y)

        for role in ("SP", "RP"):
            r = d[(d.role == role) & (d.ip > 0)]
            Xi = np.column_stack([
                np.ones(len(r)),
                (r.age.to_numpy(float) - 28.0),
                np.maximum(r.age.to_numpy(float) - 32.0, 0.0),
                np.log1p(r.ip_prev.to_numpy(float)),
            ])
            yi = np.log(r.ip.to_numpy(float))
            beta, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
            resid = yi - Xi @ beta
            self.ip_beta[role] = beta
            self.ip_sigma[role] = float(np.std(resid, ddof=Xi.shape[1]))
        return self

    def p_appear(self, age, prev_ip, velo_delta, is_sp, prev_ip2=0.0):
        X = _hazard_features(age, prev_ip, velo_delta, is_sp, prev_ip2)
        eta = np.clip(X @ self.hazard_beta, -30, 30)
        return 1.0 / (1.0 + np.exp(-eta))

    def draw_ip(self, rng, age, prev_ip, role, size=None):
        b = self.ip_beta[role]
        age = np.atleast_1d(np.asarray(age, float))
        prev_ip = np.atleast_1d(np.asarray(prev_ip, float))
        X = np.column_stack([
            np.ones_like(age), age - 28.0, np.maximum(age - 32.0, 0.0),
            np.log1p(prev_ip)])
        mu = X @ b
        n = len(mu) if size is None else size
        draw = np.exp(mu + rng.normal(0, self.ip_sigma[role], n))
        cap = 235.0 if role == "SP" else 105.0
        return np.clip(draw, 1.0, cap)

    def summary(self):
        names = ["intercept", "age-28", "max(age-30,0)^2", "log(1+IP_t)",
                 "log(1+IP_t-1)", "velo_delta", "is_SP"]
        return pd.DataFrame({"feature": names, "coef": self.hazard_beta})

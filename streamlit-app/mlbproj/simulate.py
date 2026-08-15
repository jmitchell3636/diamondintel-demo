"""
simulate.py -- the Monte Carlo forward engine.

Each path carries its own state (talent, velocity, workload, service time) and
evolves it season by season. Four separate uncertainty sources are drawn:

  1. talent posterior   -- we do not know how good he is        (reducible)
  2. talent random walk -- he genuinely changes year to year    (irreducible)
  3. aging uncertainty  -- the curve itself is an estimate      (reducible)
  4. binomial sampling  -- a season is a small sample           (irreducible)

Collapsing these into one "noise" term is the most common error in public
projection code: it makes uncertainty shrink with sample size when part of it
never should.
"""
import numpy as np
import pandas as pd
from scipy.special import logit, expit

from .core import LG, RATE_COMPONENTS, baseruns_ra9, war_from_ra9

DEFAULT_WALK = {c: 0.07 for c in RATE_COMPONENTS}


def estimate_talent_walk(obs, aging, min_denom=150):
    """Moment estimator for year-over-year drift in true talent.

    For consecutive-season pairs:
        Var(logit p_{t+1} - logit p_t | age-adjusted)
            = Var(walk) + Var(sampling_t) + Var(sampling_{t+1})
    and the sampling terms follow from the delta method,
    Var(logit p_hat) ~= 1 / (n * p * (1-p)).
    """
    num = dict(k_rate="k", bb_rate="bb", hbp_rate="hbp", gb_rate="gb",
               ld_rate="ld", hr_fb="hr", babip="hits")
    den = dict(k_rate="bf", bb_rate="bf", hbp_rate="bf", gb_rate="bip",
               ld_rate="bip", hr_fb="fb", babip="bip_nohr")
    d = obs.sort_values(["player_id", "season"])
    walk = {}
    for c in RATE_COMPONENTS:
        x, n = d[num[c]].to_numpy(float), d[den[c]].to_numpy(float)
        p = np.divide(x, n, out=np.full_like(x, np.nan), where=n > 0)
        t = pd.DataFrame({"pid": d.player_id.to_numpy(),
                          "season": d.season.to_numpy(),
                          "age": d.age.to_numpy(float), "p": p, "n": n})
        t["p2"] = t.groupby("pid")["p"].shift(-1)
        t["n2"] = t.groupby("pid")["n"].shift(-1)
        t["age2"] = t.groupby("pid")["age"].shift(-1)
        t["s2"] = t.groupby("pid")["season"].shift(-1)
        t = t[(t.s2 == t.season + 1) & (t.n >= min_denom) & (t.n2 >= min_denom)]
        t = t.dropna(subset=["p", "p2"])
        if len(t) < 40:
            walk[c] = DEFAULT_WALK[c]
            continue
        pc = np.clip(t.p.to_numpy(), 1e-3, 1 - 1e-3)
        pc2 = np.clip(t.p2.to_numpy(), 1e-3, 1 - 1e-3)
        aged = np.array([aging.delta(c, a1, a2)
                         for a1, a2 in zip(t.age, t.age2)]).ravel()
        diff = logit(pc2) - logit(pc) - aged
        v_samp = (1.0 / (t.n.to_numpy() * pc * (1 - pc))
                  + 1.0 / (t.n2.to_numpy() * pc2 * (1 - pc2)))
        v = np.var(diff) - np.mean(v_samp)
        walk[c] = float(np.sqrt(max(v, 1e-4)))
    return walk


class PitcherState:
    """Everything the engine needs to know about a pitcher at time zero."""

    def __init__(self, player_id, age, role, velo, ip_last, ip_prev=0.0,
                 service_time=0.0, history=None):
        self.player_id = player_id
        self.age = float(age)
        self.role = role
        self.velo = float(velo)
        self.ip_last = float(ip_last)
        self.ip_prev = float(ip_prev)
        self.service_time = float(service_time)
        self.history = history


class Simulator:
    def __init__(self, talent_model, aging_model, attrition_model,
                 env=LG, walk=None, aging_coef_sd=0.12, velo_decline=0.22,
                 velo_noise=0.35, leverage=1.0):
        self.tm = talent_model
        self.am = aging_model
        self.at = attrition_model
        self.env = env
        self.walk = walk or DEFAULT_WALK
        self.aging_coef_sd = aging_coef_sd
        self.velo_decline = velo_decline
        self.velo_noise = velo_noise
        self.leverage = leverage

    # ------------------------------------------------------------------
    def _posterior_draw(self, state, n_sims, rng):
        """Draw true-talent vectors from the empirical-Bayes posterior."""
        est, rel = self.tm.estimate(state.history, aging=self.am,
                                    target_age=state.age)
        draws = {}
        num = dict(k_rate="k", bb_rate="bb", hbp_rate="hbp", gb_rate="gb",
                   ld_rate="ld", hr_fb="hr", babip="hits")
        den = dict(k_rate="bf", bb_rate="bf", hbp_rate="bf", gb_rate="bip",
                   ld_rate="bip", hr_fb="fb", babip="bip_nohr")
        for c in RATE_COMPONENTS:
            p = float(np.clip(est[c], 1e-4, 1 - 1e-4))
            n_eff = float(np.nan_to_num(state.history[den[c]]).sum())
            # posterior variance of the rate, mapped to the logit scale
            var_p = p * (1 - p) / (n_eff + self.tm.m[c])
            sd_logit = np.sqrt(var_p / (p * (1 - p)) ** 2)
            draws[c] = logit(p) + rng.normal(0, sd_logit, n_sims)
        return draws, est, rel

    # ------------------------------------------------------------------
    def run(self, state, n_years=6, n_sims=5000, seed=0):
        rng = np.random.default_rng(seed)
        talent, est, rel = self._posterior_draw(state, n_sims, rng)

        # per-path multiplier on the aging curve: some arms age faster
        aging_mult = 1.0 + rng.normal(0, self.aging_coef_sd, n_sims)

        age = np.full(n_sims, state.age)
        velo = np.full(n_sims, state.velo)
        ip_last = np.full(n_sims, state.ip_last)
        ip_prev = np.full(n_sims, state.ip_prev)
        alive = np.ones(n_sims, bool)
        is_sp = 1.0 if state.role == "SP" else 0.0

        res = {k: np.zeros((n_sims, n_years)) for k in
               ("ip", "war", "ra9", "k", "bb", "hr", "so_rate", "appeared",
                "bf", "velo")}

        for t in range(n_years):
            age_next = age + 1.0

            # --- velocity path: decline plus noise, carried forward ---
            velo_new = (velo - self.velo_decline * np.maximum(age_next - 24.0, 0) / 3.0
                        + rng.normal(0, self.velo_noise, n_sims))
            velo_delta = velo_new - velo

            # --- survival ---
            p_app = self.at.p_appear(age, ip_last, velo_delta, is_sp, ip_prev)
            appear = (rng.random(n_sims) < p_app) & alive
            alive = appear

            # --- talent evolves: random walk then aging ---
            for c in RATE_COMPONENTS:
                talent[c] = talent[c] + rng.normal(0, self.walk[c], n_sims)
                talent[c] = talent[c] + aging_mult * self.am.delta(c, age, age_next)

            rates = {c: expit(talent[c]) for c in RATE_COMPONENTS}
            # FB is the residual; keep the mix coherent
            gl = rates["gb_rate"] + rates["ld_rate"]
            over = gl > 0.94
            if over.any():
                scale = np.where(over, 0.94 / np.maximum(gl, 1e-9), 1.0)
                rates["gb_rate"] = rates["gb_rate"] * scale
                rates["ld_rate"] = rates["ld_rate"] * scale

            # --- playing time ---
            ip = np.zeros(n_sims)
            if appear.any():
                idx = np.where(appear)[0]
                ip[idx] = self.at.draw_ip(rng, age_next[idx], ip_last[idx],
                                          state.role, size=len(idx))

            # --- season sampling noise: a season is a small sample ---
            bf = np.maximum(np.round(ip * 4.25), 0).astype(int)
            samp = {}
            with np.errstate(invalid="ignore"):
                for c in ("k_rate", "bb_rate", "hbp_rate"):
                    n_c = np.maximum(bf, 1)
                    samp[c] = rng.binomial(n_c, np.clip(rates[c], 1e-6, 1 - 1e-6)) / n_c
                bip = np.maximum(bf * (1 - samp["k_rate"] - samp["bb_rate"]
                                       - samp["hbp_rate"]), 1).astype(int)
                for c in ("gb_rate", "ld_rate"):
                    samp[c] = rng.binomial(bip, np.clip(rates[c], 1e-6, 1 - 1e-6)) / bip
                fbn = np.maximum(bip * (1 - samp["gb_rate"] - samp["ld_rate"]), 1).astype(int)
                samp["hr_fb"] = rng.binomial(fbn, np.clip(rates["hr_fb"], 1e-6, 1 - 1e-6)) / fbn
                samp["babip"] = rng.binomial(bip, np.clip(rates["babip"], 1e-6, 1 - 1e-6)) / bip

            ra9 = baseruns_ra9(samp, self.env, bf=np.maximum(bf, 1))
            war = war_from_ra9(ra9, ip, role=state.role,
                               leverage=self.leverage, env=self.env)
            war = np.where(appear, war, 0.0)

            res["ip"][:, t] = np.where(appear, ip, 0.0)
            res["war"][:, t] = war
            res["ra9"][:, t] = np.where(appear, ra9, np.nan)
            res["bf"][:, t] = bf
            res["k"][:, t] = np.where(appear, samp["k_rate"] * bf, 0.0)
            res["bb"][:, t] = np.where(appear, samp["bb_rate"] * bf, 0.0)
            res["so_rate"][:, t] = np.where(appear, samp["k_rate"], np.nan)
            res["appeared"][:, t] = appear.astype(float)
            res["velo"][:, t] = velo_new

            ip_prev, ip_last = ip_last, np.where(appear, ip, 0.0)
            age, velo = age_next, velo_new

        res["talent_estimate"] = est
        res["reliability"] = rel
        return res


def summarize(res, key="war", pcts=(10, 25, 50, 75, 90)):
    a = res[key]
    rows = []
    for t in range(a.shape[1]):
        col = a[:, t]
        row = {"year": t + 1, "mean": col.mean()}
        for p in pcts:
            row[f"p{p}"] = np.percentile(col, p)
        row["p_appear"] = res["appeared"][:, t].mean()
        rows.append(row)
    return pd.DataFrame(rows)

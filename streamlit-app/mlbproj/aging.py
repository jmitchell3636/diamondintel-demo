"""
aging.py -- per-component aging curves, fit with player fixed effects.

Why not the delta method: pitchers who decline lose their job, so the pairs of
consecutive seasons you observe are a biased sample -- the ones that would have
shown the steepest decline are missing. The resulting curve is too flat.

Player fixed effects fix the mechanical part of this. By demeaning within
player, each pitcher is compared only to himself, so a player observed at ages
29-31 who then washes out still contributes his 29->31 slope. What remains is
the harder selection problem (you only see pitchers good enough to appear at
all), which is handled downstream by the attrition model rather than here.

Each component is fit separately: K% and HR/FB do not age alike, and averaging
them into a single ERA-based curve throws away the distinction.
"""
import numpy as np
import pandas as pd
from scipy.special import logit, expit

from .core import RATE_COMPONENTS

AGE_KNOTS = (22.0, 25.0, 28.0, 31.0, 35.0)
REF_AGE = 27.0


def _basis(age, knots=AGE_KNOTS):
    """Natural-ish cubic spline basis: linear + quadratic + truncated cubes."""
    age = np.asarray(age, float)
    d = age - REF_AGE
    cols = [d, d ** 2]
    for k in knots[1:-1]:
        cols.append(np.maximum(age - k, 0.0) ** 3)
    return np.column_stack(cols)


class AgingModel:
    """Fits and stores logit-scale age effects for every component."""

    def __init__(self, components=RATE_COMPONENTS):
        self.components = list(components)
        self.coefs = {}
        self.grid_age = np.arange(19.0, 43.1, 0.5)
        self.curves = {}

    def fit(self, df, min_denom=40, min_seasons=2):
        num = dict(k_rate="k", bb_rate="bb", hbp_rate="hbp",
                   gb_rate="gb", ld_rate="ld", hr_fb="hr", babip="hits")
        den = dict(k_rate="bf", bb_rate="bf", hbp_rate="bf",
                   gb_rate="bip", ld_rate="bip", hr_fb="fb", babip="bip_nohr")

        for c in self.components:
            d = df[["player_id", "age"]].copy()
            d["x"] = df[num[c]].to_numpy(float)
            d["n"] = df[den[c]].to_numpy(float)
            d = d[(d.n >= min_denom) & np.isfinite(d.x)]
            # keep only players with multiple seasons -- singletons carry no
            # within-player information and just add noise to the demeaning
            counts = d.groupby("player_id").size()
            d = d[d.player_id.isin(counts[counts >= min_seasons].index)]
            if len(d) < 200:
                self.coefs[c] = np.zeros(_basis([27.0]).shape[1])
                continue

            # shrink lightly toward the league mean before the logit so that
            # small samples do not blow up at 0 or 1
            p_lg = d.x.sum() / d.n.sum()
            k_shrink = 30.0
            p = (d.x + k_shrink * p_lg) / (d.n + k_shrink)
            y = logit(np.clip(p.to_numpy(float), 1e-4, 1 - 1e-4))

            X = _basis(d.age.to_numpy(float))
            w = np.sqrt(d.n.to_numpy(float))

            # --- absorb player fixed effects by weighted within-demeaning ---
            grp = d.player_id.to_numpy()
            df_tmp = pd.DataFrame(X, columns=[f"b{i}" for i in range(X.shape[1])])
            df_tmp["_y"], df_tmp["_w"], df_tmp["_g"] = y, w, grp
            wsum = df_tmp.groupby("_g")["_w"].transform("sum")
            for col in list(df_tmp.columns[:-3]) + ["_y"]:
                wmean = (df_tmp[col] * df_tmp["_w"]).groupby(grp).transform("sum") / wsum
                df_tmp[col] = df_tmp[col] - wmean

            Xd = df_tmp[[f"b{i}" for i in range(X.shape[1])]].to_numpy()
            yd = df_tmp["_y"].to_numpy()
            Xw, yw = Xd * w[:, None], yd * w

            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            self.coefs[c] = beta

        for c in self.components:
            self.curves[c] = self._eval(c, self.grid_age)
        return self

    def _eval(self, c, age):
        return _basis(age) @ self.coefs[c]

    def delta(self, c, age_from, age_to):
        """Logit-scale change in component `c` from one age to another."""
        d = (self._eval(c, np.atleast_1d(age_to))
             - self._eval(c, np.atleast_1d(age_from)))
        return d[0] if d.shape == (1,) else d

    def adjust(self, c, p, age_from, age_to):
        """Translate an observed rate at age_from into its age_to equivalent."""
        p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
        z = logit(p) + self.delta(c, age_from, age_to)
        return expit(z)

    def curve_table(self):
        return pd.DataFrame({"age": self.grid_age,
                             **{c: self.curves[c] for c in self.components}})


def delta_method_curve(df, component="k_rate", min_denom=100):
    """Naive delta-method curve, kept only to demonstrate the bias."""
    num = dict(k_rate="k", bb_rate="bb", hr_fb="hr", babip="hits",
               gb_rate="gb", ld_rate="ld", hbp_rate="hbp")[component]
    den = dict(k_rate="bf", bb_rate="bf", hr_fb="fb", babip="bip_nohr",
               gb_rate="bip", ld_rate="bip", hbp_rate="bf")[component]
    d = df[["player_id", "season", "age"]].copy()
    d["p"] = df[num] / df[den]
    d["n"] = df[den]
    d = d[d.n >= min_denom]
    d = d.sort_values(["player_id", "season"])
    d["p_next"] = d.groupby("player_id")["p"].shift(-1)
    d["n_next"] = d.groupby("player_id")["n"].shift(-1)
    d["season_next"] = d.groupby("player_id")["season"].shift(-1)
    d = d[(d.season_next == d.season + 1) & d.p_next.notna()]
    d["age_bucket"] = d.age.round().astype(int)
    d["delta"] = logit(np.clip(d.p_next, 1e-4, 1 - 1e-4)) - \
        logit(np.clip(d.p, 1e-4, 1 - 1e-4))
    d["w"] = np.minimum(d.n, d.n_next)
    g = d.groupby("age_bucket").apply(
        lambda x: np.average(x["delta"], weights=x["w"]), include_groups=False)
    return g.sort_index().cumsum()

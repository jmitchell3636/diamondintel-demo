"""
reliability.py -- FCBL-specific stabilization constants from split-half data.

Do NOT borrow MLB regression constants. m = p(1-p) / Var(true), and the FCBL
contains D1 through JUCO arms in one league, so true-talent variance is far
wider than MLB's. Wider spread means LESS regression per batter faced. Using
MLB constants would badly over-shrink and flatten exactly the separation you
are trying to see.

Two estimators, reported side by side because they fail differently:
  * moment-based m from the full-season cross-section
  * split-half correlation, Spearman-Brown corrected to full season
Agreement between them is the sanity check.
"""
import numpy as np
import pandas as pd

NUM = dict(k_rate="k", bb_rate="bb", hbp_rate="hbp", gb_rate="gb",
           ld_rate="ld", hr_fb="hr", babip="hits")
DEN = dict(k_rate="bf", bb_rate="bf", hbp_rate="bf", gb_rate="bip",
           ld_rate="bip", hr_fb="fb", babip="bip_nohr")


# When there is not enough data to estimate a regression constant, fall back to
# heavy shrinkage rather than NaN. No information about talent spread means you
# should regress hard toward the league mean -- that is the correct answer, not
# an error. Thin denominators (HR/FB especially) hit this routinely on real data.
FALLBACK_M = 500.0


def moment_m(x, n, min_n=15, min_players=20):
    x, n = np.asarray(x, float), np.asarray(n, float)
    finite = np.isfinite(x) & np.isfinite(n) & (n > 0)
    if not finite.any():
        return np.nan, FALLBACK_M, 0.0
    p_lg_all = float(x[finite].sum() / n[finite].sum())

    ok = finite & (n >= min_n)
    x, n = x[ok], n[ok]
    if len(x) < min_players:
        # not enough qualified pitchers to decompose the variance
        return p_lg_all, FALLBACK_M, 0.0
    p_lg = x.sum() / n.sum()
    p = x / n
    w = n / n.sum()
    var_obs = np.sum(w * (p - p_lg) ** 2)
    var_samp = np.sum(w * p_lg * (1 - p_lg) / n)
    var_true = max(var_obs - var_samp, 1e-9)
    return float(p_lg), float(p_lg * (1 - p_lg) / var_true), float(np.sqrt(var_true))


def split_half(df, component, min_n=15):
    """Correlate H1 vs H2 rates, then Spearman-Brown correct to full season."""
    num, den = NUM[component], DEN[component]
    w = df.pivot_table(index="player_id", columns="half",
                       values=[num, den], aggfunc="sum")
    try:
        x1, x2 = w[(num, "H1")], w[(num, "H2")]
        n1, n2 = w[(den, "H1")], w[(den, "H2")]
    except KeyError:
        return np.nan, np.nan, 0
    ok = (n1 >= min_n) & (n2 >= min_n)
    if ok.sum() < 20:
        return np.nan, np.nan, int(ok.sum())
    p1, p2 = (x1[ok] / n1[ok]), (x2[ok] / n2[ok])
    r = float(np.corrcoef(p1, p2)[0, 1])
    r_full = 2 * r / (1 + r) if r > -1 else np.nan   # Spearman-Brown
    return r, r_full, int(ok.sum())


def reliability_table(halves_df, season_df, components=tuple(NUM)):
    rows = []
    for c in components:
        p_lg, m, sd_true = moment_m(season_df[NUM[c]], season_df[DEN[c]])
        r_half, r_full, n_pl = split_half(halves_df, c)
        med_den = float(season_df[DEN[c]].median())
        rows.append(dict(
            component=c, league_mean=p_lg, m=m, sd_true=sd_true,
            median_denominator=med_den,
            reliability_at_median=med_den / (med_den + m) if m == m else np.nan,
            split_half_r=r_half, spearman_brown_r=r_full, n_players=n_pl))
    return pd.DataFrame(rows)

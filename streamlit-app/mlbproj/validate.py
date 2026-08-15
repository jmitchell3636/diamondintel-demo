"""
validate.py -- scoring rules and a time-split backtest.

RMSE on the median is the wrong metric here: the whole output is a
distribution, and the valuation layer consumes the tails, not the centre.

  CRPS      -- proper scoring rule for a distributional forecast. Reduces to
               MAE when the forecast is a point, so it is directly comparable
               against a naive baseline. Lower is better.
  Coverage  -- does the stated 80% interval actually contain 80% of outcomes?
               A model can have excellent CRPS and useless intervals.
  PIT       -- probability integral transform; should be uniform. Its shape
               tells you HOW you are miscalibrated (U-shaped = overconfident).

Always split on time, never at random: random splits leak the future.
"""
import numpy as np
import pandas as pd

from .talent import TalentModel, marcel_baseline
from .aging import AgingModel
from .attrition import AttritionModel
from .simulate import Simulator, PitcherState, estimate_talent_walk
from .core import calibrate, LG


def crps_ensemble(samples, observed):
    """CRPS for an ensemble forecast (lower is better)."""
    x = np.sort(np.asarray(samples, float))
    y = float(observed)
    n = len(x)
    term1 = np.mean(np.abs(x - y))
    # E|X - X'| computed in O(n log n) from the sorted sample
    i = np.arange(1, n + 1)
    term2 = 2.0 * np.sum((2 * i - n - 1) * x) / (n * n)
    return term1 - 0.5 * term2


def coverage(samples_list, observed_list, level=0.80):
    lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    hits = []
    for s, y in zip(samples_list, observed_list):
        lo, hi = np.percentile(s, [lo_q, hi_q])
        hits.append(lo <= y <= hi)
    return float(np.mean(hits))


def pit_values(samples_list, observed_list):
    return np.array([np.mean(np.asarray(s) <= y)
                     for s, y in zip(samples_list, observed_list)])


def backtest(obs, truth, train_through=2019, horizons=(1, 2, 3, 4, 5),
             min_train_ip=60, n_sims=1500, seed=3, max_players=200):
    """Train on <= train_through, project forward, score against reality."""
    env = calibrate()
    train = obs[obs.season <= train_through]

    tm = TalentModel().fit(train)
    am = AgingModel().fit(train)
    at = AttritionModel().fit(train, last_season=train_through)
    walk = estimate_talent_walk(train, am)
    sim = Simulator(tm, am, at, env=env, walk=walk)

    subjects = (train[(train.season == train_through) & (train.ip >= min_train_ip)]
                .sort_values("ip", ascending=False))
    if len(subjects) > max_players:
        subjects = subjects.sample(max_players, random_state=seed)

    actual = obs[obs.season > train_through].set_index(["player_id", "season"])
    rows = []
    for _, s in subjects.iterrows():
        pid = int(s.player_id)
        h = train[train.player_id == pid].sort_values("season", ascending=False)
        st = PitcherState(pid, s.age, s.role, s.velo, s.ip,
                          float(h.iloc[1].ip) if len(h) > 1 else 0.0, history=h)
        res = sim.run(st, n_years=max(horizons), n_sims=n_sims, seed=pid)
        marc = marcel_baseline(h, tm.p_lg)

        for hz in horizons:
            season = train_through + hz
            key = (pid, season)
            actual_ip = float(actual.loc[key].ip) if key in actual.index else 0.0
            actual_k = (float(actual.loc[key].k / actual.loc[key].bf)
                        if key in actual.index else np.nan)

            samp_ip = res["ip"][:, hz - 1]
            rows.append(dict(player_id=pid, horizon=hz, metric="ip",
                             actual=actual_ip,
                             crps_model=crps_ensemble(samp_ip, actual_ip),
                             crps_naive=abs(float(s.ip) - actual_ip),
                             pred_median=np.median(samp_ip),
                             lo=np.percentile(samp_ip, 10),
                             hi=np.percentile(samp_ip, 90)))

            if not np.isnan(actual_k):
                samp_k = res["so_rate"][:, hz - 1]
                samp_k = samp_k[~np.isnan(samp_k)]
                if len(samp_k) > 50:
                    rows.append(dict(player_id=pid, horizon=hz, metric="k_rate",
                                     actual=actual_k,
                                     crps_model=crps_ensemble(samp_k, actual_k),
                                     crps_naive=abs(marc["k_rate"] - actual_k),
                                     pred_median=np.median(samp_k),
                                     lo=np.percentile(samp_k, 10),
                                     hi=np.percentile(samp_k, 90)))
    return pd.DataFrame(rows)


def score_backtest(bt):
    out = []
    for (metric, hz), g in bt.groupby(["metric", "horizon"]):
        cov = float(((g.actual >= g.lo) & (g.actual <= g.hi)).mean())
        out.append(dict(metric=metric, horizon=hz, n=len(g),
                        crps_model=g.crps_model.mean(),
                        crps_baseline=g.crps_naive.mean(),
                        skill_vs_baseline=1 - g.crps_model.mean() / g.crps_naive.mean(),
                        coverage_80=cov))
    return pd.DataFrame(out).sort_values(["metric", "horizon"])

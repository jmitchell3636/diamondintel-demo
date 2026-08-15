"""
valuation.py -- price a contract against a simulated WAR distribution.

Central principle:  E[surplus(WAR)]  !=  surplus(E[WAR]).

A guaranteed contract truncates the team's downside at zero -- you pay the same
whether he throws 200 innings at a 3.10 ERA or blows out his elbow in April. The
payoff is concave in performance, so by Jensen's inequality valuing the median
projection systematically OVERPRICES volatile pitchers. Options and opt-outs
push the same asymmetry the other way. The only correct method is to value each
simulated path and then average.

Decision timing: club options, player options and opt-outs are exercised using
ONLY information available at that moment. Using the path's realized future
would give the decision-maker perfect foresight and roughly double the modelled
cost of an opt-out. Here the decision uses a re-projection from performance
through the decision year.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .core import MarketCurve, npv

GUARANTEED, CLUB_OPTION, PLAYER_OPTION = "G", "CO", "PO"


@dataclass
class Contract:
    """A contract as a sequence of year-slots plus embedded options."""
    salaries: List[float]
    year_type: Optional[List[str]] = None       # G / CO / PO per year
    buyouts: Dict[int, float] = field(default_factory=dict)
    opt_out_after: List[int] = field(default_factory=list)
    signing_bonus: float = 0.0
    deferrals: Dict[int, tuple] = field(default_factory=dict)  # yr -> (amt, pay_yr)
    name: str = "contract"

    def __post_init__(self):
        if self.year_type is None:
            self.year_type = [GUARANTEED] * len(self.salaries)
        assert len(self.year_type) == len(self.salaries)

    @property
    def n_years(self):
        return len(self.salaries)

    @property
    def guaranteed_total(self):
        return self.signing_bonus + sum(
            s for s, t in zip(self.salaries, self.year_type) if t == GUARANTEED)

    @property
    def cbt_aav(self):
        """Luxury-tax AAV: guaranteed money over guaranteed years."""
        g = [s for s, t in zip(self.salaries, self.year_type) if t == GUARANTEED]
        return (self.signing_bonus + sum(g)) / max(len(g), 1)


# ---------------------------------------------------------------------------
def perceived_war(war_paths, t, prior_mean, shrink_years=1.6, decay=0.92):
    """Market's read on a pitcher's *forward* annual WAR at end of year t.

    Uses only years 0..t of each path. Recent seasons weighted more, shrunk
    toward the pre-contract projection, then decayed for aging. This is a
    deliberately simple stand-in for re-running the full projection at every
    decision node -- cheap, and it removes the lookahead bias that matters.
    """
    obs = war_paths[:, :t + 1]
    n = obs.shape[1]
    w = np.power(1.35, np.arange(n))          # recency weighting
    w = w / w.sum()
    recent = obs @ w
    k = shrink_years / (shrink_years + n)
    est = k * prior_mean + (1 - k) * recent
    return np.maximum(est, 0.0) * decay


def survival_curve(res):
    """Conditional year-over-year survival implied by the simulation itself.

    Needed because a market participant valuing a pitcher's remaining years
    discounts for the chance he is hurt or ineffective. Omitting this makes
    perceived forward value optimistic, which in turn makes opt-outs and club
    options fire in states where they should not.
    """
    app = res["appeared"].mean(axis=0)
    surv = np.ones_like(app)
    surv[1:] = app[1:] / np.maximum(app[:-1], 1e-9)
    return np.clip(surv, 0.0, 1.0)


def _remaining_guarantee(contract, t):
    """Guaranteed money still owed after completing year t."""
    tot = 0.0
    for j in range(t + 1, contract.n_years):
        if contract.year_type[j] == GUARANTEED:
            tot += contract.salaries[j]
        else:
            tot += contract.buyouts.get(j, 0.0)
    return tot


def value_contract(res, contract, market: MarketCurve = MarketCurve(),
                   discount_rate=0.06, n_inflation_draws=True, seed=11,
                   prior_mean=None):
    """Per-path value, cost and surplus for a contract. All figures in $."""
    war = res["war"]
    n_sims, n_years = war.shape
    T = min(contract.n_years, n_years)
    rng = np.random.default_rng(seed)

    if prior_mean is None:
        prior_mean = float(war[:, 0].mean())

    # Inflation is drawn ONCE per path and applied to all years: it is a
    # market-wide factor, perfectly correlated across a portfolio of players.
    infl = (rng.normal(market.inflation_mu, market.inflation_sd, n_sims)
            if n_inflation_draws else
            np.full(n_sims, market.inflation_mu))

    surv = survival_curve(res)
    value = np.zeros((n_sims, T))
    cost = np.zeros((n_sims, T))
    active = np.ones(n_sims, bool)
    ended_year = np.full(n_sims, T)
    opted_out = np.zeros(n_sims, bool)

    for t in range(T):
        yt = contract.year_type[t]
        sal = contract.salaries[t]
        buyout = contract.buyouts.get(t, 0.0)

        if yt in (CLUB_OPTION, PLAYER_OPTION):
            pv = perceived_war(war, t - 1, prior_mean) if t > 0 else \
                np.full(n_sims, prior_mean)
            mkt_val = market.dollars(pv, years_out=t, inflation=infl) * surv[t]
            if yt == CLUB_OPTION:
                # team picks it up only when the player is worth more than the price
                exercise = mkt_val >= sal
            else:
                # player stays only when the option pays MORE than his market
                exercise = sal >= mkt_val
            drop = active & ~exercise
            cost[drop, t] += buyout
            ended_year[drop] = t
            active = active & exercise

        if not active.any():
            break

        cost[active, t] += sal
        value[active, t] = market.dollars(war[active, t], years_out=t,
                                          inflation=infl[active])

        if t in contract.opt_out_after and t < T - 1:
            pv = perceived_war(war, t, prior_mean)
            rem_years = T - t - 1
            cum_surv = np.cumprod(surv[t + 1:t + 1 + rem_years])
            fwd = np.array([market.dollars(pv * (0.94 ** j), years_out=t + 1 + j,
                                           inflation=infl) * cum_surv[j]
                            / (1 + discount_rate) ** (t + 1 + j)
                            for j in range(rem_years)]).sum(axis=0)
            rem_guar = _remaining_guarantee(contract, t) / (1 + discount_rate) ** (t + 1)
            leaves = active & (fwd > rem_guar)
            opted_out |= leaves
            ended_year[leaves] = t + 1
            active = active & ~leaves

    cost[:, 0] += contract.signing_bonus

    # deferrals: move cash to the year it is actually paid, then discount
    for yr, (amt, pay_yr) in contract.deferrals.items():
        if yr < T:
            cost[:, yr] -= amt
            if pay_yr < T:
                cost[:, pay_yr] += amt
            else:
                cost[:, T - 1] += amt / (1 + discount_rate) ** (pay_yr - (T - 1))

    pv_value = npv(value, discount_rate)
    pv_cost = npv(cost, discount_rate)
    surplus = pv_value - pv_cost

    return dict(value=value, cost=cost, pv_value=pv_value, pv_cost=pv_cost,
                surplus=surplus, ended_year=ended_year, opted_out=opted_out,
                war=war[:, :T], inflation=infl)


# ---------------------------------------------------------------------------
def risk_metrics(surplus, risk_aversion=0.012, scale=1e6):
    """Summarise a surplus distribution the way a front office should read it.

    risk_aversion is on the $M scale under exponential utility. 0.012 implies a
    club is indifferent between a certain $10M and a coin flip on $0/$24M --
    modestly risk-averse, roughly small-market behaviour. Set to 0 for a
    risk-neutral (large-market) club.
    """
    s = np.asarray(surplus, float) / scale
    q = np.percentile(s, [5, 10, 25, 50, 75, 90, 95])
    cvar10 = s[s <= np.percentile(s, 10)].mean()
    if risk_aversion > 0:
        # numerically stable certainty equivalent
        a = risk_aversion
        m = -a * s
        ce = -(1.0 / a) * (np.max(m) + np.log(np.mean(np.exp(m - np.max(m)))))
    else:
        ce = s.mean()
    return {
        "mean_surplus_$M": s.mean(),
        "median_surplus_$M": np.median(s),
        "p5_$M": q[0], "p25_$M": q[2], "p75_$M": q[4], "p95_$M": q[6],
        "P(surplus<0)": float((s < 0).mean()),
        "CVaR10_$M": float(cvar10),
        "certainty_equivalent_$M": float(ce),
    }


def bid_ceiling(res, years, market=MarketCurve(), discount_rate=0.06,
                risk_aversion=0.012, criterion="certainty_equivalent_$M",
                lo=1e6, hi=6e7, tol=2.5e5, opt_out_after=None):
    """Highest flat AAV at which this contract still clears the club's bar."""
    def obj(aav):
        c = Contract([aav] * years, opt_out_after=opt_out_after or [])
        v = value_contract(res, c, market, discount_rate)
        return risk_metrics(v["surplus"], risk_aversion)[criterion]

    if obj(lo) < 0:
        return 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if obj(mid) >= 0:
            lo = mid
        else:
            hi = mid
    return lo


def jensen_gap(res, contract, market=MarketCurve(), discount_rate=0.06):
    """How wrong you would be valuing the mean projection instead of the paths.

    Returns (per-path surplus, naive surplus computed from mean WAR).
    """
    v = value_contract(res, contract, market, discount_rate)
    mean_war = res["war"].mean(axis=0)[:contract.n_years]
    naive_value = np.array([market.dollars(w, years_out=t)[0]
                            for t, w in enumerate(mean_war)])
    naive_cost = np.array(contract.salaries[:len(mean_war)], float)
    naive = npv(naive_value, discount_rate) - npv(naive_cost, discount_rate)
    return float(v["surplus"].mean()), float(naive)


# ---------------------------------------------------------------------------
def value_team_control(res, arb_model, service_time=0.0, role="SP",
                       super_two=False, market: MarketCurve = MarketCurve(),
                       discount_rate=0.06, career_ip=0.0, career_k=0.0,
                       min_salary=760_000.0):
    """Cost and value of the remaining pre-arb + arbitration years, per path.

    Salary is driven by each path's own simulated counting stats, because that
    is what an arbitration panel actually rules on. A path where the pitcher
    stays healthy and piles up innings gets expensive even if his rate stats
    were mediocre -- which is precisely the risk a club takes when it decides
    not to extend.
    """
    war, ip, k = res["war"], res["ip"], res["k"]
    n_sims, n_years = war.shape

    cost = np.zeros((n_sims, n_years))
    value = np.zeros((n_sims, n_years))
    st = np.full(n_sims, float(service_time))
    prior_sal = np.full(n_sims, min_salary)
    cum_ip = np.full(n_sims, float(career_ip))
    cum_k = np.full(n_sims, float(career_k))
    in_arb = np.zeros(n_sims, bool)
    arb_start = 2.128 if super_two else 3.0

    for t in range(n_years):
        cum_ip += ip[:, t]
        cum_k += k[:, t]

        free_agent = st >= 6.0
        arb_elig = (st >= arb_start) & ~free_agent

        sal = np.full(n_sims, min_salary)
        first = arb_elig & ~in_arb
        if first.any():
            sal[first] = arb_model.first_year(cum_ip[first], cum_k[first],
                                              role=role, super_two=super_two)
        repeat = arb_elig & in_arb
        if repeat.any():
            sal[repeat] = arb_model.repeat_year(prior_sal[repeat], ip[repeat, t],
                                                k[repeat, t], role=role)
        in_arb = in_arb | arb_elig

        # once he reaches free agency, team control is over
        sal = np.where(free_agent, np.nan, sal)
        cost[:, t] = np.nan_to_num(sal)
        value[:, t] = np.where(free_agent, 0.0,
                               market.dollars(war[:, t], years_out=t))
        prior_sal = np.where(arb_elig, sal, prior_sal)
        # service time accrues only when he is actually on the roster
        st = st + np.where(res["appeared"][:, t] > 0, 1.0, 0.35)

    pv_v, pv_c = npv(value, discount_rate), npv(cost, discount_rate)
    return dict(value=value, cost=cost, pv_value=pv_v, pv_cost=pv_c,
                surplus=pv_v - pv_c)

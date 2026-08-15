"""
arb.py -- arbitration salary projection.

Arbitration is NOT a WAR market. Panels weigh a narrow record -- innings, wins,
strikeouts, saves, and comparable players' salaries -- and anchor hard on the
prior year's salary. A 4-WAR pitcher with 8 wins gets paid like a 2-WAR pitcher.
Modelling arb salary as a function of value is the single biggest source of
error in public surplus calculators.

Structure:
    Arb1 salary  = f(career counting stats, role)
    Arb_{k+1}    = prior salary + raise(platform-year counting stats)
    with a floor, because salaries are rarely cut in arbitration.

COEFFICIENTS BELOW ARE STRUCTURAL PLACEHOLDERS. Call .fit() with real
arbitration outcomes before trusting any dollar figure.
"""
from dataclasses import dataclass, field
import numpy as np

MIN_SALARY = 760_000.0


@dataclass
class ArbitrationModel:
    min_salary: float = MIN_SALARY
    # --- first-time eligible: level set by career counting stats ---
    a0_sp: float = 1.05e6
    a_career_ip: float = 6_500.0
    a_career_k: float = 5_200.0
    a0_rp: float = 0.95e6
    a_career_sv: float = 62_000.0
    a_career_ip_rp: float = 4_000.0
    # --- repeat years: anchored raise ---
    r0: float = 0.35e6
    r_platform_ip: float = 9_500.0
    r_platform_k: float = 7_800.0
    r_platform_sv: float = 78_000.0
    anchor: float = 1.00          # weight on prior salary
    cut_floor: float = 0.80       # salaries rarely fall below 80% of prior
    super_two_discount: float = 0.72

    def first_year(self, career_ip, career_k, career_sv=0.0, role="SP",
                   super_two=False):
        if role == "SP":
            s = self.a0_sp + self.a_career_ip * career_ip + self.a_career_k * career_k
        else:
            s = (self.a0_rp + self.a_career_ip_rp * career_ip
                 + self.a_career_sv * career_sv + self.a_career_k * career_k * 0.4)
        s = np.maximum(s, self.min_salary * 1.02)
        if super_two:
            s = s * self.super_two_discount
        return s

    def repeat_year(self, prior_salary, platform_ip, platform_k,
                    platform_sv=0.0, role="SP"):
        raise_ = (self.r0 + self.r_platform_ip * platform_ip
                  + self.r_platform_k * platform_k
                  + (self.r_platform_sv * platform_sv if role == "RP" else 0.0))
        s = self.anchor * prior_salary + raise_
        return np.maximum(np.maximum(s, prior_salary * self.cut_floor),
                          self.min_salary)

    # ------------------------------------------------------------------
    def fit(self, df, role="SP"):
        """Fit on real arb outcomes.

        Expected columns: salary, prior_salary, arb_year (1..4), career_ip,
        career_k, career_sv, platform_ip, platform_k, platform_sv.
        """
        d1 = df[df.arb_year == 1]
        if len(d1) > 20:
            X = np.column_stack([np.ones(len(d1)), d1.career_ip, d1.career_k])
            b, *_ = np.linalg.lstsq(X, d1.salary.to_numpy(float), rcond=None)
            self.a0_sp, self.a_career_ip, self.a_career_k = map(float, b)
        dr = df[df.arb_year > 1]
        if len(dr) > 20:
            y = dr.salary.to_numpy(float) - dr.prior_salary.to_numpy(float)
            X = np.column_stack([np.ones(len(dr)), dr.platform_ip, dr.platform_k])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.r0, self.r_platform_ip, self.r_platform_k = map(float, b)
        return self


def service_class(service_time, super_two=False):
    """Map service time (years.days as a decimal) to a pay regime."""
    if service_time < (2.128 / 3 if super_two else 3.0) and not (
            super_two and service_time >= 2.128 / 3):
        pass
    if super_two and service_time >= 2.0:
        base = "ARB"
    if service_time >= 6.0:
        return "FA"
    if service_time >= 3.0 or (super_two and service_time >= 2.128):
        return "ARB"
    return "PRE_ARB"


def arb_year_number(service_time, super_two=False):
    """Which time through arbitration this is (1-indexed)."""
    start = 2.128 if super_two else 3.0
    n = int(np.floor(service_time - start)) + 1
    return int(np.clip(n, 1, 4))

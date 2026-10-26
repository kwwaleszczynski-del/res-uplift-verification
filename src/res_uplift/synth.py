"""Synthetic control with conformal inference (Chernozhukov, Wuethrich & Zhu 2021).

Counterfactual: the test unit's daily mean power as a non-negative combination of the controls' daily mean
power, ``y0_d = sum_j w_j x_jd`` with ``w_j >= 0`` (the synthetic control of Abadie, Diamond & Hainmueller 2010,
in the non-negative least squares form without the adding-up constraint, so that units of different size
can be combined). Days are weighted by their number of valid records.

* Point estimate: weights fitted on the pre-period only, ``effect = sum n_d y_d / sum n_d y0_d - 1`` over the
  post-window days.
* Interval: test inversion. Under the null of a constant relative effect ``delta0`` the post-period outcomes
  are deflated, ``y_d / (1 + delta0)``, the weights are re-fitted on *all* days, and the statistic

      S = |sum_{post} n_d u_d| / sum_{post} n_d yhat_d       (u = residual, yhat = fitted value)

  - the relative energy error over the post block - is compared with the same statistic of every cyclic shift
  (moving-block permutation) of the residual series. The p-value is the share of shifts with a statistic at
  least as large; the 90 % / 95 % confidence set is ``{delta0 : p > alpha}``. Exact when the residuals are
  exchangeable, approximately valid under stationarity and weak dependence (CWZ 2021, theorems 1 and 2).
  CWZ propose the q-norm of the residuals as default statistic and allow any function of them; the relative
  energy error is used here because wind and PV residuals grow with the power level (windy weeks would dominate
  ``sum |u_d|`` and leave the test without power) and because it matches the estimand.

Usage:
    sc = SyntheticControl()
    est = sc.estimate(sc.fit(panel, panel.y), panel, panel.y, weeks=13)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from res_uplift.core import Estimate, Panel

ALPHAS = (0.10, 0.05)


def _circular_sums(v: np.ndarray, m: int) -> np.ndarray:
    """Sums of v over the len(v) circular windows of length m (window k covers k .. k + m - 1)."""
    cs = np.concatenate(([0.0], np.cumsum(np.concatenate((v, v)))))
    return cs[m:len(v) + m] - cs[:len(v)]


def cwz_pvalue(y: np.ndarray, X: np.ndarray, n: np.ndarray, post: np.ndarray, delta0: float) -> float:
    """Moving-block permutation p-value of H0: effect = delta0 (post days must form the last block).

    ``n`` are the records per day (fit weights and energy weights of the statistic).
    """
    y0 = np.where(post, y / (1.0 + delta0), y)
    s = np.sqrt(n / n.mean())
    w, _ = nnls(X * s[:, None], y0 * s)
    fitted = X @ w
    T, m = len(y), int(post.sum())
    num = np.abs(_circular_sums(n * (y0 - fitted), m))
    den = _circular_sums(n * fitted, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        stats = np.where(den > 0, num / den, np.inf)
    return float(np.mean(stats >= stats[T - m] - 1e-12))


def confidence_sets(pfun, center: float, alphas=ALPHAS, half_width: float = 0.25, step: float = 0.005,
                    refine: int = 8) -> dict[float, tuple[float, float]]:
    """Invert a p-value function into confidence intervals (hull of the accepted grid, edges refined)."""
    cache: dict[float, float] = {}

    def p(d):
        if d not in cache:
            cache[d] = pfun(d)
        return cache[d]

    for _ in range(3):
        grid = center + np.arange(-half_width, half_width + step / 2, step)
        grid = grid[grid > -0.95]
        pv = np.array([p(float(d)) for d in grid])
        if pv[0] <= min(alphas) and pv[-1] <= min(alphas):
            break
        half_width *= 2.5
    out = {}
    for a in alphas:
        acc = np.flatnonzero(pv > a)
        if len(acc) == 0:
            best = float(grid[np.argmax(pv)])
            out[a] = (best, best)
            continue
        lo_out, lo_in = (grid[acc[0] - 1] if acc[0] > 0 else -np.inf), grid[acc[0]]
        hi_in, hi_out = grid[acc[-1]], (grid[acc[-1] + 1] if acc[-1] < len(grid) - 1 else np.inf)
        for _ in range(refine if np.isfinite(lo_out) else 0):
            mid = 0.5 * (lo_out + lo_in)
            lo_in, lo_out = (mid, lo_out) if p(float(mid)) > a else (lo_in, mid)
        for _ in range(refine if np.isfinite(hi_out) else 0):
            mid = 0.5 * (hi_in + hi_out)
            hi_in, hi_out = (mid, hi_out) if p(float(mid)) > a else (hi_in, mid)
        out[a] = (float(lo_in) if np.isfinite(lo_out) else -np.inf, float(hi_in) if np.isfinite(hi_out) else np.inf)
    return out


class SyntheticControl:
    name = "synth"
    uses_controls = True

    def fit(self, panel: Panel, y: np.ndarray) -> dict:
        return {}

    def refresh(self, state: dict, panel: Panel) -> dict:
        return state

    def weights(self, panel: Panel, y: np.ndarray) -> np.ndarray:
        pre = ~panel.post_day
        s = np.sqrt(panel.n_day[pre] / panel.n_day[pre].mean())
        w, _ = nnls(panel.daily_mean(panel.X)[pre] * s[:, None], panel.daily_mean(y)[pre] * s)
        return w

    def estimate(self, state: dict, panel: Panel, y: np.ndarray, weeks: float) -> Estimate:
        days = panel.window_days(weeks)
        yd, Xd = panel.daily_mean(y)[days], panel.daily_mean(panel.X)[days]
        n, post = panel.n_day[days], panel.post_day[days]
        if post.sum() == 0:
            return Estimate(np.nan, -np.inf, np.inf, -np.inf, np.inf)
        s = np.sqrt(n[~post] / n[~post].mean())
        w, _ = nnls(Xd[~post] * s[:, None], yd[~post] * s)
        point = float((n[post] * yd[post]).sum() / (n[post] * (Xd[post] @ w)).sum() - 1.0)
        ci = confidence_sets(lambda d: cwz_pvalue(yd, Xd, n, post, d), point)
        return Estimate(point, *ci[0.10], *ci[0.05])

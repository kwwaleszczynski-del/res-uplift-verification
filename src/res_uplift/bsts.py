"""Bayesian structural time series counterfactual (the CausalImpact approach of Brodersen et al. 2015).

Model on daily mean power, on a shifted log scale ``log(P + c)`` (``c`` = 5 % of the unit's high power level;
energy relations between units are multiplicative, and the shift keeps calm days finite), standardised with the
pre-period mean and standard deviation:

    y_d  = mu_d + x_d' beta + eps_d,     eps_d ~ N(0, s_eps^2)
    mu_d = mu_{d-1} + eta_d,             eta_d ~ N(0, g_d s_eta^2)      (local level)

with ``g_d`` the days since the last valid day.

``x_d`` holds the controls' daily mean power (same transformation) and weather covariates (wind: the controls'
mean wind speed and temperature; PV: irradiance and ambient temperature). The regression has a spike-and-slab
prior (George & McCulloch 1997) with a Zellner-type slab, as in Scott & Varian (2014). Priors follow the
CausalImpact defaults (level sd guess 0.01 sd(y) with a prior weight of 32 observations and sd(y) as upper
limit, expected model size 3, slab information weight 0.01) with one exception: CausalImpact's observation-noise prior (expected R^2 0.8,
worth 50 observations) is tuned for noisy business series. Controls of the same site explain more than 99 % of
the daily variance of SCADA energy, and that prior inflated the noise about five-fold (intervals five times
wider than the data support), so a weak prior is used (expected R^2 0.9, 1 degree of freedom).

The noise variance of a day is ``s_eps^2 / w_d`` with ``w_d`` proportional to its number of valid records times
the controls' energy level (the log of a small daily energy is noisy; this keeps calm days from driving the
level). Posterior by Gibbs sampling: (1) the whole level path from its Gaussian full conditional, whose precision
matrix is tridiagonal, so a banded Cholesky factorisation draws it in O(T) (the same draw as the simulation smoother of
Durbin & Koopman 2002), (2) inclusion indicators one at a time with beta integrated out, (3) beta, (4) both
variances from inverse-gamma full conditionals. Post-period outcomes are not used: the posterior predictive of
``y_d`` on post days (level extrapolated as a random walk, regression, observation noise) is the counterfactual,
so one fit serves every effect size and window length.

Usage:
    b = Bsts(n_iter=1000)
    state = b.fit(panel, panel.y)
    est = b.estimate(state, panel, y_with_effect, weeks=13)
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky_banded, cho_solve_banded, solve_banded

from res_uplift.core import Estimate, Panel, from_draws

WEATHER = {"wind": ["ws_ref", "temp"], "pv": ["poa", "t_amb"]}


def _inv_gamma(rng, shape: float, scale: float) -> float:
    return scale / rng.gamma(shape)


def gibbs(y: np.ndarray, X: np.ndarray, observed: np.ndarray, gaps: np.ndarray, weights: np.ndarray | None = None,
          n_iter: int = 1000,
          burn: int = 250, seed: int = 0, level_sd: float = 0.01, level_weight: float = 32.0,
          expected_r2: float = 0.9, prior_df: float = 1.0, model_size: float = 3.0,
          slab_weight: float = 0.01, diagonal_shrinkage: float = 0.5) -> dict:
    """Gibbs sampler for local level + spike-and-slab regression on standardised data.

    ``y`` has length T (values where ``observed`` is False are ignored), ``X`` is (T, p), ``gaps`` (T - 1) are the
    day gaps between consecutive rows, ``weights`` (T) scale the observation precision (noise variance
    s_eps^2 / weight). Returns the draws of the level path, beta, inclusion and both sds.
    """
    rng = np.random.default_rng(seed)
    T, p = X.shape
    obs = observed.astype(bool)
    w_all = np.ones(T) if weights is None else np.asarray(weights, dtype=float)
    n = int(obs.sum())
    Xo, yo, wo = X[obs], y[obs], w_all[obs]
    xtx = Xo.T @ (wo[:, None] * Xo)
    omega = slab_weight * ((1 - diagonal_shrinkage) * xtx + diagonal_shrinkage * np.diag(np.diag(xtx))) / n
    omega += 1e-9 * np.eye(p)
    pi = min(1.0, model_size / max(p, 1))
    log_prior = (np.log(pi) if pi > 0 else -np.inf, np.log1p(-pi) if pi < 1 else -np.inf)
    a_e, b_e = prior_df / 2, prior_df * (1 - expected_r2) / 2
    a_l, b_l = level_weight / 2, level_weight * level_sd**2 / 2

    # start: least squares with all regressors, flat level
    sw = np.sqrt(wo)
    beta = np.linalg.lstsq(np.c_[Xo, np.ones(n)] * sw[:, None], yo * sw, rcond=None)[0][:p]
    gamma = np.ones(p, dtype=bool) if pi >= 1 else np.abs(beta) > 0.05
    mu = np.full(T, float(np.average(yo - Xo @ beta, weights=wo)))
    s2_e, s2_l = max(float(np.average((yo - Xo @ beta - mu[obs]) ** 2, weights=wo)), 1e-4), level_sd**2

    keep = n_iter - burn
    out = {"mu": np.empty((keep, T)), "beta": np.empty((keep, p)), "gamma": np.empty((keep, p), dtype=bool),
           "s_e": np.empty(keep), "s_l": np.empty(keep)}
    inv_g = 1.0 / gaps
    for it in range(n_iter):
        # (1) level path | rest: precision Q = diag(obs / s2_e) + D' G^-1 D / s2_l (+ diffuse start)
        diag = obs * w_all / s2_e
        diag[:-1] += inv_g / s2_l
        diag[1:] += inv_g / s2_l
        diag[0] += 1e-4
        ab = np.vstack([np.r_[0.0, -inv_g / s2_l], diag])
        U = cholesky_banded(ab, lower=False)
        rhs = np.where(obs, w_all * (y - X @ beta), 0.0) / s2_e
        mu = cho_solve_banded((U, False), rhs) + solve_banded((0, 1), U, rng.standard_normal(T))

        # (2) inclusion indicators | level, s2_e (beta integrated out)
        z = yo - mu[obs]
        xtz = Xo.T @ (wo * z)
        cache: dict[tuple, tuple] = {}

        def marginal(g):
            key = tuple(g)
            if key not in cache:
                idx = np.flatnonzero(g)
                if len(idx) == 0:
                    cache[key] = (0.0, None, None, idx)
                else:
                    P = xtx[np.ix_(idx, idx)] + omega[np.ix_(idx, idx)]
                    cf = cho_factor(P)
                    b = cho_solve(cf, xtz[idx])
                    logdet_p = 2 * np.sum(np.log(np.diag(cf[0])))
                    logdet_o = np.linalg.slogdet(omega[np.ix_(idx, idx)])[1]
                    ll = 0.5 * (logdet_o - logdet_p) + 0.5 * float(xtz[idx] @ b) / s2_e
                    cache[key] = (ll, cf, b, idx)
            return cache[key]

        if pi < 1:
            for j in rng.permutation(p):
                g1, g0 = gamma.copy(), gamma.copy()
                g1[j], g0[j] = True, False
                l1 = marginal(g1)[0] + log_prior[0]
                l0 = marginal(g0)[0] + log_prior[1]
                gamma[j] = rng.random() < 1.0 / (1.0 + np.exp(np.clip(l0 - l1, -700, 700)))

        # (3) beta | inclusion, level, s2_e
        _, cf, b, idx = marginal(gamma)
        beta = np.zeros(p)
        if len(idx):
            L = np.linalg.cholesky(cho_solve(cf, np.eye(len(idx))))
            beta[idx] = b + np.sqrt(s2_e) * L @ rng.standard_normal(len(idx))

        # (4) variances
        e = z - Xo @ beta
        quad = float(beta[idx] @ omega[np.ix_(idx, idx)] @ beta[idx]) if len(idx) else 0.0
        s2_e = _inv_gamma(rng, a_e + (n + len(idx)) / 2, b_e + (float(wo @ (e * e)) + quad) / 2)
        d = np.diff(mu)
        for _ in range(20):  # truncated at sd(y) = 1, as in CausalImpact
            s2_l = _inv_gamma(rng, a_l + (T - 1) / 2, b_l + float(np.sum(d * d * inv_g)) / 2)
            if s2_l <= 1.0:
                break
        s2_l = min(s2_l, 1.0)

        if it >= burn:
            k = it - burn
            out["mu"][k], out["beta"][k], out["gamma"][k] = mu, beta, gamma
            out["s_e"][k], out["s_l"][k] = np.sqrt(s2_e), np.sqrt(s2_l)
    return out


class Bsts:
    name = "bsts"
    uses_controls = True

    def __init__(self, n_iter: int = 1000, burn: int = 250, seed: int = 0):
        self.n_iter, self.burn, self.seed = n_iter, burn, seed

    @staticmethod
    def shift(values: np.ndarray, pre: np.ndarray) -> np.ndarray:
        """Shift ``c`` of the log transform: 5 % of the 99th percentile of the pre-period values."""
        return 0.05 * np.maximum(np.nanpercentile(values[pre], 99, axis=0), 1e-6)

    def design(self, panel: Panel) -> np.ndarray:
        pre = ~panel.post_day
        Xd = panel.daily_mean(panel.X)
        cols = [panel.z_names.index(c) for c in WEATHER[panel.kind] if c in panel.z_names]
        logx = np.log(np.maximum(Xd, 0.0) + self.shift(Xd, pre))
        return np.column_stack([logx, panel.daily_mean(panel.Z[:, cols])])

    def fit(self, panel: Panel, y: np.ndarray) -> dict:
        X = self.design(panel)
        pre = ~panel.post_day
        yd_raw = panel.daily_mean(y)
        c = float(self.shift(yd_raw, pre))
        yd = np.log(np.maximum(yd_raw, 0.0) + c)
        y_mean, y_sd = yd[pre].mean(), yd[pre].std() or 1.0
        x_mean, x_sd = X[pre].mean(axis=0), X[pre].std(axis=0)
        use = x_sd > 1e-9
        Xs = (X[:, use] - x_mean[use]) / x_sd[use]
        ys = (yd - y_mean) / y_sd
        gaps = np.diff(panel.days).astype(float)
        # the log of a daily energy is noisier on calm (dark) days: precision grows with the energy level
        level = np.maximum(panel.daily_mean(panel.X).mean(axis=1), 0.0) + c
        weights = panel.n_day * level
        w_mean = weights[pre].mean()
        draws = gibbs(ys, Xs, pre, gaps, weights=weights / w_mean, n_iter=self.n_iter, burn=self.burn,
                      seed=self.seed)
        rng = np.random.default_rng(self.seed + 7)
        z = rng.standard_normal((len(draws["s_e"]), int(panel.post_day.sum())))
        state = {"draws": draws, "z": z, "c": c, "y_mean": y_mean, "y_sd": y_sd, "x_mean": x_mean, "x_sd": x_sd,
                 "use": use, "w_mean": w_mean, "inclusion": draws["gamma"].mean(axis=0),
                 "s_l": float(draws["s_l"].mean()), "s_e": float(draws["s_e"].mean())}
        return self.refresh(state, panel)

    def refresh(self, state: dict, panel: Panel) -> dict:
        """Posterior predictive counterfactual of the post days for the panel's current control signals."""
        post = panel.post_day
        X = self.design(panel)
        use = state["use"]
        Xs = (X[post][:, use] - state["x_mean"][use]) / state["x_sd"][use]
        level = np.maximum(panel.daily_mean(panel.X).mean(axis=1), 0.0) + state["c"]
        w = (panel.n_day * level)[post] / state["w_mean"]
        d = state["draws"]
        noise = state["z"] * d["s_e"][:, None] / np.sqrt(w)[None, :]
        cf = d["mu"][:, post] + d["beta"] @ Xs.T + noise
        return {**state, "cf": np.exp(cf * state["y_sd"] + state["y_mean"]) - state["c"]}

    def estimate(self, state: dict, panel: Panel, y: np.ndarray, weeks: float) -> Estimate:
        post_days = panel.days[panel.post_day]
        win = post_days < 7 * weeks
        if not win.any():
            return Estimate(np.nan, -np.inf, np.inf, -np.inf, np.inf)
        n = panel.n_day[panel.post_day][win]
        actual = float((n * panel.daily_mean(y)[panel.post_day][win]).sum())
        draws = actual / (state["cf"][:, win] @ n) - 1.0
        return from_draws(float(np.median(draws)), draws)

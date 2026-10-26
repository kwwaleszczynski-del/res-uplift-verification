"""Machine-learning counterfactual with blocked cross-conformal intervals for window totals.

The test unit's power is predicted record by record from the controls' power and the shared covariates
(gradient boosted trees, scikit-learn ``HistGradientBoostingRegressor``). The test unit's own signals are never
used: its anemometer sits behind the rotor and changes with the intervention.

Uncertainty of the *window total* has to respect the strong autocorrelation of prediction errors (weather
regimes, seasons), so the calibration works on windows rather than on single records, in the spirit of CV+ /
jackknife+ (Barber et al. 2021) and of the block-wise time-series variants (Xu & Xie 2021):

1. The pre-period days are split into K contiguous folds; each fold is predicted by a model trained on the other
   folds with a 2-day embargo on both sides (out-of-fold predictions). The post-period is predicted by the
   average of the K fold models.
2. Non-conformity scores are the relative errors of the out-of-fold prediction over every window of m
   consecutive valid pre-period days, ``e_w = sum_w y / sum_w yhat - 1``, where m is the number of valid days in
   the post window, capped at a quarter of the pre-period: the calibration needs at least four non-overlapping
   windows, and errors that persist longer than a season cannot be learned from one pre-year anyway. Longer post
   windows therefore keep the quarter-period interval instead of shrinking it further. (A first version capped
   at half the pre-period; its 90 % intervals covered the truth in only 45-65 % of the 26- and 52-week wind
   trials, because two overlapping half-year windows say almost nothing about the spread.)
3. With the conformal quantiles ``e_lo``, ``e_hi`` of the scores, ``(1 + effect) = (1 + est) / (1 + e)`` gives the
   interval ``[(1 + est) / (1 + e_hi) - 1, (1 + est) / (1 + e_lo) - 1]``.

Overlapping windows are not exchangeable, so coverage is approximate (valid under stationarity and mixing as the
pre-period grows); the injected-effect validation measures how close it is.

Usage:
    m = MlCounterfactual()
    state = m.fit(panel, panel.y)
    est = m.estimate(state, panel, y_with_effect, weeks=13)
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from res_uplift.core import Estimate, Panel


def conformal_quantiles(scores: np.ndarray, alpha: float) -> tuple[float, float]:
    """Two-sided split-conformal quantiles with the (n + 1) finite-sample correction (inf if too few scores)."""
    s = np.sort(scores[np.isfinite(scores)])
    n = len(s)
    k_lo = int(np.floor((n + 1) * alpha / 2))
    k_hi = int(np.ceil((n + 1) * (1 - alpha / 2)))
    lo = s[k_lo - 1] if k_lo >= 1 else -np.inf
    hi = s[k_hi - 1] if k_hi <= n else np.inf
    return float(lo), float(hi)


def window_errors(y_day: np.ndarray, yhat_day: np.ndarray, n_day: np.ndarray, m: int) -> np.ndarray:
    """Relative error of the predicted total over every run of m consecutive days."""
    cy = np.concatenate(([0.0], np.cumsum(n_day * y_day)))
    ch = np.concatenate(([0.0], np.cumsum(n_day * yhat_day)))
    sy, sh = cy[m:] - cy[:-m], ch[m:] - ch[:-m]
    with np.errstate(divide="ignore", invalid="ignore"):
        return sy / sh - 1.0


class MlCounterfactual:
    name = "ml"
    uses_controls = True

    def __init__(self, n_folds: int = 5, embargo_days: int = 2, max_iter: int = 200, learning_rate: float = 0.08,
                 max_leaf_nodes: int = 31, min_samples_leaf: int = 40, seed: int = 0):
        self.n_folds, self.embargo = n_folds, embargo_days
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                           min_samples_leaf=min_samples_leaf, l2_regularization=1.0, early_stopping=False,
                           random_state=seed)

    def features(self, panel: Panel) -> np.ndarray:
        return np.column_stack([panel.X, panel.Z])

    def fit(self, panel: Panel, y: np.ndarray) -> dict:
        F = self.features(panel)
        pre = ~panel.post
        pre_days = panel.days[~panel.post_day]
        folds = np.array_split(np.arange(len(pre_days)), self.n_folds)
        oof = np.full(len(y), np.nan)
        models = []
        for f in folds:
            lo, hi = pre_days[f[0]], pre_days[f[-1]]
            test = pre & (panel.day >= lo) & (panel.day <= hi)
            train = pre & ((panel.day < lo - self.embargo) | (panel.day > hi + self.embargo))
            model = HistGradientBoostingRegressor(**self.params).fit(F[train], y[train])
            oof[test] = model.predict(F[test])
            models.append(model)
        return self.refresh({"models": models, "oof": oof}, panel)

    def refresh(self, state: dict, panel: Panel) -> dict:
        """Post-period prediction (average of the fold models) for the panel's current control signals."""
        pred = state["oof"].copy()
        if panel.post.any():
            F = self.features(panel)[panel.post]
            pred[panel.post] = np.mean([m.predict(F) for m in state["models"]], axis=0)
        return {**state, "pred_day": panel.daily_mean(pred)}

    def estimate(self, state: dict, panel: Panel, y: np.ndarray, weeks: float) -> Estimate:
        y_day, p_day, n_day = panel.daily_mean(y), state["pred_day"], panel.n_day
        pre, post = ~panel.post_day, panel.post_day & (panel.days < 7 * weeks)
        m = int(post.sum())
        if m == 0:
            return Estimate(np.nan, -np.inf, np.inf, -np.inf, np.inf)
        point = float((n_day[post] * y_day[post]).sum() / (n_day[post] * p_day[post]).sum() - 1.0)
        m_cal = min(m, int(pre.sum()) // 4)
        scores = window_errors(y_day[pre], p_day[pre], n_day[pre], max(m_cal, 1))
        out = []
        for alpha in (0.10, 0.05):
            e_lo, e_hi = conformal_quantiles(scores, alpha)
            lo = (1 + point) / (1 + e_hi) - 1.0 if np.isfinite(e_hi) else -1.0
            hi = (1 + point) / (1 + e_lo) - 1.0 if e_lo > -1.0 else np.inf
            out += [lo, hi]
        return Estimate(point, *out)

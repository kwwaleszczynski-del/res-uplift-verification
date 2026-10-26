"""Ratio estimators with a moving-block bootstrap: the industry test-control method, DiD and a naive baseline.

All three compare the test unit's energy with a reference in the pre-period and carry that relation into the
post-period:

* **binned** (industry practice for side-by-side upgrade tests, e.g. vortex generators): in every bin of the
  controls' reference wind speed (1 m/s) and wind direction (30 deg) - for PV: irradiance (100 W/m2) x sun
  azimuth (30 deg) - the pre-period ratio of test to control energy predicts the post-period test energy,

      effect = sum_b E_test,post(b) / sum_b r_pre(b) E_ctrl,post(b) - 1,   r_pre(b) = E_test,pre(b) / E_ctrl,pre(b).

  This is the power-ratio form of the test/control turbine method described for turbine upgrades by
  Hwangbo et al. (2017) and Ding (2019, ch. 7); the binning follows IEC 61400-12-1.
* **did**: difference-in-differences of log energy, the same ratio with a single bin,
  ``log(E_test,post / E_ctrl,post) - log(E_test,pre / E_ctrl,pre)``.
* **naive**: the test unit against its own pre-period power curve (own nacelle anemometer, 0.5 m/s bins) or,
  for PV, its own pre-period yield per irradiance bin (a performance-ratio comparison). No controls - this is
  what a before/after comparison with weather normalisation does.

Intervals: moving-block bootstrap over days (blocks of 7 days, Kuensch 1989), resampled independently in the
pre- and post-period; percentile intervals.

Usage:
    m = Binned()
    state = m.fit(panel, panel.y)
    est = m.estimate(state, y_with_effect, weeks=13)
"""
from __future__ import annotations

import numpy as np

from res_uplift.core import Estimate, Panel, block_counts, from_draws


class RatioMethod:
    name = "ratio"
    uses_controls = True

    def __init__(self, n_boot: int = 1000, block_days: int = 7, min_pre: int = 6, seed: int = 0):
        self.n_boot, self.block_days, self.min_pre, self.seed = n_boot, block_days, min_pre, seed

    def labels(self, panel: Panel) -> np.ndarray:
        return panel.bins

    def reference(self, panel: Panel) -> np.ndarray:
        return panel.X.sum(axis=1)

    def _day_bin_sums(self, panel: Panel, sel: np.ndarray, y: np.ndarray, codes: np.ndarray, n_bins: int):
        """(days, bins) sums of test power, reference and record counts over the selected records."""
        day_pos = np.searchsorted(panel.days, panel.day[sel])
        day_pos = day_pos - day_pos.min() if len(day_pos) else day_pos
        n_days = int(day_pos.max()) + 1 if len(day_pos) else 0
        flat = day_pos * n_bins + codes[sel]
        size = n_days * n_bins

        def acc(w):
            return np.bincount(flat, weights=w, minlength=size).reshape(n_days, n_bins)

        return acc(y[sel]), acc(self.reference(panel)[sel]), acc(np.ones(sel.sum())), day_pos

    def fit(self, panel: Panel, y: np.ndarray) -> dict:
        labels = self.labels(panel)
        uniq, codes = np.unique(labels, return_inverse=True)
        n_bins = len(uniq)
        pre = ~panel.post
        st, sc, sn, _ = self._day_bin_sums(panel, pre, y, codes, n_bins)
        keep_days = sn.sum(axis=1) > 0
        st, sc, sn = st[keep_days], sc[keep_days], sn[keep_days]
        ok = (sn.sum(axis=0) >= self.min_pre) & (sc.sum(axis=0) > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = st.sum(axis=0) / sc.sum(axis=0)
            rng = np.random.default_rng(self.seed)
            w = block_counts(len(st), self.n_boot, self.block_days, rng)
            r_boot = (w @ st) / (w @ sc)
        r_boot = np.where(np.isfinite(r_boot), r_boot, r[None, :])
        return {"codes": codes, "n_bins": n_bins, "r": r, "r_boot": r_boot, "ok": ok}

    def refresh(self, state: dict, panel: Panel) -> dict:
        return state

    def estimate(self, state: dict, panel: Panel, y: np.ndarray, weeks: float) -> Estimate:
        sel = panel.post & (panel.day < 7 * weeks)
        if sel.sum() == 0:
            return Estimate(np.nan, -np.inf, np.inf, -np.inf, np.inf)
        st, sc, sn, day_pos = self._day_bin_sums(panel, sel, y, state["codes"], state["n_bins"])
        st, sc = st[:, state["ok"]], sc[:, state["ok"]]
        keep = sn.sum(axis=1) > 0
        st, sc = st[keep], sc[keep]
        r, r_boot = state["r"][state["ok"]], state["r_boot"][:, state["ok"]]
        point = st.sum() / (r * sc.sum(axis=0)).sum() - 1.0
        rng = np.random.default_rng(self.seed + 1 + int(weeks * 7))
        w = block_counts(len(st), self.n_boot, self.block_days, rng)
        draws = (w @ st).sum(axis=1) / ((w @ sc) * r_boot).sum(axis=1) - 1.0
        return from_draws(point, draws)


class Binned(RatioMethod):
    name = "binned"


class DiD(RatioMethod):
    name = "did"

    def labels(self, panel: Panel) -> np.ndarray:
        return np.zeros(len(panel.y), dtype=np.int64)


class Naive(RatioMethod):
    """Before/after against the test unit's own reference; ignores the controls."""

    name = "naive"
    uses_controls = False

    def labels(self, panel: Panel) -> np.ndarray:
        return panel.own_bins

    def reference(self, panel: Panel) -> np.ndarray:
        return np.ones(len(panel.y))

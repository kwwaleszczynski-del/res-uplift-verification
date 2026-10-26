"""Every estimator recovers a known effect on a simple synthetic case.

The cases are simple on purpose: identical turbines (only gains differ) and stationary weather, so that every
method's assumptions hold. Interval *coverage* is tested by simulation in test_conformal.py; here the 95 %
interval only has to reach the truth within one percentage point.
"""
import numpy as np
import pandas as pd
import pytest

from res_uplift.bsts import Bsts
from res_uplift.core import build_panel
from res_uplift.inject import Scenario, apply, true_effect
from res_uplift.mlcf import MlCounterfactual
from res_uplift.ratio import Binned, DiD, Naive
from res_uplift.synth import SyntheticControl

FAST = {
    "binned": lambda: Binned(n_boot=400),
    "did": lambda: DiD(n_boot=400),
    "naive": lambda: Naive(n_boot=400),
    "synth": SyntheticControl,
    "bsts": lambda: Bsts(n_iter=400, burn=100),
    "ml": lambda: MlCounterfactual(max_iter=60, n_folds=3),
}


def _panel(fleet, t0):
    t0 = pd.Timestamp(t0, tz="UTC")
    return build_panel(fleet, fleet.units[0], fleet.units[1:], (t0 - pd.Timedelta(days=365), t0),
                       (t0, t0 + pd.Timedelta(days=13 * 7)))


@pytest.mark.parametrize("name", ["binned", "did", "naive", "synth", "bsts", "ml"])
def test_wind_estimators_recover_uniform_effect(wind, name):
    panel = _panel(wind, "2019-01-15")
    y_obs, y_cf = apply(panel, Scenario("uniform", 0.03))
    m = FAST[name]()
    est = m.estimate(m.fit(panel, y_obs), panel, y_obs, 13)
    truth = true_effect(panel, y_obs, y_cf, 13)
    assert abs(truth - 0.03) < 1e-9
    assert abs(est.effect - truth) < 0.012, est
    assert est.lo95 - 0.01 <= truth <= est.hi95 + 0.01, est
    assert est.lo95 <= est.lo90 <= est.effect <= est.hi90 <= est.hi95


@pytest.mark.parametrize("name", ["binned", "did", "synth", "bsts", "ml"])
def test_pv_estimators_recover_step_effect(pv, name):
    panel = _panel(pv, "2017-04-01")
    y_obs, y_cf = apply(panel, Scenario("uniform", 0.02))
    m = FAST[name]()
    est = m.estimate(m.fit(panel, y_obs), panel, y_obs, 13)
    assert abs(est.effect - 0.02) < 0.006, est
    assert est.lo95 - 0.005 <= 0.02 <= est.hi95 + 0.005, est


def test_zero_effect_is_not_a_success(wind):
    panel = _panel(wind, "2019-03-01")
    for name in ("binned", "synth", "ml"):
        m = FAST[name]()
        est = m.estimate(m.fit(panel, panel.y), panel, panel.y, 13)
        assert abs(est.effect) < 0.012 and est.lo95 < 0.01, (name, est)


def test_refresh_follows_changed_controls(wind):
    from dataclasses import replace
    panel = _panel(wind, "2019-01-15")
    for name in ("ml", "bsts", "synth", "binned"):
        m = FAST[name]()
        state = m.fit(panel, panel.y)
        before = m.estimate(state, panel, panel.y, 13).effect
        lowered = replace(panel, X=np.where(panel.post[:, None], panel.X * 0.95, panel.X))
        after = m.estimate(m.refresh(state, lowered), lowered, panel.y, 13).effect
        assert after > before + 0.005, (name, before, after)  # weaker controls -> lower counterfactual


def test_synthetic_control_weights_are_non_negative(wind):
    panel = _panel(wind, "2019-01-15")
    w = SyntheticControl().weights(panel, panel.y)
    assert (w >= 0).all() and 0.8 < w.sum() < 1.2

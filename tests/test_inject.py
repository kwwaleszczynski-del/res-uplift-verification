import numpy as np
import pandas as pd
import pytest

from res_uplift.core import build_panel
from res_uplift.inject import Scenario, apply, partial_load_weight, true_effect


def _panel(fleet, t0="2018-09-01", post_days=120):
    t0 = pd.Timestamp(t0, tz="UTC")
    return build_panel(fleet, fleet.units[0], fleet.units[1:], (t0 - pd.Timedelta(days=200), t0),
                       (t0, t0 + pd.Timedelta(days=post_days)))


def test_uniform_effect_is_exact(wind):
    p = _panel(wind)
    y_obs, y_cf = apply(p, Scenario("uniform", 0.02))
    assert np.allclose(y_obs[~p.post], p.y[~p.post]) and np.allclose(y_cf, p.y)
    assert np.isclose(true_effect(p, y_obs, y_cf, 4), 0.02)


def test_partial_load_effect_only_below_rated(wind):
    p = _panel(wind)
    y_obs, y_cf = apply(p, Scenario("partial_load", 0.05))
    ratio = y_obs[p.post] / np.maximum(y_cf[p.post], 1e-9)
    high = y_cf[p.post] >= 0.95 * p.rated
    low = (y_cf[p.post] > 10) & (y_cf[p.post] <= 0.75 * p.rated)
    assert np.allclose(ratio[high], 1.0) and np.allclose(ratio[low], 1.05)
    assert 0 < true_effect(p, y_obs, y_cf, 13) < 0.05
    assert partial_load_weight(np.array([0.5, 0.85, 1.0]), 1.0).tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_soiling_wash_changes_pre_period_and_gives_positive_effect(pv):
    p = _panel(pv, "2017-03-01")
    sc = Scenario("soiling_wash", 0.03)
    assert sc.changes_pre
    y_obs, y_cf = apply(p, sc)
    days_before = -p.day[~p.post]
    last_month, early = (days_before <= 30) & (days_before >= 1), days_before > 90
    assert np.allclose(y_obs[~p.post][early], p.y[~p.post][early])
    assert (y_obs[~p.post][last_month] < p.y[~p.post][last_month]).all()
    assert np.allclose(y_obs[~p.post], y_cf[~p.post])
    e2, e13 = true_effect(p, y_obs, y_cf, 2), true_effect(p, y_obs, y_cf, 13)
    assert 0.025 < e2 < 0.031 and 0 < e13 < e2  # full gain right after the wash, less once it has re-soiled
    late = p.post & (p.day > 70)
    assert np.allclose(y_obs[late], y_cf[late])  # re-soiled to the same level after 60 days


def test_unknown_kind_raises(wind):
    with pytest.raises(ValueError):
        apply(_panel(wind), Scenario("teleport", 0.01))

"""Shared synthetic fleets (no network, no real data)."""
import numpy as np
import pandas as pd
import pytest

from res_uplift.core import Panel
from res_uplift.synthetic import pv_fleet, wind_fleet


@pytest.fixture(scope="session")
def wind():
    return wind_fleet(n_units=4, days=560, seed=11, gust=0.01, speed_bias=0.0)


@pytest.fixture(scope="session")
def pv():
    return pv_fleet(n_units=4, days=600, seed=5, freq="30min", soiling=False)


def make_daily_panel(rng, n_pre=200, n_post=30, n_controls=3, noise=0.03, effect=0.0, per_day=1, phi=0.5,
                     unit="ns"):
    """Panel built directly from arrays: controls share a lognormal daily weather signal, the test is a fixed
    mix of them times AR(1) multiplicative noise; the post period is scaled by (1 + effect)."""
    n_days = n_pre + n_post
    weather = np.exp(rng.normal(0, 0.5, n_days))
    gains = rng.uniform(0.5, 1.5, n_controls)
    X = weather[:, None] * gains[None, :] * np.exp(rng.normal(0, 0.05, (n_days, n_controls)))
    w = rng.dirichlet(np.ones(n_controls))
    e = np.zeros(n_days)
    for t in range(1, n_days):
        e[t] = phi * e[t - 1] + rng.normal(0, noise * np.sqrt(1 - phi**2))
    y = (X @ w) * np.exp(e)
    day = np.r_[np.arange(-n_pre, 0), np.arange(n_post)]
    y = np.where(day >= 0, y * (1 + effect), y)
    rep = np.repeat(np.arange(n_days), per_day)
    t0 = pd.Timestamp("2020-01-01", tz="UTC")
    hour_ns = np.tile(np.arange(per_day), n_days) * 3600 * 10**9
    t_ns = (t0 + pd.to_timedelta(day[rep], unit="D")).as_unit("ns").asi8 + hour_ns
    Z = np.column_stack([np.log(weather[rep]), np.zeros(len(rep))])
    return Panel(kind="wind", t_ns=t_ns, day=day[rep], y=y[rep], X=X[rep], Z=Z, z_names=["ws_ref", "temp"],
                 bins=np.zeros(len(rep), dtype=np.int64), own_bins=np.zeros(len(rep), dtype=np.int64), rated=10.0,
                 meta={"test": "T", "controls": [f"C{j}" for j in range(n_controls)],
                       "pre": (int(t0.value), int(t0.value)), "post": (int(t0.value), int(t0.value))})

import numpy as np
import pandas as pd

from res_uplift.core import build_panel
from res_uplift.inject import Scenario, apply
from res_uplift.wake import control_factors, deficit, disturbed, disturbed_sector_width, thrust_coefficient


def test_deficit_shape():
    x = np.array([3.0, 5.0, 8.0])  # the Gaussian model holds beyond the near wake (x / d > ~3)
    d = deficit(x, 0.0, 0.8)
    assert (np.diff(d) < 0).all() and (d > 0).all() and (d < 0.6).all()
    assert deficit(-3.0, 0.0, 0.8) == 0.0
    assert np.isclose(deficit(5.0, 0.5, 0.8), deficit(5.0, -0.5, 0.8))
    assert deficit(5.0, 0.0, 0.8) > deficit(5.0, 0.0, 0.4)
    assert thrust_coefficient(25.0) < thrust_coefficient(6.0)


def test_iec_disturbed_sector_width():
    assert abs(disturbed_sector_width(4.0) - 59.2) < 0.5
    assert disturbed_sector_width(10.0) < disturbed_sector_width(3.0)


def test_downstream_control_loses_power_when_test_gains(wind):
    t0 = pd.Timestamp("2018-09-01", tz="UTC")
    # units sit on a west-east line 400 m apart: W2 is downstream of W1 in westerly wind
    p = build_panel(wind, "W1", ["W2", "W4"], (t0 - pd.Timedelta(days=100), t0), (t0, t0 + pd.Timedelta(days=60)))
    y_obs, _ = apply(p, Scenario("partial_load", 0.03))
    f = control_factors(wind, p, np.where(p.post, y_obs / np.maximum(p.y, 1e-9) - 1, 0))
    westerly = (np.abs(((p.ref_dir - 270) + 180) % 360 - 180) < 10) & p.post & (p.X[:, 0] < 1000)
    easterly = (np.abs(((p.ref_dir - 90) + 180) % 360 - 180) < 10) & p.post
    assert westerly.sum() > 10 and easterly.sum() > 10
    assert (f[westerly, 0] < 1).mean() > 0.9 and f[westerly, 0].mean() > 0.95
    assert np.allclose(f[easterly, 0], 1.0)
    assert f[westerly, 1].mean() > f[westerly, 0].mean()  # W4 is three times farther away
    d = disturbed(wind, p)
    assert d[westerly].all() and not d[easterly].any()

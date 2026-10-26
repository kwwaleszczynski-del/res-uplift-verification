import numpy as np
import pandas as pd

from res_uplift.core import block_counts, build_panel, circular_mean_deg, to_ns
from res_uplift.synthetic import wind_fleet


def test_to_ns_does_not_depend_on_datetime_resolution():
    a = pd.date_range("2019-03-01", periods=5, freq="10min", tz="UTC", unit="us")
    b = pd.date_range("2019-03-01", periods=5, freq="10min", tz="UTC", unit="ns")
    s = pd.date_range("2019-03-01", periods=5, freq="10min", tz="UTC", unit="s")
    assert np.array_equal(to_ns(a), to_ns(b)) and np.array_equal(to_ns(a), to_ns(s))
    assert to_ns(pd.Timestamp("2019-03-01", tz="UTC").as_unit("s")) == to_ns("2019-03-01 00:00:00+00:00")


def test_panel_identical_for_us_and_ns_fleets():
    t0 = pd.Timestamp("2018-09-01", tz="UTC")
    pre, post = (t0 - pd.Timedelta(days=200), t0), (t0, t0 + pd.Timedelta(days=60))
    panels = [build_panel(wind_fleet(n_units=3, days=300, seed=2, unit=u), "W1", ["W2", "W3"], pre, post)
              for u in ("us", "ns", "s")]
    for p in panels[1:]:
        assert np.array_equal(p.day, panels[0].day)
        assert np.array_equal(p.t_ns, panels[0].t_ns)
        assert np.allclose(p.y, panels[0].y)


def test_day_numbering_and_windows(wind):
    t0 = pd.Timestamp("2018-07-01", tz="UTC")
    gap_end = t0 + pd.Timedelta(days=3)
    p = build_panel(wind, "W1", ["W2", "W3"], (t0 - pd.Timedelta(days=120), t0),
                    (gap_end, gap_end + pd.Timedelta(days=70)))
    assert p.day[~p.post].max() == -1 and p.day[p.post].min() == 0
    t = p.t_ns[p.post]
    assert t.min() >= to_ns(gap_end) and p.t_ns[~p.post].max() < to_ns(t0)
    w = p.window(4)
    assert (p.day[w & p.post] < 28).all() and (~p.post <= w).all()
    assert np.allclose(p.daily_mean(p.y) * p.n_day, np.bincount(p.inv, weights=p.y))
    assert p.n_controls == 2


def test_min_records_drops_short_days(wind):
    t0 = pd.Timestamp("2018-07-01", tz="UTC")
    p = build_panel(wind, "W1", ["W2"], (t0 - pd.Timedelta(days=60), t0), (t0, t0 + pd.Timedelta(days=30)),
                    min_records=24)
    assert (p.n_day >= 24).all()


def test_block_counts_are_contiguous_blocks():
    rng = np.random.default_rng(0)
    c = block_counts(30, 200, 7, rng)
    assert c.shape == (200, 30) and np.allclose(c.sum(axis=1), 30)
    assert c.mean() == 1.0


def test_circular_mean_wraps():
    a = np.array([[350.0, 10.0], [90.0, np.nan]])
    m = circular_mean_deg(a)
    assert abs(((m[0] + 180) % 360) - 180) < 1e-9 and abs(m[1] - 90) < 1e-9

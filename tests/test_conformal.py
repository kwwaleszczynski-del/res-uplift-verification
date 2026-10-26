"""The conformal and Bayesian intervals have (close to) nominal coverage in simulation."""
import numpy as np
import pytest

from conftest import make_daily_panel
from res_uplift.bsts import Bsts
from res_uplift.mlcf import MlCounterfactual, conformal_quantiles, window_errors
from res_uplift.synth import SyntheticControl, cwz_pvalue


def test_conformal_quantiles_finite_sample_rule():
    s = np.arange(1, 20, dtype=float)  # 19 scores
    assert conformal_quantiles(s, 0.10) == (1.0, 19.0)  # ranks floor(20 * 0.05) = 1 and ceil(20 * 0.95) = 19
    lo, hi = conformal_quantiles(s[:5], 0.10)
    assert lo == -np.inf and hi == np.inf  # too few scores for a 90 % interval


def test_window_errors_match_direct_sums():
    rng = np.random.default_rng(1)
    y, yhat, n = rng.uniform(1, 2, 20), rng.uniform(1, 2, 20), rng.integers(1, 5, 20).astype(float)
    e = window_errors(y, yhat, n, 5)
    assert len(e) == 16
    assert np.isclose(e[3], (n[3:8] * y[3:8]).sum() / (n[3:8] * yhat[3:8]).sum() - 1)


def test_cwz_pvalue_is_uniform_under_the_null_and_powerful_under_an_effect():
    rng = np.random.default_rng(7)
    p_null, p_alt = [], []
    for _ in range(300):
        p = make_daily_panel(rng, n_pre=150, n_post=20, noise=0.03)
        yd, Xd, post = p.daily_mean(p.y), p.daily_mean(p.X), p.post_day
        p_null.append(cwz_pvalue(yd, Xd, p.n_day, post, 0.0))
        p_alt.append(cwz_pvalue(np.where(post, yd * 1.06, yd), Xd, p.n_day, post, 0.0))
    p_null, p_alt = np.array(p_null), np.array(p_alt)
    assert 0.05 <= np.mean(p_null <= 0.10) <= 0.16
    assert 0.01 <= np.mean(p_null <= 0.05) <= 0.10
    assert np.mean(p_alt <= 0.10) > 0.9


def test_synthetic_control_interval_coverage():
    rng = np.random.default_rng(3)
    sc, hits = SyntheticControl(), []
    for _ in range(80):
        p = make_daily_panel(rng, n_pre=150, n_post=28, noise=0.03, effect=0.02)
        est = sc.estimate(sc.fit(p, p.y), p, p.y, 4)
        hits.append(est.lo90 <= 0.02 <= est.hi90)
    assert 0.80 <= np.mean(hits) <= 0.99


def test_ml_conformal_interval_coverage():
    rng = np.random.default_rng(5)
    m, hits, widths = MlCounterfactual(max_iter=40, n_folds=4, min_samples_leaf=20), [], []
    for _ in range(40):
        p = make_daily_panel(rng, n_pre=240, n_post=28, noise=0.03, effect=0.0, per_day=4)
        est = m.estimate(m.fit(p, p.y), p, p.y, 4)
        hits.append(est.lo90 <= 0.0 <= est.hi90)
        widths.append(est.hi90 - est.lo90)
    assert 0.78 <= np.mean(hits) <= 1.0
    assert np.median(widths) < 0.1


def test_bsts_recovers_effect_with_calibrated_interval():
    rng = np.random.default_rng(11)
    b, hits = Bsts(n_iter=300, burn=100), []
    for k in range(15):
        p = make_daily_panel(rng, n_pre=150, n_post=28, noise=0.03, effect=0.03)
        est = b.estimate(b.fit(p, p.y), p, p.y, 4)
        hits.append(est.lo90 <= 0.03 <= est.hi90)
        if k == 0:
            assert abs(est.effect - 0.03) < 0.03
    assert np.mean(hits) >= 0.7


@pytest.mark.parametrize("phi", [0.0, 0.8])
def test_cwz_handles_autocorrelated_noise(phi):
    rng = np.random.default_rng(int(phi * 10) + 1)
    rej = []
    for _ in range(200):
        p = make_daily_panel(rng, n_pre=200, n_post=14, noise=0.03, phi=phi)
        rej.append(cwz_pvalue(p.daily_mean(p.y), p.daily_mean(p.X), p.n_day, p.post_day, 0.0) <= 0.10)
    assert np.mean(rej) <= 0.18

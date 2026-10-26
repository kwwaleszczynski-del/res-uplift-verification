import numpy as np
import pandas as pd

from res_uplift import planner


def _fake_trials(rng, method, sd, half, bias=0.0, weeks=(2, 4, 8, 13, 26, 52), n=300):
    """Trial rows of a scale-equivariant estimator whose error spread shrinks as 1/sqrt(weeks)."""
    rows = []
    for w in weeks:
        s, h = sd / np.sqrt(w), half / np.sqrt(w)
        e0 = bias + rng.normal(0, s, n)  # zero-effect estimate of each trial
        for size in (0.0, 0.01, 0.02, 0.03):
            f = 1 + size  # a uniform effect scales estimate and bounds: (1 + x) -> (1 + x)(1 + size)
            est, lo, hi = f * (1 + e0) - 1, f * (1 + e0 - h) - 1, f * (1 + e0 + h) - 1
            rows.append(pd.DataFrame({"tech": "wind", "kind": "uniform" if size else "none", "method": method,
                                      "n_controls": 5, "weeks": w, "size": size, "true": size, "effect": est,
                                      "lo90": lo, "hi90": hi, "lo95": lo - 0.2 * h, "hi95": hi + 0.2 * h,
                                      "noise_pct": 3.0}))
    return pd.concat(rows, ignore_index=True)


def test_placebo_power_reproduces_simulated_success_exactly():
    rng = np.random.default_rng(1)
    trials = _fake_trials(rng, "m", 0.04, 0.066)
    cs = planner.cells(trials)
    emp = planner.empirical_power(trials)
    chk = emp[(emp["kind"] == "uniform") & (emp["size"] > 0)].merge(cs, on=planner.KEYS)
    pred = np.array([planner.power(r, r["size"]) for _, r in chk.iterrows()])
    assert np.allclose(pred, chk["success"].to_numpy())


def test_required_weeks_behave_and_recommend_the_better_method():
    rng = np.random.default_rng(0)
    trials = pd.concat([_fake_trials(rng, "good", 0.02, 0.033), _fake_trials(rng, "noisy", 0.04, 0.066)])
    cs = planner.cells(trials)
    good = cs[cs["method"] == "good"]
    row13 = good[good["weeks"] == 13].iloc[0]
    assert row13["fsr"] < 0.10 and row13["cover90"] > 0.85 and len(row13["lo"]) == 300
    w1, w2 = planner.required_weeks(good, 0.01), planner.required_weeks(good, 0.02)
    assert np.isfinite(w1) and w2 < w1
    assert planner.required_weeks(good, 0.01, v=1.5) > w1
    best, need = planner.recommend(cs, "wind", 5, 0.02)
    assert best == "good" and need["noisy"] > need["good"]


def test_untrustworthy_cells_are_skipped():
    rng = np.random.default_rng(2)
    trials = _fake_trials(rng, "biased", 0.03, 0.02, bias=0.01)  # intervals too narrow and shifted upwards
    cs = planner.cells(trials)
    assert (cs["fsr"] > planner.MAX_FSR).any()
    assert planner.required_weeks(cs, 0.01, trustworthy_only=False) <= planner.required_weeks(cs, 0.01)

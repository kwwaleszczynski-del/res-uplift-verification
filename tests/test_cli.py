from pathlib import Path

import pandas as pd
import pytest

from res_uplift.cli import parse_args
from res_uplift.validation import DESIGNS, draw_trials, run_trial, summarise, with_zero


def test_parse_args_defaults_and_choices():
    a = parse_args(["validate", "--jobs", "3", "--tech", "pv"])
    assert a.command == "validate" and a.jobs == 3 and a.tech == "pv" and a.data_dir == Path("data")
    with pytest.raises(SystemExit):
        parse_args(["teleport"])


def test_validation_trial_end_to_end_on_synthetic_fleet(wind):
    from dataclasses import replace
    design = replace(DESIGNS["wind"], first_t0="2018-12-01", last_t0="2019-01-31", trials_per_unit=1,
                     weeks=(4, 13), sizes=(0.0, 0.02), n_controls=(1, 99))
    trials = draw_trials(wind, design)
    assert len(trials) == len(wind.units)
    t = trials[0]
    assert all(t["test"] not in cs for cs in t["control_sets"]) and len(t["control_sets"][-1]) == 3
    rows = run_trial(t, fleet=wind, design=design, bsts_iter=200)
    df = pd.DataFrame(rows)
    assert set(df["method"]) == {"binned", "did", "naive", "synth", "bsts", "ml"}
    assert set(df["kind"]) == {"none", "uniform", "partial_load"}
    s = summarise(with_zero(df))
    assert {"bias", "rmse", "cover90", "cover95", "success"} <= set(s.columns)
    assert (s[s["size"] == 0]["kind"].isin(["uniform", "partial_load"])).all()

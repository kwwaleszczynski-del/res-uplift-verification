"""Case studies: real interventions from the logs, estimated with every method.

Wind (Kelmarsh status logs): the longest service stops after component faults - a drivetrain oscillation fault
followed by 25 days of on-site work, converter faults, a hydraulic fault. PV (PVDAQ): multi-week inverter
outages after which the inverter came back (repaired or replaced). None of them was meant to raise output; the
question an asset manager asks is whether the unit came back as good as before.

For each event the test unit is compared with all other units that have data in both periods; the pre-period
is the 365 days before the event (or what the data allow), the post windows start when the unit is back.
Reference ("placebo"): the zero-effect injected trials of the same unit with all controls - what each method
reports at random dates when nothing was done.

Usage:
    run_all(Path("data"), Path("results"))
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from res_uplift.core import build_panel
from res_uplift.data import get_fleet
from res_uplift.validation import _coverages, make_methods

log = logging.getLogger(__name__)
PV_TZ = "Etc/GMT+8"
CASES = [
    {"id": "T6-2021", "tech": "wind", "unit": "T6", "start": "2021-02-05 08:46", "end": "2021-03-05 13:58",
     "label": "Drivetrain oscillation fault, then 25 days of on-site maintenance"},
    {"id": "T6-2017", "tech": "wind", "unit": "T6", "start": "2017-11-11 20:05", "end": "2017-11-16 10:32",
     "label": "Frequency converter fault, 4.6 days down"},
    {"id": "T3-2020", "tech": "wind", "unit": "T3", "start": "2020-10-24 12:40", "end": "2020-11-04 13:40",
     "label": "Converter fault and a 6-day pitch battery test, 11 days"},
    {"id": "T1-2021", "tech": "wind", "unit": "T1", "start": "2021-07-25 17:55", "end": "2021-07-29 14:36",
     "label": "Low hydraulic pressure, 3.9 days down"},
    {"id": "T5-2021", "tech": "wind", "unit": "T5", "start": "2021-10-30 12:16", "end": "2021-11-16 14:13",
     "label": "Converter fault, then 11 days of on-site maintenance"},
    {"id": "A-2016", "tech": "pv", "unit": "A", "start": "2016-06-04", "end": "2016-07-08",
     "label": "Inverter failure, 34 days off, back after repair"},
    {"id": "B-2017", "tech": "pv", "unit": "B", "start": "2017-08-24", "end": "2017-09-25",
     "label": "Inverter outage, 32 days"},
    {"id": "D1-2017", "tech": "pv", "unit": "D1", "start": "2017-10-17", "end": "2017-12-05",
     "label": "Inverter 1 of building D out, 49 days"},
]
WEEKS = (4, 13, 26, 52)


def _ts(case: dict, key: str) -> pd.Timestamp:
    t = pd.Timestamp(case[key])
    return t.tz_localize("UTC") if case["tech"] == "wind" else t.tz_localize(PV_TZ).tz_convert("UTC")


def run_case(case: dict, fleet, bsts_iter: int = 1000) -> tuple[list[dict], dict]:
    start, end = _ts(case, "start"), _ts(case, "end")
    data_end = fleet.time[-1]
    pre_start = max(start - pd.Timedelta(days=365), fleet.time[0])
    weeks = [w for w in WEEKS if end + pd.Timedelta(weeks=w) <= data_end + pd.Timedelta(days=1)]
    if not weeks:  # less than four weeks after the event: use what is there
        weeks = [round((data_end - end) / pd.Timedelta(weeks=1), 1)]
    post_end = end + pd.Timedelta(weeks=max(weeks))
    c_pre, c_post = _coverages(fleet, pre_start, start), _coverages(fleet, end, post_end)
    ti = fleet.index(case["unit"])
    controls = [u for j, u in enumerate(fleet.units)
                if j != ti and c_pre[j] >= 0.5 * c_pre.max() and c_post[j] >= 0.5 * c_post.max()]
    panel = build_panel(fleet, case["unit"], controls, (pre_start, start), (end, post_end))
    rows = []
    series = {}
    for m in make_methods(seed=7, bsts_iter=bsts_iter):
        state = m.fit(panel, panel.y)
        for w in weeks:
            est = m.estimate(state, panel, panel.y, w)
            rows.append({"case": case["id"], "tech": case["tech"], "unit": case["unit"], "label": case["label"],
                         "method": m.name, "weeks": w, **est.as_dict(), "n_controls": len(controls),
                         "controls": "+".join(controls), "n_pre_days": int((~panel.post_day).sum()),
                         "n_post_days": int((panel.post_day & (panel.days < 7 * w)).sum())})
        if m.name == "ml":
            series["ml"] = state["pred_day"]
        if m.name == "synth":
            series["synth"] = panel.daily_mean(panel.X) @ m.weights(panel, panel.y)
    day0 = {False: start.normalize(), True: end.normalize()}
    dates = [(day0[bool(d >= 0)] + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in panel.days]
    keep = panel.days >= -120
    plot = {"id": case["id"], "label": case["label"], "unit": case["unit"], "tech": case["tech"],
            "start": str(start), "end": str(end), "controls": controls,
            "date": [d for d, k in zip(dates, keep) if k],
            "actual": np.round(panel.daily_mean(panel.y)[keep], 2).tolist(),
            "synth": np.round(series["synth"][keep], 2).tolist(), "ml": np.round(series["ml"][keep], 2).tolist(),
            "post": panel.post_day[keep].astype(int).tolist()}
    return rows, plot


def placebo(trials: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """5th-95th percentile of zero-effect trial estimates of the same unit, method, window and control count."""
    zero = trials[trials["kind"].eq("none")]
    out = []
    for _, r in rows.iterrows():
        z = zero[(zero["tech"] == r["tech"]) & (zero["test"] == r["unit"]) & (zero["method"] == r["method"])
                 & (zero["weeks"] == r["weeks"]) & (zero["n_controls"] == r["n_controls"])]["effect"]
        out.append({"placebo_n": len(z), "placebo_p05": z.quantile(0.05) if len(z) else np.nan,
                    "placebo_p95": z.quantile(0.95) if len(z) else np.nan,
                    "placebo_rank": float((z < r["effect"]).mean()) if len(z) else np.nan})
    return pd.concat([rows.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def run_all(data_dir: Path, results_dir: Path, bsts_iter: int = 1000) -> pd.DataFrame:
    rows, plots = [], []
    fleets = {}
    for case in CASES:
        if case["tech"] not in fleets:
            fleets[case["tech"]] = get_fleet(case["tech"], data_dir)
        r, p = run_case(case, fleets[case["tech"]], bsts_iter)
        rows += r
        plots.append(p)
        log.info("%s: %s", case["id"], ", ".join(f"{x['method']} {100 * x['effect']:+.1f}%" for x in r
                                                if x["weeks"] == max(y["weeks"] for y in r)))
    df = pd.DataFrame(rows)
    trial_files = sorted((Path(data_dir) / "validation").glob("trials_*.parquet"))
    if trial_files:
        df = placebo(pd.concat([pd.read_parquet(f) for f in trial_files], ignore_index=True), df)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "case_studies.csv", index=False, float_format="%.5g")
    with open(results_dir / "case_series.json", "w", encoding="utf-8") as fh:
        json.dump(plots, fh)
    return df

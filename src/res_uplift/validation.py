"""Injected-effect validation on real data, and the wake spillover experiment.

Protocol
--------
* A *trial* is a test unit (balanced over units), an intervention date ``t0`` drawn at random from the eligible
  range, and control sets of 1, 2, 3 units and all other units, drawn at random from the units with enough data.
* Pre-period: the 365 days before ``t0``; post windows of 2 to 52 weeks after it (windows past the end of the data
  are skipped).
* A known effect is injected into the test unit's real power (``inject``): sizes 0 to 5 %, shapes uniform and
  partial-load (wind) or uniform and soiling-then-wash (PV). Every method estimates every (trial, effect,
  window) and is scored against the realised true effect of that window.
* Real data already contain whatever happened at the site (weather years, unlogged changes, sensor drift), so a
  zero injected effect measures each method's false-success rate on real noise.

Spillover (wind): the test turbine gets a partial-load uplift and, through the Gaussian wake model, every
control downstream of it loses power in proportion (``wake``). The bias is measured with all records and with
the IEC disturbed sectors removed.

Usage:
    trials = draw_trials(fleet, DESIGNS["wind"])
    rows = run(fleet, trials, jobs=8)          # list of dicts, one per (trial, control set, effect, window, method)
    summary = summarise(pd.DataFrame(rows))
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from res_uplift.bsts import Bsts
from res_uplift.core import Fleet, Panel, build_panel, to_ns
from res_uplift.inject import KINDS, Scenario, apply, true_effect
from res_uplift.mlcf import MlCounterfactual
from res_uplift.ratio import Binned, DiD, Naive
from res_uplift.synth import SyntheticControl

log = logging.getLogger(__name__)
METHODS = ("binned", "did", "naive", "synth", "bsts", "ml")
LABELS = {"binned": "Test-control bins (industry)", "did": "Difference-in-differences", "naive": "Naive before/after",
          "synth": "Synthetic control + CWZ conformal", "bsts": "Bayesian structural time series",
          "ml": "GBM counterfactual + conformal"}


@dataclass(frozen=True)
class Design:
    tech: str
    first_t0: str
    last_t0: str
    sizes: tuple = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
    weeks: tuple = (2, 4, 8, 13, 26, 52)
    n_controls: tuple = (1, 2, 3, 99)  # 99 = all other units
    trials_per_unit: int = 12
    pre_days: int = 365
    seed: int = 2026

    @property
    def kinds(self) -> tuple:
        return KINDS[self.tech]


DESIGNS = {
    "wind": Design("wind", first_t0="2017-05-01", last_t0="2020-12-31"),
    "pv": Design("pv", first_t0="2017-01-01", last_t0="2018-06-30"),
}


def make_methods(seed: int = 0, bsts_iter: int = 1000) -> list:
    return [Binned(seed=seed), DiD(seed=seed), Naive(seed=seed), SyntheticControl(),
            Bsts(n_iter=bsts_iter, burn=bsts_iter // 4, seed=seed), MlCounterfactual(seed=seed)]


def _coverages(fleet: Fleet, a, b) -> np.ndarray:
    t = fleet.t_ns
    sel = (t >= to_ns(a)) & (t < to_ns(b))
    return fleet.valid[sel].mean(axis=0) if sel.any() else np.zeros(len(fleet.units))


def draw_trials(fleet: Fleet, design: Design) -> list[dict]:
    """Random (test unit, t0, control sets); units need half the best unit's valid share in pre and post."""
    rng = np.random.default_rng(design.seed)
    first, last = pd.Timestamp(design.first_t0, tz="UTC"), pd.Timestamp(design.last_t0, tz="UTC")
    n_days = (last - first).days
    trials = []
    for _ in range(design.trials_per_unit):
        for ti, unit in enumerate(fleet.units):
            for _attempt in range(100):
                t0 = first + pd.Timedelta(days=int(rng.integers(0, n_days + 1)))
                c_pre = _coverages(fleet, t0 - pd.Timedelta(days=design.pre_days), t0)
                c_post = _coverages(fleet, t0, t0 + pd.Timedelta(weeks=13))
                ok = (c_pre >= 0.5 * c_pre.max()) & (c_post >= 0.5 * c_post.max())
                candidates = [fleet.units[j] for j in np.flatnonzero(ok) if j != ti]
                if ok[ti] and candidates:
                    break
            else:
                continue
            sets = []
            for j in design.n_controls:
                k = min(j, len(candidates))
                chosen = sorted(rng.choice(candidates, size=k, replace=False).tolist(), key=fleet.units.index)
                if chosen not in sets:
                    sets.append(chosen)
            trials.append({"tech": design.tech, "trial": len(trials), "test": unit, "t0": str(t0.date()),
                           "control_sets": sets, "seed": int(rng.integers(0, 2**31 - 1))})
    return trials


def weekly_variability(panel: Panel) -> float:
    """Standard deviation (%) of the weekly log ratio of test to control energy in the pre-period."""
    pre = ~panel.post
    week = np.floor_divide(panel.day[pre], 7)
    uniq, inv = np.unique(week, return_inverse=True)
    sy = np.bincount(inv, weights=panel.y[pre])
    sx = np.bincount(inv, weights=panel.X[pre].sum(axis=1))
    nd = np.bincount(np.unique(np.c_[inv, panel.day[pre]], axis=0)[:, 0], minlength=len(uniq))
    ok = (nd >= 3) & (sy > 0) & (sx > 0)
    return float(100 * np.std(np.log(sy[ok] / sx[ok]), ddof=1)) if ok.sum() > 4 else np.nan


def _panel_for(fleet: Fleet, design: Design, test: str, controls: list, t0: pd.Timestamp):
    data_end = fleet.time[-1] + (fleet.time[-1] - fleet.time[-2])
    weeks = [w for w in design.weeks if t0 + pd.Timedelta(weeks=w) <= data_end]
    if not weeks:
        return None, []
    end = t0 + pd.Timedelta(weeks=max(weeks))
    panel = build_panel(fleet, test, controls, (t0 - pd.Timedelta(days=design.pre_days), t0), (t0, end))
    if (~panel.post_day).sum() < 120 or panel.post_day.sum() < 7:
        return None, []
    return panel, weeks


def _row(trial: dict, panel: Panel, kind: str, size: float, weeks: float, method: str, est, truth: float,
         noise: float, variant: str = "") -> dict:
    return {"tech": trial["tech"], "trial": trial["trial"], "test": trial["test"], "t0": trial["t0"],
            "controls": "+".join(panel.meta["controls"]), "n_controls": panel.n_controls, "kind": kind, "size": size,
            "weeks": weeks, "method": method, **est.as_dict(), "true": truth,
            "n_pre_days": int((~panel.post_day).sum()),
            "n_post_days": int((panel.post_day & (panel.days < 7 * weeks)).sum()), "noise_pct": noise,
            "variant": variant}


def run_trial(trial: dict, fleet: Fleet | None = None, design: Design | None = None, bsts_iter: int = 1000) -> list:
    fleet = fleet if fleet is not None else _WORKER["fleet"]
    design = design or DESIGNS[trial["tech"]]
    t0 = pd.Timestamp(trial["t0"], tz="UTC")
    rows = []
    for controls in trial["control_sets"]:
        panel, weeks = _panel_for(fleet, design, trial["test"], controls, t0)
        if panel is None:
            continue
        noise = weekly_variability(panel)
        methods = make_methods(trial["seed"], bsts_iter)
        base = {m.name: m.fit(panel, panel.y) for m in methods}
        for kind in design.kinds:
            for size in design.sizes:
                if size == 0 and kind != design.kinds[0]:
                    continue  # the zero effect is the same for both shapes
                sc = Scenario(kind, size)
                y_obs, y_cf = apply(panel, sc)
                for m in methods:
                    state = m.fit(panel, y_obs) if sc.changes_pre else base[m.name]
                    for w in weeks:
                        est = m.estimate(state, panel, y_obs, w)
                        rows.append(_row(trial, panel, kind if size > 0 else "none", size, w, m.name, est,
                                         true_effect(panel, y_obs, y_cf, w), noise))
    return rows


# --- wake spillover -------------------------------------------------------------------------------------------

SPILL_SIZES = (0.0, 0.01, 0.02, 0.03, 0.05)
SPILL_WEEKS = (13, 26, 52)


def draw_spill_trials(fleet: Fleet, design: Design, per_unit: int = 6) -> list[dict]:
    """Every turbine as test, each other turbine alone as control plus all of them together."""
    rng = np.random.default_rng(design.seed + 1)
    base = draw_trials(fleet, replace(design, trials_per_unit=per_unit))
    out = []
    for t in base:
        others = [u for u in fleet.units if u != t["test"]]
        out.append({**t, "control_sets": [[u] for u in others] + [others], "seed": int(rng.integers(0, 2**31 - 1))})
    return out


def wake_exposure(fleet: Fleet, panel: Panel) -> float:
    """Share of the controls' pre-period energy produced while inside the test turbine's disturbed sector."""
    from res_uplift.wake import disturbed

    pre = ~panel.post
    d = disturbed(fleet, panel)
    e = panel.X.sum(axis=1)
    return float(e[pre & d].sum() / e[pre].sum())


def run_spill_trial(trial: dict, fleet: Fleet | None = None, design: Design | None = None,
                    bsts_iter: int = 1000) -> list:
    from res_uplift import wake

    fleet = fleet if fleet is not None else _WORKER["fleet"]
    design = replace(design or DESIGNS["wind"], weeks=SPILL_WEEKS)
    t0 = pd.Timestamp(trial["t0"], tz="UTC")
    rows = []
    for controls in trial["control_sets"]:
        full, weeks = _panel_for(fleet, design, trial["test"], controls, t0)
        if full is None:
            continue
        exposure = wake_exposure(fleet, full)
        sector = full.subset(~wake.disturbed(fleet, full))
        for variant, panel in (("all_records", full), ("sector_filter", sector)):
            if (~panel.post_day).sum() < 60 or panel.post_day.sum() < 7:
                continue
            methods = [m for m in make_methods(trial["seed"], bsts_iter) if m.uses_controls]
            states = {m.name: m.fit(panel, panel.y) for m in methods}
            for size in SPILL_SIZES:
                sc = Scenario("partial_load", size)
                y_obs, y_cf = apply(panel, sc)
                uplift = np.where(panel.post, y_obs / np.maximum(panel.y, 1e-9) - 1, 0.0)
                spilled = replace(panel, X=panel.X * wake.control_factors(fleet, panel, uplift))
                for spill, pnl in (("no_spill", panel), ("spill", spilled)):
                    for m in methods:
                        # the models were fitted on the pre-period only; refresh re-predicts the post-period
                        state = m.refresh(states[m.name], pnl) if spill == "spill" else states[m.name]
                        for w in weeks:
                            est = m.estimate(state, pnl, y_obs, w)
                            r = _row(trial, pnl, "partial_load", size, w, m.name, est,
                                     true_effect(pnl, y_obs, y_cf, w), np.nan, variant=f"{variant}|{spill}")
                            r["wake_exposure"] = exposure
                            rows.append(r)
    return rows


# --- running in parallel ----------------------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(tech: str, data_dir: str) -> None:
    from threadpoolctl import threadpool_limits

    from res_uplift.data import get_fleet

    threadpool_limits(1)
    _WORKER["fleet"] = get_fleet(tech, data_dir)
    logging.basicConfig(level=logging.WARNING)


def run(trials: list[dict], tech: str, data_dir, jobs: int = 1, spill: bool = False, fleet: Fleet | None = None,
        bsts_iter: int = 1000) -> list[dict]:
    func = run_spill_trial if spill else run_trial
    rows: list[dict] = []
    if jobs <= 1:
        if fleet is None:
            from res_uplift.data import get_fleet
            fleet = get_fleet(tech, data_dir)
        for k, t in enumerate(trials):
            rows += func(t, fleet=fleet, bsts_iter=bsts_iter)
            log.info("trial %d/%d done", k + 1, len(trials))
        return rows
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(tech, str(data_dir))) as ex:
        for k, r in enumerate(ex.map(func, trials, chunksize=1)):
            rows += r
            if (k + 1) % max(1, len(trials) // 20) == 0:
                log.info("%s: %d/%d trials done", "spillover" if spill else tech, k + 1, len(trials))
    return rows


# --- metrics ----------------------------------------------------------------------------------------------------

def with_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Copy the zero-effect rows (kind 'none') into every effect shape so that each shape has a size-0 point."""
    parts = [df[df["kind"].ne("none")]]
    for tech, d in df.groupby("tech"):
        zero = d[d["kind"].eq("none")]
        for kind in sorted(set(d["kind"]) - {"none"}):
            parts.append(zero.assign(kind=kind))
    return pd.concat(parts, ignore_index=True)


def summarise(df: pd.DataFrame, by=("tech", "kind", "method", "n_controls", "weeks", "size")) -> pd.DataFrame:
    """Bias, RMSE, interval coverage, success rate (lower 90 % bound > 0) and interval width per group."""
    d = df[np.isfinite(df["effect"])].copy()  # windows without a single valid day give no estimate
    d["err"] = d["effect"] - d["true"]
    d["cov90"] = (d["lo90"] <= d["true"]) & (d["true"] <= d["hi90"])
    d["cov95"] = (d["lo95"] <= d["true"]) & (d["true"] <= d["hi95"])
    d["success"] = d["lo90"] > 0
    d["half90"] = d["effect"] - d["lo90"]
    g = d.groupby(list(by))
    out = pd.DataFrame({
        "n": g.size(),
        "true_mean": g["true"].mean(),
        "bias": g["err"].mean(),
        "rmse": g["err"].apply(lambda e: float(np.sqrt(np.mean(e**2)))),
        "sd_err": g["err"].std(ddof=1),
        "cover90": g["cov90"].mean(),
        "cover95": g["cov95"].mean(),
        "success": g["success"].mean(),
        "half90_median": g["half90"].median(),
    })
    return out.reset_index()


def spill_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Paired spillover metrics per method, number of controls and window.

    ``added_bias``: median of (estimate with spillover - estimate without) / true effect over the same trials;
    ``width_ratio``: median 90 % half-width with the IEC disturbed sectors removed divided by all records
    (no spillover), i.e. the price of the filter; ``exposure``: median wake exposure of the control set.
    """
    d = df[df["size"] > 0]
    key = ["trial", "controls", "method", "weeks", "size"]
    a = d[d["variant"] == "all_records|no_spill"].set_index(key)
    b = d[d["variant"] == "all_records|spill"].set_index(key)
    f = d[d["variant"] == "sector_filter|no_spill"].set_index(key)
    x = pd.DataFrame({"added_bias": (b["effect"] - a["effect"]) / a["true"], "n_controls": a["n_controls"],
                      "exposure": a["wake_exposure"],
                      "width_ratio": (f["effect"] - f["lo90"]) / (a["effect"] - a["lo90"])}).reset_index()
    g = x.groupby(["method", "n_controls", "weeks"])
    return g.agg(added_bias=("added_bias", "median"), width_ratio=("width_ratio", "median"),
                 exposure=("exposure", "median"), n=("added_bias", "size")).reset_index()


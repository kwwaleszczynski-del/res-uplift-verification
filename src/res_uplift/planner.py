"""Test-plan calculator: how long a post-period is needed to show an expected uplift, and with which method.

For a uniform (multiplicative) effect every method here is scale-equivariant: multiplying the test unit's
post-period output by ``1 + delta`` multiplies the estimate and both interval bounds by the same factor
(``1 + bound`` -> ``(1 + bound)(1 + delta)``), because the counterfactual does not use post-period outcomes (or,
for the synthetic-control test inversion, transforms with them). The success rate for any ``delta`` can
therefore be read off the zero-effect trials, without assuming a distribution:

    power(delta, L) = share of zero-effect trials with  (1 + lo90) (1 + delta) > 1,

which reproduces the simulated success rates of the injected sizes exactly (checked in ``build``). A
variability multiplier ``v`` (the site's weekly test/control ratio scatter divided by the reference value of the
trials) scales the lower bounds. Only *trustworthy* windows count: false-success rate at zero effect <= 10 % and
90 % coverage >= 80 % (from the injected uniform effects). The required length is the shortest such window with
power >= target, interpolated log-linearly between the simulated windows; the recommended method is the
trustworthy one needing the shortest window. Partial-load and soiling effects are not multiplicative; their
simulated success rates are reported separately (``power_empirical.csv``).

Usage:
    planner.build(Path("data/validation"), Path("results"))
    weeks = required_weeks(cells_of_one_method, delta=0.01, target=0.9)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from res_uplift.validation import with_zero

log = logging.getLogger(__name__)
MAX_FSR, MIN_COVER = 0.10, 0.80
KEYS = ["tech", "method", "n_controls", "weeks"]


def cells(trials: pd.DataFrame) -> pd.DataFrame:
    """Per (tech, method, controls, window): zero-effect lower bounds, false-success rate, coverage, noise."""
    d = trials[np.isfinite(trials["effect"])]
    zero = d[d["kind"].eq("none")]
    uni = d[d["kind"].isin(["none", "uniform"])]
    cover = uni.assign(c=(uni["lo90"] <= uni["true"]) & (uni["true"] <= uni["hi90"])).groupby(KEYS)["c"].mean()
    out = []
    for key, g in zero.groupby(KEYS):
        # an infinite lower bound never succeeds, and neither does -100 %
        lo = np.sort(np.clip(g["lo90"].to_numpy(), -1.0, None))
        out.append({**dict(zip(KEYS, key)), "fsr": float((lo > 0).mean()), "cover90": float(cover.get(key, np.nan)),
                    "noise_ref": float(g["noise_pct"].median()), "n": len(lo), "lo": lo})
    return pd.DataFrame(out)


def power(cell, delta: float, v: float = 1.0) -> float:
    lo = np.asarray(cell["lo"], dtype=float) * v
    return float(np.mean((1 + lo) * (1 + delta) > 1)) if len(lo) else np.nan


def trustworthy(cell) -> bool:
    return cell["fsr"] <= MAX_FSR and cell["cover90"] >= MIN_COVER


def required_weeks(cs: pd.DataFrame, delta: float, target: float = 0.9, v: float = 1.0,
                   trustworthy_only: bool = True) -> float:
    """Shortest window (weeks) with power >= target for the cells of one (tech, method, n_controls)."""
    cs = cs.sort_values("weeks")
    if trustworthy_only:
        cs = cs[[trustworthy(r) for _, r in cs.iterrows()]]
    if cs.empty:
        return np.inf
    w = cs["weeks"].to_numpy(float)
    p = np.array([power(r, delta, v) for _, r in cs.iterrows()])
    above = np.flatnonzero(p >= target)
    if len(above) == 0:
        return np.inf
    k = above[0]
    if k == 0 or p[k] <= p[k - 1]:
        return float(w[k])
    f = (target - p[k - 1]) / (p[k] - p[k - 1])  # log-linear interpolation between neighbouring windows
    return float(np.exp(np.log(w[k - 1]) + f * (np.log(w[k]) - np.log(w[k - 1]))))


def recommend(cs: pd.DataFrame, tech: str, n_controls: int, delta: float, target: float = 0.9,
              v: float = 1.0) -> tuple[str | None, dict]:
    """Method needing the shortest trustworthy window, and the required weeks of every method."""
    g = cs[(cs["tech"] == tech) & (cs["n_controls"] == n_controls)]
    need = {m: required_weeks(gm, delta, target, v) for m, gm in g.groupby("method")}
    finite = {m: w for m, w in need.items() if np.isfinite(w) and m != "naive"}
    return (min(finite, key=finite.get) if finite else None), need


def empirical_power(trials: pd.DataFrame) -> pd.DataFrame:
    """Share of successes (lower 90 % bound > 0) and mean true effect per cell and injected size.

    Trials without a single valid day in the window (no estimate) are left out, as in ``cells``.
    """
    d = with_zero(trials[np.isfinite(trials["effect"])])
    d = d.assign(success=d["lo90"] > 0)
    g = d.groupby(["tech", "kind", "method", "n_controls", "weeks", "size"])
    return g.agg(success=("success", "mean"), true_mean=("true", "mean"), n=("success", "size")).reset_index()


def weeks_table(cs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in cs.groupby(["tech", "method", "n_controls"]):
        row = dict(zip(["tech", "method", "n_controls"], key))
        for d in (0.01, 0.02):
            tag = f"{int(round(d * 100))}pct"
            row[f"weeks_{tag}_p90"] = required_weeks(g, d, 0.9)
            row[f"weeks_{tag}_p80"] = required_weeks(g, d, 0.8)
            last = g.sort_values("weeks").iloc[-1]
            row[f"power_{tag}_at_{int(last['weeks'])}wk"] = power(last, d)
        rows.append(row)
    return pd.DataFrame(rows)


def build(validation_dir: Path, results_dir: Path) -> None:
    parts = [pd.read_parquet(p) for p in sorted(Path(validation_dir).glob("trials_*.parquet"))]
    if not parts:
        raise FileNotFoundError(f"no trial results in {validation_dir}; run 'validate' first")
    trials = pd.concat(parts, ignore_index=True)
    cs = cells(trials)
    results_dir.mkdir(parents=True, exist_ok=True)
    cs.drop(columns="lo").to_csv(results_dir / "plan_cells.csv", index=False, float_format="%.5g")
    with open(results_dir / "plan_bounds.json", "w", encoding="utf-8") as fh:
        json.dump([{**{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items() if k != "lo"},
                    "lo": np.round(r["lo"], 5).tolist()} for r in cs.to_dict("records")], fh)
    emp = empirical_power(trials)
    emp.to_csv(results_dir / "power_empirical.csv", index=False, float_format="%.5g")
    weeks_table(cs).to_csv(results_dir / "weeks_needed.csv", index=False, float_format="%.4g")
    # check: the equivariance argument reproduces the simulated success rates of the uniform effects
    chk = emp[(emp["kind"] == "uniform") & (emp["size"] > 0)].merge(cs, on=KEYS)
    pred = np.array([power(r, r["size"]) for _, r in chk.iterrows()])
    log.info("placebo-based power vs simulated success (uniform effects): max |difference| %.3f",
             float(np.max(np.abs(pred - chk["success"].to_numpy()))) if len(chk) else np.nan)

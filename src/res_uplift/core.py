"""Shared building blocks: the fleet, the analysis panel, effect estimates and the moving-block bootstrap.

An intervention (upgrade, repair, cleaning) on one *test* unit is judged against *control* units of the same
site that were left alone. Every method works on a :class:`Panel`: the valid records (test unit and every control
in normal operation) of a pre-period before and a post-period after the intervention, with the test unit's power
``y``, the controls' power ``X``, shared covariates ``Z`` (weather and reference signals measured by the controls
or a met station) and bin labels for the binned methods.

The estimand is the relative change of the test unit's energy over the post-window records,

    effect = sum(y_post) / sum(y0_post) - 1,

with ``y0`` the power the test unit would have produced without the intervention.

Usage:
    panel = build_panel(fleet, test=2, controls=[0, 1, 3], pre=(t0 - 365 days, t0), post=(t0, t0 + 182 days))
    est = method.estimate(method.fit(panel, panel.y), panel.y, weeks=13)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NS_PER_DAY = 86_400 * 10**9
NS_PER_HOUR = 3_600 * 10**9


def to_ns(t) -> np.ndarray | int:
    """Datetimes as int64 nanoseconds since the epoch (UTC), whatever resolution pandas chose to store them in."""
    if isinstance(t, (pd.Timestamp, str)):
        return int(pd.Timestamp(t).as_unit("ns").value)
    return pd.DatetimeIndex(t).as_unit("ns").asi8


@dataclass
class Fleet:
    """Synchronous records of several units of one site, as (n_records, n_units) arrays on a regular grid.

    ``unit_signals`` are measured per unit (wind: density-normalised nacelle wind speed ``ws``, nacelle direction
    ``dir``, temperature ``temp``, turbulence intensity ``ti``); ``shared`` signals are common to the site (PV: plane
    of array irradiance ``poa``, ambient temperature ``t_amb``, sun position ``sun_el`` / ``sun_az``).
    """

    kind: str  # "wind" or "pv"
    time: pd.DatetimeIndex
    units: list[str]
    power: np.ndarray
    valid: np.ndarray
    rated: np.ndarray
    unit_signals: dict[str, np.ndarray] = field(default_factory=dict)
    shared: dict[str, np.ndarray] = field(default_factory=dict)
    x: np.ndarray | None = None  # m east (wind layout)
    y: np.ndarray | None = None  # m north
    diameter: float | None = None
    events: pd.DataFrame | None = None

    @property
    def t_ns(self) -> np.ndarray:
        return to_ns(self.time)

    def index(self, unit) -> int:
        return self.units.index(unit) if not isinstance(unit, (int, np.integer)) else int(unit)

    def coverage(self, unit, start, end) -> float:
        """Share of records in [start, end) on which the unit is valid."""
        t = self.t_ns
        sel = (t >= to_ns(start)) & (t < to_ns(end))
        return float(self.valid[sel, self.index(unit)].mean()) if sel.any() else 0.0


@dataclass
class Panel:
    """Valid records of one test unit and its controls around one intervention, pre-period first.

    ``day`` counts days from the intervention: -1 is the last day before it started, 0 the first day after it
    ended. Records of days with too few valid records were dropped when the panel was built.
    """

    kind: str
    t_ns: np.ndarray
    day: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    z_names: list[str]
    bins: np.ndarray  # bin label for the industry method (reference wind speed x direction, or irradiance x sun)
    own_bins: np.ndarray  # bin label from the test unit's own reference (own anemometer / irradiance)
    rated: float
    meta: dict = field(default_factory=dict)
    ref_ws: np.ndarray | None = None
    ref_dir: np.ndarray | None = None

    def __post_init__(self):
        self.post = self.day >= 0
        self.days, self.inv = np.unique(self.day, return_inverse=True)
        self.n_day = np.bincount(self.inv, minlength=len(self.days)).astype(float)
        self.post_day = self.days >= 0

    @property
    def n_controls(self) -> int:
        return self.X.shape[1]

    def window(self, weeks: float) -> np.ndarray:
        """Records of the pre-period plus the first ``weeks`` of the post-period."""
        return ~self.post | (self.day < 7 * weeks)

    def window_days(self, weeks: float) -> np.ndarray:
        """Same as :meth:`window` for the per-day arrays."""
        return ~self.post_day | (self.days < 7 * weeks)

    def daily_mean(self, values: np.ndarray) -> np.ndarray:
        """Mean of record values per day (works for 1-D and 2-D arrays)."""
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            return np.bincount(self.inv, weights=values, minlength=len(self.days)) / self.n_day
        return np.column_stack([self.daily_mean(values[:, k]) for k in range(values.shape[1])])

    def subset(self, keep: np.ndarray) -> "Panel":
        """Panel restricted to the records in ``keep`` (e.g. sectors without wake interaction)."""
        return Panel(kind=self.kind, t_ns=self.t_ns[keep], day=self.day[keep], y=self.y[keep], X=self.X[keep],
                     Z=self.Z[keep], z_names=self.z_names, bins=self.bins[keep], own_bins=self.own_bins[keep],
                     rated=self.rated, meta=dict(self.meta),
                     ref_ws=None if self.ref_ws is None else self.ref_ws[keep],
                     ref_dir=None if self.ref_dir is None else self.ref_dir[keep])


@dataclass
class Estimate:
    """Relative effect with two-sided 90 % and 95 % intervals (0.02 = +2 %)."""

    effect: float
    lo90: float
    hi90: float
    lo95: float
    hi95: float

    @property
    def success(self) -> bool:
        """The claim "the intervention increased output": lower bound of the 90 % interval above zero
        (a one-sided test at the 5 % level)."""
        return self.lo90 > 0.0

    def as_dict(self) -> dict:
        return {"effect": self.effect, "lo90": self.lo90, "hi90": self.hi90, "lo95": self.lo95, "hi95": self.hi95}


def from_draws(point: float, draws: np.ndarray) -> Estimate:
    """Estimate with percentile intervals from bootstrap or posterior draws."""
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 20:
        return Estimate(point, -np.inf, np.inf, -np.inf, np.inf)
    q = np.quantile(d, [0.05, 0.95, 0.025, 0.975])
    return Estimate(float(point), *map(float, q))


def block_counts(n: int, n_rep: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap (Kuensch 1989) of ``n`` consecutive days as an (n_rep, n) matrix of day counts.

    Each replicate glues ceil(n / block) randomly placed blocks of ``block`` consecutive days and keeps the
    first n days; the counts say how often each original day was drawn.
    """
    block = max(1, min(block, n))
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_rep, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_rep, -1)[:, :n]
    counts = np.zeros((n_rep, n))
    np.add.at(counts, (np.repeat(np.arange(n_rep), n), idx.ravel()), 1.0)
    return counts


def circular_mean_deg(angles: np.ndarray, axis: int = 1) -> np.ndarray:
    """Circular mean [0, 360) of angles in degrees, ignoring NaN."""
    a = np.deg2rad(angles)
    s, c = np.nansum(np.sin(a), axis=axis), np.nansum(np.cos(a), axis=axis)
    out = np.rad2deg(np.arctan2(s, c)) % 360.0
    out[np.all(np.isnan(a), axis=axis)] = np.nan
    return out


def build_panel(fleet: Fleet, test, controls, pre, post, min_records: int | None = None,
                extra_keep: np.ndarray | None = None) -> Panel:
    """Panel of the valid records of ``test`` and ``controls`` in the pre and post periods.

    ``pre`` and ``post`` are (start, end) pairs of timestamps; the gap between them (the intervention itself)
    is left out. A record is valid when the test unit and every control are in normal operation and the
    covariates are available. Days with fewer than ``min_records`` valid records are dropped (default: six hours
    for wind, three hours for PV).
    """
    ti, ci = fleet.index(test), [fleet.index(c) for c in controls]
    if ti in ci or not ci:
        raise ValueError("controls must be non-empty and must not contain the test unit")
    t = fleet.t_ns
    pre_a, pre_b, post_a, post_b = (to_ns(x) for x in (*pre, *post))
    in_pre, in_post = (t >= pre_a) & (t < pre_b), (t >= post_a) & (t < post_b)
    keep = (in_pre | in_post) & fleet.valid[:, ti] & fleet.valid[:, ci].all(axis=1)
    if extra_keep is not None:
        keep &= extra_keep
    day = np.where(in_post, np.floor_divide(t - post_a, NS_PER_DAY), np.floor_divide(t - pre_b, NS_PER_DAY))

    if fleet.kind == "wind":
        Z, names, bins, own_bins, ref_ws, ref_dir = _wind_features(fleet, ti, ci)
    else:
        Z, names, bins, own_bins, ref_ws, ref_dir = _pv_features(fleet, ti, ci)
    step_h = float(np.median(np.diff(t[:1000]))) / NS_PER_HOUR
    keep &= np.isfinite(Z).all(axis=1) & np.isfinite(fleet.power[:, ti]) & np.isfinite(fleet.power[:, ci]).all(axis=1)
    if min_records is None:  # a quarter of the day for wind, three daylight hours for PV
        min_records = max(1, int(round((6.0 if fleet.kind == "wind" else 3.0) / step_h)))
    d = day[keep]
    uniq, inv, cnt = np.unique(d, return_inverse=True, return_counts=True)
    ok_day = np.zeros(len(keep), dtype=bool)
    ok_day[np.flatnonzero(keep)] = cnt[inv] >= min_records
    keep &= ok_day

    order = np.flatnonzero(keep)  # records are already time-ordered, so pre comes first
    meta = {"test": fleet.units[ti], "controls": [fleet.units[c] for c in ci], "pre": (pre_a, pre_b),
            "post": (post_a, post_b), "step_h": step_h}
    return Panel(kind=fleet.kind, t_ns=t[order], day=day[order], y=fleet.power[order, ti].astype(float),
                 X=fleet.power[order][:, ci].astype(float), Z=Z[order], z_names=names, bins=bins[order],
                 own_bins=own_bins[order], rated=float(fleet.rated[ti]), meta=meta,
                 ref_ws=None if ref_ws is None else ref_ws[order],
                 ref_dir=None if ref_dir is None else ref_dir[order])


def _hour_features(t_ns: np.ndarray, offset_h: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    h = ((t_ns % NS_PER_DAY) / NS_PER_HOUR + offset_h) % 24.0
    return np.sin(2 * np.pi * h / 24.0), np.cos(2 * np.pi * h / 24.0)


def _wind_features(fleet: Fleet, ti: int, ci: list[int]):
    """Controls' mean wind speed, direction, temperature and turbulence, plus hour of day (a stability proxy)."""
    sig = fleet.unit_signals
    with warnings.catch_warnings():  # records where every control is missing are dropped later
        warnings.simplefilter("ignore", RuntimeWarning)
        ws = np.nanmean(sig["ws"][:, ci], axis=1)
        direction = circular_mean_deg(sig["dir"][:, ci])
        temp = np.nanmean(sig["temp"][:, ci], axis=1)
        ti_c = np.nanmean(sig["ti"][:, ci], axis=1)
    hs, hc = _hour_features(fleet.t_ns)
    rad = np.deg2rad(direction)
    Z = np.column_stack([ws, np.sin(rad), np.cos(rad), temp, np.nan_to_num(ti_c, nan=0.1), hs, hc])
    names = ["ws_ref", "dir_sin", "dir_cos", "temp", "ti", "hour_sin", "hour_cos"]
    ws_bin = np.clip(np.floor(np.nan_to_num(ws) + 0.5), 0, 25)
    sector = np.floor(((np.nan_to_num(direction) + 15.0) % 360.0) / 30.0)
    bins = (ws_bin * 12 + sector).astype(np.int64)
    own = sig["ws"][:, ti]
    own_bins = np.clip(np.floor(np.nan_to_num(own, nan=-1) / 0.5 + 0.5), -1, 60).astype(np.int64)
    return Z, names, bins, own_bins, ws, direction


def _pv_features(fleet: Fleet, ti: int, ci: list[int]):
    """Irradiance, ambient temperature and sun position (all shared by the site's units)."""
    sh = fleet.shared
    az = np.deg2rad(sh["sun_az"])
    Z = np.column_stack([sh["poa"], sh["t_amb"], sh["sun_el"], np.sin(az), np.cos(az)])
    names = ["poa", "t_amb", "sun_el", "sun_az_sin", "sun_az_cos"]
    poa_bin = np.clip(np.floor(np.nan_to_num(sh["poa"]) / 100.0), 0, 14)
    sector = np.floor(np.nan_to_num(sh["sun_az"]) / 30.0) % 12
    bins = (poa_bin * 12 + sector).astype(np.int64)
    own_bins = np.clip(np.floor(np.nan_to_num(sh["poa"]) / 50.0), 0, 30).astype(np.int64)
    return Z, names, bins, own_bins, sh["poa"], None

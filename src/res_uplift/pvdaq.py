"""NREL PVDAQ: the roof systems of one school campus in Las Vegas as a PV fleet of comparable units.

Source: Deline, C., Perry, K., Deceglie, M., Muller, M., Sekulic, W., & Jordan, D. (2021). Photovoltaic Data
Acquisition (PVDAQ) Public Datasets. Open Energy Data Initiative (OEDI), NREL. https://doi.org/10.25984/1846021,
CC BY 4.0. One CSV per system and day in the public bucket ``BASE_URL``; 15-minute values in local standard time
(UTC-8, no daylight saving).

Units (PVDAQ system metadata): Andre Agassi Preparatory Academy, buildings A (system 34, 146.6 kWp, south,
11 deg), B (1276, 68.5 kWp, south, 5 deg), D (1278, 171.4 kWp, south, two inverters D1 and D2) and the gymnasium
G (35, 121.7 kWp, facing west) - one site, one met station (plane-of-array irradiance and temperatures are
identical in all files). Building C (1277) only reports from November 2017 and is not used. Power channels are
in hectowatts until 4 August 2018 and in watts afterwards (the switch is detected, see ``kw_factor``).

A record of a unit is valid with irradiance >= 100 W/m2, no inverter error code and output above 2 % of the
unit's typical power at that irradiance (otherwise the inverter is off). The ambient temperature sensor drops out
for weeks in winter; those gaps are filled with a seasonal x diurnal harmonic fit, as the temperature is only a
covariate.

Usage:
    files = fetch([2016, 2017, 2018], Path("data/raw/pvdaq"))
    fleet = load([2016, 2017, 2018], Path("data/raw/pvdaq"))
"""
from __future__ import annotations

import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from res_uplift.core import Fleet

log = logging.getLogger(__name__)

BASE_URL = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/pvdata"
LATITUDE, LONGITUDE, TZ = 36.1952, -115.1582, "Etc/GMT+8"
# unit -> (system id, AC power channel prefix, inverter error channel prefix)
UNITS = {
    "A": (34, "ac_power_hw", "inverter_error_code"),
    "B": (1276, "ac_power_hw", "inverter_error_code"),
    "D1": (1278, "inv1_ac_power_hw", "inv1_error_code"),
    "D2": (1278, "inv2_ac_power_hw", "inv2_error_code"),
    "G": (35, "ac_power_hw", "inverter_error_code"),
}
SYSTEMS = sorted({s for s, _, _ in UNITS.values()})
MET_PRIORITY = [1276, 34, 1278, 35]  # same sensors in every file; take the first one available
STEP = "15min"


def day_url(system_id: int, day: pd.Timestamp) -> str:
    return (f"{BASE_URL}/system_id={system_id}/year={day.year}/month={day.month}/day={day.day}/"
            f"system_{system_id}__date_{day:%Y_%m_%d}.csv")


def fetch(years: list[int], raw_dir: Path, systems=SYSTEMS, workers: int = 8, refresh: bool = False) -> int:
    """Download the daily files of the given years (cached files are kept); returns the number downloaded."""
    jobs = []
    for sid in systems:
        out = Path(raw_dir) / f"system_{sid}"
        out.mkdir(parents=True, exist_ok=True)
        for day in pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D"):
            path = out / f"system_{sid}__date_{day:%Y_%m_%d}.csv"
            if refresh or not path.exists():
                jobs.append((sid, day, path))

    def get(job):
        sid, day, path = job
        try:
            with urllib.request.urlopen(day_url(sid, day), timeout=60) as r:
                path.write_bytes(r.read())
            return 1
        except urllib.error.HTTPError as e:  # days without data do not exist in the bucket
            if e.code in (403, 404):
                return 0
            raise

    with ThreadPoolExecutor(workers) as pool:
        n = sum(pool.map(get, jobs))
    log.info("PVDAQ: %d of %d missing daily files downloaded into %s", n, len(jobs), raw_dir)
    return n


def _column(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Channel by name without its numeric suffix (``ac_power_hw__2695`` -> ``ac_power_hw``)."""
    cols = [c for c in df.columns if c.rsplit("__", 1)[0] == prefix]
    return df[cols[0]] if cols else pd.Series(np.nan, index=df.index)


def _fahrenheit(x: pd.Series) -> pd.Series:
    """degF -> degC; exactly 0 and -40 degF are sensor dropouts at this site."""
    return ((x - 32) * 5 / 9).where((x != 0) & (x > -40))


def read_system(files: list[Path]) -> pd.DataFrame:
    """Daily CSV files of one system -> frame on UTC timestamps with the raw channels."""
    frames = [pd.read_csv(f) for f in files if f.exists() and f.stat().st_size > 0]
    frames = [f for f in frames if "measured_on" in f.columns]
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    t = pd.to_datetime(raw["measured_on"]).dt.round(STEP).dt.tz_localize(TZ).dt.tz_convert("UTC")
    raw = raw.set_index(t).drop(columns="measured_on")
    return raw[~raw.index.duplicated()].sort_index()


def load(years: list[int], raw_dir: Path) -> Fleet:
    """Fleet of the units in ``UNITS`` on a regular 15-minute UTC grid."""
    import pvlib

    days = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
    systems = {sid: read_system([Path(raw_dir) / f"system_{sid}" / f"system_{sid}__date_{d:%Y_%m_%d}.csv"
                                 for d in days]) for sid in SYSTEMS}
    start = pd.Timestamp(f"{min(years)}-01-01", tz=TZ).tz_convert("UTC")
    end = pd.Timestamp(f"{max(years) + 1}-01-01", tz=TZ).tz_convert("UTC")
    grid = pd.date_range(start, end, freq=STEP, inclusive="left")

    poa, t_amb = pd.Series(np.nan, index=grid), pd.Series(np.nan, index=grid)
    for sid in MET_PRIORITY:
        df = systems[sid].reindex(grid)
        if df.empty:
            continue
        poa = poa.fillna(_column(df, "poa_irradiance"))
        t_amb = t_amb.fillna(_fahrenheit(_column(df, "ambient_temp_f")))
    med = t_amb.rolling(5, center=True, min_periods=3).median()
    t_amb = t_amb.where((t_amb - med).abs() <= 10.0).interpolate(limit=4)
    t_amb = fill_climatology(t_amb)
    poa = poa.clip(lower=0.0)

    power, error = [], []
    for unit, (sid, p_col, e_col) in UNITS.items():
        df = systems[sid].reindex(grid)
        raw = _column(df, p_col)
        power.append((raw * kw_factor(raw, poa)).clip(lower=0.0).to_numpy())
        error.append(_column(df, e_col).to_numpy())
    power, error = np.column_stack(power), np.column_stack(error)

    solpos = pvlib.solarposition.get_solarposition(grid, LATITUDE, LONGITUDE)
    rated = np.nanpercentile(np.where(poa.to_numpy()[:, None] > 600, power, np.nan), 99.5, axis=0)
    valid = typical_output_ok(power, poa.to_numpy(), rated)
    valid &= (np.nan_to_num(error) == 0) & (poa.to_numpy() >= 100.0)[:, None] & np.isfinite(t_amb.to_numpy())[:, None]
    log.info("PVDAQ fleet: %d records, valid share per unit %s", len(grid),
             dict(zip(UNITS, np.round(valid.mean(axis=0) / max((poa >= 100).mean(), 1e-9), 3))))
    return Fleet(kind="pv", time=grid, units=list(UNITS), power=power, valid=valid, rated=rated,
                 shared={"poa": poa.to_numpy(), "t_amb": t_amb.to_numpy(),
                         "sun_el": solpos["apparent_elevation"].to_numpy(), "sun_az": solpos["azimuth"].to_numpy()})


def kw_factor(raw: pd.Series, poa: pd.Series, jump: float = 10.0) -> pd.Series:
    """Per-record factor to kW: 0.1 while the channel reports hectowatts, 0.001 once it reports watts.

    The "_hw" power channels of these systems switch from hectowatts to watts on 4 August 2018 (values jump
    100-fold in the middle of a day). Daylight records (irradiance > 50 W/m2) whose power per irradiance exceeds
    ``jump`` times the median of the whole period look like watts; the switch is the first such record after
    which at least 95 % of daylight records look like watts, and every record from it on is converted as watt.
    """
    ratio = (raw.where(raw > 0) / poa.where(poa > 50)).to_numpy()  # zeros (outages) say nothing about units
    day = np.isfinite(ratio)
    looks_watt = ratio[day] > jump * np.nanmedian(ratio)
    factor = np.full(len(raw), 0.1)
    if looks_watt.any():
        tail_share = np.cumsum(looks_watt[::-1])[::-1] / np.arange(len(looks_watt), 0, -1)
        k = np.flatnonzero(looks_watt & (tail_share >= 0.95))
        if len(k):
            switch = np.flatnonzero(day)[k[0]]
            factor[switch:] = 0.001
            log.info("%s: watt from %s", raw.name, raw.index[switch])
    return pd.Series(factor, index=raw.index)


def fill_climatology(t: pd.Series) -> pd.Series:
    """Fill gaps with a seasonal x diurnal harmonic fit (the ambient sensor drops out for weeks in winter).

    Ambient temperature is only a covariate here; a climatological value keeps those records usable.
    """
    time = t.index.tz_convert(TZ)
    doy = 2 * np.pi * np.asarray(time.day_of_year, dtype=float) / 365.25
    hod = 2 * np.pi * (np.asarray(time.hour, dtype=float) + np.asarray(time.minute, dtype=float) / 60) / 24
    cols = [np.ones(len(t))]
    for k in (1, 2):
        cols += [np.sin(k * doy), np.cos(k * doy), np.sin(k * hod), np.cos(k * hod)]
    cols += [np.sin(doy) * np.sin(hod), np.sin(doy) * np.cos(hod), np.cos(doy) * np.sin(hod), np.cos(doy) * np.cos(hod)]
    A = np.column_stack(cols)
    ok = np.isfinite(t.to_numpy())
    if ok.sum() < 100:
        return t
    coef = np.linalg.lstsq(A[ok], t.to_numpy()[ok], rcond=None)[0]
    return t.fillna(pd.Series(A @ coef, index=t.index))


def typical_output_ok(power: np.ndarray, poa: np.ndarray, rated: np.ndarray, floor: float = 0.02) -> np.ndarray:
    """Output above ``floor`` x the unit's median output in the same 50 W/m2 irradiance bin (inverter on)."""
    ok = np.isfinite(power) & (power > 0)
    bins = np.floor(np.nan_to_num(poa) / 50.0).astype(int)
    for j in range(power.shape[1]):
        p = power[:, j]
        med = pd.Series(np.where(ok[:, j], p, np.nan)).groupby(bins).transform("median").to_numpy()
        ok[:, j] &= ~(p < floor * np.maximum(med, 0.05 * rated[j]))
    return ok

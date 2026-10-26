"""Synthetic fleets with a known truth, for the tests and as a supplementary PV check.

* :func:`wind_fleet` - a row of turbines sharing an autocorrelated wind (log-AR(1) with a seasonal mean and a
  slowly turning direction), each with its own anemometer bias, local gusts, optionally a slowly wandering
  performance (``perf_noise``) and random outages. Power from the Senvion MM92 catalogue curve.
* :func:`pv_fleet` - roofs at one site simulated with pvlib: Haurwitz clear sky, daily clearness drawn from a
  beta distribution plus short-term cloud noise, Erbs decomposition, transposition to the plane of array, SAPM
  cell temperature, PVWatts DC and a clipped inverter. Each roof soils at its own rate; rain days (shared) wash
  all roofs.

Usage:
    fleet = wind_fleet(n_units=4, days=730, seed=1)
    pv = pv_fleet(n_units=4, days=730, seed=1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from res_uplift.core import Fleet

# Senvion MM92 catalogue curve (kW at 1.225 kg/m3, 2..25 m/s), Open Energy Platform turbine library
MM92_WS = np.arange(2.0, 26.0)
MM92_KW = np.array([0.0, 22.0, 93.1, 207.2, 388.2, 642.7, 991.2, 1355.7, 1780.0, 2012.6, 2048.0] + [2055.0] * 13)
RATED_KW = 2050.0


def _ar1(rng, n: int, phi: float, sd: float, size=None) -> np.ndarray:
    shape = (n,) if size is None else (n, size)
    e = rng.standard_normal(shape) * sd * np.sqrt(1 - phi**2)
    x = np.empty(shape)
    x[0] = rng.standard_normal(shape[1:]) * sd
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _outages(rng, n: int, m: int, per_unit: int, mean_len: int) -> np.ndarray:
    valid = np.ones((n, m), dtype=bool)
    for j in range(m):
        for _ in range(per_unit):
            a = rng.integers(0, n)
            valid[a:a + int(rng.exponential(mean_len)) + 1, j] = False
    return valid


def wind_fleet(n_units: int = 4, days: int = 730, start: str = "2018-01-01", freq: str = "1h", seed: int = 0,
               perf_noise: float = 0.0, gust: float = 0.03, speed_bias: float = 0.03, unit: str = "ns") -> Fleet:
    rng = np.random.default_rng(seed)
    time = pd.date_range(start, periods=int(days * pd.Timedelta("1D") / pd.Timedelta(freq)), freq=freq, tz="UTC",
                         unit=unit)
    n = len(time)
    steps_per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))
    doy = np.asarray(time.day_of_year, dtype=float)
    phi = 0.985 ** (24 / steps_per_day)
    ws = np.exp(np.log(7.0) + 0.12 * np.cos(2 * np.pi * (doy - 15) / 365.25) + _ar1(rng, n, phi, 0.45))
    direction = (240 + np.cumsum(rng.standard_normal(n) * 6.0)) % 360.0
    hour = np.asarray(time.hour, dtype=float)
    temp = 10 - 7 * np.cos(2 * np.pi * (doy - 15) / 365.25) + 3 * np.sin(2 * np.pi * (hour - 9) / 24)
    bias = rng.normal(0.0, speed_bias, n_units)
    local = 1.0 + _ar1(rng, n, 0.9 ** (24 / steps_per_day), gust, n_units)  # gusts, decorrelate in hours
    ws_u = ws[:, None] * (1 + bias[None, :]) * local
    gain = rng.normal(1.0, 0.02, n_units)
    perf = 1.0 + _ar1(rng, n, 0.999, perf_noise, n_units)
    power = np.interp(ws_u, MM92_WS, MM92_KW, left=0.0, right=0.0) * gain * perf
    power = np.clip(power * (1 + 0.02 * rng.standard_normal((n, n_units))), 0.0, RATED_KW)
    anemo = ws_u * (1 + 0.02 * rng.standard_normal((n, n_units)))
    dirs = (direction[:, None] + rng.normal(0, 3, n_units)[None, :] + rng.normal(0, 4, (n, n_units))) % 360.0
    valid = _outages(rng, n, n_units, per_unit=max(1, days // 120), mean_len=steps_per_day)
    return Fleet(kind="wind", time=time, units=[f"W{j + 1}" for j in range(n_units)], power=power, valid=valid,
                 rated=np.full(n_units, RATED_KW),
                 unit_signals={"ws": anemo, "dir": dirs, "temp": np.repeat(temp[:, None], n_units, axis=1),
                               "ti": np.full((n, n_units), 0.1)},
                 x=np.arange(n_units) * 400.0, y=np.zeros(n_units), diameter=92.0)


def pv_fleet(n_units: int = 4, days: int = 730, start: str = "2016-01-01", freq: str = "15min", seed: int = 0,
             soiling: bool = True, rain_prob: float = 0.03, unit: str = "ns", latitude: float = 36.2,
             longitude: float = -115.2, tilt: float = 10.0) -> Fleet:
    import pvlib

    rng = np.random.default_rng(seed)
    time = pd.date_range(start, periods=int(days * pd.Timedelta("1D") / pd.Timedelta(freq)), freq=freq, tz="UTC",
                         unit=unit)
    n = len(time)
    solpos = pvlib.solarposition.get_solarposition(time, latitude, longitude)
    zen = solpos["apparent_zenith"].to_numpy()
    cs = pvlib.clearsky.haurwitz(solpos["apparent_zenith"])["ghi"].to_numpy()
    day = (time.as_unit("ns").asi8 // (86_400 * 10**9))
    day = day - day.min()
    kt_day = rng.beta(6.0, 1.2, day.max() + 1)
    steps_per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))
    cloud = np.clip(kt_day[day] + _ar1(rng, n, 0.9 ** (96 / steps_per_day), 0.08), 0.05, 1.05)
    ghi = cs * cloud
    erbs = pvlib.irradiance.erbs(ghi, zen, time)
    poa = pvlib.irradiance.get_total_irradiance(tilt, 180.0, zen, solpos["azimuth"].to_numpy(), erbs["dni"],
                                                ghi, erbs["dhi"])["poa_global"]
    poa = np.nan_to_num(np.asarray(poa, dtype=float))
    doy = np.asarray(time.day_of_year, dtype=float)
    hour = np.asarray(time.hour, dtype=float)
    t_amb = 20 - 11 * np.cos(2 * np.pi * (doy - 15) / 365.25) + 6 * np.sin(2 * np.pi * (hour - 9) / 24)
    wind = np.full(n, 2.0)
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["close_mount_glass_glass"]
    t_cell = pvlib.temperature.sapm_cell(poa, t_amb, wind, **params)
    size = rng.uniform(40, 170, n_units)
    rain = rng.random(day.max() + 1) < rain_prob
    power = np.empty((n, n_units))
    for j in range(n_units):
        dc = pvlib.pvsystem.pvwatts_dc(poa * rng.uniform(0.97, 1.03), t_cell, size[j], -0.004)
        loss = np.zeros(day.max() + 1)
        if soiling:
            rate = rng.uniform(0.0003, 0.0015)
            for d in range(1, len(loss)):
                loss[d] = 0.0 if rain[d] else min(loss[d - 1] + rate, 0.15)
        ac = np.minimum(np.asarray(dc) * 0.96 * (1 - loss[day]), 0.92 * size[j])
        power[:, j] = np.clip(ac * (1 + 0.01 * rng.standard_normal(n)), 0.0, None)
    valid = (poa > 100.0)[:, None] & _outages(rng, n, n_units, per_unit=max(1, days // 180), mean_len=steps_per_day)
    return Fleet(kind="pv", time=time, units=[f"PV{j + 1}" for j in range(n_units)], power=power, valid=valid,
                 rated=0.92 * size,
                 shared={"poa": poa, "t_amb": t_amb, "sun_el": 90.0 - zen, "sun_az": solpos["azimuth"].to_numpy()})

"""Kelmarsh wind farm SCADA (6 x Senvion MM92, 2016-2021) as a fleet of comparable units.

Data: Plumley, C. (2022). Kelmarsh wind farm data (Zenodo, record 16807551 for the 2016-2021 SCADA archives),
CC BY 4.0. Each yearly ``Kelmarsh_SCADA_<year>_<id>.zip`` holds a Greenbyte 10-minute export (~300 signals)
and a status (event) log per turbine; ``Kelmarsh_WT_static.csv`` has positions and hub heights.

Reading and cleaning are a compact port of the author's earlier project wind-power-curve-kelmarsh
(github.com/kwwaleszczynski-del/wind-power-curve-kelmarsh, MIT): records are normal unless they fall before the
commercial operation date, miss wind or power, have an anemometer fault (out of range, frozen, the two sensors
disagree), overlap a stop or power-reduction event of the status log, show blades pitched out at low power for
the wind speed (curtailment), or look iced (cold and far below the turbine's warm-weather curve). The wind speed
is the mean of the two nacelle anemometers (the SCADA "Wind speed" signal changes definition in April 2018) and
is normalised to 1.225 kg/m3 (IEC 61400-12-1).

Usage:
    fleet = load_fleet(range(2016, 2022), raw_dir=Path(os.environ["KELMARSH_DATA_DIR"]),
                       cache_dir=Path("data/processed"))
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from res_uplift.core import Fleet, to_ns

log = logging.getLogger(__name__)

TIME_COL = "# Date and time"
COLUMNS = {
    "Wind speed (m/s)": "ws_scada",
    "Wind speed, Standard deviation (m/s)": "ws_std",
    "Wind speed Sensor 1 (m/s)": "ws1",
    "Wind speed Sensor 2 (m/s)": "ws2",
    "Nacelle position (°)": "nacelle",
    "Power (kW)": "power",
    "Nacelle ambient temperature (°C)": "temp",
    "Blade angle (pitch position) A (°)": "pitch_a",
    "Blade angle (pitch position) B (°)": "pitch_b",
    "Blade angle (pitch position) C (°)": "pitch_c",
}
STATUS_COLUMNS = {"Timestamp start": "start", "Timestamp end": "end", "Status": "status", "Code": "code",
                  "Message": "message", "IEC category": "iec_category"}
TURBINE_RE = re.compile(r"_Kelmarsh_(\d+)_")
STEP = pd.Timedelta("10min")
RATED_KW = 2050.0
EARTH_RADIUS_M = 6_371_000.0
GRID_CATEGORIES = {"Out of Electrical Specification", "Requested Shutdown"}
DERATE_PATTERN = r"P red|externally reduced|[Rr]educed power|[Pp]ower reduction|[Pp]ower limit"


def _header_row(text: str, marker: str) -> int:
    for i, line in enumerate(io.StringIO(text)):
        if line.startswith(marker):
            return i
        if i > 50:
            break
    raise ValueError(f"header line starting with {marker!r} not found")


def read_turbine_csv(text: str, turbine: int) -> pd.DataFrame:
    """One 10-minute export -> frame with the kept signals, columns turbine + timestamp (UTC)."""
    skip = _header_row(text, TIME_COL)
    wanted = {TIME_COL, *COLUMNS}
    df = pd.read_csv(io.StringIO(text), skiprows=skip, usecols=lambda c: c in wanted, na_values=["NaN"])
    out = pd.DataFrame({"timestamp": pd.to_datetime(df[TIME_COL]).dt.tz_localize("UTC")})
    for src, dst in COLUMNS.items():
        out[dst] = df[src].to_numpy(dtype="float32") if src in df else np.float32(np.nan)
    out["pitch"] = out[["pitch_a", "pitch_b", "pitch_c"]].mean(axis=1).astype("float32")
    out = out.drop(columns=["pitch_a", "pitch_b", "pitch_c"])
    out.insert(0, "turbine", np.int8(turbine))
    return out


def read_status_csv(text: str, turbine: int) -> pd.DataFrame:
    skip = _header_row(text, "Timestamp start")
    df = pd.read_csv(io.StringIO(text), skiprows=skip, usecols=list(STATUS_COLUMNS), dtype=str)
    df = df.rename(columns=STATUS_COLUMNS)
    for c in ("start", "end"):  # events still open at export time have end "-"
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d %H:%M:%S", errors="coerce").dt.tz_localize("UTC")
    df["code"] = pd.to_numeric(df["code"], errors="coerce")
    df.insert(0, "turbine", np.int8(turbine))
    return df


def read_zip(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scada, status = [], []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            base = name.rsplit("/", 1)[-1]
            m = TURBINE_RE.search(base)
            if not m or not base.endswith(".csv"):
                continue
            text = z.read(name).decode("utf-8-sig")
            if base.startswith("Turbine_Data"):
                scada.append(read_turbine_csv(text, int(m.group(1))))
            elif base.startswith("Status"):
                status.append(read_status_csv(text, int(m.group(1))))
    if not scada:
        raise ValueError(f"{path.name}: no turbine data files found")
    return pd.concat(scada, ignore_index=True), pd.concat(status, ignore_index=True)


def read_static(raw_dir: Path) -> pd.DataFrame:
    """Positions (m east / north of the farm centre), hub height, elevation, rotor diameter, COD per turbine."""
    df = pd.read_csv(Path(raw_dir) / "Kelmarsh_WT_static.csv", encoding="utf-8-sig")
    lat0, lon0 = np.deg2rad(df["Latitude"].mean()), np.deg2rad(df["Longitude"].mean())
    lat, lon = np.deg2rad(df["Latitude"]), np.deg2rad(df["Longitude"])
    return pd.DataFrame({
        "turbine": df["Title"].str.extract(r"(\d+)$", expand=False).astype(int),
        "x_m": EARTH_RADIUS_M * np.cos(lat0) * (lon - lon0),
        "y_m": EARTH_RADIUS_M * (lat - lat0),
        "hub_height_m": df["Hub Height (m)"],
        "elevation_m": df["Elevation (m)"],
        "diameter_m": df["Rotor Diameter (m)"],
        "cod": pd.to_datetime(df["Commercial Operations Date"], dayfirst=True).dt.tz_localize("UTC"),
    }).set_index("turbine").sort_index()


def load_raw(years, raw_dir: Path, cache_dir: Path, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SCADA records and status events for the years, from the Parquet cache or the archives."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    scada, status = [], []
    for year in years:
        s_path, e_path = cache_dir / f"kelmarsh_scada_{year}.parquet", cache_dir / f"kelmarsh_status_{year}.parquet"
        if s_path.exists() and e_path.exists() and not refresh:
            s, e = pd.read_parquet(s_path), pd.read_parquet(e_path)
        else:
            archives = sorted(Path(raw_dir).glob(f"Kelmarsh_SCADA_{year}_*.zip"))
            if not archives:
                raise FileNotFoundError(f"no Kelmarsh archive for {year} in {raw_dir}")
            s, e = read_zip(archives[0])
            s.to_parquet(s_path)
            e.to_parquet(e_path)
            log.info("Kelmarsh %d: %d records, %d status events", year, len(s), len(e))
        scada.append(s)
        status.append(e)
    s = pd.concat(scada, ignore_index=True).drop_duplicates(["turbine", "timestamp"])
    return s.sort_values(["turbine", "timestamp"]).reset_index(drop=True), \
        pd.concat(status, ignore_index=True).drop_duplicates()


def event_mask(t_ns: np.ndarray, starts, ends) -> np.ndarray:
    """True for 10-minute intervals [t, t + 10 min) overlapping any event [start, end) (open events: one step)."""
    step = int(STEP / pd.Timedelta("1ns"))
    starts = to_ns(pd.DatetimeIndex(starts))
    ends = pd.DatetimeIndex(ends)
    ends = np.where(ends.isna(), starts + step, to_ns(ends.fillna(pd.Timestamp(0, tz="UTC"))))
    first = np.searchsorted(t_ns, starts - step, side="right")
    last = np.searchsorted(t_ns, ends, side="left")
    diff = np.zeros(len(t_ns) + 1, dtype=np.int64)
    np.add.at(diff, first, 1)
    np.add.at(diff, last, -1)
    return np.cumsum(diff[:-1]) > 0


def frozen(x: np.ndarray, n: int = 3) -> np.ndarray:
    """True where a value repeats exactly for at least n consecutive records (a stuck sensor)."""
    same = np.r_[False, x[1:] == x[:-1]]
    run_id = np.cumsum(~same)
    run_len = np.bincount(run_id)[run_id]
    return (run_len >= n) & ~np.isnan(x)


def classify(df: pd.DataFrame, events: pd.DataFrame, cod: pd.Timestamp) -> np.ndarray:
    """Normal-operation flag for the records of ONE turbine (time-sorted frame with the signals)."""
    t = to_ns(df["timestamp"])
    ws, p, pitch = df["ws"].to_numpy(float), df["power"].to_numpy(float), df["pitch"].to_numpy(float)
    bad = (t < to_ns(cod)) | np.isnan(ws) | np.isnan(p)
    ws1, ws2 = df["ws1"].to_numpy(float), df["ws2"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        pair = np.abs(ws1 - ws2) > np.maximum(1.0, 0.15 * 0.5 * (ws1 + ws2))
    bad |= (ws < 0) | (ws > 40) | (frozen(df["ws_scada"].to_numpy(float)) & (ws > 1.0)) | np.nan_to_num(pair)

    is_stop = events["status"].eq("Stop")
    is_derate = events["message"].fillna("").str.contains(DERATE_PATTERN) | \
        events["iec_category"].eq("Partial Performance")
    for sel in (is_stop, is_derate & ~is_stop):
        ev = events[sel]
        if len(ev):
            bad |= event_mask(t, ev["start"], ev["end"])
    bad |= (p <= 0) & (ws >= 4.5)

    # pitched out at low power for the wind speed: derating or curtailment (above 5 m/s)
    b = np.floor(ws / 0.5 + 0.5)
    frame = pd.DataFrame({"b": b, "pitch": pitch, "p": p})
    med_pitch = frame.groupby("b")["pitch"].transform("median").to_numpy()
    med_p = frame.groupby("b")["p"].transform("median").to_numpy()
    bad |= (pitch > med_pitch + 4.0) & (p < 0.85 * med_p) & (ws >= 5.0)

    # icing: cold and below 70 % of the warm-weather median curve
    warm = ~bad & (df["temp"].to_numpy(float) > 5)
    curve = pd.Series(p[warm]).groupby(b[warm]).median()
    expected = np.interp(b, curve.index.to_numpy(float), curve.to_numpy(float), left=0.0, right=curve.iloc[-1])
    bad |= (df["temp"].to_numpy(float) <= 2) & (expected > 100) & (p < 0.7 * expected)
    return ~bad


def air_density(temp_c: np.ndarray, elevation_m: float) -> np.ndarray:
    pressure = 101325.0 * (1.0 - 2.25577e-5 * elevation_m) ** 5.25588
    return pressure / (287.05 * (temp_c + 273.15))


def load_fleet(years, raw_dir: Path, cache_dir: Path, refresh: bool = False) -> Fleet:
    """Fleet arrays on a complete 10-minute grid (cached as one Parquet file per year range)."""
    years = sorted(years)
    cache = cache_dir / f"kelmarsh_fleet_{years[0]}_{years[-1]}.parquet"
    static = read_static(raw_dir)
    if cache.exists() and not refresh:
        wide = pd.read_parquet(cache)
    else:
        scada, status = load_raw(years, raw_dir, cache_dir, refresh)
        parts = []
        for turbine, df in scada.groupby("turbine"):
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["ws"] = df[["ws1", "ws2"]].mean(axis=1, skipna=False).fillna(df["ws_scada"])
            df["normal"] = classify(df, status[status["turbine"].eq(turbine)], static.loc[turbine, "cod"])
            height = static.loc[turbine, "elevation_m"] + static.loc[turbine, "hub_height_m"]
            temp = df["temp"].fillna(df["temp"].median()).to_numpy(float)
            df["ws_n"] = df["ws"] * (air_density(temp, height) / 1.225) ** (1 / 3)
            with np.errstate(divide="ignore", invalid="ignore"):
                df["ti"] = np.where(df["ws_scada"] >= 3.0, df["ws_std"] / df["ws_scada"], np.nan)
            parts.append(df[["turbine", "timestamp", "power", "ws_n", "nacelle", "temp", "ti", "normal"]])
        wide = pd.concat(parts, ignore_index=True)
        wide.to_parquet(cache)
        status.to_parquet(cache_dir / f"kelmarsh_events_{years[0]}_{years[-1]}.parquet")
        log.info("Kelmarsh fleet cached in %s (%d records)", cache, len(wide))
    turbines = sorted(wide["turbine"].unique())
    ts = pd.DatetimeIndex(wide["timestamp"])
    grid = pd.date_range(ts.min(), ts.max(), freq=STEP)

    def arr(col, dtype=float):
        w = wide.pivot(index="timestamp", columns="turbine", values=col)
        return w.reindex(index=grid, columns=turbines).to_numpy(dtype=dtype, na_value=np.nan if dtype is float else 0)

    events_path = cache_dir / f"kelmarsh_events_{years[0]}_{years[-1]}.parquet"
    events = pd.read_parquet(events_path) if events_path.exists() else None
    st = static.loc[turbines]
    normal = arr("normal", dtype=float) > 0.5
    return Fleet(kind="wind", time=grid, units=[f"T{t}" for t in turbines], power=arr("power"), valid=normal,
                 rated=np.full(len(turbines), RATED_KW),
                 unit_signals={"ws": arr("ws_n"), "dir": arr("nacelle"), "temp": arr("temp"), "ti": arr("ti")},
                 x=st["x_m"].to_numpy(), y=st["y_m"].to_numpy(), diameter=float(st["diameter_m"].iloc[0]),
                 events=events)

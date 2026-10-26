"""Data locations and cached fleets.

The Kelmarsh archives are read from ``$KELMARSH_DATA_DIR`` if set, else from ``<data_dir>/raw/kelmarsh``; the
PVDAQ daily files from ``$PVDAQ_DATA_DIR`` or ``<data_dir>/raw/pvdaq``. Prepared fleets are cached in
``<data_dir>/processed``.

Usage:
    fleet = get_fleet("wind", Path("data"))
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from res_uplift.core import Fleet

WIND_YEARS = list(range(2016, 2022))
PV_YEARS = [2016, 2017, 2018]


def kelmarsh_dir(data_dir: Path) -> Path:
    return Path(os.environ.get("KELMARSH_DATA_DIR", Path(data_dir) / "raw" / "kelmarsh"))


def pvdaq_dir(data_dir: Path) -> Path:
    return Path(os.environ.get("PVDAQ_DATA_DIR", Path(data_dir) / "raw" / "pvdaq"))


def get_fleet(tech: str, data_dir: Path, refresh: bool = False) -> Fleet:
    processed = Path(data_dir) / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    if tech == "wind":
        from res_uplift import kelmarsh
        return kelmarsh.load_fleet(WIND_YEARS, kelmarsh_dir(data_dir), processed, refresh=refresh)
    if tech != "pv":
        raise ValueError(f"unknown technology {tech!r}")
    cache = processed / f"pvdaq_fleet_{PV_YEARS[0]}_{PV_YEARS[-1]}.npz"
    if cache.exists() and not refresh:
        z = np.load(cache, allow_pickle=False)
        return Fleet(kind="pv", time=pd.DatetimeIndex(pd.to_datetime(z["time"], utc=True)), units=list(z["units"]),
                     power=z["power"], valid=z["valid"], rated=z["rated"],
                     shared={k: z[k] for k in ("poa", "t_amb", "sun_el", "sun_az")})
    from res_uplift import pvdaq
    fleet = pvdaq.load(PV_YEARS, pvdaq_dir(data_dir))
    np.savez_compressed(cache, time=fleet.t_ns, units=np.array(fleet.units), power=fleet.power, valid=fleet.valid,
                        rated=fleet.rated, **fleet.shared)
    return fleet

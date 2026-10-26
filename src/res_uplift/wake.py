"""Wake spillover: how an upgrade of the test turbine changes the power of controls standing in its wake.

A turbine that extracts more energy also slows the wind behind it more. With the Gaussian wake model of
Bastankhah & Porte-Agel (2014) the velocity deficit at a downstream turbine is

    D = (1 - sqrt(1 - Ct / (8 (sigma/d)^2))) exp(-r^2 / (2 sigma^2)),   sigma/d = k x/d + 0.2 sqrt(beta),
    beta = (1 + sqrt(1 - Ct)) / (2 sqrt(1 - Ct)),

with x the downstream and r the cross-stream distance. An uplift ``u`` of the test turbine's power is assumed to
raise its thrust coefficient by the same fraction (``Ct' = Ct (1 + kappa u)``, kappa = 1 - power and thrust grow
together when the rotor extracts more momentum); a control in partial load then produces
``((1 - D') / (1 - D))^3`` times its observed power. The control looks worse, so the counterfactual is too low
and the effect is overestimated - the spillover the methods have to be protected against.

The disturbed-sector rule of IEC 61400-12-1 (annex A) gives the directions to exclude so that no control sits in
the test turbine's wake: a sector of width ``1.3 arctan(2.5 d / L + 0.15) + 10 deg`` around the direction from
the control to the test turbine (L = distance).

Usage:
    factor = control_factors(fleet, panel, uplift=y_obs / panel.y - 1)   # (n_records, n_controls)
    keep = ~disturbed(fleet, panel)                                      # records with no control in the wake
"""
from __future__ import annotations

import numpy as np

from res_uplift.core import Fleet, Panel
from res_uplift.inject import partial_load_weight

# generic thrust curve of a pitch-regulated 2 MW, 92 m rotor (shape typical of published curves of this class)
CT_WS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25], dtype=float)
CT = np.array([0.85, 0.83, 0.81, 0.80, 0.79, 0.77, 0.73, 0.65, 0.53, 0.42, 0.33, 0.27, 0.22, 0.15, 0.09, 0.05])
K_STAR = 0.04  # wake growth rate for onshore turbulence (Bastankhah & Porte-Agel 2014: 0.02-0.05)


def thrust_coefficient(ws) -> np.ndarray:
    return np.interp(np.asarray(ws, dtype=float), CT_WS, CT, left=0.0, right=CT[-1])


def deficit(x_d, r_d, ct, k: float = K_STAR) -> np.ndarray:
    """Velocity deficit (fraction of free wind) at x_d rotor diameters downstream, r_d diameters off the axis."""
    x_d, r_d, ct = np.broadcast_arrays(np.asarray(x_d, float), np.asarray(r_d, float), np.asarray(ct, float))
    ct = np.clip(ct, 1e-6, 0.95)
    beta = 0.5 * (1 + np.sqrt(1 - ct)) / np.sqrt(1 - ct)
    sigma = k * np.maximum(x_d, 0.0) + 0.2 * np.sqrt(beta)
    c = 1 - np.sqrt(np.clip(1 - ct / (8 * sigma**2), 0.0, 1.0))
    return np.where(x_d > 0, c * np.exp(-(r_d**2) / (2 * sigma**2)), 0.0)


def bearing_deg(x0, y0, x1, y1) -> float:
    """Compass bearing from point 0 to point 1 (degrees clockwise from north)."""
    return float(np.rad2deg(np.arctan2(x1 - x0, y1 - y0)) % 360.0)


def pair_coordinates(fleet: Fleet, test: int, control: int, wind_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Downstream and cross-stream distance (rotor diameters) of the control from the test turbine.

    ``wind_dir`` is the meteorological direction the wind comes from.
    """
    dx, dy = fleet.x[control] - fleet.x[test], fleet.y[control] - fleet.y[test]
    to = np.deg2rad(np.asarray(wind_dir) + 180.0)  # direction the wind blows to
    ex, ey = np.sin(to), np.cos(to)
    x = dx * ex + dy * ey
    r = np.abs(dx * ey - dy * ex)
    return x / fleet.diameter, r / fleet.diameter


def control_factors(fleet: Fleet, panel: Panel, uplift: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    """(n_records, n_controls) factors on the controls' power caused by the test turbine's uplift."""
    test = fleet.index(panel.meta["test"])
    ct = thrust_coefficient(panel.ref_ws)
    ct_new = np.minimum(ct * (1.0 + kappa * np.asarray(uplift)), 0.95)
    out = np.ones(panel.X.shape)
    for j, c in enumerate(panel.meta["controls"]):
        x, r = pair_coordinates(fleet, test, fleet.index(c), panel.ref_dir)
        d0, d1 = deficit(x, r, ct), deficit(x, r, ct_new)
        ratio = ((1 - d1) / (1 - d0)) ** 3
        w = partial_load_weight(panel.X[:, j], fleet.rated[fleet.index(c)])
        out[:, j] = 1.0 + (ratio - 1.0) * w
    return out


def disturbed_sector_width(distance_d: float) -> float:
    """IEC 61400-12-1 annex A: width (degrees) of the sector disturbed by a turbine L = distance_d diameters away."""
    return 1.3 * np.rad2deg(np.arctan(2.5 / distance_d + 0.15)) + 10.0


def disturbed(fleet: Fleet, panel: Panel, max_distance_d: float = 20.0) -> np.ndarray:
    """True for records in which at least one control lies in the test turbine's disturbed sector."""
    test = fleet.index(panel.meta["test"])
    out = np.zeros(len(panel.y), dtype=bool)
    for c in panel.meta["controls"]:
        j = fleet.index(c)
        dist = np.hypot(fleet.x[j] - fleet.x[test], fleet.y[j] - fleet.y[test]) / fleet.diameter
        if dist > max_distance_d:
            continue
        centre = bearing_deg(fleet.x[j], fleet.y[j], fleet.x[test], fleet.y[test])
        half = disturbed_sector_width(dist) / 2
        out |= np.abs((panel.ref_dir - centre + 180.0) % 360.0 - 180.0) <= half
    return out

"""Known effects injected into real data, and the true effect they produce.

Effect shapes (``size`` is a fraction, 0.02 = 2 %):

uniform       post-period power x (1 + size): a power uprate, a repaired string or a corrected loss that adds a
              fixed share of output at every operating point.
partial_load  wind only: power x (1 + size g), g = 1 up to 75 % of rated power and falling linearly to 0 at
              95 %. Vortex generators, leading-edge repairs and pitch or yaw tuning act where the rotor is not
              yet power-limited, so the energy gain is smaller than ``size``.
soiling_wash  PV only: the test unit soils linearly for ``soil_days`` before the intervention, reaching a loss of
              ``size`` at the wash; the wash removes it, soiling resumes at the same rate and saturates again at
              ``size`` (the dirt level a roof settles at between rains). The counterfactual stays at ``size``.
              The wash therefore pays off fully for a few weeks and less over longer windows. Unlike the other
              two shapes this one also changes the pre-period (a downward trend just before the wash), which
              is what a real cleaning campaign looks like.

The true effect of a window is the realised change of the test unit's energy over the post-window records,
``sum(y_obs) / sum(y_cf) - 1``; every method is scored against it.

Usage:
    y_obs, y_cf = apply(panel, Scenario("partial_load", 0.02))
    truth = true_effect(panel, y_obs, y_cf, weeks=13)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from res_uplift.core import NS_PER_DAY, Panel

KINDS = {"wind": ("uniform", "partial_load"), "pv": ("uniform", "soiling_wash")}
SOIL_DAYS = 60


@dataclass(frozen=True)
class Scenario:
    kind: str
    size: float

    @property
    def changes_pre(self) -> bool:
        return self.kind == "soiling_wash" and self.size > 0


def partial_load_weight(power: np.ndarray, rated: float, full: float = 0.75, zero: float = 0.95) -> np.ndarray:
    """1 in partial load, linear fade to 0 between ``full`` and ``zero`` times rated power."""
    return np.clip((zero - np.asarray(power) / rated) / (zero - full), 0.0, 1.0)


def apply(panel: Panel, scenario: Scenario, soil_days: int = SOIL_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """Observed test power with the effect, and the counterfactual (what the test unit would have produced)."""
    y, post, s = panel.y, panel.post, scenario.size
    if scenario.kind == "uniform":
        return np.where(post, y * (1.0 + s), y), y.copy()
    if scenario.kind == "partial_load":
        return np.where(post, y * (1.0 + s * partial_load_weight(y, panel.rated)), y), y.copy()
    if scenario.kind == "soiling_wash":
        pre_end, post_start = panel.meta["pre"][1], panel.meta["post"][0]
        rate = s / soil_days
        tau_pre = (panel.t_ns - pre_end) / NS_PER_DAY  # negative before the wash
        tau_post = (panel.t_ns - post_start) / NS_PER_DAY
        loss_pre = rate * np.clip(tau_pre + soil_days, 0.0, None)
        loss_obs = np.where(post, np.minimum(rate * tau_post, s), loss_pre)
        loss_cf = np.where(post, s, loss_pre)
        return y * (1.0 - np.clip(loss_obs, 0, 0.5)), y * (1.0 - np.clip(loss_cf, 0, 0.5))
    raise ValueError(f"unknown effect kind {scenario.kind!r}")


def true_effect(panel: Panel, y_obs: np.ndarray, y_cf: np.ndarray, weeks: float) -> float:
    sel = panel.post & (panel.day < 7 * weeks)
    return float(y_obs[sel].sum() / y_cf[sel].sum() - 1.0) if sel.any() else np.nan

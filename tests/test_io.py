"""Readers of the Kelmarsh (Greenbyte) and PVDAQ formats on small hand-made files."""
import numpy as np
import pandas as pd

from res_uplift import kelmarsh, pvdaq

HEADER = ('# Date and time,Wind speed (m/s),"Wind speed, Standard deviation (m/s)",Wind speed Sensor 1 (m/s),'
          'Wind speed Sensor 2 (m/s),Nacelle position (°),Power (kW),Nacelle ambient temperature (°C),'
          'Blade angle (pitch position) A (°),Blade angle (pitch position) B (°),Blade angle (pitch position) C (°)')


def _turbine_text(n=300, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2018-03-01", periods=n, freq="10min")
    ws = 4 + 6 * rng.random(n)
    p = np.clip(2050 * ((ws - 3) / 9) ** 3, 0, 2050)
    lines = ["# Turbine export", "# Kelmarsh 1", HEADER]
    for k in range(n):
        lines.append(f"{t[k]:%Y-%m-%d %H:%M:%S},{ws[k]:.2f},1.0,{ws[k] + 0.05:.2f},{ws[k] - 0.05:.2f},250,"
                     f"{p[k]:.1f},8.0,0.5,0.5,0.5")
    return "\n".join(lines) + "\n", t


def test_read_turbine_csv_keeps_signals_and_utc_time():
    text, t = _turbine_text(20)
    df = kelmarsh.read_turbine_csv(text, 3)
    assert len(df) == 20 and (df["turbine"] == 3).all()
    assert str(df["timestamp"].dt.tz) == "UTC" and df["timestamp"].iloc[0] == pd.Timestamp(t[0], tz="UTC")
    assert np.isclose(df["ws_std"].iloc[0], 1.0) and np.isclose(df["pitch"].iloc[0], 0.5)


def test_status_csv_and_event_mask():
    text = ("# Status export\nTimestamp start,Timestamp end,Duration,Status,Code,Message,IEC category\n"
            "2018-03-01 01:05:00,2018-03-01 02:00:00,00:55:00,Stop,20,Manual stop - on site,Scheduled Maintenance\n"
            "2018-03-01 05:00:00,-,-,Stop,3110,Frequency converter error,Forced outage\n")
    ev = kelmarsh.read_status_csv(text, 1)
    assert ev["end"].isna().iloc[1] and ev["code"].iloc[0] == 20
    t = pd.date_range("2018-03-01", periods=48, freq="10min", tz="UTC", unit="us")
    m = kelmarsh.event_mask(kelmarsh.to_ns(t), ev["start"], ev["end"])
    hit = t[m]
    assert hit[0] == pd.Timestamp("2018-03-01 01:00", tz="UTC") and pd.Timestamp("2018-03-01 01:50", tz="UTC") in hit
    assert pd.Timestamp("2018-03-01 02:00", tz="UTC") not in hit
    assert pd.Timestamp("2018-03-01 05:00", tz="UTC") in hit and pd.Timestamp("2018-03-01 05:10", tz="UTC") not in hit


def test_classify_flags_stops_frozen_sensor_and_curtailment():
    text, _ = _turbine_text(300, seed=1)
    df = kelmarsh.read_turbine_csv(text, 1)
    df["ws"] = df[["ws1", "ws2"]].mean(axis=1)
    df.loc[10:14, "ws_scada"] = 7.77  # frozen anemometer
    df.loc[50:60, ["pitch", "power"]] = [20.0, 5.0]  # pitched out at low power
    events = pd.DataFrame({"turbine": [1], "start": [df["timestamp"].iloc[100]], "end": [df["timestamp"].iloc[110]],
                           "status": ["Stop"], "code": [20], "message": ["Manual stop"], "iec_category": ["x"]})
    normal = kelmarsh.classify(df, events, cod=pd.Timestamp("2016-04-15", tz="UTC"))
    assert not normal[10:15].any() and not normal[100:110].any()
    assert not normal[50:61][df["ws"].iloc[50:61].to_numpy() >= 5].any()
    assert normal[200:300].mean() > 0.9


def test_frozen_detects_runs():
    x = np.array([1.0, 2.0, 2.0, 2.0, 3.0, np.nan, np.nan, np.nan])
    assert kelmarsh.frozen(x).tolist() == [False, True, True, True, False, False, False, False]


def _pvdaq_day(path, system, day, p_col, e_col, power_hw, error=0):
    t = pd.date_range(day, periods=96, freq="15min")
    hour = t.hour + t.minute / 60
    poa = np.clip(1000 * np.sin(np.pi * (hour - 6) / 12), 0, None)
    df = pd.DataFrame({"measured_on": t.strftime("%Y-%m-%d %H:%M:%S"),
                       f"{p_col}__1": power_hw * poa / 1000, f"{e_col}__2": error,
                       "poa_irradiance__3": poa, "ambient_temp_f__4": 77.0, "system_id": system})
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / f"system_{system}__date_{pd.Timestamp(day):%Y_%m_%d}.csv", index=False)


def test_pvdaq_load_units_validity_and_scaling(tmp_path):
    (tmp_path / "system_1278").mkdir()
    for day in ("2016-06-14", "2016-06-15"):
        _pvdaq_day(tmp_path / "system_34", 34, day, "ac_power_hw", "inverter_error_code", 1200)
        _pvdaq_day(tmp_path / "system_1276", 1276, day, "ac_power_hw", "inverter_error_code", 500,
                   error=7 if day == "2016-06-15" else 0)
        _pvdaq_day(tmp_path / "system_35", 35, day, "ac_power_hw", "inverter_error_code", 900)
        t = pd.date_range(day, periods=96, freq="15min")
        hour = t.hour + t.minute / 60
        poa = np.clip(1000 * np.sin(np.pi * (hour - 6) / 12), 0, None)
        pd.DataFrame({"measured_on": t.strftime("%Y-%m-%d %H:%M:%S"), "inv1_ac_power_hw__1": 950 * poa / 1000,
                      "inv2_ac_power_hw__2": 360 * poa / 1000, "inv1_error_code__3": 0, "inv2_error_code__4": 0,
                      "poa_irradiance__5": poa, "ambient_temp_f__6": 77.0}).to_csv(
            tmp_path / "system_1278" / f"system_1278__date_{pd.Timestamp(day):%Y_%m_%d}.csv", index=False)
    fleet = pvdaq.load([2016], tmp_path)
    assert fleet.units == ["A", "B", "D1", "D2", "G"]
    noon = fleet.time == pd.Timestamp("2016-06-14 12:00", tz="Etc/GMT+8").tz_convert("UTC")
    assert np.isclose(fleet.power[noon, 0][0], 120.0)  # hectowatt -> kW
    assert np.isclose(fleet.shared["t_amb"][noon][0], 25.0)
    day2 = ((fleet.time >= pd.Timestamp("2016-06-15", tz="Etc/GMT+8"))
            & (fleet.time < pd.Timestamp("2016-06-16", tz="Etc/GMT+8")))
    assert fleet.valid[noon, 0].all() and not fleet.valid[day2, 1].any()  # inverter error code -> invalid
    dark = fleet.shared["poa"] < 100
    assert not fleet.valid[dark].any()

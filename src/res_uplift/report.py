"""README figures (PNG) and the data file of the static dashboard (docs/index.html + docs/data.js).

Usage:
    report.build(Path("data"), Path("results"), Path("docs"))
"""
from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from res_uplift.validation import LABELS  # noqa: E402

log = logging.getLogger(__name__)
# categorical slots in fixed order (validated for colour-vision deficiency on adjacent pairs, light and dark)
COLORS = {"binned": ("#2a78d6", "#3987e5"), "ml": ("#eb6834", "#d95926"), "synth": ("#1baf7a", "#199e70"),
          "bsts": ("#eda100", "#c98500"), "did": ("#e87ba4", "#d55181"), "naive": ("#008300", "#008300")}
ORDER = ["binned", "ml", "synth", "bsts", "did", "naive"]
SHORT = {"binned": "Test-control bins", "ml": "GBM + conformal", "synth": "Synthetic control + CWZ",
         "bsts": "BSTS (CausalImpact)", "did": "Diff-in-diff", "naive": "Naive before/after"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
KIND_LABEL = {"uniform": "uniform", "partial_load": "partial-load (vortex generator type)",
              "soiling_wash": "soiling, then wash"}


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)


def _figure(nrows, ncols, w=10.5, h=6.2):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _weeks_axis(ax, weeks):
    ax.set_xscale("log")
    ax.set_xticks(weeks)
    ax.set_xticklabels([str(w) for w in weeks])
    ax.minorticks_off()


def _legend(fig, methods, ncol=6, extra=()):
    handles = [plt.Line2D([], [], color=COLORS[m][0], marker="o", markersize=5, linewidth=1.8) for m in methods]
    labels = [SHORT[m] for m in methods]
    for h, lab in extra:  # line styles that are not methods
        handles.append(h)
        labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=8.5, labelcolor=INK2)


def max_controls(df: pd.DataFrame, tech: str) -> int:
    return int(df[df["tech"] == tech]["n_controls"].max())


def fig_method_comparison(summary: pd.DataFrame, out: Path) -> None:
    """RMSE, 90 % coverage and false-success rate against window length, all controls, uniform effects."""
    techs = [t for t in ("wind", "pv") if (summary["tech"] == t).any()]
    fig, axes = _figure(len(techs), 3, h=3.3 * len(techs))
    for r, tech in enumerate(techs):
        s = summary[(summary["tech"] == tech) & (summary["kind"] == "uniform")
                    & (summary["n_controls"] == max_controls(summary, tech))]
        weeks = sorted(s["weeks"].unique())
        for m in ORDER:
            g = s[s["method"] == m]
            allsz = g.groupby("weeks").apply(lambda x: np.average(x["rmse"] ** 2, weights=x["n"]) ** 0.5,
                                             include_groups=False)
            cov = g.groupby("weeks").apply(lambda x: np.average(x["cover90"], weights=x["n"]), include_groups=False)
            fsr = g[g["size"] == 0].set_index("weeks")["success"]
            kw = dict(color=COLORS[m][0], marker="o", markersize=3.5, linewidth=1.6)
            axes[r, 0].plot(allsz.index, 100 * allsz.values, **kw)
            axes[r, 1].plot(cov.index, 100 * cov.values, **kw)
            axes[r, 2].plot(fsr.index, 100 * fsr.values, **kw)
        name = "Wind (Kelmarsh, 5 controls)" if tech == "wind" else "PV (PVDAQ Las Vegas, all controls)"
        xlab = "post-period window (weeks)" if r == len(techs) - 1 else None
        _style(axes[r, 0], f"{name}: RMSE", xlab, "RMSE (% points)")
        _style(axes[r, 1], "90 % interval coverage", xlab, "coverage (%)")
        _style(axes[r, 2], "false 'success' at zero effect", xlab, "share of trials (%)")
        axes[r, 1].axhline(90, color=INK2, linewidth=0.8, linestyle="--")
        axes[r, 2].axhline(5, color=INK2, linewidth=0.8, linestyle="--")
        axes[r, 1].set_ylim(0, 102)
        for ax in axes[r]:
            _weeks_axis(ax, weeks)
        axes[r, 0].set_yscale("log")
        ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10) if axes[r, 0].get_ylim()[0] <= t <= axes[r, 0].get_ylim()[1]]
        axes[r, 0].set_yticks(ticks)
        axes[r, 0].set_yticklabels([f"{t:g}" for t in ticks])
        axes[r, 0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    _legend(fig, ORDER)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_power(summary: pd.DataFrame, out: Path) -> None:
    """Empirical power (share of trials with lower 90 % bound > 0) for +1 % and +2 % uniform effects."""
    techs = [t for t in ("wind", "pv") if (summary["tech"] == t).any()]
    fig, axes = _figure(len(techs), 2, h=3.1 * len(techs))
    for r, tech in enumerate(techs):
        s = summary[(summary["tech"] == tech) & (summary["kind"] == "uniform")
                    & (summary["n_controls"] == max_controls(summary, tech))]
        weeks = sorted(s["weeks"].unique())
        for c, size in enumerate([0.01, 0.02]):
            ax = axes[r, c]
            for m in ORDER:
                g = s[(s["method"] == m) & np.isclose(s["size"], size)].sort_values("weeks")
                ax.plot(g["weeks"], 100 * g["success"], color=COLORS[m][0], marker="o", markersize=3.5,
                        linewidth=1.6)
            ax.axhline(90, color=INK2, linewidth=0.8, linestyle="--")
            _style(ax, f"{'Wind' if tech == 'wind' else 'PV'}: true uplift +{size:.0%}",
                   "post-period window (weeks)" if r == len(techs) - 1 else None, "power (% of trials detected)")
            _weeks_axis(ax, weeks)
            ax.set_ylim(0, 102)
    _legend(fig, ORDER)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_spillover(spill: pd.DataFrame, out: Path) -> None:
    """Bias added by wake spillover (paired) against wake exposure, and the price of the sector filter."""
    from res_uplift.validation import spill_summary

    d = spill[(spill["size"] > 0) & (spill["weeks"] == 26)]
    key = ["trial", "controls", "method", "weeks", "size"]
    a = d[d["variant"] == "all_records|no_spill"].set_index(key)
    b = d[d["variant"] == "all_records|spill"].set_index(key)
    x = pd.DataFrame({"added": (b["effect"] - a["effect"]) / a["true"], "exp": a["wake_exposure"],
                      "J": a["n_controls"]}).reset_index()
    fig, axes = _figure(1, 2, h=3.9)
    methods = ["binned", "ml", "synth", "bsts", "did"]
    for m in methods:
        g = x[x["method"] == m].groupby(["trial", "controls"]).agg(e=("exp", "mean"), a=("added", "median"),
                                                                    J=("J", "first"))
        axes[0, 0].scatter(100 * g["e"], 100 * g["a"], s=np.where(g["J"] > 1, 26, 12), color=COLORS[m][0],
                           edgecolor=SURFACE, linewidth=0.8, alpha=0.85)
    axes[0, 0].axhline(0, color=AXIS, linewidth=0.8)
    _style(axes[0, 0], "Bias added by wake spillover (26 weeks; large dots: all five controls)",
           "controls' energy produced inside the test turbine's disturbed sector (%)",
           "added bias (% of the true effect)")
    summ = spill_summary(spill)
    summ = summ[summ["weeks"] == 26]
    width = 0.16
    for k, m in enumerate(methods):
        vals = [summ[(summ["method"] == m) & (summ["n_controls"] == j)]["width_ratio"].median() for j in (1, 5)]
        axes[0, 1].bar(np.arange(2) + (k - 2) * width, vals, width * 0.88, color=COLORS[m][0])
    axes[0, 1].axhline(1, color=AXIS, linewidth=0.8)
    axes[0, 1].set_xticks(range(2))
    axes[0, 1].set_xticklabels(["one control", "all five controls"], fontsize=8.5)
    _style(axes[0, 1], "Price of removing the disturbed sectors (26 weeks)", None,
           "90 % interval width, filtered / all records")
    _legend(fig, methods, ncol=5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_cases(cases: pd.DataFrame, out: Path) -> None:
    """Forest plot: every method's estimate and 90 % interval for each event, longest common window."""
    ids = list(dict.fromkeys(cases["case"]))
    fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(ids) * 1.6 + 1.2))
    fig.patch.set_facecolor(SURFACE)
    yticks, ylabels = [], []
    for i, cid in enumerate(ids):
        c = cases[cases["case"] == cid]
        w = c["weeks"].max()
        c = c[c["weeks"] == w]
        base = -i * (len(ORDER) + 2)
        for k, m in enumerate(ORDER):
            r = c[c["method"] == m]
            if r.empty:
                continue
            r = r.iloc[0]
            y = base - k
            if "placebo_p05" in r and np.isfinite(r["placebo_p05"]):  # what the method says when nothing happened
                ax.fill_between([100 * r["placebo_p05"], 100 * r["placebo_p95"]], y - 0.4, y + 0.4, color=GRID,
                                linewidth=0)
            lo, hi = max(r["lo90"], -0.3), min(r["hi90"], 0.3)
            ax.plot([100 * lo, 100 * hi], [y, y], color=COLORS[m][0], linewidth=1.6)
            ax.plot(100 * r["effect"], y, "o", color=COLORS[m][0], markersize=4.5)
        yticks.append(base - 2.5)
        ylabels.append(f"{cid} ({w:g} wk)\n" + textwrap.fill(c["label"].iloc[0], 34))
    ax.axvline(0, color=AXIS, linewidth=0.9)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7.5, color=INK2)
    _style(ax, "Real events: change of the unit's output after it came back (90 % intervals; grey: placebo range)",
           "estimated change of energy (%)", None)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-12, 12)
    _legend(fig, ORDER)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_case_series(series: dict, out: Path) -> None:
    """Weekly mean power of the test unit and two counterfactuals around one event."""
    df = pd.DataFrame({k: series[k] for k in ("date", "actual", "synth", "ml", "post")})
    df["date"] = pd.to_datetime(df["date"])
    w = df.set_index("date").resample("W").mean()
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    fig.patch.set_facecolor(SURFACE)
    ax.plot(w.index, w["actual"], color=INK, linewidth=1.6, label=f"{series['unit']} measured")
    ax.plot(w.index, w["ml"], color=COLORS["ml"][0], linewidth=1.6, label="GBM counterfactual")
    ax.plot(w.index, w["synth"], color=COLORS["synth"][0], linewidth=1.6, label="synthetic control")
    ax.axvspan(pd.Timestamp(series["start"]).tz_localize(None), pd.Timestamp(series["end"]).tz_localize(None),
               color=GRID, alpha=0.8, linewidth=0)
    _style(ax, f"{series['id']}: {series['label']}\nweekly mean of valid records; before the event the "
           "counterfactuals are out-of-sample", None, "mean power (kW)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_weeks(bounds: list[dict], out: Path) -> None:
    """Power for +1 % and +2 % uniform effects against window length, all controls (trustworthy windows solid)."""
    from res_uplift import planner

    cs = pd.DataFrame(bounds)
    techs = [t for t in ("wind", "pv") if (cs["tech"] == t).any()]
    fig, axes = _figure(1, 2 * len(techs), w=10.5, h=3.6)
    methods = [m for m in ORDER if m != "naive"]
    for t, tech in enumerate(techs):
        J = int(cs[cs["tech"] == tech]["n_controls"].max())
        for c, delta in enumerate((0.01, 0.02)):
            ax = axes[0, 2 * t + c]
            for m in methods:
                g = cs[(cs["tech"] == tech) & (cs["method"] == m) & (cs["n_controls"] == J)].sort_values("weeks")
                p = np.array([planner.power(r, delta) for _, r in g.iterrows()])
                ok = np.array([planner.trustworthy(r) for _, r in g.iterrows()])
                ax.plot(g["weeks"], 100 * p, color=COLORS[m][0], linewidth=1.0, linestyle=":")
                ax.plot(g["weeks"], np.where(ok, 100 * p, np.nan), color=COLORS[m][0], linewidth=1.8, marker="o",
                        markersize=3.5)
            ax.axhline(90, color=INK2, linewidth=0.8, linestyle="--")
            ax.axhline(80, color=AXIS, linewidth=0.8, linestyle="--")
            _weeks_axis(ax, sorted(g["weeks"].unique()))
            ax.set_ylim(0, 102)
            _style(ax, f"{'Wind' if tech == 'wind' else 'PV'}, {J} controls: +{delta:.0%}", "weeks",
                   "power (%)" if c == 0 and t == 0 else None)
    styles = [(plt.Line2D([], [], color=INK2, linewidth=1.0, linestyle=":"),
               f"not trustworthy at that window (false success > {100 * planner.MAX_FSR:.0f} % or coverage "
               f"< {100 * planner.MIN_COVER:.0f} %)"),
              (plt.Line2D([], [], color=INK2, linewidth=0.8, linestyle="--"), "80 % and 90 % power")]
    _legend(fig, methods, ncol=4, extra=styles)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def dashboard_data(results: Path, out: Path) -> None:
    def records(df, cols, digits=5):
        d = df[cols].copy()
        for c in d.columns:
            if d[c].dtype.kind == "f":
                d[c] = d[c].round(digits).replace([np.inf, -np.inf], np.nan)
        return json.loads(d.to_json(orient="records"))

    data = {"methods": [{"id": m, "label": LABELS[m], "short": SHORT[m], "light": COLORS[m][0],
                         "dark": COLORS[m][1]} for m in ORDER],
            "kinds": KIND_LABEL}
    parts = [pd.read_csv(p) for p in sorted(results.glob("validation_*.csv"))]
    if parts:
        s = pd.concat(parts, ignore_index=True)
        data["summary"] = records(s, ["tech", "kind", "method", "n_controls", "weeks", "size", "n", "true_mean", "bias",
                                      "rmse", "cover90", "cover95", "success", "half90_median"])
    if (results / "plan_bounds.json").exists():
        data["cells"] = json.loads((results / "plan_bounds.json").read_text(encoding="utf-8"))
    if (results / "spillover_summary.csv").exists():
        sp = pd.read_csv(results / "spillover_summary.csv")
        data["spill"] = records(sp, ["method", "n_controls", "weeks", "added_bias", "width_ratio", "exposure", "n"])
    if (results / "case_studies.csv").exists():
        cs = pd.read_csv(results / "case_studies.csv")
        cols = [c for c in ["case", "tech", "unit", "label", "method", "weeks", "effect", "lo90", "hi90", "lo95",
                            "hi95", "n_controls", "controls", "n_pre_days", "n_post_days", "placebo_p05",
                            "placebo_p95", "placebo_n"] if c in cs.columns]
        data["cases"] = records(cs, cols)
        series = json.loads((results / "case_series.json").read_text(encoding="utf-8"))
        weekly = []
        for s_ in series:
            df = pd.DataFrame({k: s_[k] for k in ("date", "actual", "synth", "ml")})
            df["date"] = pd.to_datetime(df["date"])
            w = df.set_index("date").resample("W").mean().dropna(how="all")
            weekly.append({k: s_[k] for k in ("id", "label", "unit", "tech", "start", "end", "controls")}
                          | {"date": [d.strftime("%Y-%m-%d") for d in w.index],
                             **{k: np.round(w[k].to_numpy(), 1).tolist() for k in ("actual", "synth", "ml")}})
        data["series"] = weekly
    out.write_text("window.DATA = " + json.dumps(data, separators=(",", ":"), allow_nan=False).replace("NaN", "null")
                   + ";\n", encoding="utf-8")


def build(data_dir: Path, results_dir: Path, docs_dir: Path) -> None:
    img = docs_dir / "img"
    img.mkdir(parents=True, exist_ok=True)
    parts = [pd.read_csv(p) for p in sorted(results_dir.glob("validation_*.csv"))]
    if parts:
        summary = pd.concat(parts, ignore_index=True)
        fig_method_comparison(summary, img / "method_comparison.png")
        fig_power(summary, img / "power_curves.png")
    if (results_dir / "plan_bounds.json").exists():
        fig_weeks(json.loads((results_dir / "plan_bounds.json").read_text(encoding="utf-8")), img / "weeks_needed.png")
    spill_path = data_dir / "validation" / "spillover_wind.parquet"
    if spill_path.exists():
        fig_spillover(pd.read_parquet(spill_path), img / "spillover.png")
    if (results_dir / "case_studies.csv").exists():
        fig_cases(pd.read_csv(results_dir / "case_studies.csv"), img / "case_studies.png")
        series = json.loads((results_dir / "case_series.json").read_text(encoding="utf-8"))
        for s in series:
            if s["id"] in ("T6-2021", "A-2016"):
                fig_case_series(s, img / f"case_{s['id'].replace('-', '_').lower()}.png")
    dashboard_data(results_dir, docs_dir / "data.js")
    log.info("figures in %s, dashboard data in %s", img, docs_dir / "data.js")

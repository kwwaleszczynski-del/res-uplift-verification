"""Command line interface.

Usage:
    python -m res_uplift fetch-pv                  # PVDAQ daily files of the four Las Vegas systems, 2016-2018 (~50 MB)
    python -m res_uplift prepare                   # read Kelmarsh archives and PVDAQ files into cached fleets
    python -m res_uplift validate --jobs 10        # injected-effect validation, wind and PV (~10 + 15 min)
    python -m res_uplift spillover --jobs 10       # wake spillover experiment (wind)
    python -m res_uplift cases                     # case studies on real events from the logs
    python -m res_uplift plan                      # test-plan tables (weeks needed per method, effect, controls)
    python -m res_uplift report                    # README figures and the HTML dashboard in docs/
    python -m res_uplift all --jobs 10             # everything after fetch-pv

The Kelmarsh archives (Zenodo 16807551, ~1.8 GB) are read from $KELMARSH_DATA_DIR if set, else from
data/raw/kelmarsh; PVDAQ files from $PVDAQ_DATA_DIR or data/raw/pvdaq.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

STAGES = ("prepare", "validate", "spillover", "cases", "plan", "report")
log = logging.getLogger("res_uplift")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="res-uplift", description="Causal verification of upgrades and O&M "
                                "interventions on wind turbines and PV systems, validated with injected effects.")
    p.add_argument("command", choices=("fetch-pv", *STAGES, "all"))
    p.add_argument("--tech", choices=("wind", "pv", "all"), default="all", help="technology for validate")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="raw data, caches and trial results")
    p.add_argument("--results-dir", type=Path, default=Path("results"), help="tables committed with the repo")
    p.add_argument("--docs-dir", type=Path, default=Path("docs"), help="dashboard and README figures")
    p.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    p.add_argument("--trials-per-unit", type=int, default=None, help="override the number of trials per unit")
    p.add_argument("--bsts-iter", type=int, default=1000, help="MCMC iterations of the BSTS sampler")
    p.add_argument("--refresh", action="store_true", help="rebuild caches and rerun cached stages")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def techs(args) -> list[str]:
    return ["wind", "pv"] if args.tech == "all" else [args.tech]


def cmd_fetch_pv(args) -> None:
    from res_uplift import data, pvdaq
    pvdaq.fetch(data.PV_YEARS, data.pvdaq_dir(args.data_dir))


def cmd_prepare(args) -> None:
    from res_uplift.data import get_fleet
    for tech in ("wind", "pv"):
        f = get_fleet(tech, args.data_dir, refresh=args.refresh)
        log.info("%s fleet: %d records x %d units, valid share %s", tech, *f.power.shape,
                 dict(zip(f.units, f.valid.mean(axis=0).round(3))))


def _trials_path(args, tech: str, spill: bool = False) -> Path:
    return args.data_dir / "validation" / (f"spillover_{tech}.parquet" if spill else f"trials_{tech}.parquet")


def cmd_validate(args, spill: bool = False) -> None:
    from dataclasses import replace

    from res_uplift.data import get_fleet
    from res_uplift.validation import (DESIGNS, draw_spill_trials, draw_trials, run, spill_summary, summarise,
                                      with_zero)

    for tech in (["wind"] if spill else techs(args)):
        out = _trials_path(args, tech, spill)
        if out.exists() and not args.refresh:
            log.info("%s exists, skipping (use --refresh to rerun)", out)
            continue
        design = DESIGNS[tech]
        if args.trials_per_unit:
            design = replace(design, trials_per_unit=args.trials_per_unit)
        fleet = get_fleet(tech, args.data_dir)
        trials = draw_spill_trials(fleet, design, per_unit=args.trials_per_unit or 6) if spill \
            else draw_trials(fleet, design)
        log.info("%s%s: %d trials, %d control sets", tech, " spillover" if spill else "", len(trials),
                 sum(len(t["control_sets"]) for t in trials))
        t = time.time()
        rows = run(trials, tech, args.data_dir, jobs=args.jobs, spill=spill, bsts_iter=args.bsts_iter,
                   fleet=fleet if args.jobs <= 1 else None)
        df = pd.DataFrame(rows)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        log.info("%d rows in %.0f min -> %s", len(df), (time.time() - t) / 60, out)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        if spill:
            spill_summary(df).to_csv(args.results_dir / "spillover_summary.csv", index=False, float_format="%.5g")
        else:
            summarise(with_zero(df)).to_csv(args.results_dir / f"validation_{tech}.csv", index=False,
                                            float_format="%.5g")
            with open(args.results_dir / f"trials_{tech}.json", "w", encoding="utf-8") as fh:
                json.dump(trials, fh, indent=1)


def cmd_cases(args) -> None:
    from res_uplift import cases
    cases.run_all(args.data_dir, args.results_dir, bsts_iter=args.bsts_iter)


def cmd_plan(args) -> None:
    from res_uplift import planner
    planner.build(args.data_dir / "validation", args.results_dir)


def cmd_report(args) -> None:
    from res_uplift import report
    report.build(args.data_dir, args.results_dir, args.docs_dir)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    steps = {"fetch-pv": cmd_fetch_pv, "prepare": cmd_prepare, "validate": cmd_validate,
             "spillover": lambda a: cmd_validate(a, spill=True), "cases": cmd_cases, "plan": cmd_plan,
             "report": cmd_report}
    for name in (STAGES if args.command == "all" else (args.command,)):
        log.info("== %s", name)
        steps[name](args)

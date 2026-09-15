"""Measure mixedlm-rs, statsmodels and lme4 on the same data, on one machine.

`bench/performance.py` times this package against statsmodels and carries
lme4 forward as a *historical* number from another run, because it has no R.
This closes that gap when R is present: all three packages fit byte-identical
fixtures in the same session, so the lme4 column is a measurement rather than
a citation.

    python bench/three_way.py                    # writes bench/three_way.json
    python bench/three_way.py --quick            # 3 timing cases, 20 fixtures

Two experiments
---------------
**timing** -- the scaling cases from performance.py: a random intercept and
slope on 2,000 to 500,264 rows. Each package is timed from a data frame in
memory to a fitted model, `--reps` times, minimum reported, every repetition
kept. Before any timing is reported, each Python fit is checked against lme4
with the tolerances in `bench/tolerances.py`: a fast wrong answer is not a
benchmark result.

**accuracy** -- the 120 randomised fixtures from `differential_table.py`,
same generator, same seeds: badly scaled predictors, near-zero variances,
unbalanced groups, REML and ML. These are where optimisers disagree, so this is
where "matches lme4" is worth measuring. For each fixture it records how far
each Python package's optimum lands from lme4's, in deviance units.

lme4 is a reference here, not ground truth. Its optimiser can stop short too,
so a package can land *above* it, and the output keeps the sign rather than
folding that into "disagrees".

Data
----
Each fixture is written to CSV once, and **both sides read that file**:
lme4 through `read.csv`, the Python packages through `pd.read_csv`. The
Python fits do not use the in-memory frame the file was written from, so a
lossy round trip could not make the two sides fit different numbers.

No R, no number
---------------
Without an Rscript this exits before measuring anything. It does not fall back
to the historical lme4 figures: those are already in performance.json, labelled
as what they are.

A busy machine, no timing
-------------------------
The first full run of this script was recorded while other jobs held the CPU
at 100%. Every agreement check passed and it exited 0 -- with lme4 taking
longer on 200,000 rows than on 500,264, and this package ten times slower than
its own earlier record at the largest size. Contention is not constant, so it
distorts even the *ratios* between packages, and nothing in the output said
so.

So before each timed case the background CPU load is sampled with nothing of
ours running, and recorded. Above `MAX_BACKGROUND_BUSY` the timing experiment
stops and writes no timings. Accuracy is unaffected by load -- an optimiser's
end point does not depend on how fast it got there -- so the two experiments
are recorded separately (`--part`), and a busy machine costs only the timings.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tests"))
warnings.filterwarnings("ignore")

import mixedlm_rs as mlm
import statsmodels.formula.api as smf
import tolerances
from performance import CASES, environment, make
from test_fuzz import random_case

OUT = HERE / "three_way.json"
ACCURACY_SEEDS = 120                       # as differential_table.py
ACCURACY_SEED_BASE = 10_000                # as differential_table.py
# statsmodels is timed once above this, as in performance.py: a single fit is
# minutes, and repeating it sharpens nothing about a ratio in the hundreds.
SM_SINGLE_REP_ABOVE = 50_000
# Fraction of all logical CPUs busy with *other* work, sampled while nothing of
# ours runs. All three packages parallelise (rayon here, BLAS under statsmodels
# and lme4), so a machine that is a sixth occupied is already not the machine
# the numbers claim to describe.
MAX_BACKGROUND_BUSY = 0.15
LOAD_SAMPLE_SECONDS = 2.0


class BusyMachine(RuntimeError):
    pass


def cpu_busy_fraction(seconds=LOAD_SAMPLE_SECONDS):
    """System-wide CPU busy fraction over `seconds`, or None if unmeasurable."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class FileTime(ctypes.Structure):
            _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

        def snapshot():
            idle, kernel, user = FileTime(), FileTime(), FileTime()
            ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
            value = lambda ft: (ft.high << 32) | ft.low      # noqa: E731
            # Kernel time includes idle time on Windows.
            return value(idle), value(kernel) + value(user)

        idle0, total0 = snapshot()
        time.sleep(seconds)
        idle1, total1 = snapshot()
    elif os.path.exists("/proc/stat"):
        def snapshot():
            with open("/proc/stat") as fh:
                fields = [int(v) for v in fh.readline().split()[1:]]
            return fields[3] + fields[4], sum(fields)       # idle + iowait

        idle0, total0 = snapshot()
        time.sleep(seconds)
        idle1, total1 = snapshot()
    else:
        return None
    total = total1 - total0
    return None if total <= 0 else 1.0 - (idle1 - idle0) / total


def require_quiet(case, samples):
    busy = cpu_busy_fraction()
    samples.append({"case": case, "background_busy": busy})
    if busy is not None and busy > MAX_BACKGROUND_BUSY:
        raise BusyMachine(
            f"before {case}: {busy:.0%} of the CPU is busy with other work "
            f"(limit {MAX_BACKGROUND_BUSY:.0%}). Timings taken now would "
            "describe the contention, not the packages.")
    return busy


def find_rscript() -> str | None:
    explicit = os.environ.get("RSCRIPT")
    if explicit:
        return explicit
    found = shutil.which("Rscript")
    if found:
        return found
    # R's Windows installer does not put itself on PATH.
    candidates = sorted(glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe"))
    return candidates[-1] if candidates else None


def run_lme4(rscript, manifest_rows, workdir, label):
    manifest = pathlib.Path(workdir) / f"{label}-manifest.csv"
    out = pathlib.Path(workdir) / f"{label}-lme4.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    proc = subprocess.run([rscript, str(HERE / "three_way.R"),
                           str(manifest), str(out)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"lme4 run failed:\n{proc.stdout}\n{proc.stderr}")
    frame = pd.read_csv(out)
    frame["warnings"] = frame["warnings"].fillna("").astype(str)
    frame["error"] = frame["error"].fillna("").astype(str)
    versions = pathlib.Path(str(out) + ".versions").read_text().split()
    return frame, {"lme4": versions[1], "R": f"{versions[3]}.{versions[4]}"}


def time_reps(fit, reps):
    times, result = [], None
    for _ in range(reps):
        start = time.perf_counter()
        result = fit()
        times.append(time.perf_counter() - start)
    return times, result


def lme4_view(row):
    """One lme4 repetition as the arrays the Python results expose."""
    fe = np.array([row.beta0, row.beta1, row.beta2], float)
    se = np.array([row.se0, row.se1, row.se2], float)
    cov = np.array([[row.re00, row.re01], [row.re01, row.re11]], float)
    return fe, se, cov


def agreement(result, row):
    """How far a Python fit is from lme4's, on the scales tolerances.py uses."""
    fe, se, cov = lme4_view(row)
    se = np.where(np.isfinite(se) & (se > 0), se, 1.0)
    theirs = np.asarray(result.cov_re, float)
    return {
        "fixed_effects_in_lme4_standard_errors":
            float(np.max(np.abs(np.asarray(result.fe_params, float) - fe) / se)),
        "cov_re_relative":
            float(np.max(np.abs(theirs - cov)
                         / max(1.0, float(np.max(np.abs(cov)))))),
        # Positive: this package found a higher likelihood than lme4 did.
        "deviance_gap": float(2.0 * (float(result.llf) - float(row.loglik))),
    }


# ------------------------------------------------------------------- timing
def timing(rscript, cases, reps, workdir):
    fixtures, manifest = [], []
    for ngroups, nper, with_sm in cases:
        name = f"g{ngroups}"
        path = pathlib.Path(workdir) / f"timing-{name}.csv"
        make(ngroups, nper).to_csv(path, index=False)
        fixtures.append((name, path, ngroups, with_sm))
        manifest.append({"case": name, "file": str(path), "re": "slope",
                         "reml": "TRUE", "reps": reps})

    rows, fits, load = [], {}, []
    for name, path, ngroups, with_sm in fixtures:
        df = pd.read_csv(path)
        require_quiet(f"{name} (python)", load)
        ours_t, ours = time_reps(
            lambda: mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                                re_formula="~x1").fit(), reps)
        row = {"case": name, "n": len(df), "groups": ngroups,
               "mixedlm_rs": {"seconds": min(ours_t), "repetitions": ours_t,
                              "converged": bool(ours.converged)},
               "statsmodels": None, "lme4": None}
        fits[(name, "mixedlm_rs")] = ours
        if with_sm:
            sm_reps = 1 if len(df) > SM_SINGLE_REP_ABOVE else reps
            sm_t, sm = time_reps(
                lambda: smf.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                                    re_formula="~x1").fit(), sm_reps)
            row["statsmodels"] = {"seconds": min(sm_t), "repetitions": sm_t,
                                  "converged": bool(sm.converged)}
            fits[(name, "statsmodels")] = sm
        else:
            row["statsmodels_note"] = ("not timed: at this scale statsmodels "
                                       "dominates the run, as in performance.py")
        sm_txt = ("-" if row["statsmodels"] is None
                  else f"{row['statsmodels']['seconds']:.3f}s")
        print(f"  {len(df):>8,d} rows  mixedlm-rs {min(ours_t):.4f}s  "
              f"statsmodels {sm_txt}", flush=True)
        rows.append(row)

    # lme4 runs every case in one R process, so the machine is checked on both
    # sides of it: load that arrives mid-run shows up in the sample after.
    print("  lme4 ...", flush=True)
    require_quiet("lme4 batch (before)", load)
    lme4, versions = run_lme4(rscript, manifest, workdir, "timing")
    require_quiet("lme4 batch (after)", load)
    problems = []
    for row in rows:
        mine = lme4[lme4.case == row["case"]]
        if (mine.error != "").any():
            problems.append(f"{row['case']}: lme4 failed: {mine.error.iloc[0]}")
            continue
        best = mine.loc[mine.seconds.idxmin()]
        row["lme4"] = {"seconds": float(mine.seconds.min()),
                       "repetitions": mine.seconds.tolist(),
                       "warnings": best.warnings or None,
                       "singular": bool(best.singular)}
        for key in ("mixedlm_rs", "statsmodels"):
            result = fits.get((row["case"], key))
            if result is None:
                continue
            gap = agreement(result, best)
            row[key]["agreement_with_lme4"] = gap
            fe = gap["fixed_effects_in_lme4_standard_errors"]
            if fe > tolerances.FIXED_EFFECT_SE:
                problems.append(f"{row['case']} {key}: fixed effects "
                                f"{fe:.3g} SEs from lme4")
            if gap["cov_re_relative"] > tolerances.COV_RE_REL:
                problems.append(f"{row['case']} {key}: cov_re "
                                f"{gap['cov_re_relative']:.3g} relative "
                                "from lme4")
            if gap["deviance_gap"] < -tolerances.DEVIANCE_ABS:
                problems.append(f"{row['case']} {key}: deviance "
                                f"{gap['deviance_gap']:.3g} below lme4")
        print(f"  {row['n']:>8,d} rows  lme4 {row['lme4']['seconds']:.4f}s",
              flush=True)
    return rows, versions, problems, load


# ----------------------------------------------------------------- accuracy
def accuracy(rscript, seeds, workdir):
    fixtures, manifest = [], []
    for seed in range(seeds):
        df, re_formula, reml = random_case(
            np.random.default_rng(ACCURACY_SEED_BASE + seed))
        path = pathlib.Path(workdir) / f"accuracy-{seed}.csv"
        df.to_csv(path, index=False)
        fixtures.append((seed, path, re_formula, reml))
        manifest.append({"case": f"s{seed}", "file": str(path),
                         "re": "slope" if re_formula else "intercept",
                         "reml": "TRUE" if reml else "FALSE", "reps": 1})

    print(f"  lme4 on {seeds} fixtures ...", flush=True)
    lme4, versions = run_lme4(rscript, manifest, workdir, "accuracy")
    rows = []
    for seed, path, re_formula, reml in fixtures:
        df = pd.read_csv(path)
        ref = lme4[lme4.case == f"s{seed}"].iloc[0]
        failed = bool(ref.error)
        row = {"seed": ACCURACY_SEED_BASE + seed, "n": len(df),
               "groups": int(df["g"].nunique()),
               "random_effects": ("intercept and slope" if re_formula
                                  else "intercept"),
               "reml": reml,
               "lme4": {"loglik": None if failed else float(ref.loglik),
                        "error": ref.error or None,
                        "warnings": ref.warnings or None,
                        "singular": None if failed else bool(ref.singular)}}
        for key, api in (("mixedlm_rs", mlm), ("statsmodels", smf)):
            try:
                result = api.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                                     re_formula=re_formula).fit(reml=reml)
                entry = {"loglik": float(result.llf),
                         "converged": bool(result.converged), "error": None}
                if not failed and np.isfinite(result.llf):
                    entry["deviance_gap_to_lme4"] = float(
                        2.0 * (float(result.llf) - float(ref.loglik)))
            except Exception as exc:        # a failure is a result
                entry = {"loglik": None, "converged": False,
                         "error": f"{type(exc).__name__}: {exc}"[:300]}
            row[key] = entry
        rows.append(row)
    return rows, versions


def block_environment(reps, versions):
    env = environment(reps)
    env["mixedlm_rs_path"] = mlm.__file__
    env["dependencies"].update({"R": versions["R"], "lme4": versions["lme4"]})
    env["lme4_optimizer"] = "lmer defaults"
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("all", "timing", "accuracy"),
                    default="all",
                    help="record one experiment and keep the other block of "
                         "an existing output file as it is")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rscript = find_rscript()
    if not rscript:
        print("no Rscript found (set RSCRIPT); nothing measured, nothing "
              "written. The historical lme4 figures in performance.json "
              "remain the only lme4 record.", file=sys.stderr)
        return 2

    out = pathlib.Path(args.out)
    payload = {"timing": None, "accuracy": None}
    if args.part != "all" and out.is_file():
        payload.update(json.loads(out.read_text(encoding="utf-8")))

    cases = CASES[:3] if args.quick else CASES
    seeds = 20 if args.quick else ACCURACY_SEEDS
    print(f"mixedlm-rs {mlm.__version__} from {mlm.__file__}", flush=True)
    status, problems = 0, []

    with tempfile.TemporaryDirectory() as work:
        if args.part in ("all", "timing"):
            print("\ntiming", flush=True)
            try:
                rows, versions, problems, load = timing(rscript, cases,
                                                        args.reps, work)
                payload["timing"] = {
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "environment": block_environment(args.reps, versions),
                    "quick": args.quick,
                    "method": "same CSV fixture read by all three packages; "
                              "timed from an in-memory data frame to a fitted "
                              "model; minimum of the repetitions; lme4 timed "
                              "inside R with Sys.time()",
                    "max_background_busy": MAX_BACKGROUND_BUSY,
                    "background_load": load,
                    "rows": rows,
                }
            except BusyMachine as exc:
                # Nothing from this attempt is kept, including a previous
                # timing block: a file that still showed old timings beside a
                # refusal would invite reading them as current.
                payload["timing"] = None
                print(f"\ntiming refused: {exc}", file=sys.stderr)
                status = 3

        if args.part in ("all", "accuracy"):
            print("\naccuracy", flush=True)
            rows, versions = accuracy(rscript, seeds, work)
            payload["accuracy"] = {
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "environment": block_environment(1, versions),
                "quick": args.quick,
                "tolerance": {"deviance_abs": tolerances.DEVIANCE_ABS,
                              "source": "bench/tolerances.py"},
                "method": "fixtures from tests/test_fuzz.py random_case, "
                          f"seeds {ACCURACY_SEED_BASE}.."
                          f"{ACCURACY_SEED_BASE + seeds - 1}, as "
                          "bench/differential_table.py; deviance_gap_to_lme4 "
                          "is 2 * (loglik - lme4 loglik), positive means a "
                          "higher likelihood than lme4 found",
                "rows": rows,
            }

    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    if problems:
        print("\nthe timed fits do not agree with lme4; no timing is "
              "reportable:")
        for line in problems:
            print(f"  - {line}")
        return 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())

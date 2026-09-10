"""Record the headline performance numbers with enough provenance to audit.

`bench/baseline.py` does this for the *correctness* sweeps. This does it for
the timings, because a speedup in a README is only a claim unless a reader can
find the machine, the build, the repetition count and the thread settings that
produced it.

    python bench/performance.py                 # full, writes bench/performance.json
    python bench/performance.py --quick         # small cases only, for a smoke run
    python bench/performance.py --reps 5

What it measures and what it cannot
-----------------------------------
It re-measures **this package against statsmodels**, which is the comparison
both are installed for. It cannot re-measure **lme4** or **pymer4**: those need
R, rpy2 and a writable R library, and a machine without them can produce no
number at all rather than a wrong one. Those columns are therefore carried
forward as *historical* records, tagged with the environment that produced them
and never silently mixed with a fresh measurement -- `historical` in the output
is a separate block from `measured`.

Timing method
-------------
Each case runs `--reps` times and the **minimum** is reported, not the mean. A
timing distribution is bounded below by the true cost and has an unbounded
right tail made of scheduler noise, so the minimum is the better estimator of
"how long does this take" -- and it is the convention lme4's own benchmarks
use. Every repetition is kept in the output so a reader can see the spread
rather than trusting that choice.

Correctness is checked before any timing is reported: a fast wrong answer is
not a benchmark result. The thresholds come from `bench/tolerances.py`, the
same ones the correctness sweeps use.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import mixedlm_rs as mlm                                          # noqa: E402
import statsmodels.formula.api as smf                             # noqa: E402
import tolerances                                                 # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "performance.json"

# (groups, rows per group, time statsmodels too). The last is the scale from
# statsmodels#9097, where a user reported mixedlm taking 41 minutes against
# 1-2 seconds for R's lmer; statsmodels is not timed there because it would
# dominate the run.
CASES = [
    (100, 20, True),
    (500, 20, True),
    (1_000, 20, True),
    (5_000, 8, True),
    (20_000, 5, True),
    (50_000, 4, True),
    (125_066, 4, False),
]
QUICK_CASES = CASES[:3]

# Measured elsewhere, on a machine with R. Carried forward verbatim rather than
# re-measured here, and reported separately so it cannot be mistaken for a
# fresh number. Re-measure with bench/vs_lme4.py on a machine with the R
# toolchain installed.
HISTORICAL = {
    "source": "bench/vs_lme4.py, recorded 2026-09-08",
    "environment": {
        "note": "R toolchain present; not available on the recording machine "
                "for the current candidate",
        "R": "4.6.1", "lme4": "2.0.6", "rpy2": "3.6.7", "pymer4": "0.9.2",
    },
    "caveat": "pymer4 timings are the end-to-end cost of fitting through "
              "pymer4, which also runs lmerTest for Satterthwaite degrees of "
              "freedom and extracts results as R objects. They are not a "
              "measurement of rpy2 marshalling alone.",
    "rows": [
        {"n": 10_000, "groups": 500, "ours": 0.010, "lme4": 0.080, "pymer4": 1.71},
        {"n": 40_000, "groups": 5_000, "ours": 0.017, "lme4": 0.330, "pymer4": 11.67},
        {"n": 100_000, "groups": 20_000, "ours": 0.038, "lme4": 1.120, "pymer4": 118.10},
        {"n": 200_000, "groups": 50_000, "ours": 0.078, "lme4": 2.510, "pymer4": 747.28},
        {"n": 500_264, "groups": 125_066, "ours": 0.169, "lme4": 7.330, "pymer4": None},
    ],
}


def environment(reps: int) -> dict:
    import hashlib

    from mixedlm_rs import _mixedlm_rs

    versions = {}
    for name in ("numpy", "scipy", "pandas", "patsy", "statsmodels"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                    capture_output=True, text=True,
                                    check=True).stdout.strip())
    except Exception:
        commit, dirty = None, None

    extension = getattr(_mixedlm_rs, "__file__", None)
    digest = None
    if extension and pathlib.Path(extension).is_file():
        digest = hashlib.sha256(
            pathlib.Path(extension).read_bytes()).hexdigest()

    # Thread settings are part of the number: this package factorises the
    # per-group blocks in a rayon parallel loop, so a run pinned to one thread
    # and a run on twelve are different experiments.
    return {
        "commit": commit,
        "working_tree_dirty": dirty,
        "mixedlm_rs": mlm.__version__,
        "extension_sha256": digest,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", "unset "
                                            "(rayon uses all logical cpus)"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "repetitions": reps,
        "statistic": "minimum of the repetitions",
        "dependencies": versions,
    }


def make(ngroups, nper, seed=0):
    rng = np.random.default_rng(seed)
    codes = np.repeat(np.arange(ngroups), nper)
    n = len(codes)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    b0 = rng.standard_normal(ngroups)
    b1 = rng.standard_normal(ngroups) * 0.6
    y = (1 + 2 * x1 - 0.5 * x2 + b0[codes] + b1[codes] * x1
         + rng.standard_normal(n) * 0.5)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": codes})


def time_fit(fit, reps):
    """All repetitions, so the spread is visible rather than asserted."""
    times = []
    result = None
    for _ in range(reps):
        start = time.perf_counter()
        result = fit()
        times.append(time.perf_counter() - start)
    return times, result


def measure(ngroups, nper, do_sm, reps) -> dict:
    df = make(ngroups, nper)
    ours_times, ours = time_fit(
        lambda: mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                            re_formula="~x1").fit(), reps)

    row = {
        "n": len(df), "groups": ngroups,
        "ours_seconds": min(ours_times),
        "ours_repetitions": ours_times,
        "ours_converged": bool(ours.converged),
        "statsmodels_seconds": None,
        "statsmodels_repetitions": None,
        "statsmodels_converged": None,
        "speedup": None,
        "agreement": None,
    }
    if not do_sm:
        row["statsmodels_note"] = (
            "not timed: at this scale statsmodels dominates the run; see "
            "docs/BENCHMARKS.md for the reported figure")
        return row

    # statsmodels is timed once regardless of --reps: at the larger sizes a
    # single fit is minutes, and repeating it would dominate the run without
    # sharpening a ratio that is already three orders of magnitude.
    sm_reps = 1 if len(df) > 50_000 else reps
    sm_times, theirs = time_fit(
        lambda: smf.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                            re_formula="~x1").fit(), sm_reps)

    se = np.asarray(ours.bse_fe, float)
    se = np.where(np.isfinite(se) & (se > 0), se, 1.0)
    fe_gap = float(np.max(np.abs(ours.fe_params
                                 - np.asarray(theirs.fe_params)) / se))
    re_gap = float(np.max(np.abs(np.asarray(ours.cov_re)
                                 - np.asarray(theirs.cov_re))
                          / max(1.0, float(np.max(np.abs(ours.cov_re))))))
    row.update({
        "statsmodels_seconds": min(sm_times),
        "statsmodels_repetitions": sm_times,
        "statsmodels_converged": bool(theirs.converged),
        "speedup": min(sm_times) / min(ours_times),
        "agreement": {
            "fixed_effects_in_standard_errors": fe_gap,
            "cov_re_relative": re_gap,
            "delta_loglik": float(ours.llf - theirs.llf),
        },
    })
    return row


def check(row) -> list[str]:
    """A speedup beside an unchecked answer is not a result."""
    if row["agreement"] is None or not row["statsmodels_converged"]:
        return []
    problems = []
    gap = row["agreement"]
    if gap["fixed_effects_in_standard_errors"] > tolerances.FIXED_EFFECT_SE:
        problems.append(
            f"n={row['n']}: fixed effects differ by "
            f"{gap['fixed_effects_in_standard_errors']:.3g} SEs")
    if gap["cov_re_relative"] > tolerances.COV_RE_REL:
        problems.append(f"n={row['n']}: cov_re differs by "
                        f"{gap['cov_re_relative']:.3g} relative")
    if gap["delta_loglik"] < -tolerances.DEVIANCE_ABS / 2:
        problems.append(f"n={row['n']}: our log-likelihood is "
                        f"{gap['delta_loglik']:.3g} below the reference's")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    cases = QUICK_CASES if args.quick else CASES
    env = environment(args.reps)
    print(f"commit {env['commit']} (dirty={env['working_tree_dirty']})")
    print(f"{env['logical_cpus']} logical cpus, "
          f"RAYON_NUM_THREADS={env['rayon_num_threads']}")
    print(f"{args.reps} repetitions per case, reporting the minimum\n")

    print(f"{'n':>9s} {'groups':>8s} {'statsmodels':>12s} {'mixedlm-rs':>12s} "
          f"{'speedup':>9s}  agreement")
    rows, problems = [], []
    for ngroups, nper, do_sm in cases:
        row = measure(ngroups, nper, do_sm, args.reps)
        rows.append(row)
        problems += check(row)
        sm = ("not run" if row["statsmodels_seconds"] is None
              else f"{row['statsmodels_seconds']:11.3f}s")
        sp = "-" if row["speedup"] is None else f"{row['speedup']:8.0f}x"
        ag = ("-" if row["agreement"] is None else
              f"fe {row['agreement']['fixed_effects_in_standard_errors']:.1e} "
              f"re {row['agreement']['cov_re_relative']:.1e}")
        print(f"{row['n']:9,d} {row['groups']:8,d} {sm:>12s} "
              f"{row['ours_seconds']:11.3f}s {sp:>9s}  {ag}")

    payload = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": env,
        "quick": args.quick,
        "measured": {
            "comparison": "mixedlm-rs against statsmodels, same machine, "
                          "same process, agreement checked before timing",
            "rows": rows,
        },
        "historical": HISTORICAL,
    }
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=1) + "\n",
                                      encoding="utf-8")
    print(f"\nwrote {args.out}")

    if problems:
        print("\nthe fits do not agree; no timing here is reportable:")
        for line in problems:
            print(f"  - {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark mixedlm-rs against lme4 (and, where available, pymer4).

Three implementations of the same model, on byte-identical data:

  * mixedlm-rs   -- this package
  * lme4         -- the R reference, timed inside R by bench/vs_lme4.R
  * pymer4       -- lme4 called from Python through rpy2

pymer4 is the closest competitor in the market: it is the only way a Python user
gets genuine lme4 results today. It cannot be faster than lme4 itself, since it
*is* lme4 plus an rpy2 round trip, so the lme4 row is pymer4's floor.

Run:
    python bench/vs_lme4.py --write     # write fixtures + time mixedlm-rs
    Rscript bench/vs_lme4.R bench/fixtures
    python bench/vs_lme4.py --report    # join and print the comparison
"""

import argparse
import json
import pathlib
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"

# (name, groups, obs per group, random-effects structure, REML)
CASES = [
    ("sleepstudy-like",   18,   10, "slope",     True),
    ("small-500x20",     500,   20, "slope",     True),
    ("mid-1000x20",     1000,   20, "slope",     True),
    ("mid-5000x8",      5000,    8, "slope",     True),
    ("large-20000x5",  20000,    5, "slope",     True),
    ("large-50000x4",  50000,    4, "slope",     True),
    ("huge-125066x4", 125066,    4, "slope",     True),
    ("intercept-5000",  5000,    8, "intercept", True),
    ("ml-5000",         5000,    8, "slope",     False),
]


def make(ngroups, nper, seed=0):
    rng = np.random.default_rng(seed)
    codes = np.repeat(np.arange(ngroups), nper)
    n = len(codes)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = (1 + 2 * x1 - 0.5 * x2
         + rng.standard_normal(ngroups)[codes]
         + rng.standard_normal(ngroups)[codes] * 0.6 * x1
         + rng.standard_normal(n) * 0.5)
    return pd.DataFrame(dict(y=y, x1=x1, x2=x2, g=codes))


def time_best(fn, reps=3):
    fn()                                    # warm up
    best, out = np.inf, None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def write_and_time_ours():
    import mixedlm_rs as mlm

    FIXTURES.mkdir(exist_ok=True)
    manifest, results = [], []
    for name, ng, nper, re_kind, reml in CASES:
        df = make(ng, nper)
        fname = f"{name}.csv"
        df.to_csv(FIXTURES / fname, index=False)
        manifest.append(dict(name=name, file=fname, re=re_kind,
                             reml=str(reml).upper(), n=len(df), groups=ng))

        rf = "~x1" if re_kind == "slope" else None
        dt, r = time_best(lambda d=df, rf=rf, reml=reml:
                          mlm.mixedlm("y ~ x1 + x2", d, groups=d["g"],
                                      re_formula=rf).fit(reml=reml))
        results.append(dict(name=name, n=len(df), groups=ng, seconds=dt,
                            logLik=float(r.llf),
                            beta=[float(v) for v in r.fe_params],
                            sigma=float(np.sqrt(r.scale)),
                            converged=bool(r.converged)))
        print(f"  {name:<22s} {dt:8.3f} s   (mixedlm-rs)")

    pd.DataFrame(manifest).to_csv(FIXTURES / "manifest.csv", index=False)
    (FIXTURES / "ours_results.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {len(CASES)} fixtures to {FIXTURES}")


def _setup_r_env():
    """Windows needs all of this set *before* rpy2 is imported.

    Without R's bin/x64 on PATH, R cannot load its own stats.dll, and pymer4
    dies with a LoadLibrary failure that names an unrelated package. Python 3.8+
    additionally needs os.add_dll_directory. R_LIBS_USER points at the user
    library because R's own library directory under Program Files is read-only.
    """
    import os
    r_home = os.environ.get("R_HOME") or "C:/Program Files/R/R-4.6.1"
    rbin = os.path.join(r_home, "bin", "x64")
    os.environ["R_HOME"] = r_home
    os.environ.setdefault("R_LIBS_USER", "C:/Users/Fatin/R/win-library")
    if os.path.isdir(rbin):
        os.environ["PATH"] = rbin + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(rbin)
        except (AttributeError, OSError):
            pass


def time_pymer4():
    """pymer4, if the R toolchain is present. Returns {} when unavailable.

    pymer4 0.9.2 exposes `lmer` (lowercase) and takes a polars frame, not the
    `Lmer` class and pandas frame of earlier releases.
    """
    _setup_r_env()
    try:
        import polars as pl
        from pymer4.models import lmer
    except Exception as exc:
        print(f"  pymer4 unavailable: {type(exc).__name__}: {str(exc)[:90]}")
        return {}

    out = {}
    for name, ng, nper, re_kind, reml in CASES:
        df = pl.read_csv(FIXTURES / f"{name}.csv")
        form = ("y ~ x1 + x2 + (1 + x1|g)" if re_kind == "slope"
                else "y ~ x1 + x2 + (1|g)")
        try:
            def run(d=df, f=form, reml=reml):
                m = lmer(f, data=d)
                m.fit(REML=reml, summarize=False)
                return m
            dt, _ = time_best(run, reps=3)
            out[name] = dt
            print(f"  {name:<22s} {dt:8.3f} s   (pymer4)")
        except Exception as exc:
            print(f"  {name:<22s}   FAILED   {type(exc).__name__}: {str(exc)[:60]}")
    return out


def report():
    ours = {r["name"]: r for r in json.loads((FIXTURES / "ours_results.json").read_text())}
    lme4_path = FIXTURES / "lme4_results.csv"
    lme4 = {}
    if lme4_path.exists():
        for _, row in pd.read_csv(lme4_path).iterrows():
            lme4[row["name"]] = row
    pm = time_pymer4()

    print()
    hdr = (f"{'fixture':<20s} {'n':>8s} {'groups':>8s} | {'mixedlm-rs':>11s} | "
           f"{'lme4 (R)':>10s} | {'pymer4':>9s} | {'vs lme4':>8s} | {'dlogLik':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for name, ng, nper, re_kind, reml in CASES:
        o = ours.get(name)
        if o is None:
            continue
        L = lme4.get(name)
        lt = f"{L['seconds']:9.3f}s" if L is not None and np.isfinite(L["seconds"]) else "        -"
        pt = f"{pm[name]:8.3f}s" if name in pm else "        -"
        if L is not None and np.isfinite(L["seconds"]):
            ratio = f"{L['seconds'] / o['seconds']:7.1f}x"
            dll = f"{o['logLik'] - L['logLik']:+10.6f}"
        else:
            ratio, dll = "       -", "         -"
        print(f"{name:<20s} {o['n']:8,d} {o['groups']:8,d} | {o['seconds']:10.3f}s | "
              f"{lt} | {pt} | {ratio} | {dll}")
    print("-" * len(hdr))
    print("dlogLik = ours minus lme4; positive means we found the better optimum.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.write:
        write_and_time_ours()
    if a.report or not a.write:
        report()

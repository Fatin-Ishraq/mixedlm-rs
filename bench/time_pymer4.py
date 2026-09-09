"""Time pymer4 on the fixtures written by bench/vs_lme4.py --write.

Kept separate from vs_lme4.py because pymer4 is slow and fragile enough that it
needs its own process: results are appended to CSV as each fixture finishes, so
a hang on a later case does not discard the earlier ones. Run with `python -u`.

pymer4 calls lme4 through rpy2, so what a Python user pays for an lme4 fit is
what this measures.

What this measures, and what it does not
----------------------------------------
This is the *end-to-end cost of fitting through pymer4*, which is the number a
Python user actually pays. It is NOT a measurement of serialisation alone.
`pymer4.models.lmer.fit` does more than hand `lmer` a data frame: it routes
through lmerTest for Satterthwaite degrees of freedom, and pulls fixed effects,
random effects, and fit statistics back across the bridge as R objects. The gap
between this and a bare `lmer` call in R therefore covers marshalling *and*
that extra inference, and nothing here separates the two. Attributing the whole
difference to the bridge would need a profiler decomposition that has not been
done.
"""

import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
OUT = FIXTURES / "pymer4_results.csv"

# Must happen before rpy2 is imported, or R cannot load its own stats.dll.
R_HOME = os.environ.get("R_HOME") or "C:/Program Files/R/R-4.6.1"
RBIN = os.path.join(R_HOME, "bin", "x64")
os.environ["R_HOME"] = R_HOME
os.environ.setdefault(
    "R_LIBS_USER",
    os.path.expanduser("~/R/win-library" if os.name == "nt"
                       else "~/R/library"))
if os.path.isdir(RBIN):
    os.environ["PATH"] = RBIN + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(RBIN)
    except (AttributeError, OSError):
        pass

import polars as pl
from pymer4.models import lmer

CASES = [
    ("sleepstudy-like",   18,   10, "slope",     True),
    ("small-500x20",     500,   20, "slope",     True),
    ("mid-1000x20",     1000,   20, "slope",     True),
    ("mid-5000x8",      5000,    8, "slope",     True),
    ("intercept-5000",  5000,    8, "intercept", True),
    ("ml-5000",         5000,    8, "slope",     False),
    ("large-20000x5",  20000,    5, "slope",     True),
    ("large-50000x4",  50000,    4, "slope",     True),
    ("huge-125066x4", 125066,    4, "slope",     True),
]

REPS = int(os.environ.get("PYMER4_REPS", "1"))


def main():
    only = set(sys.argv[1:]) or None
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("name,seconds,logLik,note\n")
        fh.flush()
        for name, _ng, _nper, re_kind, reml in CASES:
            if only and name not in only:
                continue
            path = FIXTURES / f"{name}.csv"
            if not path.exists():
                continue
            df = pl.read_csv(path)
            form = ("y ~ x1 + x2 + (1 + x1|g)" if re_kind == "slope"
                    else "y ~ x1 + x2 + (1|g)")
            print(f"  {name:<20s} n={df.height:>7,d} ...", end=" ", flush=True)
            try:
                best, ll = float("inf"), float("nan")
                for rep in range(REPS + 1):      # 1 warm-up, then REPS timed
                    t0 = time.perf_counter()
                    m = lmer(form, data=df)
                    m.fit(REML=reml, summarize=False)
                    dt = time.perf_counter() - t0
                    if rep:                      # the warm-up is not a timing
                        best = min(best, dt)
                    try:
                        ll = float(m.result_fit_stats["log_likelihood"][0])
                    except Exception:
                        pass
                print(f"{best:8.3f} s", flush=True)
                fh.write(f"{name},{best},{ll},ok\n")
            except Exception as exc:
                msg = f"{type(exc).__name__}: {str(exc)[:60]}".replace(",", ";")
                print(f"FAILED  {msg}", flush=True)
                fh.write(f"{name},,,{msg}\n")
            fh.flush()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

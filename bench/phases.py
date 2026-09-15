"""Time the phases of a fit separately, for a candidate build against a baseline.

`bench/performance.py` times whole fits against other packages. This is the
development tool for changing this package: it splits a fit into the phases an
optimisation can move -- constructing the core, one criterion evaluation, one
gradient, one full solution, the whole fit, first inference, bulk result
access and a refit for a new response -- and times each under a pinned thread
budget.

    python bench/phases.py                            # installed package only
    python bench/phases.py --baseline OLD_SITE_DIR    # candidate vs a baseline
    python bench/phases.py --quick --threads 1,4

`--baseline` names a directory from which a *different* build of `mixedlm_rs`
imports (for example `pip install --target OLD_SITE_DIR old.whl`).

Controls
--------
* Every (build, thread budget) runs in its own process, so each gets a fresh
  rayon pool, with BLAS pinned to one thread and `RAYON_NUM_THREADS` set.
* Builds are run in alternating order over `--rounds` rounds (ABBA), so a drift
  in machine state is shared rather than charged to whichever ran second.
* Each phase records its first (cold) call separately from the warm samples,
  and every sample is kept, not only the median.
* The extension's sha256 and the Python sources' digest are recorded for each
  build, from inside the process that was timed.
* Before each round the background CPU load is sampled with nothing of ours
  running; above `MAX_BACKGROUND_BUSY` the run stops without writing timings,
  for the reason given in `bench/three_way.py`.
* The fitted deviance and evaluation count are recorded next to the timings, so
  a faster build that fits a different optimum cannot pass unnoticed.

The results are exploratory measurements on whatever machine runs this; nothing
in the test suite reads them.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "phases.json"
MAX_BACKGROUND_BUSY = 0.15

# (name, kind, groups, p). Kinds: "slope" is a continuous random slope (no two
# groups share Z'Z); "intercept" is a balanced random intercept; "unbalanced"
# a random intercept with 1-8 rows per group; "visits" a random intercept and
# a slope in visit number, which repeats across groups.
CASES = [
    ("slope-18", "slope", 18, 3),
    ("slope-500", "slope", 500, 3),
    ("slope-50k", "slope", 50_000, 3),
    ("slope-125k", "slope", 125_066, 3),
    ("intercept-125k", "intercept", 125_066, 3),
    ("unbalanced-125k", "unbalanced", 125_066, 3),
    ("visits-125k", "visits", 125_066, 3),
    ("slope-20k-p30", "slope", 20_000, 30),
]
QUICK = ["slope-18", "slope-50k", "intercept-125k", "visits-125k"]


def design(kind, m, p, seed=913):
    rng = np.random.default_rng(seed)
    if kind == "unbalanced":
        nper = rng.integers(1, 9, size=m)
    else:
        nper = np.full(m, 4 if m > 1000 else 10)
    codes = np.repeat(np.arange(m, dtype=np.int64), nper)
    n = len(codes)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    if kind in ("intercept", "unbalanced"):
        Z = np.ones((n, 1))
    elif kind == "slope":
        Z = X[:, :2].copy()
    elif kind == "visits":
        start = np.r_[0, np.cumsum(nper)[:-1]]
        Z = np.column_stack([np.ones(n), (np.arange(n) - np.repeat(start, nper)).astype(float)])
    else:
        raise ValueError(kind)
    b = rng.normal(size=(m, Z.shape[1])) * 0.6
    y = X @ np.linspace(0.5, 1, p) + np.einsum("ij,ij->i", Z, b[codes]) + rng.normal(size=n) * 0.5
    return (np.ascontiguousarray(y), np.ascontiguousarray(X), np.ascontiguousarray(Z), codes, m)


# --------------------------------------------------------------------- worker
def _time(fn, reps):
    t0 = time.perf_counter()
    fn()
    cold = (time.perf_counter() - t0) * 1e3
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return {"cold_ms": cold, "samples_ms": samples}


def worker(args):
    if args.package:
        sys.path.insert(0, args.package)
    sys.path.insert(1, str(HERE))
    warnings.simplefilter("ignore")
    import mixedlm_rs as mlm
    from baseline import _extension_identity, _python_sources_digest
    from mixedlm_rs import LmmCore

    out = {"identity": {**_extension_identity(), **_python_sources_digest()},
           "version": mlm.__version__, "cases": {}}
    wanted = args.cases.split(",")
    for name, kind, m, p in CASES:
        if name in wanted:
            out["cases"][name] = measure(mlm, LmmCore, kind, m, p, args.reps)
    print(json.dumps(out))


def measure(mlm, LmmCore, kind, m, p, reps):
    """Every phase of one case, in one process."""
    y, X, Z, codes, m = design(kind, m, p)
    q = Z.shape[1]
    theta = [0.7 if r == c else 0.1 for c in range(q) for r in range(c, q)]
    rec = {"n": len(y), "m": m, "p": p, "q": q}
    core = LmmCore(y, X, Z, codes, m)
    rec["evaluator"] = getattr(core, "evaluator", "blocks")
    rec["construct"] = _time(lambda: LmmCore(y, X, Z, codes, m), reps)
    rec["deviance"] = _time(lambda: core.deviance(theta, True), reps * 3)
    rec["gradient"] = _time(lambda: core.deviance_grad(theta, True), reps * 3)
    rec["solution"] = _time(lambda: core.solution(theta, True), reps)

    def fit(endog=y):
        return mlm.MixedLM(endog, X, codes, exog_re=Z).fit()

    rec["fit"] = _time(fit, max(1, reps // 2))
    res = fit()
    rec["nfev"] = int(res._res["nfev"])
    rec["fit_deviance"] = float(-2 * res.llf)
    rec["converged"] = bool(res.converged)

    fresh = fit()
    t0 = time.perf_counter()
    _ = fresh.bse
    rec["first_bse_ms"] = (time.perf_counter() - t0) * 1e3
    fresh = fit()
    t0 = time.perf_counter()
    if hasattr(fresh, "random_effects_cov_array"):
        _ = fresh.random_effects_cov_array
        rec["re_cov_accessor"] = "random_effects_cov_array"
    elif m <= 20_000:
        _ = fresh.random_effects_cov
        rec["re_cov_accessor"] = "random_effects_cov"
    rec["first_re_cov_ms"] = (time.perf_counter() - t0) * 1e3 if "re_cov_accessor" in rec else None

    y2 = y + np.random.default_rng(1).normal(size=len(y)) * 0.1
    model = mlm.MixedLM(y, X, codes, exog_re=Z)
    model.fit()
    if hasattr(model, "with_endog"):
        rec["refit"] = _time(lambda: model.with_endog(y2).fit(), max(1, reps // 2))
    else:
        rec["refit"] = _time(lambda: fit(y2), max(1, reps // 2))
    return rec


# --------------------------------------------------------------------- driver
def _busy():
    try:
        from three_way import cpu_busy_fraction
    except Exception:                                  # pragma: no cover
        return None
    return cpu_busy_fraction()


def _summary(samples):
    return {"median_ms": statistics.median(samples), "min_ms": min(samples),
            "samples_ms": [round(s, 5) for s in samples]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", help="directory a baseline build imports from")
    ap.add_argument("--threads", default=f"1,{os.cpu_count()}")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--package", default="", help=argparse.SUPPRESS)
    ap.add_argument("--cases", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(args)

    cases = QUICK if args.quick else [c[0] for c in CASES]
    builds = [("candidate", "")]
    if args.baseline:
        builds.append(("baseline", str(pathlib.Path(args.baseline).resolve())))
    threads = [t.strip() for t in args.threads.split(",") if t.strip()]
    runs, loads = {}, []
    for rnd in range(args.rounds):
        busy = _busy()
        loads.append({"round": rnd, "background_busy": busy})
        if busy is not None and busy > MAX_BACKGROUND_BUSY:
            sys.exit(f"round {rnd}: {busy:.0%} of the CPU is busy with other work; "
                     "no timings written")
        order = builds if rnd % 2 == 0 else builds[::-1]
        for label, package in order:
            for t in threads:
                env = dict(os.environ, RAYON_NUM_THREADS=t, OMP_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                cmd = [sys.executable, __file__, "--worker", "--package", package,
                       "--cases", ",".join(cases), "--reps", str(args.reps)]
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                      check=True)
                runs.setdefault((label, t), []).append(json.loads(proc.stdout))
                print(f"round {rnd} {label} threads={t} done", flush=True)

    result = {"settings": {"rounds": args.rounds, "reps": args.reps, "threads": threads,
                           "cases": cases, "python": sys.version.split()[0],
                           "numpy": np.__version__},
              "background_load": loads, "builds": {}}
    for (label, t), per_round in runs.items():
        build = result["builds"].setdefault(label, {"identity": per_round[0]["identity"],
                                                    "version": per_round[0]["version"],
                                                    "threads": {}})
        table = {}
        for name in cases:
            recs = [r["cases"][name] for r in per_round]
            row = {k: recs[0][k] for k in ("n", "m", "p", "q", "evaluator", "nfev",
                                           "fit_deviance", "converged")}
            for phase in ("construct", "deviance", "gradient", "solution", "fit", "refit"):
                row[phase] = _summary([s for r in recs for s in r[phase]["samples_ms"]])
                row[phase]["cold_ms"] = [r[phase]["cold_ms"] for r in recs]
            for once in ("first_bse_ms", "first_re_cov_ms"):
                vals = [r[once] for r in recs if r.get(once) is not None]
                row[once] = _summary(vals) if vals else None
            row["re_cov_accessor"] = recs[0].get("re_cov_accessor")
            table[name] = row
        build["threads"][t] = table
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

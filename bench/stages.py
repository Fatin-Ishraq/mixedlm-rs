"""The staged benchmark that makes the contribution legible.

Reporting a single "Nx faster than statsmodels" number would misattribute the
win. Each stage is measured on the same fixtures so the algorithmic, structural,
language and gradient contributions can be read off separately:

  S0  statsmodels.MixedLM                               baseline
  S1  profiled REML, pure NumPy, per-group Python loop   the algorithmic win
  S2  S1 + batched block-diagonal linear algebra         the structural win
  S3  Rust core, numeric gradient (scipy L-BFGS-B)       the language win
  S4  Rust core, analytic gradient (scipy L-BFGS-B)      the beyond-lme4 win
  S5  Rust core, analytic gradient, in-Rust optimiser    owning the whole loop

S1 and S2 are reproducible by anyone in NumPy. Saying so openly is what makes
the rest of the table credible.

Convergence status is reported for every row: where statsmodels does not
converge, a speedup ratio compares against a fit that did not happen.
"""

import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "proto"))

import preml  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from mixedlm_rs import LmmCore  # noqa: E402

CASES = [(500, 20), (1000, 20), (2000, 10), (5000, 8), (20000, 5)]


def make(ngroups, nper, seed=0):
    rng = np.random.default_rng(seed)
    n = ngroups * nper
    codes = np.repeat(np.arange(ngroups), nper).astype(np.int64)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = (1 + 2 * x1 - 0.5 * x2
         + np.repeat(rng.standard_normal(ngroups), nper)
         + rng.standard_normal(n) * 0.5)
    df = pd.DataFrame(dict(y=y, x1=x1, x2=x2, g=codes))
    X = np.ascontiguousarray(np.column_stack([np.ones(n), x1, x2]))
    Z = np.ascontiguousarray(np.column_stack([np.ones(n), x1]))
    return df, y, X, Z, codes, ngroups


def timeit(fn, repeat=3):
    best, out = np.inf, None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def s0(df):
    r = smf.mixedlm("y ~ x1 + x2", df, groups=df["g"], re_formula="~x1").fit()
    return np.asarray(r.fe_params, float), bool(r.converged), int(getattr(r, "nit", 0) or 0)


def s1(y, X, Z, codes):
    r = preml.fit_lmm_looped(y, X, Z, codes, reml=True)
    return r["beta"], r["converged"], r["nfev"]


def s2(y, X, Z, codes):
    r = preml.fit_lmm(y, X, Z, codes, reml=True)
    return r["beta"], r["converged"], r["nfev"]


def s3_s4(core, analytic):
    bounds = [(0.0, None) if np.isfinite(v) else (None, None)
              for v in core.lower_bounds()]
    th0 = core.default_theta()
    if analytic:
        res = minimize(lambda t: tuple(
            (lambda fg: (fg[0], np.asarray(fg[1])))(core.deviance_grad(list(t), True))),
            th0, jac=True, method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500})
    else:
        res = minimize(lambda t: core.deviance(list(t), True), th0,
                       method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500})
    sol = core.solution(list(res.x), True)
    return np.asarray(sol["beta"]), bool(res.success), int(res.nfev)


def s5(core):
    sol = core.fit(core.default_theta(), True, 300, 1e-8, 1e-12)
    return np.asarray(sol["beta"]), bool(sol["converged"]), int(sol["fev"])


def run_case(ngroups, nper):
    df, y, X, Z, codes, m = make(ngroups, nper)
    core = LmmCore(y, X, Z, codes, m)
    out = {"n": ngroups * nper, "groups": ngroups}
    out["S0"] = (*timeit(lambda: s0(df), 1),)
    out["S1"] = (*timeit(lambda: s1(y, X, Z, codes), 1),)
    out["S2"] = (*timeit(lambda: s2(y, X, Z, codes)),)
    out["S3"] = (*timeit(lambda: s3_s4(core, False)),)
    out["S4"] = (*timeit(lambda: s3_s4(core, True)),)
    out["S5"] = (*timeit(lambda: s5(core)),)
    return out


def main():
    rows = [run_case(*c) for c in CASES]
    t = lambda r, k: r[k][0]          # noqa: E731
    beta = lambda r, k: r[k][1][0]    # noqa: E731
    conv = lambda r, k: r[k][1][1]    # noqa: E731
    fev = lambda r, k: r[k][1][2]     # noqa: E731

    print("\nAgreement: max |beta| difference, mixedlm-rs vs statsmodels")
    print("-" * 80)
    for r in rows:
        d = np.max(np.abs(beta(r, "S4") - beta(r, "S0")))
        flag = "" if conv(r, "S0") else "   <- statsmodels did NOT converge"
        print(f"  n={r['n']:>7,d} groups={r['groups']:>6,d}   max|dbeta| = {d:.2e}{flag}")

    print("\nStaged timings (seconds; best of 3, S0/S1 single run)")
    print("-" * 108)
    print(f"{'n':>8s} {'groups':>7s} | {'S0 sm':>9s} {'cv':>5s} | {'S1 loop':>8s} | "
          f"{'S2 batch':>8s} | {'S3 rust':>8s} | {'S4 grad':>8s} | {'S5 rustopt':>10s} | {'S4/S0':>6s}")
    print("-" * 108)
    for r in rows:
        print(f"{r['n']:8,d} {r['groups']:7,d} | {t(r,'S0'):8.2f}s {str(conv(r,'S0'))[:5]:>5s} | "
              f"{t(r,'S1'):7.3f}s | {t(r,'S2'):7.3f}s | {t(r,'S3'):7.4f}s | "
              f"{t(r,'S4'):7.4f}s | {t(r,'S5'):9.4f}s | {t(r,'S0')/t(r,'S4'):5.0f}x")
    print("-" * 108)

    med = lambda f: float(np.median([f(r) for r in rows]))  # noqa: E731
    print("\nStage-to-stage multipliers (median across fixtures)")
    print(f"  S0 -> S1   profiled REML alone           {med(lambda r: t(r,'S0')/t(r,'S1')):8.1f}x")
    print(f"  S1 -> S2   + batched block Cholesky      {med(lambda r: t(r,'S1')/t(r,'S2')):8.1f}x")
    print(f"  S2 -> S3   + Rust core                   {med(lambda r: t(r,'S2')/t(r,'S3')):8.1f}x")
    print(f"  S3 -> S4   + analytic gradient           {med(lambda r: t(r,'S3')/t(r,'S4')):8.1f}x")
    print(f"  S0 -> S4   end to end                    {med(lambda r: t(r,'S0')/t(r,'S4')):8.1f}x")

    print("\nObjective evaluations -- what the analytic gradient actually buys")
    print(f"{'n':>8s} {'groups':>7s} | {'S3 numeric':>11s} | {'S4 analytic':>12s} | {'S5 in-rust':>11s}")
    print("-" * 60)
    for r in rows:
        print(f"{r['n']:8,d} {r['groups']:7,d} | {fev(r,'S3'):11d} | "
              f"{fev(r,'S4'):12d} | {fev(r,'S5'):11d}")

    nc = sum(1 for r in rows if not conv(r, "S0"))
    print(f"\nstatsmodels failed to converge on {nc}/{len(rows)} fixtures; "
          f"mixedlm-rs converged on {sum(1 for r in rows if conv(r,'S4'))}/{len(rows)}")
    print("\nNote: S5 (in-Rust optimiser) needs ~3x the objective evaluations of S4,")
    print("      because scipy's L-BFGS-B line search is better than the hand-rolled")
    print("      projected one. S4 is therefore the default path; see docs/DESIGN.md.")


if __name__ == "__main__":
    main()

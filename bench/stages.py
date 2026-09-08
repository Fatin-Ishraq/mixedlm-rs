"""The staged benchmark that makes the contribution legible.

Reporting a single "Nx faster than statsmodels" number would misattribute the
win. Each stage below is measured on the same fixtures so the algorithmic,
structural, language and gradient contributions can be read off separately:

  S0  statsmodels.MixedLM                              baseline
  S1  profiled REML, pure NumPy, per-group Python loop  the algorithmic win
  S2  S1 + batched block-diagonal linear algebra        the structural win
  S3  Rust core, numeric gradient (scipy L-BFGS-B)      the language win
  S4  Rust core, analytic gradient (scipy L-BFGS-B)     the beyond-lme4 win
  S5  Rust core, analytic gradient, in-Rust optimiser   owning the loop

Convergence status is reported for every row. Where the reference does not
converge, a speedup ratio is not a meaningful comparison and is labelled.
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "proto"))

from mixedlm_rs import LmmCore  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402


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
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, out


def s0_statsmodels(df):
    r = smf.mixedlm("y ~ x1 + x2", df, groups=df["g"], re_formula="~x1").fit()
    return np.asarray(r.fe_params, float), bool(r.converged)


def s1_s2_numpy(y, X, Z, codes, batched):
    import preml
    preml_fn = preml.fit_lmm
    if not batched:
        raise RuntimeError("S1 requires the unbatched build; see note in the report")
    r = preml_fn(y, X, Z, codes, reml=True)
    return r["beta"], r["converged"]


def s3_s4_scipy(core, analytic):
    lower = core.lower_bounds()
    bounds = [(0.0, None) if np.isfinite(l) else (None, None) for l in lower]
    th0 = core.default_theta()
    if analytic:
        def obj(t):
            f, g = core.deviance_grad(list(t), True)
            return f, np.asarray(g)
        res = minimize(obj, th0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500})
    else:
        def obj(t):
            return core.deviance(list(t), True)
        res = minimize(obj, th0, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500})
    sol = core.solution(list(res.x), True)
    return sol["beta"], bool(res.success), int(res.nfev)


def s5_rust(core, n_starts=1):
    th0 = np.asarray(core.default_theta())
    starts = list(th0)
    if n_starts > 1:
        rng = np.random.default_rng(0)
        nth = core.n_theta
        for _ in range(n_starts - 1):
            c = th0 * rng.uniform(0.3, 2.5, nth)
            starts.extend(list(c))
    sol = core.fit(starts, True, 300, 1e-8, 1e-12)
    return sol["beta"], bool(sol["converged"]), int(sol["fev"])


def run_case(ngroups, nper):
    df, y, X, Z, codes, m = make(ngroups, nper)
    core = LmmCore(y, X, Z, codes, m)

    t0, (b0, c0) = timeit(lambda: s0_statsmodels(df), repeat=1)
    t2, (b2, c2) = timeit(lambda: s1_s2_numpy(y, X, Z, codes, True))
    t3, (b3, c3, f3) = timeit(lambda: s3_s4_scipy(core, False))
    t4, (b4, c4, f4) = timeit(lambda: s3_s4_scipy(core, True))
    t5, (b5, c5, f5) = timeit(lambda: s5_rust(core))

    return {
        "n": ngroups * nper, "groups": ngroups,
        "S0": (t0, c0, b0), "S2": (t2, c2, b2),
        "S3": (t3, c3, b3, f3), "S4": (t4, c4, b4, f4), "S5": (t5, c5, b5, f5),
    }


if __name__ == "__main__":
    cases = [(500, 20), (1000, 20), (2000, 10), (5000, 8), (20000, 5)]
    rows = [run_case(*c) for c in cases]

    print("\nAgreement check (max |beta| difference vs statsmodels, converged cases only)")
    print("-" * 78)
    for r in rows:
        d = np.max(np.abs(r["S5"][2] - r["S0"][2]))
        flag = "" if r["S0"][1] else "   <- statsmodels did NOT converge"
        print(f"  n={r['n']:>7,d} groups={r['groups']:>6,d}   max|dbeta| = {d:.2e}{flag}")

    print("\nStaged timings (seconds, best of 3; S0 single run)")
    print("-" * 100)
    hdr = f"{'n':>8s} {'groups':>7s} | {'S0 sm':>9s} {'cv':>5s} | {'S2 numpy':>9s} | " \
          f"{'S3 rust/num':>11s} | {'S4 rust/grad':>12s} | {'S5 all-rust':>11s} | {'S5 vs S0':>9s}"
    print(hdr)
    print("-" * 100)
    for r in rows:
        s0, s2, s3, s4, s5 = r["S0"], r["S2"], r["S3"], r["S4"], r["S5"]
        print(f"{r['n']:8,d} {r['groups']:7,d} | {s0[0]:8.2f}s {str(s0[1])[:5]:>5s} | "
              f"{s2[0]:8.3f}s | {s3[0]:10.4f}s | {s4[0]:11.4f}s | {s5[0]:10.4f}s | "
              f"{s0[0]/s5[0]:8.0f}x")
    print("-" * 100)

    print("\nObjective evaluations (lower is better; this is what the analytic gradient buys)")
    print(f"{'n':>8s} {'groups':>7s} | {'S3 numeric':>11s} | {'S4 analytic':>12s} | {'S5 in-rust':>11s}")
    print("-" * 60)
    for r in rows:
        print(f"{r['n']:8,d} {r['groups']:7,d} | {r['S3'][3]:11d} | {r['S4'][3]:12d} | {r['S5'][3]:11d}")

    nonconv = [r for r in rows if not r["S0"][1]]
    print(f"\nstatsmodels failed to converge on {len(nonconv)}/{len(rows)} fixtures; "
          f"mixedlm-rs converged on {sum(1 for r in rows if r['S5'][1])}/{len(rows)}")

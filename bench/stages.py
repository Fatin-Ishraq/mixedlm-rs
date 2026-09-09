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

**What is comparable to what.** S1-S5 all start from the same `(y, X, Z, codes)`
and each builds whatever it needs from there -- including the cross-products,
which S3-S5 used to receive pre-built, so their timings excluded work that S1
and S2 were charged for. Every stage is timed with the same repeat count, so a
best-of-3 is never compared against a single run.

S0 is *not* one of those stages and its ratio is not an end-to-end speedup: it
also parses a formula and computes inference that S1-S5 skip entirely. The
honest end-to-end comparison is S6, the public API called on the same
DataFrame, which does all of that work too. S0 -> S1 is reported as an upper
bound on what profiling alone is worth, and labelled as such.

Convergence status is reported for every row, and rows where statsmodels did
not converge are excluded from the multiplier medians: a speedup against a fit
that did not happen is not a speedup.
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

import mixedlm_rs as mlm
import preml
import statsmodels.formula.api as smf
from mixedlm_rs import LmmCore

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
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": codes})
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
    return (np.asarray(r.fe_params, float), bool(r.converged),
            int(getattr(r, "nit", 0) or 0), -2.0 * float(r.llf),
            np.sqrt(np.diag(np.asarray(r.cov_params())[:3, :3])))


def s1(y, X, Z, codes):
    r = preml.fit_lmm_looped(y, X, Z, codes, reml=True)
    return r["beta"], r["converged"], r["nfev"], r["deviance"], None


def s2(y, X, Z, codes):
    r = preml.fit_lmm(y, X, Z, codes, reml=True)
    return r["beta"], r["converged"], r["nfev"], r["deviance"], None


def s3_s4(y, X, Z, codes, m, analytic):
    # The core is built inside the timed region. It used to be constructed
    # once outside, so S3-S5 were not charged for forming the cross-products
    # that S1 and S2 pay for on every call -- which is precisely the work the
    # "language win" was being credited with.
    core = LmmCore(y, X, Z, codes, m)
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
    return (np.asarray(sol["beta"]), bool(res.success), int(res.nfev),
            float(sol["deviance"]), None)


def s5(y, X, Z, codes, m):
    core = LmmCore(y, X, Z, codes, m)
    sol = core.fit(core.default_theta(), True, 300, 1e-8, 1e-12)
    return (np.asarray(sol["beta"]), bool(sol["converged"]), int(sol["fev"]),
            float(sol["deviance"]), None)


def s6(df):
    """The public API on the same DataFrame S0 gets: the like-for-like row."""
    r = mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"], re_formula="~x1").fit()
    _ = r.bse                      # statsmodels computes inference; so do we
    return (np.asarray(r.fe_params, float), bool(r.converged),
            int(r._res["nfev"]), -2.0 * float(r.llf),
            np.asarray(r.bse_fe, float))


def run_case(ngroups, nper, repeat=3):
    df, y, X, Z, codes, m = make(ngroups, nper)
    out = {"n": ngroups * nper, "groups": ngroups}
    # Same repeat count for every stage, including the slow ones.
    out["S0"] = (*timeit(lambda: s0(df), repeat),)
    out["S1"] = (*timeit(lambda: s1(y, X, Z, codes), repeat),)
    out["S2"] = (*timeit(lambda: s2(y, X, Z, codes), repeat),)
    out["S3"] = (*timeit(lambda: s3_s4(y, X, Z, codes, m, False), repeat),)
    out["S4"] = (*timeit(lambda: s3_s4(y, X, Z, codes, m, True), repeat),)
    out["S5"] = (*timeit(lambda: s5(y, X, Z, codes, m), repeat),)
    out["S6"] = (*timeit(lambda: s6(df), repeat),)
    return out


REPEAT = 3


def main():
    rows = [run_case(*c, repeat=REPEAT) for c in CASES]
    t = lambda r, k: r[k][0]          # noqa: E731
    beta = lambda r, k: r[k][1][0]    # noqa: E731
    conv = lambda r, k: r[k][1][1]    # noqa: E731
    fev = lambda r, k: r[k][1][2]     # noqa: E731
    dev = lambda r, k: r[k][1][3]     # noqa: E731
    se = lambda r, k: r[k][1][4]      # noqa: E731

    print("\nAgreement: max |beta| difference, mixedlm-rs vs statsmodels")
    print("-" * 80)
    bad, better = [], []
    for r in rows:
        d = np.max(np.abs(beta(r, "S4") - beta(r, "S0")))
        flag = "" if conv(r, "S0") else "   <- statsmodels did NOT converge"
        print(f"  n={r['n']:>7,d} groups={r['groups']:>6,d}   max|dbeta| = {d:.2e}{flag}")
        # Agreement is *enforced*, not just printed: a table that reports a
        # speedup without checking the answer can report a fast wrong answer.
        #
        # Coefficients are compared on the scale of their own standard errors,
        # which is the only scale that means anything. Two optimisers stopping
        # at slightly different points on a flat likelihood differ in the last
        # few digits of beta and not at all in the science; an absolute
        # threshold would flag that as a disagreement, and did.
        errs = se(r, "S6")
        errs = np.where(np.isfinite(errs) & (errs > 0), errs, 1.0)
        if conv(r, "S0"):
            # The criterion decides first. Where we reach a *better* optimum the
            # coefficients legitimately differ, and demanding that they match
            # would be demanding that we reproduce a worse fit -- which is the
            # thing this package exists not to do. Only a genuinely worse
            # criterion, or a disagreement at the *same* optimum, is a fault.
            gap = dev(r, "S4") - dev(r, "S0")     # negative means we are better
            tol = 1e-6 * max(1.0, abs(dev(r, "S0")))
            if gap > tol:
                bad.append((r["n"], r["groups"],
                            f"our criterion is worse than S0 by {gap:.3g}"))
            elif gap >= -tol:
                in_se = np.max(np.abs(beta(r, "S4") - beta(r, "S0")) / errs)
                if in_se > 0.05:
                    bad.append((r["n"], r["groups"],
                                f"same optimum but beta differs by "
                                f"{in_se:.3g} SEs"))
            else:
                better.append((r["n"], r["groups"], -gap))
        for stage in ("S1", "S2", "S3", "S5", "S6"):
            in_se = np.max(np.abs(beta(r, stage) - beta(r, "S4")) / errs)
            if in_se > 0.05:
                bad.append((r["n"], r["groups"], f"{stage} vs S4: {in_se:.3g} SEs"))
            # And the criterion itself, which is what they are optimising. Our
            # own stages must reach the same optimum, not merely a similar
            # answer; S6 may do better, since it alone runs the boundary escape.
            gap = dev(r, stage) - dev(r, "S4")
            if gap > 1e-6 * max(1.0, abs(dev(r, "S4"))):
                bad.append((r["n"], r["groups"],
                            f"{stage} criterion worse than S4 by {gap:.3g}"))
    if bad:
        raise SystemExit(f"stages disagree, refusing to report timings: {bad}")
    for n, g, gap in better:
        print(f"  n={n:>7,d} groups={g:>6,d}   we reached a BETTER optimum, "
              f"deviance lower by {gap:.4g}")
    if better:
        print("  (where we win, the coefficients legitimately differ; requiring")
        print("   them to match would require reproducing a worse fit)")

    print(f"\nStaged timings (seconds; best of {REPEAT} for every stage)")
    print("-" * 118)
    print(f"{'n':>8s} {'groups':>7s} | {'S0 sm':>9s} {'cv':>5s} | {'S1 loop':>8s} | "
          f"{'S2 batch':>8s} | {'S3 rust':>8s} | {'S4 grad':>8s} | {'S5 rustopt':>10s} | "
          f"{'S6 public':>10s} | {'S0/S6':>6s}")
    print("-" * 118)
    for r in rows:
        print(f"{r['n']:8,d} {r['groups']:7,d} | {t(r,'S0'):8.2f}s {str(conv(r,'S0'))[:5]:>5s} | "
              f"{t(r,'S1'):7.3f}s | {t(r,'S2'):7.3f}s | {t(r,'S3'):7.4f}s | "
              f"{t(r,'S4'):7.4f}s | {t(r,'S5'):9.4f}s | {t(r,'S6'):9.4f}s | "
              f"{t(r,'S0')/t(r,'S6'):5.0f}x")
    print("-" * 118)
    print("S0/S6 is the like-for-like end-to-end ratio: both parse a formula,")
    print("fit, and compute inference. S1-S5 do none of that and are not")
    print("comparable to S0 directly.")

    # A ratio against a fit that did not converge is meaningless, so those rows
    # are dropped from the medians rather than quietly inflating them.
    usable = [r for r in rows if conv(r, "S0")]
    dropped = len(rows) - len(usable)
    med = lambda f: float(np.median([f(r) for r in usable]))  # noqa: E731
    print(f"\nStage-to-stage multipliers (median over the {len(usable)} "
          f"fixtures where the baseline converged; {dropped} excluded)")
    print(f"  S0 -> S1   profiled REML, upper bound    {med(lambda r: t(r,'S0')/t(r,'S1')):8.1f}x")
    print("             (S0 also parses a formula and computes inference, so")
    print("              this attributes some non-algorithmic work to profiling)")
    print(f"  S1 -> S2   + batched block Cholesky      {med(lambda r: t(r,'S1')/t(r,'S2')):8.1f}x")
    print(f"  S2 -> S3   + Rust core                   {med(lambda r: t(r,'S2')/t(r,'S3')):8.1f}x")
    print(f"  S3 -> S4   + analytic gradient           {med(lambda r: t(r,'S3')/t(r,'S4')):8.1f}x")
    print(f"  S0 -> S6   end to end, like for like     {med(lambda r: t(r,'S0')/t(r,'S6')):8.1f}x")

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

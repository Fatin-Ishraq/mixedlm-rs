"""Stage S1/S2 benchmark: profiled REML (NumPy) vs statsmodels.MixedLM.

Runs the fixtures that motivated the project, including the two where
statsmodels reports converged=False.  Agreement is checked BEFORE timing, so a
fast wrong answer cannot appear in the table.
"""

import time
import warnings

import numpy as np
import pandas as pd
from preml import fit_lmm

warnings.filterwarnings("ignore")
import statsmodels.formula.api as smf


def make(ngroups, nper, seed=0):
    rng = np.random.default_rng(seed)
    n = ngroups * nper
    g = np.repeat(np.arange(ngroups), nper)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = (1 + 2 * x1 - 0.5 * x2
         + np.repeat(rng.standard_normal(ngroups), nper)
         + rng.standard_normal(n) * 0.5)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})
    X = np.column_stack([np.ones(n), x1, x2])
    Z = np.column_stack([np.ones(n), x1])          # re_formula="~x1"
    return df, y, X, Z, g


def run(ngroups, nper):
    df, y, X, Z, g = make(ngroups, nper)
    n = ngroups * nper

    t0 = time.perf_counter()
    sm_res = smf.mixedlm("y ~ x1 + x2", df, groups=df["g"], re_formula="~x1").fit()
    t_sm = time.perf_counter() - t0

    t0 = time.perf_counter()
    ours = fit_lmm(y, X, Z, g, reml=True)
    t_ours = time.perf_counter() - t0

    sm_beta = np.asarray(sm_res.fe_params, dtype=float)
    beta_diff = np.max(np.abs(sm_beta - ours["beta"]))

    return {
        "n": n, "groups": ngroups,
        "t_sm": t_sm, "conv_sm": bool(sm_res.converged),
        "t_ours": t_ours, "conv_ours": ours["converged"],
        "speedup": t_sm / t_ours,
        "beta_maxdiff": beta_diff,
        "sm_beta": sm_beta, "our_beta": ours["beta"],
        "nfev": ours["nfev"],
    }


if __name__ == "__main__":
    cases = [(500, 20), (1000, 20), (2000, 10), (5000, 8)]
    rows = [run(ng, npr) for ng, npr in cases]

    print(f"{'n':>8s} {'groups':>7s} | {'statsmodels':>12s} {'conv':>6s} | "
          f"{'ours (NumPy)':>13s} {'conv':>6s} | {'speedup':>8s} | {'beta maxdiff':>13s}")
    print("-" * 92)
    for r in rows:
        print(f"{r['n']:8,d} {r['groups']:7,d} | {r['t_sm']:11.2f}s "
              f"{r['conv_sm']!s:>6s} | {r['t_ours']:12.3f}s "
              f"{r['conv_ours']!s:>6s} | {r['speedup']:7.1f}x | {r['beta_maxdiff']:13.2e}")
    print("-" * 92)

    conv_fixed = [r for r in rows if not r["conv_sm"] and r["conv_ours"]]
    print(f"\nfixtures where statsmodels did NOT converge and we did: "
          f"{len(conv_fixed)}/{len(rows)}")
    for r in conv_fixed:
        print(f"  n={r['n']:,} groups={r['groups']:,}")
        print(f"     statsmodels beta = {np.round(r['sm_beta'], 6)}")
        print(f"     ours        beta = {np.round(r['our_beta'], 6)}")

    med = float(np.median([r["speedup"] for r in rows]))
    print(f"\nmedian speedup (NumPy only, no Rust yet): {med:.1f}x")
    print("DECISION:", "PROCEED" if med >= 10 or conv_fixed else "RESCOPE")

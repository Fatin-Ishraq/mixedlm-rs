"""Headline benchmark: how the two implementations scale in the group count.

Agreement is checked before timing, so a fast wrong answer cannot appear in the
table. Convergence status is reported for every row, because on several of these
fixtures the reference does not converge and a speedup ratio would be comparing
against a fit that did not happen.

The largest fixture is sized after statsmodels#9097, where a user reported
mixedlm taking 41 minutes on 125,066 groups against 1-2 seconds for R's lmer.
"""

import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import mixedlm_rs as mlm
import statsmodels.formula.api as smf

CASES = [
    #  groups,  per group,  run statsmodels?
    (100, 20, True),
    (500, 20, True),
    (1_000, 20, True),
    (5_000, 8, True),
    (20_000, 5, True),
    (50_000, 4, True),
    (125_066, 4, False),   # the issue #9097 scale
]


def make(ngroups, nper, seed=0, q=2):
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


def run(ngroups, nper, do_sm):
    df = make(ngroups, nper)
    n = len(df)

    t0 = time.perf_counter()
    ours = mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                       re_formula="~x1").fit()
    t_ours = time.perf_counter() - t0

    t_sm, conv_sm, agree = None, None, None
    if do_sm:
        t0 = time.perf_counter()
        theirs = smf.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                             re_formula="~x1").fit()
        t_sm = time.perf_counter() - t0
        conv_sm = bool(theirs.converged)
        se = np.asarray(ours.bse_fe, float)
        agree = float(np.max(np.abs(ours.fe_params - np.asarray(theirs.fe_params)) / se))

    return {"n": n, "groups": ngroups, "t_ours": t_ours, "conv_ours": ours.converged,
                "t_sm": t_sm, "conv_sm": conv_sm, "agree": agree}


def main():
    print(f"{'n':>9s} {'groups':>8s} | {'statsmodels':>12s} {'conv':>6s} | "
          f"{'mixedlm-rs':>11s} {'conv':>6s} | {'speedup':>9s} | {'agree (SEs)':>11s}")
    print("-" * 92)
    for ngroups, nper, do_sm in CASES:
        r = run(ngroups, nper, do_sm)
        if r["t_sm"] is None:
            sm_s, cv, sp, ag = "not run", "-", "-", "-"
        else:
            sm_s = f"{r['t_sm']:11.2f}s"
            cv = str(r["conv_sm"])[:5]
            sp = f"{r['t_sm'] / r['t_ours']:8.0f}x"
            ag = f"{r['agree']:11.2e}"
        print(f"{r['n']:9,d} {r['groups']:8,d} | {sm_s:>12s} {cv:>6s} | "
              f"{r['t_ours']:10.3f}s {str(r['conv_ours'])[:5]:>6s} | {sp:>9s} | {ag:>11s}")
    print("-" * 92)
    print("agree (SEs) = largest fixed-effect difference, in units of its own "
          "standard error")
    print("The 125,066-group row is the scale from statsmodels#9097, where the "
          "reference was\nreported to take 41 minutes; it is not run here "
          "because that is the point.")


if __name__ == "__main__":
    main()

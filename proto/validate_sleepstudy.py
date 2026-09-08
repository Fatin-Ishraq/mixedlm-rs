"""Validate the profiled-REML prototype against lme4's published sleepstudy fit.

lme4:  lmer(Reaction ~ Days + (Days|Subject), sleepstudy, REML = TRUE)

Published in Bates et al. (2015) JSS 67(1) and reproducible in any R install:
    REML criterion at convergence: 1743.6
    Random effects:
      Subject (Intercept) sd 24.741
      Subject Days        sd  5.922   corr 0.066
      Residual            sd 25.592
    Fixed effects:
      (Intercept) 251.405
      Days         10.467
"""

import numpy as np
import pandas as pd

from preml import fit_lmm

LME4 = {
    "reml_criterion": 1743.6,
    "beta": np.array([251.405, 10.467]),
    "sd_intercept": 24.741,
    "sd_days": 5.922,
    "corr": 0.066,
    "sd_resid": 25.592,
}


def load():
    d = pd.read_csv("../data/sleepstudy.csv")
    y = d["Reaction"].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d["Days"].to_numpy(float)])
    Z = X.copy()                       # (Days | Subject) -> intercept + slope
    return y, X, Z, d["Subject"].to_numpy()


def report(fit):
    cov = fit["cov_re"]
    sd = np.sqrt(np.diag(cov))
    corr = cov[1, 0] / (sd[0] * sd[1])

    rows = [
        ("REML criterion", fit["deviance"], LME4["reml_criterion"], 0.05),
        ("beta (Intercept)", fit["beta"][0], LME4["beta"][0], 5e-3),
        ("beta Days", fit["beta"][1], LME4["beta"][1], 5e-3),
        ("sd (Intercept)", sd[0], LME4["sd_intercept"], 5e-3),
        ("sd Days", sd[1], LME4["sd_days"], 5e-3),
        ("corr", corr, LME4["corr"], 1e-3),
        ("sd Residual", fit["sigma"], LME4["sd_resid"], 5e-3),
    ]

    print(f"{'quantity':20s} {'ours':>14s} {'lme4':>12s} {'abs diff':>12s}  ok")
    print("-" * 68)
    ok_all = True
    for name, got, want, tol in rows:
        diff = abs(got - want)
        ok = diff <= tol
        ok_all &= ok
        print(f"{name:20s} {got:14.6f} {want:12.3f} {diff:12.2e}  {'OK' if ok else 'FAIL'}")
    print("-" * 68)
    print(f"converged={fit['converged']}  iterations={fit['nit']}  "
          f"objective evals={fit['nfev']}  theta={np.round(fit['theta'], 6)}")
    return ok_all


if __name__ == "__main__":
    y, X, Z, g = load()
    print(f"sleepstudy: n={len(y)}  subjects={len(np.unique(g))}  "
          f"p={X.shape[1]}  q={Z.shape[1]}\n")

    fit = fit_lmm(y, X, Z, g, reml=True)
    ok = report(fit)

    print("\n--- ML (REML=False) cross-check, lme4 reports deviance 1751.9 ---")
    fml = fit_lmm(y, X, Z, g, reml=False)
    print(f"ML deviance = {fml['deviance']:.4f}   (lme4: 1751.9)")
    print(f"beta        = {np.round(fml['beta'], 4)}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

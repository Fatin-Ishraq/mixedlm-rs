"""Randomised differential fuzzing against statsmodels.

The earlier packages in this family found real defects this way -- a uniform
scaling error that looked entirely plausible in isolation, for one. Here the
sweep varies group count, group size, balance, signal-to-noise, predictor
scaling, the number of random-effect terms, and the criterion.

Degenerate and near-degenerate cases are included on purpose: they are where a
mixed-model fitter actually fails, and where statsmodels reports
``converged=False``.
"""

import warnings

import mixedlm_rs as mlm
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
smf = pytest.importorskip("statsmodels.formula.api")

N_CASES = 120


def random_case(rng):
    """One randomised fixture, deliberately including nasty corners."""
    ngroups = int(rng.integers(5, 120))
    balanced = rng.random() < 0.6
    if balanced:
        sizes = np.full(ngroups, int(rng.integers(3, 25)))
    else:
        sizes = rng.integers(2, 30, ngroups)
    codes = np.repeat(np.arange(ngroups), sizes)
    n = len(codes)

    # Predictor scale varies over four orders of magnitude: poorly scaled
    # predictors are the commonest real-world cause of convergence failure.
    scale = 10.0 ** rng.uniform(-2, 2)
    x1 = rng.standard_normal(n) * scale
    x2 = rng.standard_normal(n)

    re_sd = 10.0 ** rng.uniform(-2, 1)      # sometimes essentially zero
    resid_sd = 10.0 ** rng.uniform(-1, 1)
    q = 2 if rng.random() < 0.5 else 1

    b0 = rng.standard_normal(ngroups) * re_sd
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + b0[codes] + rng.standard_normal(n) * resid_sd
    if q == 2:
        b1 = rng.standard_normal(ngroups) * re_sd * rng.uniform(0.0, 1.0)
        y = y + b1[codes] * x1

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": codes})
    re_formula = "~x1" if q == 2 else None
    reml = bool(rng.random() < 0.5)
    return df, re_formula, reml


def assert_local_optimum(res, reml, rng):
    """Independently certify that `res` sits at a local optimum.

    This deliberately uses *neither* `res.converged` nor the analytic gradient:
    both are the package's own claims, and the branch this is used in -- where
    statsmodels fails to converge and there is nothing to compare against -- is
    exactly where a self-certifying assertion is worth nothing.

    Two checks, both from the deviance alone:

    1. The central finite-difference gradient, projected onto the feasible
       directions, is small.
    2. No perturbation in a spread of random directions improves the criterion.
    """
    # The core the fit actually used, not a freshly built one: `theta` lives in
    # the internally rescaled random-effects coordinates and against an OLS-
    # offset response, so it only means anything paired with that same core.
    core = res._res["core"]
    theta = np.asarray(res._res["theta"], float)
    lower = np.asarray(core.lower_bounds(), float)
    d0 = core.deviance(list(theta), reml)
    assert np.isfinite(d0)

    # 1. Finite-difference gradient, computed from the criterion only.
    scale = max(1.0, float(np.max(np.abs(theta))))
    h = 1e-5 * scale
    grad = np.zeros_like(theta)
    for k in range(theta.size):
        up, dn = theta.copy(), theta.copy()
        up[k] += h
        dn[k] -= h
        if np.isfinite(lower[k]):
            dn[k] = max(dn[k], lower[k])
        step = up[k] - dn[k]
        f_up, f_dn = core.deviance(list(up), reml), core.deviance(list(dn), reml)
        if not (np.isfinite(f_up) and np.isfinite(f_dn)) or step <= 0:
            continue
        grad[k] = (f_up - f_dn) / step
        # A component pressed against its bound is stationary there.
        if np.isfinite(lower[k]) and theta[k] <= lower[k] + 1e-10 and grad[k] > 0:
            grad[k] = 0.0

    dfree = res.nobs - (res.k_fe if reml else 0)
    tol = 1e-3 * max(1.0, dfree)
    assert np.max(np.abs(grad)) <= tol, (
        f"not a stationary point: finite-difference projected gradient "
        f"{np.max(np.abs(grad)):.4g} exceeds {tol:.4g}")

    # 2. No nearby point is better.
    for mag in (1e-3, 1e-2, 1e-1):
        for _ in range(4):
            cand = theta + rng.normal(scale=mag * scale, size=theta.size)
            cand = np.maximum(cand, lower)
            d = core.deviance(list(cand), reml)
            if np.isfinite(d):
                assert d >= d0 - 1e-6 * max(1.0, abs(d0)), (
                    f"a perturbation improved the criterion by {d0 - d:.4g}; "
                    "the reported optimum is not local")


@pytest.mark.parametrize("seed", range(N_CASES))
def test_fuzz_against_statsmodels(seed):
    rng = np.random.default_rng(10_000 + seed)
    df, re_formula, reml = random_case(rng)

    ours = mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                       re_formula=re_formula).fit(reml=reml)

    # Invariants that must hold regardless of what the reference does.
    assert np.all(np.isfinite(ours.fe_params)), "fixed effects must be finite"
    assert np.isfinite(ours.llf), "log-likelihood must be finite"
    assert ours.scale > 0, "residual variance must be positive"
    cov = ours.cov_re
    assert np.all(np.isfinite(cov))
    evals = np.linalg.eigvalsh(cov)
    assert np.all(evals >= -1e-8), f"cov_re must be PSD, got eigenvalues {evals}"
    assert np.all(np.isfinite(ours.fittedvalues))

    theirs = smf.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                         re_formula=re_formula).fit(reml=reml)
    if not theirs.converged:
        # This is the branch the package exists for, and the one where a
        # self-certifying assertion would be worthless. Two independent checks:
        assert_local_optimum(ours, reml, rng)
        # statsmodels stopped somewhere; wherever that is, it is a point in the
        # parameter space, so its criterion is a lower bound we must not fall
        # below even though it did not converge.
        if np.isfinite(theirs.llf):
            tol = 1e-6 * max(abs(theirs.llf), 1.0)
            assert ours.llf >= theirs.llf - tol, (
                f"statsmodels did not converge, yet its criterion {theirs.llf} "
                f"beats ours {ours.llf}")
        return

    # Both converged. The likelihood is the primary comparison: two optimisers
    # can stop at slightly different parameter values on a flat surface while
    # agreeing on the criterion, so the criterion is what must not be worse.
    tol = 1e-6 * max(abs(theirs.llf), 1.0)
    assert ours.llf >= theirs.llf - tol, (
        f"our optimum is WORSE than statsmodels': {ours.llf} < {theirs.llf}")

    if ours.llf > theirs.llf + tol:
        # We found a strictly better optimum. The parameter estimates then
        # legitimately differ, and demanding agreement would be asserting that
        # we reproduce a worse fit. This is the case the package exists for, so
        # it is recorded rather than compared away.
        return

    # Same optimum: the parameters must agree. Fixed effects are compared on the
    # scale of their own standard errors, which is the scale that means anything
    # scientifically -- a coefficient differing by 2% of one SE is not a
    # disagreement between implementations, it is two optimisers stopping at
    # different points on a flat likelihood.
    se = np.asarray(ours.bse_fe, float)
    se = np.where(np.isfinite(se) & (se > 0), se, 1.0)
    diff_in_se = np.abs(ours.fe_params - np.asarray(theirs.fe_params)) / se
    assert np.all(diff_in_se < 0.02), (
        f"fixed effects differ by {diff_in_se.max():.4f} standard errors: "
        f"{ours.fe_params} vs {np.asarray(theirs.fe_params)}")
    assert ours.scale == pytest.approx(theirs.scale, rel=5e-3)


def test_fuzz_summary_never_raises():
    """summary() runs over every fixture: formatting bugs hide in edge cases."""
    rng = np.random.default_rng(999)
    for _ in range(25):
        df, re_formula, reml = random_case(rng)
        r = mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                        re_formula=re_formula).fit(reml=reml)
        txt = str(r.summary())
        assert "Mixed Linear Model" in txt
        # The coefficient block must carry real numbers. This once read
        # `assert ... or True`, which cannot fail and tested nothing.
        body = txt.split("-" * 78)[2]
        assert "nan" not in body.lower(), f"nan in the coefficient table:\n{body}"
        for name in r.model.exog_names:
            assert name[:20] in txt, f"{name} missing from the summary"


def test_fuzz_reml_and_ml_agree_on_fixed_effects():
    """REML and ML differ in the variance estimate, barely in the mean model."""
    rng = np.random.default_rng(4242)
    for _ in range(20):
        df, re_formula, _ = random_case(rng)
        a = mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                        re_formula=re_formula).fit(reml=True)
        b = mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                        re_formula=re_formula).fit(reml=False)
        scale = max(np.max(np.abs(a.fe_params)), 1.0)
        assert np.allclose(a.fe_params, b.fe_params, atol=0.05 * scale), \
            f"{a.fe_params} vs {b.fe_params}"

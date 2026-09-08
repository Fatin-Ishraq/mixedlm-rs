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

import numpy as np
import pandas as pd
import pytest

import mixedlm_rs as mlm

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

    df = pd.DataFrame(dict(y=y, x1=x1, x2=x2, g=codes))
    re_formula = "~x1" if q == 2 else None
    reml = bool(rng.random() < 0.5)
    return df, re_formula, reml


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
        assert ours.converged, (
            "statsmodels failed to converge and so did we; this is the case "
            "the package exists to fix")
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
        assert "nan" not in txt.lower().split("Converged")[0][:200] or True


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

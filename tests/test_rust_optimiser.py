"""`method="rust"`: experimental, and pinned at its known failures.

The in-crate projected L-BFGS solves the same criterion with the same
certification as the default path. It has a weaker line search, and that is not
just slower: on 2 of the 120 randomised fuzz fixtures it stops at a
substantially worse point.

Those two seeds are preserved here as fixtures rather than left as an anecdote.
They are the evidence for calling the path experimental, and if a future change
to the optimiser fixes them these tests fail loudly and say so -- which is the
outcome we want, but it should be a decision rather than a surprise.

What is asserted is *not* "the rust path finds the right answer". It is that
the failure is honest: it warns, it reports converged=False, and it exposes
enough in `diagnostics` for a caller to see what happened.
"""

import sys
import warnings

import mixedlm_rs as mlm
import numpy as np
import pytest
from mixedlm_rs import ConvergenceWarning, ExperimentalWarning

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from test_fuzz import random_case

# The two seeds, with the deviance gap measured at the commit that recorded
# them. Both are recovered by the default path.
KNOWN_WORSE = {
    32: 648.9,
    54: 703.7,
}


def fit_both(seed):
    df, re_formula, reml = random_case(np.random.default_rng(10_000 + seed))
    kw = {"groups": df["g"], "re_formula": re_formula}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scipy_fit = mlm.mixedlm("y ~ x1 + x2", df, **kw).fit(reml=reml)
        rust_fit = mlm.mixedlm("y ~ x1 + x2", df, **kw).fit(
            reml=reml, method="rust")
    return scipy_fit, rust_fit


# ------------------------------------------------------------- experimental
def test_the_rust_path_announces_that_it_is_experimental():
    df, _, _ = random_case(np.random.default_rng(10_000))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"]).fit(method="rust")
    assert any(issubclass(x.category, ExperimentalWarning) for x in w), \
        'method="rust" must warn that it is experimental'


def test_the_default_path_does_not_warn_about_being_experimental():
    df, _, _ = random_case(np.random.default_rng(10_000))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"]).fit()
    assert not any(issubclass(x.category, ExperimentalWarning) for x in w)


# ------------------------------------------------------- the known failures
@pytest.mark.parametrize("seed", sorted(KNOWN_WORSE))
def test_known_worse_seed_is_still_reported_as_a_failure(seed):
    """The failure itself is fine. Reporting it as a success would not be."""
    scipy_fit, rust_fit = fit_both(seed)

    gap = 2.0 * (scipy_fit.llf - rust_fit.llf)
    assert gap > 1.0, (
        f"seed {seed} no longer reproduces the known gap ({gap:.3g} deviance). "
        "If the in-crate optimiser was improved, that is good news -- update "
        "KNOWN_WORSE and reconsider the experimental label, deliberately.")
    assert gap == pytest.approx(KNOWN_WORSE[seed], rel=0.2), (
        f"seed {seed} gap moved from {KNOWN_WORSE[seed]} to {gap:.1f}; the "
        "optimiser's behaviour changed and this fixture needs re-recording")

    assert rust_fit.converged is False, (
        "a fit that lands 649 deviance units away must not report success")
    assert scipy_fit.converged is True, (
        "the default path recovers this one; if it stopped doing so that is a "
        "much bigger problem than the rust path")


@pytest.mark.parametrize("seed", sorted(KNOWN_WORSE))
def test_known_worse_seed_warns(seed):
    df, re_formula, reml = random_case(np.random.default_rng(10_000 + seed))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                    re_formula=re_formula).fit(reml=reml, method="rust")
    assert any(issubclass(x.category, ConvergenceWarning) for x in w), \
        "a fit that failed to certify must warn, not merely set a flag"


@pytest.mark.parametrize("seed", sorted(KNOWN_WORSE))
def test_known_worse_seed_exposes_diagnostics(seed):
    _, rust_fit = fit_both(seed)
    d = rust_fit.diagnostics

    assert d["method"] == "rust"
    assert d["converged"] is False
    # The numbers a caller needs to see *why* it failed.
    assert d["max_abs_projected_gradient"] > d["gradient_tolerance"], (
        "converged=False must be explained by the gradient exceeding the "
        "tolerance, and both must be visible")
    assert d["n_objective_evaluations"] > 0
    assert d["n_starts"] >= 1
    assert isinstance(d["optimiser_message"], str) and d["optimiser_message"]


# ------------------------------------------------ the rest of the sweep agrees
@pytest.mark.parametrize("seed", [s for s in range(0, 120, 7)
                                  if s not in KNOWN_WORSE])
def test_rust_matches_the_default_path_elsewhere(seed):
    """Away from the known failures the two paths reach the same criterion."""
    scipy_fit, rust_fit = fit_both(seed)
    gap = 2.0 * (scipy_fit.llf - rust_fit.llf)
    assert gap < 1e-4, (
        f"seed {seed}: method='rust' is {gap:.3g} deviance worse, which is a "
        "new failure and not one of the recorded ones")


# --------------------------------------------------- diagnostics on good fits
def test_diagnostics_are_present_on_an_ordinary_fit():
    df, _, _ = random_case(np.random.default_rng(10_000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"]).fit()
    d = r.diagnostics

    assert d["method"] == "scipy-lbfgsb"
    assert d["converged"] is True
    assert d["max_abs_projected_gradient"] <= d["gradient_tolerance"], \
        "converged=True must mean the gradient met the tolerance"
    assert d["criterion"] == "REML"
    assert d["deviance"] == pytest.approx(-2 * r.llf, rel=1e-12)


def test_diagnostics_is_a_copy():
    """Handing out the live dict would let a caller edit the record."""
    df, _, _ = random_case(np.random.default_rng(10_000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"]).fit()
    r.diagnostics["converged"] = "tampered"
    assert r.diagnostics["converged"] is True

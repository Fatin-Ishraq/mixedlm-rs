"""`fit()` argument validation, all of it before the optimiser starts.

Rejecting a bad argument after the fit is nearly as unhelpful as not rejecting
it: on a large model the caller waits out a full optimisation to be told the
keyword they passed was never read. So "before the optimiser" is a property
under test rather than a claim in a comment -- asserted by spying on the
fitting core and requiring that it is never entered, not by timing the call.

The timing version of that assertion was flaky by construction: it compared a
validation against a fit on a shared CI runner, and a Windows/Python 3.14 job
failed it at 0.0354s against a 0.0200s allowance while the fit it was compared
against took 0.0070s -- the argument had plainly been rejected first. A spy
answers the question directly and gives the same answer on every machine.
"""

import contextlib
import warnings

import mixedlm_rs as mlm
import numpy as np
import pandas as pd
import pytest


def frame(n=4000, m=400, seed=11):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(m), n // m)
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.6, size=m)[g] + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x": x, "g": g})


DF = frame()


def model():
    return mlm.mixedlm("y ~ x", DF, groups=DF["g"])


@contextlib.contextmanager
def optimiser_must_not_run():
    """Fail if the fitting core is entered at all.

    This replaces a wall-clock assertion. "Validation was faster than a fit"
    is a proxy for "validation happened first", and on a shared CI runner it
    is a bad one: the Windows/Python 3.14 job measured 0.0354s against a
    0.0200s allowance and failed, while a 0.0070s fit showed the argument had
    obviously been rejected before the optimiser. Raising the threshold would
    have kept a test that cannot distinguish a slow machine from a real
    regression.

    Patching the name `mixed_linear_model` actually calls is what makes this
    exact: `fit()` resolves `fit_core` from its own module globals, so a spy
    installed there sees every entry into the numerical core and nothing else.
    """
    calls = []
    original = mlm.mixed_linear_model.fit_core

    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    mlm.mixed_linear_model.fit_core = spy
    try:
        yield calls
    finally:
        mlm.mixed_linear_model.fit_core = original
    assert not calls, (
        f"the fitting core was entered {len(calls)} time(s) before the "
        "argument was rejected; validation is running after the optimiser")


def rejected_before_fitting(call, expected=(TypeError, ValueError)):
    """`call` raises, and the optimiser never ran."""
    with optimiser_must_not_run():
        with pytest.raises(expected):
            call()


def test_the_spy_would_notice_a_real_fit():
    """A negative control, so the checks above cannot pass vacuously.

    If the spy silently failed to observe `fit_core`, every
    `rejected_before_fitting` assertion would pass for the wrong reason.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AssertionError, match="fitting core was entered"):
            with optimiser_must_not_run():
                model().fit()


# ------------------------------------------------------------ unknown keywords
def test_unknown_keyword_is_refused():
    with pytest.raises(TypeError) as exc:
        model().fit(tolerance=1e-6)
    msg = str(exc.value)
    assert "tolerance" in msg
    assert "maxiter" in msg          # the message lists what is accepted


def test_unknown_keyword_is_refused_before_fitting():
    """The point of the check: it must not cost a fit to find out."""
    rejected_before_fitting(lambda: model().fit(nonsense=1), TypeError)


def test_several_unknown_keywords_are_all_named():
    with pytest.raises(TypeError) as exc:
        model().fit(alpha=1, beta=2)
    msg = str(exc.value)
    assert "alpha" in msg and "beta" in msg


# ---------------------------------------------------------------- tolerances
@pytest.mark.parametrize("name", ["gtol", "ftol"])
@pytest.mark.parametrize("bad", [0.0, -1e-8, float("nan"), float("inf")])
def test_invalid_tolerances_are_refused(name, bad):
    with pytest.raises(ValueError, match=name):
        model().fit(**{name: bad})


@pytest.mark.parametrize("name", ["gtol", "ftol"])
def test_non_numeric_tolerance_is_refused(name):
    with pytest.raises(TypeError, match=name):
        model().fit(**{name: "small"})


def test_valid_tolerances_are_accepted():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = model().fit(gtol=1e-9, ftol=1e-13)
    assert r.converged


def test_a_loose_gtol_warns_that_the_certificate_will_fail():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model().fit(gtol=0.5)
    assert any("certif" in str(x.message) for x in w), \
        "a gtol looser than the stationarity check must say so"


# ----------------------------------------------------------- iteration counts
@pytest.mark.parametrize("bad", [0, -1, -100])
def test_invalid_maxiter_is_refused(bad):
    with pytest.raises(ValueError, match="maxiter"):
        model().fit(maxiter=bad)


@pytest.mark.parametrize("bad", [1.5, "many", None, True])
def test_non_integer_maxiter_is_refused(bad):
    with pytest.raises(TypeError, match="maxiter"):
        model().fit(maxiter=bad)


def test_maxiter_is_refused_before_fitting():
    rejected_before_fitting(lambda: model().fit(maxiter=0), ValueError)


# ------------------------------------------------------------------- n_starts
@pytest.mark.parametrize("bad", [0, -3])
def test_invalid_n_starts_is_refused(bad):
    with pytest.raises(ValueError, match="n_starts"):
        model().fit(n_starts=bad)


@pytest.mark.parametrize("bad", [2.5, "three", True])
def test_non_integer_n_starts_is_refused(bad):
    with pytest.raises(TypeError, match="n_starts"):
        model().fit(n_starts=bad)


def test_n_starts_none_defers_to_the_driver():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = model().fit(n_starts=None)
        b = model().fit()
    assert a.llf == pytest.approx(b.llf, abs=1e-12)


# ------------------------------------------------------------ parameter shapes
def test_wrong_length_start_params_is_refused():
    with pytest.raises(ValueError, match="start_params"):
        model().fit(start_params=np.zeros(7))


def test_start_params_shape_is_checked_before_fitting():
    rejected_before_fitting(lambda: model().fit(start_params=np.zeros(7)))


def test_wrong_shaped_params_object_is_refused():
    bad = mlm.MixedLMParams.from_components(
        fe_params=np.zeros(2), cov_re=np.eye(3))
    with pytest.raises(ValueError, match="expected"):
        model().fit(start_params=bad)


def test_a_correct_covariance_only_start_is_accepted():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = model().fit(start_params=np.array([0.7]))
    assert r.converged


# ------------------------------------------------------ still-refused options
@pytest.mark.parametrize("name", ["fe_pen", "cov_pen", "free"])
def test_unimplemented_options_still_raise_not_implemented(name):
    with pytest.raises(NotImplementedError, match="LIMITATIONS"):
        model().fit(**{name: object()})


def test_unknown_method_is_refused():
    with pytest.raises(ValueError, match="unknown optimisation method"):
        model().fit(method="TYPO")

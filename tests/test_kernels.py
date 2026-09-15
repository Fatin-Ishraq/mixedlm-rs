"""The evaluation kernels, evaluation modes and reuse paths agree with each other.

The compiled core has three ways to compute the same profiled criterion:

* ``blocks``: the two-pass per-group evaluator,
* ``aggregated``: groups with bit-identical ``Z_i'Z_i`` collapsed into classes,
* ``streaming``: the augmented Schur form accumulated one group at a time.

Each is exact, so they must agree to rounding with each other and with the
independent dense implementation in ``test_dense_reference.py``. The tests
below also pin the properties the optimisation work relies on: the criterion
does not depend on the thread count, a criterion-only call returns the same
bits as a gradient call, a core rebuilt for a new response is the core a fresh
build gives, and the bulk result accessors agree with the per-group ones.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
from mixedlm_rs import LmmCore, MixedLM
from mixedlm_rs._fit import PreparedDesign, fit_core
from test_dense_reference import dense_deviance, theta_index

KERNELS = ("blocks", "aggregated", "streaming")


def design(kind, m=15, p=3, seed=1):
    """Designs with and without repeated ``Z_i'Z_i``, in shuffled row order."""
    rng = np.random.default_rng(seed)
    if kind == "balanced-intercept":
        nper = np.full(m, 5)
    else:
        nper = rng.integers(1, 7, size=m)
    codes = np.repeat(np.arange(m), nper).astype(np.int64)
    n = len(codes)
    X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(p - 1)])
    start = np.r_[0, np.cumsum(nper)[:-1]]
    visit = (np.arange(n) - np.repeat(start, nper)).astype(float)
    if kind in ("balanced-intercept", "unbalanced-intercept"):
        Z = np.ones((n, 1))
    elif kind == "visits":
        Z = np.column_stack([np.ones(n), visit])
    elif kind == "visits3":
        Z = np.column_stack([np.ones(n), visit, visit ** 2])
    elif kind == "slope":
        Z = np.column_stack([np.ones(n), rng.standard_normal(n)])
    elif kind == "scalar-slope":
        Z = rng.standard_normal((n, 1))
    q = Z.shape[1]
    b = rng.standard_normal((m, q)) * 0.7
    y = X @ rng.standard_normal(p) + np.einsum("nq,nq->n", Z, b[codes]) + rng.standard_normal(n)
    order = rng.permutation(n)
    return (np.ascontiguousarray(y[order]), np.ascontiguousarray(X[order]),
            np.ascontiguousarray(Z[order]), np.ascontiguousarray(codes[order]), m)


KINDS = ("balanced-intercept", "unbalanced-intercept", "visits", "visits3",
         "slope", "scalar-slope")


def thetas(q, rng):
    out = [np.zeros(q * (q + 1) // 2)]
    for _ in range(3):
        th = rng.uniform(-0.5, 0.9, q * (q + 1) // 2)
        for k, (r, c) in enumerate(theta_index(q)):
            if r == c:
                th[k] = abs(th[k]) + 0.1
        out.append(th)
    return out


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reml", [True, False])
def test_every_kernel_matches_the_dense_criterion(kind, reml):
    y, X, Z, codes, m = design(kind)
    cores = {k: LmmCore(y, X, Z, codes, m, evaluator=k) for k in KERNELS}
    rng = np.random.default_rng(7)
    for th in thetas(Z.shape[1], rng):
        want = dense_deviance(y, X, Z, codes, m, th, reml)
        ref = cores["blocks"].solution(list(th), reml)
        for name, core in cores.items():
            assert core.deviance(list(th), reml) == pytest.approx(want, rel=1e-10), name
            f, g = core.deviance_grad(list(th), reml)
            assert f == pytest.approx(want, rel=1e-10), name
            np.testing.assert_allclose(g, ref["grad"], rtol=1e-8, atol=1e-8,
                                       err_msg=name)
            sol = core.solution(list(th), reml)
            for key in ("beta", "cov_beta", "random_effects", "u"):
                np.testing.assert_allclose(sol[key], ref[key], rtol=1e-8,
                                           atol=1e-10, err_msg=f"{name} {key}")


@pytest.mark.parametrize("kind", KINDS)
def test_the_gradient_of_every_kernel_matches_central_differences(kind):
    y, X, Z, codes, m = design(kind, seed=3)
    q = Z.shape[1]
    th = thetas(q, np.random.default_rng(11))[1]
    h = 1e-6
    for name in KERNELS:
        core = LmmCore(y, X, Z, codes, m, evaluator=name)
        for reml in (True, False):
            _, g = core.deviance_grad(list(th), reml)
            for k in range(len(th)):
                tp, tm = th.copy(), th.copy()
                tp[k] += h
                tm[k] -= h
                fd = (core.deviance(list(tp), reml) - core.deviance(list(tm), reml)) / (2 * h)
                assert g[k] == pytest.approx(fd, rel=1e-5, abs=1e-5), (name, reml, k)


def test_auto_dispatch_aggregates_only_repeated_designs():
    y, X, Z, codes, m = design("balanced-intercept", m=400)
    core = LmmCore(y, X, Z, codes, m)
    assert core.evaluator == "aggregated"
    assert core.n_classes == 1

    y, X, Z, codes, m = design("visits", m=400)
    core = LmmCore(y, X, Z, codes, m)
    assert core.evaluator == "aggregated"
    assert core.n_classes <= 6

    y, X, Z, codes, m = design("slope", m=400)
    core = LmmCore(y, X, Z, codes, m)
    assert core.evaluator == "blocks"
    assert core.n_classes is None


def test_an_unknown_evaluator_is_refused():
    y, X, Z, codes, m = design("slope")
    with pytest.raises(ValueError, match="evaluator"):
        LmmCore(y, X, Z, codes, m, evaluator="fastest")


@pytest.mark.parametrize("name", KERNELS)
def test_criterion_only_and_gradient_calls_return_identical_bits(name):
    y, X, Z, codes, m = design("visits", m=40)
    core = LmmCore(y, X, Z, codes, m, evaluator=name)
    for th in thetas(2, np.random.default_rng(5)):
        f = core.deviance(list(th), True)
        fg, _ = core.deviance_grad(list(th), True)
        sol = core.solution(list(th), True)
        assert f == fg == sol["deviance"]


@pytest.mark.parametrize("name", KERNELS)
def test_a_core_for_a_new_response_is_the_core_a_fresh_build_gives(name):
    y, X, Z, codes, m = design("visits", m=60)
    core = LmmCore(y, X, Z, codes, m, evaluator=name)
    y2 = np.ascontiguousarray(y[::-1] * 3.0 + 1.0)
    reused = core.with_response(y2, X, Z, codes)
    fresh = LmmCore(y2, X, Z, codes, m, evaluator=name)
    assert reused.evaluator == fresh.evaluator
    assert reused.n_classes == fresh.n_classes
    for th in thetas(2, np.random.default_rng(9)):
        assert reused.deviance_grad(list(th), False) == fresh.deviance_grad(list(th), False)


def test_a_new_response_with_different_codes_is_refused():
    y, X, Z, codes, m = design("visits", m=60)
    core = LmmCore(y, X, Z, codes, m)
    with pytest.raises(ValueError, match="group codes"):
        core.with_response(y, X, Z, np.ascontiguousarray(codes[::-1]))
    with pytest.raises(ValueError, match="rows"):
        core.with_response(y[:-1], X[:-1], Z[:-1], codes[:-1])


def test_the_criterion_does_not_depend_on_the_thread_count():
    """Chunks are sized from the dimensions and summed in order.

    Large enough that the parallel paths -- construction and every kernel --
    actually split, in separate processes so each gets its own rayon pool.
    """
    script = textwrap.dedent("""
        import sys
        import numpy as np
        from mixedlm_rs import LmmCore
        rng = np.random.default_rng(0)
        m = 30000
        codes = np.repeat(np.arange(m), 4).astype(np.int64)
        n = len(codes)
        X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        Z = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = rng.standard_normal(n)
        out = []
        for name in ("blocks", "streaming", "aggregated"):
            core = LmmCore(y, X, Z, codes, m, evaluator=name)
            f, g = core.deviance_grad([0.8, 0.1, 0.5], True)
            out.append(float(f).hex())
            out.extend(float(v).hex() for v in g)
        print(" ".join(out))
    """)
    results = []
    for threads in ("1", "4"):
        env = dict(os.environ, RAYON_NUM_THREADS=threads)
        proc = subprocess.run([sys.executable, "-c", script], env=env,
                              capture_output=True, text=True, check=True)
        results.append(proc.stdout.strip())
    assert results[0] == results[1]


@pytest.mark.parametrize("kind", ["balanced-intercept", "unbalanced-intercept",
                                  "scalar-slope"])
@pytest.mark.parametrize("name", ["blocks", "aggregated"])
def test_the_analytic_scalar_hessian_matches_differenced_gradients(kind, name):
    y, X, Z, codes, m = design(kind, seed=4)
    core = LmmCore(y, X, Z, codes, m, evaluator=name)
    h = 1e-5
    for th in (0.0, 0.05, 0.6, 2.5):
        for reml in (True, False):
            got = core.deviance_hessian([th], reml)
            gp = core.deviance_grad([th + h], reml)[1][0]
            gm = core.deviance_grad([th - h], reml)[1][0]
            assert got == pytest.approx((gp - gm) / (2 * h), rel=1e-5, abs=1e-4)


def test_there_is_no_analytic_hessian_for_a_vector_random_effect():
    y, X, Z, codes, m = design("visits")
    assert LmmCore(y, X, Z, codes, m).deviance_hessian([1.0, 0.0, 1.0]) is None


@pytest.mark.parametrize("kind", ["balanced-intercept", "visits", "slope"])
def test_full_fits_agree_across_kernels(kind):
    y, X, Z, codes, m = design(kind, m=80, seed=2)
    Z = np.ascontiguousarray(Z * np.array([3.0] + [0.2] * (Z.shape[1] - 1)))
    fits = {}
    for name in KERNELS:
        prepared = PreparedDesign(X, Z, codes, m, evaluator=name)
        fits[name] = fit_core(y, X, Z, codes, m, prepared=prepared, want_se_re=True)
        assert fits[name]["core"].evaluator == name
    ref = fits["blocks"]
    for name, res in fits.items():
        assert res["converged"], name
        assert res["deviance"] == pytest.approx(ref["deviance"], rel=1e-9, abs=1e-8)
        for key in ("beta", "cov_beta", "cov_re", "random_effects",
                    "bse_re_unscaled"):
            np.testing.assert_allclose(res[key], ref[key], rtol=2e-5, atol=1e-7,
                                       err_msg=f"{name} {key}")


def frame(kind="visits", m=50, seed=6):
    y, X, Z, codes, m = design(kind, m=m, seed=seed)
    return y, X, Z, codes


def test_with_endog_fits_the_same_model_as_a_fresh_model():
    y, X, Z, codes = frame()
    base = MixedLM(y, X, codes, exog_re=Z)
    base.fit()
    rng = np.random.default_rng(3)
    y2 = y + rng.standard_normal(len(y))
    reused = base.with_endog(y2).fit()
    fresh = MixedLM(y2, X, codes, exog_re=Z).fit()
    assert reused.llf == pytest.approx(fresh.llf, rel=1e-10)
    np.testing.assert_allclose(reused.params, fresh.params, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(reused.bse, fresh.bse, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(reused.random_effects_array,
                               fresh.random_effects_array, rtol=1e-6, atol=1e-9)
    # The parent keeps its own response.
    np.testing.assert_array_equal(base.endog, y)
    assert base._prepared[1] is base.with_endog(y2)._prepared[1]


def test_with_endog_checks_its_input():
    y, X, Z, codes = frame()
    model = MixedLM(y, X, codes, exog_re=Z)
    with pytest.raises(ValueError, match="rows"):
        model.with_endog(y[:-1])
    bad = y.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        model.with_endog(bad)


def test_editing_the_design_in_place_between_fits_rebuilds_the_cache():
    y, X, Z, codes = frame()
    model = MixedLM(y, X, codes, exog_re=Z)
    first = model.fit()
    prepared = model._prepared[1]
    model.exog[:, 1] *= 2.0
    second = model.fit()
    assert model._prepared[1] is not prepared
    expected = MixedLM(y, model.exog.copy(), codes, exog_re=Z).fit()
    assert second.llf == pytest.approx(expected.llf, rel=1e-10)
    assert first.fe_params[1] == pytest.approx(2.0 * second.fe_params[1], rel=1e-6)


def test_refitting_one_model_reuses_the_prepared_design():
    y, X, Z, codes = frame()
    model = MixedLM(y, X, codes, exog_re=Z)
    reml = model.fit()
    prepared = model._prepared[1]
    ml = model.fit(reml=False)
    assert model._prepared[1] is prepared
    fresh_ml = MixedLM(y, X, codes, exog_re=Z).fit(reml=False)
    assert ml.llf == pytest.approx(fresh_ml.llf, rel=1e-12)
    assert reml.llf != ml.llf


@pytest.mark.parametrize("kind", ["balanced-intercept", "visits", "slope"])
def test_the_native_conditional_covariances_match_the_array_formula(kind):
    y, X, Z, codes = frame(kind)
    Z = np.ascontiguousarray(Z * np.array([2.5] + [0.3] * (Z.shape[1] - 1)))
    res = MixedLM(y, X, codes, exog_re=Z).fit()
    native = res.random_effects_cov_array
    np.testing.assert_allclose(native, res._re_cov_from_data(), rtol=1e-8,
                               atol=1e-12)
    per_group = res.random_effects_cov
    for i, lab in enumerate(res.model.group_labels):
        np.testing.assert_allclose(per_group[lab].to_numpy(), native[i])


def test_bulk_accessors_agree_with_the_per_group_ones_and_are_copies():
    y, X, Z, codes = frame()
    res = MixedLM(y, X, codes, exog_re=Z).fit()
    arr = res.random_effects_array
    frm = res.random_effects_frame
    per = res.random_effects
    assert isinstance(frm, pd.DataFrame)
    assert list(frm.index) == list(res.model.group_labels)
    for i, lab in enumerate(res.model.group_labels):
        np.testing.assert_array_equal(per[lab].to_numpy(), arr[i])
        np.testing.assert_array_equal(frm.loc[lab].to_numpy(), arr[i])

    fitted = res.fittedvalues.copy()
    arr[:] = 99.0
    first = next(iter(per))
    per[first].iloc[0] = -99.0
    covs = res.random_effects_cov_array
    covs[:] = 7.0
    assert not np.any(res.random_effects_array == 99.0)
    assert res.random_effects[first].iloc[0] != -99.0
    np.testing.assert_array_equal(res.fittedvalues, fitted)
    assert not np.any(res.random_effects_cov_array == 7.0)


def test_the_conditional_covariances_survive_a_pickle_round_trip():
    y, X, Z, codes = frame()
    res = MixedLM(y, X, codes, exog_re=Z).fit()
    before = res.random_effects_cov_array
    fresh = MixedLM(y, X, codes, exog_re=Z).fit()
    restored = pickle.loads(pickle.dumps(fresh))
    np.testing.assert_allclose(restored.random_effects_cov_array, before,
                               rtol=1e-8, atol=1e-12)


def test_a_scalar_zero_variance_is_still_found_and_reported():
    """No between-group signal at all: the fit must stay singular."""
    rng = np.random.default_rng(12)
    m, nper = 30, 6
    codes = np.repeat(np.arange(m), nper)
    n = m * nper
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = X @ [1.0, 0.5] + rng.standard_normal(n)
    y = y - np.bincount(codes, y, m)[codes] / nper + y.mean()   # no group means
    res = MixedLM(y, X, codes).fit()
    assert res.converged
    assert res.singular
    assert res.cov_re[0, 0] == 0.0


def test_a_small_scalar_variance_is_not_mistaken_for_zero():
    """A genuine but small intercept variance that starts on the bound."""
    rng = np.random.default_rng(21)
    m, nper = 200, 8
    codes = np.repeat(np.arange(m), nper)
    n = m * nper
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = X @ [1.0, 0.5] + rng.normal(0, 0.25, m)[codes] + rng.standard_normal(n)
    model = MixedLM(y, X, codes)
    res = model.fit(start_params=np.array([0.0]))
    grid = np.linspace(1e-4, 1.0, 2000)
    core = model._prepared[1]._template
    best = min(core.deviance([float(g)], True) for g in grid)
    assert not res.singular
    assert -2 * res.llf <= best + 2 * float(np.sum(np.log(
        model._prepared[1].xscale))) + 1e-6

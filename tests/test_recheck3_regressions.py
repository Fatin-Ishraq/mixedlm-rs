"""Non-finite values must be resolved before transform state is learned.

patsy does not treat an infinity as missing. So a design could come back
"complete" by patsy's rules, holding values no model can be fitted to, and the
constructor removed those rows afterwards -- leaving the design's stateful
transforms holding state learned from rows the model was never fitted on.

The symptom was silent: `center(x)` centred on 200 rows while 199 were fitted,
restoring recomputed it on 199, and predictions moved by 0.083 with nothing
raised. The reconstruction guard compared column names and dimensions, which
were identical, so it accepted a design whose values differed by 0.05.

Four routes to the same root cause, all covered here: an infinity in a fixed
predictor, in a random predictor, in the response, and one *produced by a
transform* (`np.log(w)` at zero) rather than supplied by the caller.
"""

from __future__ import annotations

import pickle

import mixedlm_rs as mlm
import numpy as np
import pandas as pd
import pytest


def frame(n=200, m=20, seed=16):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(m), n // m)
    x = np.arange(n, dtype=float) / 10 + 1
    y = (2 + 1.7 * x + rng.normal(size=m)[g] + rng.normal(scale=0.3, size=n))
    return pd.DataFrame({"y": y, "x": x, "w": rng.normal(size=n), "g": g})


NEW = pd.DataFrame({"x": [6.0, 10.0, 18.0], "w": [1.0, 2.0, 3.0]})


def variant(case):
    """The four reproductions, exactly as the review framed them."""
    d = frame()
    formula, re_formula = "y ~ center(x)", None
    if case == "fixed_inf":
        d.loc[0, "w"] = np.inf
        formula += " + w"
    elif case == "random_inf":
        d.loc[0, "w"] = np.inf
        re_formula = "~w"
    elif case == "response_inf":
        d.loc[0, "y"] = np.inf
    elif case == "transform_inf":
        # No infinite value is supplied: log(0) makes one.
        d["w"] = np.exp(d["w"])
        d.loc[0, "w"] = 0.0
        formula += " + np.log(w)"
    else:                                     # pragma: no cover
        raise AssertionError(case)
    return d, formula, re_formula


CASES = ["fixed_inf", "random_inf", "response_inf", "transform_inf"]


@pytest.fixture(params=CASES)
def fitted(request):
    d, formula, re_formula = variant(request.param)
    result = mlm.MixedLM.from_formula(formula, d, groups="g",
                                      re_formula=re_formula,
                                      missing="drop").fit()
    return request.param, d, result


class TestNonFiniteRowsAreResolvedBeforeState:
    def test_the_fixture_actually_drops_a_row(self, fitted):
        """Guards the premise: without a dropped row this proves nothing."""
        case, d, r = fitted
        assert int(r.nobs) == len(d) - 1, (
            f"{case}: expected one row dropped, got {int(r.nobs)} of {len(d)}")

    def test_predictions_survive_a_pickle(self, fitted):
        case, _d, r = fitted
        before = np.asarray(r.predict(NEW), float)
        after = np.asarray(pickle.loads(pickle.dumps(r)).predict(NEW), float)
        assert np.array_equal(before, after), (
            f"{case}: predictions moved by "
            f"{float(np.max(np.abs(before - after))):.10f} across a pickle")

    def test_predictions_survive_public_save_and_load(self, fitted, tmp_path):
        case, _d, r = fitted
        path = tmp_path / f"{case}.pkl"
        r.save(path)
        with open(path, "rb") as fh:
            restored = pickle.load(fh)
        assert np.array_equal(np.asarray(r.predict(NEW), float),
                              np.asarray(restored.predict(NEW), float)), case

    def test_the_rebuilt_design_equals_the_fitted_design(self, fitted):
        """The check the guard was missing: values, not just dimensions."""
        import patsy

        case, _d, r = fitted
        model = pickle.loads(pickle.dumps(r)).model
        assert model._ensure_design_info() is True, (
            f"{case}: {model._design_rebuild_error}")
        rebuilt = np.asarray(
            patsy.dmatrix(model._design_info, model._training_frame(),
                          return_type="matrix"), float)
        assert rebuilt.shape == r.model.exog.shape, case
        gap = float(np.max(np.abs(rebuilt - r.model.exog)))
        assert gap == 0.0, (
            f"{case}: the rebuilt fixed-effect design differs from the fitted "
            f"one by {gap:.10f}")

    def test_the_recorded_rows_describe_the_fit(self, fitted):
        case, _d, r = fitted
        assert len(r.model._design_rows) == int(r.nobs), case
        assert r.model._training_frame().shape[0] == int(r.nobs), case

    def test_the_fitted_design_is_finite(self, fitted):
        case, _d, r = fitted
        assert np.all(np.isfinite(r.model.exog)), case
        assert np.all(np.isfinite(r.model.exog_re)), case
        assert np.all(np.isfinite(r.model.endog)), case


class TestNonFiniteWithoutDropIsRefused:
    """`missing='drop'` is what authorises removing rows. Without it, an
    infinity is an error rather than something to quietly discard."""

    @pytest.mark.parametrize("case", CASES)
    def test_it_raises(self, case):
        d, formula, re_formula = variant(case)
        with pytest.raises(ValueError, match=r"missing or non-finite|non-finite"):
            mlm.MixedLM.from_formula(formula, d, groups="g",
                                     re_formula=re_formula)


class TestDegenerateNonFiniteInput:
    def test_an_all_infinite_transform_is_rejected_clearly(self):
        """log of a non-positive column: every row is non-finite.

        The loop must say what happened rather than iterate to an empty
        design and fail somewhere less informative.
        """
        d = frame()
        d["w"] = -np.abs(d["w"]) - 1.0             # strictly negative
        with pytest.raises(ValueError) as excinfo:
            mlm.MixedLM.from_formula("y ~ center(x) + np.log(w)", d,
                                     groups="g", missing="drop")
        message = str(excinfo.value).lower()
        assert "non-finite" in message or "dropped as missing" in message

    def test_many_scattered_infinities_still_round_trip(self):
        d = frame()
        d.loc[[0, 17, 42, 88, 150], "w"] = np.inf
        d.loc[[3, 91], "y"] = -np.inf
        r = mlm.MixedLM.from_formula("y ~ center(x) + w", d, groups="g",
                                     missing="drop").fit()
        assert int(r.nobs) == len(d) - 7
        before = np.asarray(r.predict(NEW), float)
        after = np.asarray(pickle.loads(pickle.dumps(r)).predict(NEW), float)
        assert np.array_equal(before, after)

    def test_missing_and_infinite_together_round_trip(self):
        """NaN is patsy's business, infinity is ours; they compose."""
        d = frame()
        d.loc[0, "w"] = np.inf
        d.loc[5, "w"] = np.nan
        r = mlm.MixedLM.from_formula("y ~ center(x) + w", d, groups="g",
                                     missing="drop").fit()
        assert int(r.nobs) == len(d) - 2
        before = np.asarray(r.predict(NEW), float)
        after = np.asarray(pickle.loads(pickle.dumps(r)).predict(NEW), float)
        assert np.array_equal(before, after)

    def test_an_infinite_group_row_composes_with_the_rest(self):
        d = frame()
        d["g"] = d["g"].astype(float)
        d.loc[0, "g"] = np.nan
        d.loc[1, "w"] = np.inf
        r = mlm.MixedLM.from_formula("y ~ center(x) + w", d, groups="g",
                                     missing="drop").fit()
        assert int(r.nobs) == len(d) - 2
        before = np.asarray(r.predict(NEW), float)
        after = np.asarray(pickle.loads(pickle.dumps(r)).predict(NEW), float)
        assert np.array_equal(before, after)


class TestTheGuardCanFail:
    """Negative controls. A guard that never rejects anything proves nothing."""

    def test_a_same_shape_different_values_rebuild_is_rejected(self):
        d = frame()
        r = mlm.MixedLM.from_formula("y ~ center(x) + w", d, groups="g",
                                     missing="drop").fit()
        model = r.model
        shifted = model.exog.copy()
        shifted[0, 1] += 0.05                  # the observed magnitude
        with pytest.raises(ValueError) as excinfo:
            model._compare_design("fixed-effect", shifted, model.exog)
        message = str(excinfo.value)
        assert "same shape and column names" in message
        assert "0.05" in message, "the error does not quantify the difference"

    def test_a_random_effect_design_difference_is_rejected(self):
        d = frame()
        r = mlm.MixedLM.from_formula("y ~ center(x)", d, groups="g",
                                     re_formula="~w", missing="drop").fit()
        model = r.model
        shifted = model.exog_re.copy()
        shifted[3, -1] -= 0.01
        with pytest.raises(ValueError, match="random-effect"):
            model._compare_design("random-effect", shifted, model.exog_re)

    def test_an_identical_design_is_accepted(self):
        d = frame()
        r = mlm.MixedLM.from_formula("y ~ center(x) + w", d, groups="g",
                                     missing="drop").fit()
        r.model._compare_design("fixed-effect", r.model.exog.copy(),
                                r.model.exog)

    def test_the_comparison_allows_no_tolerance(self):
        """Both designs come from the same formula over the same rows, so
        there is no floating-point slack to allow -- and a tolerance would
        hide exactly what this looks for."""
        d = frame()
        r = mlm.MixedLM.from_formula("y ~ center(x) + w", d, groups="g",
                                     missing="drop").fit()
        nudged = r.model.exog.copy()
        nudged[0, 1] = np.nextafter(nudged[0, 1], np.inf)
        with pytest.raises(ValueError, match="different values"):
            r.model._compare_design("fixed-effect", nudged, r.model.exog)

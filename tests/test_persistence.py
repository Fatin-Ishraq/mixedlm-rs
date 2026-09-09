"""Saving and restoring a fit, and what survives the trip.

patsy declines to pickle a ``DesignInfo`` (pydata/patsy#26). Dropping it and
stopping there broke ``predict`` on raw new data after a round trip, and -- the
part that made it worse -- the resulting error advised the caller to "pass a
DataFrame of new data and leave transform=True", which is exactly what had just
failed.

The design is rebuilt instead, from the frame the model was fitted on. That
frame has to be the training one: it is what patsy's stateful transforms
learned their state from, so a categorical coding or a centring constant is
reproduced rather than recomputed on whatever new data happens to arrive.
"""

import pickle
import warnings

import mixedlm_rs as mlm
import numpy as np
import pandas as pd
import pytest


def frame(n=300, m=30, seed=4):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(m), n // m)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    grp = rng.choice(["a", "b", "c"], size=n)
    y = (1.0 + 0.5 * x - 0.3 * z + rng.normal(scale=0.6, size=m)[g]
         + rng.normal(scale=0.5, size=n))
    return pd.DataFrame({"y": y, "x": x, "z": z, "grp": grp, "g": g})


DF = frame()
NEW = pd.DataFrame({"x": [0.5, -1.0, 2.0], "z": [1.0, 0.0, -0.5],
                    "grp": ["a", "b", "c"]})


def fit(formula="y ~ x + z", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mlm.mixedlm(formula, DF, groups=DF["g"], **kw).fit()


def roundtrip(result):
    return pickle.loads(pickle.dumps(result))


# --------------------------------------------------------- the numbers survive
def test_the_fitted_numbers_survive():
    r = fit()
    back = roundtrip(r)

    assert back.llf == pytest.approx(r.llf, abs=1e-12)
    assert np.allclose(back.fe_params, r.fe_params, rtol=0, atol=0)
    assert np.allclose(back.cov_re, r.cov_re, rtol=0, atol=0)
    assert back.scale == r.scale
    assert back.converged == r.converged
    assert back.singular == r.singular
    assert back.diagnostics == r.diagnostics


def test_lazy_standard_errors_survive():
    """`bse_re` is computed on first access from the compiled core, which is
    not picklable. It has to be realised before the core is dropped."""
    r = fit()
    back = roundtrip(r)          # never touched bse_re before pickling
    assert np.allclose(back.bse_re, r.bse_re, equal_nan=True)
    assert np.allclose(back.bse, r.bse, equal_nan=True)


def test_summary_still_renders():
    back = roundtrip(fit())
    assert "Mixed Linear Model" in str(back.summary())


# ------------------------------------------------------ formula prediction
def test_formula_prediction_survives_a_pickle_round_trip():
    """The regression."""
    r = fit()
    before = r.predict(NEW)
    after = roundtrip(r).predict(NEW)
    assert np.allclose(before, after, rtol=0, atol=0)


def test_formula_prediction_survives_save_and_load(tmp_path):
    r = fit()
    path = tmp_path / "fit.pkl"
    r.save(path)
    back = mlm.MixedLMResults.load(path)
    assert np.allclose(back.predict(NEW), r.predict(NEW), rtol=0, atol=0)


def test_categorical_levels_are_reproduced_not_recomputed():
    """The reason the *training* frame is what gets retained.

    A design rebuilt from the new data would code `grp` against whatever levels
    happen to appear there. Rebuilding from the training frame reproduces the
    coding the model was actually fitted with.
    """
    r = fit("y ~ x + C(grp)")
    before = r.predict(NEW)
    back = roundtrip(r)
    after = back.predict(NEW)
    assert np.allclose(before, after, rtol=0, atol=0)

    # New data missing a level must still land in the original coding.
    partial = NEW.iloc[:1]                      # only grp == "a"
    assert np.allclose(back.predict(partial), before[:1], rtol=0, atol=0)


def test_centred_transform_keeps_the_training_mean():
    r = fit("y ~ center(x) + z")
    before = r.predict(NEW)
    after = roundtrip(r).predict(NEW)
    assert np.allclose(before, after, rtol=0, atol=0)


def test_random_slope_design_survives():
    r = fit(re_formula="~x")
    back = roundtrip(r)
    assert np.allclose(back.predict(NEW), r.predict(NEW), rtol=0, atol=0)
    assert np.allclose(back.cov_re, r.cov_re, rtol=0, atol=0)


# ----------------------------------------------------- opting out of the data
def test_save_without_data_still_restores_the_numbers(tmp_path):
    r = fit()
    path = tmp_path / "lean.pkl"
    r.save(path, with_data=False)
    back = mlm.MixedLMResults.load(path)

    assert back.llf == pytest.approx(r.llf, abs=1e-12)
    assert np.allclose(back.fe_params, r.fe_params, rtol=0, atol=0)


def test_save_without_data_is_smaller(tmp_path):
    r = fit()
    fat, lean = tmp_path / "fat.pkl", tmp_path / "lean.pkl"
    r.save(fat)
    r.save(lean, with_data=False)
    assert lean.stat().st_size < fat.stat().st_size


def test_save_without_data_does_not_mutate_the_live_result(tmp_path):
    r = fit()
    r.save(tmp_path / "lean.pkl", with_data=False)
    # The frame must be put back: the in-memory result keeps working.
    assert r.model.data_frame is not None
    assert np.allclose(r.predict(NEW), fit().predict(NEW), rtol=0, atol=0)


def test_prediction_from_a_prebuilt_design_still_works_without_data(tmp_path):
    r = fit()
    path = tmp_path / "lean.pkl"
    r.save(path, with_data=False)
    back = mlm.MixedLMResults.load(path)

    design = np.column_stack([np.ones(len(NEW)), NEW["x"], NEW["z"]])
    assert np.allclose(back.predict(design, transform=False), r.predict(NEW),
                       rtol=0, atol=1e-12)


def test_the_error_names_the_real_cause_and_the_real_remedy(tmp_path):
    """It must not advise the thing that just failed."""
    r = fit()
    path = tmp_path / "lean.pkl"
    r.save(path, with_data=False)
    back = mlm.MixedLMResults.load(path)

    with pytest.raises(ValueError) as exc:
        back.predict(NEW)
    msg = str(exc.value)

    assert "design metadata is not available" in msg
    assert "with_data=False" in msg, "the message must name what dropped it"
    assert "transform=False" in msg, "and the way out"
    # The old advice, which was to do exactly what had just failed.
    assert "leave transform=True" not in msg


def test_a_hand_built_model_predicts_after_a_round_trip():
    """No formula at all: nothing to rebuild, and nothing to complain about."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mlm.MixedLM(
            DF["y"].to_numpy(),
            np.column_stack([np.ones(len(DF)), DF["x"].to_numpy()]),
            DF["g"].to_numpy(),
        ).fit()
    back = roundtrip(r)
    design = np.column_stack([np.ones(3), [0.5, -1.0, 2.0]])
    assert np.allclose(back.predict(design), r.predict(design), rtol=0, atol=0)

"""The fourth review's findings, each reproduced before it was fixed.

Three were silent-or-broken persistence defects with a common shape: the model
fitted correctly, and only a save/load round trip revealed that the recorded
bookkeeping did not describe the fit. The fourth was capture doing work nobody
asked for.

Restoration is exercised in a *fresh interpreter* where it matters. An
in-process `pickle.loads` can succeed for the wrong reason -- the module is
still imported, the alias is still bound in `sys.modules`, the class is still
defined -- so a round trip inside one process is a weaker claim than it looks.
"""

from __future__ import annotations

import pathlib
import pickle
import subprocess
import sys
import textwrap
import warnings

import mixedlm_rs as mlm
import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def frame(n=200, m=20, seed=16):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(m), n // m).astype(float)
    x = np.arange(n, dtype=float) / 10 + 1
    y = (2 + 1.7 * x + rng.normal(size=m)[g.astype(int)]
         + rng.normal(scale=0.3, size=n))
    return pd.DataFrame({"y": y, "x": x, "z": rng.normal(size=n), "g": g})


NEW = pd.DataFrame({"x": [6.0, 10.0, 18.0], "z": [0.1, -0.2, 0.3]})


def round_trips(result, new=NEW):
    """Predictions are bit-identical across a pickle."""
    before = np.asarray(result.predict(new), float)
    after = np.asarray(pickle.loads(pickle.dumps(result)).predict(new), float)
    assert np.array_equal(before, after), (
        f"predictions moved by {np.max(np.abs(before - after)):.6g} "
        "across a pickle")
    return before


def in_fresh_process(body):
    """Run `body` in a new interpreter; return its stdout.

    The point of a subprocess here is that nothing from this session survives
    into it: no imported module, no alias, no class definition. A restore that
    only works because the fitting process is still alive is not a restore.
    """
    script = textwrap.dedent(body)
    done = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          cwd=str(ROOT.parent))
    assert done.returncode == 0, (
        f"the fresh process failed:\n{done.stdout}\n{done.stderr}")
    return done.stdout


# ================================================= 1. missing group labels
class TestMissingGroupSerialisation:
    """One row-selection record, covering every step that drops rows.

    `_design_rows` recorded the rows patsy kept, and the constructor then
    dropped rows with a missing group label without reconciling it. Fitting
    199 of 200 rows and rebuilding produced 200, so restoring raised -- and
    blamed the formula. Worse than the count: `center()` had learned its mean
    over 200 rows while the model was fitted on 199, so the fitted design and
    any rebuildable design were different objects.
    """

    def test_a_missing_group_label_round_trips(self):
        d = frame()
        d.loc[0, "g"] = np.nan
        r = mlm.mixedlm("y ~ center(x)", d, groups="g", missing="drop").fit()
        assert r.nobs == 199
        assert len(r.model._design_rows) == 199, (
            "the recorded design rows do not describe the fitted rows")
        round_trips(r)

    def test_the_transform_learned_the_state_of_the_fitted_rows(self):
        """Not just the same row *count* -- the same design.

        Rebuilding is compared against the design that was actually fitted,
        column by column, so a transform that relearned different state fails
        here rather than silently predicting from a different model.
        """
        d = frame()
        d.loc[0, "g"] = np.nan
        r = mlm.mixedlm("y ~ center(x)", d, groups="g", missing="drop").fit()

        model = pickle.loads(pickle.dumps(r)).model
        assert model._ensure_design_info() is True
        import patsy
        rebuilt = np.asarray(
            patsy.dmatrix(model._design_info, model._training_frame(),
                          return_type="matrix"), float)
        assert rebuilt.shape == r.model.exog.shape
        assert np.allclose(rebuilt, r.model.exog, rtol=0, atol=0), (
            "the rebuilt design differs from the fitted one; a stateful "
            "transform relearned different state")

    @pytest.mark.parametrize("formula,re_formula", [
        # A stateful transform over a complete column, combined with rows lost
        # to a missing group and to a missing predictor elsewhere in the
        # formula. That combination is the finding: the transform's state must
        # come from the rows that survive *both* selections.
        ("y ~ center(x)", None),
        ("y ~ center(x)", "~z"),
        ("y ~ center(x) + w", "~z"),
        ("y ~ standardize(x)", "~z"),
        ("y ~ bs(x, df=3) + w", None),
        ("y ~ C(k) + w", None),
        ("y ~ center(x) + C(k) + w", "~z"),
    ])
    def test_missing_groups_and_predictors_together(self, formula, re_formula):
        """Missing groups, missing FE and RE predictors, and combinations.

        `w` and `z` carry the missing predictor values; `x` is complete. That
        split is forced by patsy, not by this package: `center()` and
        `standardize()` compute their state over the raw column including
        NaN, so `center(x)` on a column with one missing value yields a NaN
        mean and drops every row -- upstream, before anything here is
        involved. See `test_a_stateful_transform_over_a_missing_column_is_a_
        patsy_limitation`, which pins that behaviour so this fixture's shape
        stays explained.
        """
        d = frame()
        d["k"] = np.where(np.arange(len(d)) % 5 == 0, "b", "a")
        # Not linspace: `x` is itself linear in the row index, so a
        # linearly spaced `w` is an exact combination of it and the
        # design is rank deficient by construction.
        d["w"] = np.random.default_rng(4).normal(size=len(d))
        d.loc[0, "g"] = np.nan            # missing group
        d.loc[3, "w"] = np.nan            # missing fixed-effect predictor
        d.loc[7, "z"] = np.nan            # missing random-effect predictor
        r = mlm.MixedLM.from_formula(formula, d, groups="g",
                                     re_formula=re_formula,
                                     missing="drop").fit()
        assert r.nobs < len(d), "the fixture no longer drops any rows"
        new = NEW.copy()
        new["k"] = ["a"] * len(new)
        new["w"] = [0.2, 0.4, 0.6]
        round_trips(r, new)

    def test_a_stateful_transform_over_a_missing_column_is_a_patsy_limitation(
            self):
        """Pinned so the fixture above is understood rather than worked around.

        patsy's `center()` computes its mean over the raw column, NaN
        included, so a single missing value makes the mean NaN and every row
        is then dropped as missing. That happens inside patsy with no
        involvement from this package -- `dmatrix('center(x)', ...)` on such a
        column returns zero rows while `dmatrix('x', ...)` returns the
        complete cases.
        """
        import patsy
        d = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0]})
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            centred = np.asarray(
                patsy.dmatrix("center(x)", d, NA_action="drop"))
            plain = np.asarray(patsy.dmatrix("x", d, NA_action="drop"))
        assert centred.shape[0] == 0, (
            "patsy now handles NaN inside center(); the fixture split in "
            "test_missing_groups_and_predictors_together can be simplified")
        assert plain.shape[0] == 4

        # And this package reports it as an empty selection rather than
        # something obscure further down.
        d2 = frame()
        d2.loc[3, "x"] = np.nan
        with pytest.raises(ValueError, match=r"dropped as missing|no rows"):
            mlm.mixedlm("y ~ center(x)", d2, groups="g", missing="drop")

    def test_public_save_and_load_round_trips(self, tmp_path):
        d = frame()
        d.loc[0, "g"] = np.nan
        r = mlm.mixedlm("y ~ center(x)", d, groups="g", missing="drop").fit()
        path = tmp_path / "model.pkl"
        r.save(path)
        with open(path, "rb") as fh:
            restored = pickle.load(fh)
        assert np.array_equal(np.asarray(r.predict(NEW), float),
                              np.asarray(restored.predict(NEW), float))

    def test_a_missing_group_without_drop_is_refused(self):
        d = frame()
        d.loc[0, "g"] = np.nan
        with pytest.raises(ValueError, match="missing `groups` label"):
            mlm.mixedlm("y ~ x", d, groups="g")

    def test_the_row_trace_is_reported_on_a_mismatch(self):
        """A row-count mismatch is this package's bug, and says so."""
        d = frame()
        d.loc[0, "g"] = np.nan
        r = mlm.mixedlm("y ~ center(x)", d, groups="g", missing="drop").fit()
        steps = dict(r.model._row_selection_steps)
        assert steps["groups present"] == 199
        assert steps["complete cases"] == 199
        trace = r.model._row_trace()
        assert "groups present 199" in trace


# ============================================ 2. module-qualified functions
class TestModuleQualifiedFormulas:
    """`import numpy as numeric` then `y ~ numeric.log(x)`.

    Capture skipped modules outright, so the alias was unbound at rebuild time
    and prediction failed with a bare NameError out of patsy. A module cannot
    be pickled, but its importable name can, and re-importing reproduces it.
    """

    def test_a_module_alias_round_trips(self):
        import numpy as numeric  # noqa: F401  (resolved by patsy, not here)
        r = mlm.MixedLM.from_formula("y ~ numeric.log(x)", frame(),
                                     groups="g").fit()
        round_trips(r)

    def test_a_module_alias_in_the_random_effects_formula_round_trips(self):
        import numpy as numeric  # noqa: F401
        r = mlm.MixedLM.from_formula("y ~ x", frame(), groups="g",
                                     re_formula="~numeric.log(x)").fit()
        round_trips(r)

    def test_a_submodule_alias_round_trips(self):
        import numpy.linalg as la  # noqa: F401
        d = frame()
        r = mlm.MixedLM.from_formula("y ~ x", d, groups="g").fit()
        # The alias is captured only when the formula names it.
        assert "la" not in (r.model._formula_namespace or {})

    def test_only_the_root_of_an_attribute_chain_is_captured(self):
        import numpy as numeric  # noqa: F401
        r = mlm.MixedLM.from_formula("y ~ numeric.log(x)", frame(),
                                     groups="g").fit()
        captured = r.model._formula_namespace
        assert "numeric" in captured, "the alias the formula needs was not kept"
        assert "log" not in captured, (
            "`log` is an attribute of the alias, not a name to resolve on its "
            "own; capturing it means the formula was scanned as text")

    def test_restoration_works_in_a_fresh_interpreter(self, tmp_path):
        """The real test: nothing from the fitting session is still alive."""
        model_path = tmp_path / "aliased.pkl"
        fit = f'''
            import pickle, sys, warnings
            warnings.simplefilter("ignore")
            import numpy as numeric
            import pandas as pd
            import mixedlm_rs as mlm
            sys.path.insert(0, {str(ROOT)!r})
            from tests.test_recheck2_regressions import frame
            r = mlm.MixedLM.from_formula("y ~ numeric.log(x)", frame(),
                                         groups="g").fit()
            new = pd.DataFrame({{"x": [6.0, 10.0, 18.0]}})
            print(list(r.predict(new)))
            with open({str(model_path)!r}, "wb") as fh:
                pickle.dump(r, fh)
        '''
        before = in_fresh_process(fit).strip().splitlines()[-1]

        restore = f'''
            import pickle, warnings
            warnings.simplefilter("ignore")
            import pandas as pd
            with open({str(model_path)!r}, "rb") as fh:
                r = pickle.load(fh)
            new = pd.DataFrame({{"x": [6.0, 10.0, 18.0]}})
            print(list(r.predict(new)))
        '''
        after = in_fresh_process(restore).strip().splitlines()[-1]
        assert before == after, (
            f"a module-aliased formula did not survive a process boundary:\\n"
            f"  before {before}\\n  after  {after}")

    def test_a_closure_is_still_refused_and_named(self):
        """Unsupported, documented, and diagnosed -- not silently wrong."""
        d = frame()

        def local_transform(v):
            return np.asarray(v, float) ** 2

        r = mlm.MixedLM.from_formula("y ~ local_transform(x)", d,
                                     groups="g").fit()
        with pytest.raises(ValueError) as excinfo:
            pickle.loads(pickle.dumps(r)).predict(NEW)
        message = str(excinfo.value)
        assert "local_transform" in message, "the transform is not named"
        assert "were dropped" not in message, (
            "the error blames discarded training data")

    def test_the_closure_limitation_is_documented(self):
        text = (ROOT / "docs" / "LIMITATIONS.md").read_text(encoding="utf-8")
        assert "no universal formula round-trip guarantee" in text
        assert "module-level" in text


# ==================================== 3. what capture may and may not touch
class TestNamespaceCaptureIsMinimal:
    """Columns win, and nothing is serialised during construction.

    Capture was a regex over the formula text with no knowledge of the data
    frame, so `y ~ x` captured a caller variable named `x` that patsy never
    consults -- and, because portability was probed by pickling, ran that
    object's `__reduce__` during `from_formula`.
    """

    def test_a_column_name_shadowing_a_local_is_not_captured(self):
        calls = []

        class Tracked:
            def __reduce__(self):
                calls.append("__reduce__")
                return (str, ("tracked",))

        x = Tracked()                     # noqa: F841  (the point of the test)
        model = mlm.MixedLM.from_formula("y ~ x", frame(), groups="g")

        assert calls == [], (
            "constructing a model ran an unrelated local object's "
            f"serialisation hooks: {calls}")
        assert "x" not in (model._formula_namespace or {}), (
            "`x` is a column of the data frame; patsy resolves it there and "
            "never consults the caller's namespace")

    def test_unused_local_arrays_are_not_retained(self):
        x = frame()["x"].to_numpy().copy()        # noqa: F841
        y = frame()["y"].to_numpy().copy()        # noqa: F841
        model = mlm.MixedLM.from_formula("y ~ x", frame(), groups="g")
        retained = {k: v for k, v in (model._formula_namespace or {}).items()
                    if isinstance(v, np.ndarray)}
        assert not retained, (
            f"the model retained unrelated caller arrays: "
            f"{[(k, v.shape) for k, v in retained.items()]}")

    def test_nothing_is_pickled_during_construction(self):
        """Even for a name the formula *does* use.

        Whether a captured object survives is decided at save time. Deciding it
        at construction meant every fit paid for a serialise-and-reconstruct of
        every candidate object.
        """
        calls = []

        class Tracked:
            def __call__(self, v):
                return np.asarray(v, float)

            def __reduce__(self):
                calls.append("__reduce__")
                return (str, ("tracked",))

        transform = Tracked()          # noqa: F841 - patsy resolves it
        model = mlm.MixedLM.from_formula("y ~ transform(x)", frame(),
                                         groups="g")
        assert calls == [], (
            f"construction serialised a captured object: {calls}")
        assert "transform" in (model._formula_namespace or {})

    def test_a_name_the_formula_needs_is_still_captured(self):
        """The other side: minimal capture must not become no capture."""
        import numpy as numeric  # noqa: F401
        model = mlm.MixedLM.from_formula("y ~ numeric.log(x)", frame(),
                                         groups="g")
        assert "numeric" in (model._formula_namespace or {})

    def test_capture_is_empty_for_an_ordinary_formula(self):
        model = mlm.MixedLM.from_formula("y ~ x + z", frame(), groups="g")
        assert not (model._formula_namespace or {}), (
            "a formula over plain columns needs nothing from the caller")

    def test_patsy_builtins_are_not_captured(self):
        model = mlm.MixedLM.from_formula("y ~ center(x) + C(g)", frame(),
                                         groups="g")
        captured = model._formula_namespace or {}
        assert "center" not in captured and "C" not in captured, (
            "patsy supplies its own transforms; they are not caller state")


# =========================================== 8. migration documentation
class TestMigrationConclusion:
    """The three-check shortcut promised more than the contract delivers."""

    @pytest.fixture(scope="class")
    def section(self):
        text = (ROOT / "docs" / "COMPATIBILITY.md").read_text(encoding="utf-8")
        return text[text.index("## Deciding whether to switch"):]

    def test_the_absolute_promise_is_gone(self, section):
        assert "the import swap really is the whole migration" not in section

    @pytest.mark.parametrize("omitted", [
        "free", "fe_pen", "profile_re", "fit_regularized",
        "vc_formula", "use_sparse", "by label",
    ])
    def test_the_previously_omitted_categories_are_named(self, section, omitted):
        assert omitted in section, (
            f"the migration checklist does not mention {omitted!r}, which the "
            "contract above documents as unsupported or divergent")

    def test_it_defers_to_the_full_contract(self, section):
        assert "not a warranty" in section or "not this list" in section, (
            "the conclusion should point at the tables as the complete "
            "statement rather than standing alone")

"""The third review's findings, held down by tests that reproduce them.

Every test here failed before the corresponding fix. Two of them were silent
wrong answers -- a fitted model on scrambled clusters, and predictions that
moved after a pickle -- which is the class of defect a test suite exists for,
because nothing else catches it.

Where the review gave concrete numbers, those numbers are asserted against an
explicitly constructed reference rather than against a recorded constant, so
the test says *why* the value is right.
"""

from __future__ import annotations

import pathlib
import pickle
import subprocess
import sys
import warnings
import zipfile
from typing import ClassVar

import mixedlm_rs as mlm
import numpy as np
import pandas as pd
import pytest
from mixedlm_rs import mixed_linear_model as _mlm

ROOT = pathlib.Path(__file__).resolve().parents[1]


def clustered(n=300, m=20, seed=0, sd_u=1.0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(m), n // m)
    u = rng.normal(0, sd_u, m)
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + u[g] + rng.normal(0, 0.4, size=n)
    return pd.DataFrame({"y": y, "x": x, "g": g},
                        index=[f"r{i}" for i in range(n)])


# =============================================== 1. group alignment (P1)
class TestGroupAlignment:
    """A labelled group Series is aligned by label, or rejected.

    The bug: an already-subset Series was taken in its supplied order and its
    index discarded. Shuffling it -- which `Series.sample`, `sort_values` or a
    groupby round-trip all do -- reassigned 137 of 150 observations to the
    wrong cluster and returned a plausible variance.
    """

    @staticmethod
    def reference(data, labels):
        return mlm.MixedLM.from_formula("y ~ x", data, groups="g",
                                        subset=labels).fit()

    def test_shuffled_pre_subset_series_matches_the_reference_fit(self):
        data = clustered()
        labels = data.index[:150]
        shuffled = data.loc[labels, "g"].sample(frac=1.0, random_state=1)

        want = self.reference(data, labels)
        got = mlm.MixedLM.from_formula("y ~ x", data, groups=shuffled,
                                       subset=labels).fit()

        assert float(got.cov_re[0, 0]) == pytest.approx(
            float(want.cov_re[0, 0]), rel=1e-12), (
            "a shuffled group Series fitted a different model")
        assert float(got.llf) == pytest.approx(float(want.llf), rel=1e-12)

    def test_reversed_pre_subset_series_matches_the_reference_fit(self):
        data = clustered()
        labels = data.index[:150]
        reversed_ = data.loc[labels, "g"].iloc[::-1]
        want = self.reference(data, labels)
        got = mlm.MixedLM.from_formula("y ~ x", data, groups=reversed_,
                                       subset=labels).fit()
        assert float(got.llf) == pytest.approx(float(want.llf), rel=1e-12)

    def test_alignment_is_by_label_not_position(self):
        """The direct statement of the contract, without fitting anything."""
        data = clustered(n=40, m=4)
        positions = _mlm._subset_positions(data, data.index[:20])
        shuffled = data.loc[data.index[:20], "g"].sample(frac=1.0,
                                                         random_state=7)
        aligned = _mlm._align_groups(shuffled, data, positions)
        expected = np.asarray(data["g"])[positions]
        assert np.array_equal(aligned, expected)

    def test_a_series_carrying_the_unsubset_index_is_aligned(self):
        data = clustered(n=60, m=6)
        labels = data.index[10:40]
        full = data["g"].sample(frac=1.0, random_state=3)
        positions = _mlm._subset_positions(data, labels)
        aligned = _mlm._align_groups(full, data, positions)
        assert np.array_equal(aligned, np.asarray(data["g"])[positions])

    def test_a_series_missing_selected_labels_is_rejected(self):
        data = clustered(n=60, m=6)
        labels = data.index[:30]
        short = data.loc[data.index[:20], "g"]
        positions = _mlm._subset_positions(data, labels)
        with pytest.raises(ValueError, match="missing 10 of the 30"):
            _mlm._align_groups(short, data, positions)

    def test_duplicate_group_labels_are_rejected_not_guessed(self):
        data = clustered(n=60, m=6)
        positions = _mlm._subset_positions(data, data.index[:30])
        dup = pd.Series(np.arange(30), index=["a"] * 30)
        with pytest.raises(ValueError, match="duplicate index labels"):
            _mlm._align_groups(dup, data, positions)

    def test_labels_outside_the_frame_are_rejected(self):
        data = clustered(n=60, m=6)
        positions = _mlm._subset_positions(data, data.index[:30])
        stray = pd.Series(np.arange(30), index=[f"z{i}" for i in range(30)])
        with pytest.raises(ValueError, match="not in the index of `data`"):
            _mlm._align_groups(stray, data, positions)

    def test_a_bare_range_indexed_series_is_still_positional(self):
        """The compatible case has to keep working.

        A Series built from an array carries a default RangeIndex, which says
        nothing beyond position, so positional alignment is right there.
        """
        data = clustered(n=60, m=6).reset_index(drop=True)
        positions = _mlm._subset_positions(data, np.arange(60) < 30)
        plain = pd.Series(np.asarray(data["g"]))
        aligned = _mlm._align_groups(plain, data, positions)
        assert np.array_equal(aligned, np.asarray(data["g"])[positions])

    def test_a_plain_array_is_still_positional(self):
        data = clustered(n=60, m=6)
        positions = _mlm._subset_positions(data, data.index[:30])
        arr = np.asarray(data["g"])
        assert np.array_equal(_mlm._align_groups(arr, data, positions),
                              arr[positions])

    def test_a_non_unique_frame_index_is_rejected_for_a_series(self):
        data = clustered(n=40, m=4)
        data.index = ["a"] * 40
        positions = np.arange(20, dtype=np.intp)
        with pytest.raises(ValueError, match="non-unique index"):
            _mlm._align_groups(pd.Series(np.arange(40), index=list("b" * 40)),
                               data, positions)


# ============================================= 2 & 3. persistence (P1/P2)
def _missing_in_re_predictor(n=200, m=20, seed=3):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(m), n // m)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1 + 0.5 * x + 0.3 * z + rng.normal(0, 0.5, n)
    d = pd.DataFrame({"y": y, "x": x, "z": z, "g": g})
    d.loc[d.index[::4], "z"] = np.nan          # NA in the RE predictor only
    return d


NEW = pd.DataFrame({"x": [0.1, 0.5, 1.2], "z": [0.0, 1.0, -0.5],
                    "g": [0, 1, 2]})


class TestSerialisationKeepsPredictions:
    """The rebuild has to use the rows the fit used, not the rows retained.

    Missing-data handling drops rows across the fixed *and* random formulas
    together. Rebuilding from the pre-filter frame let `center()` relearn a
    different mean and `C()` relearn an extra level -- moving every prediction
    by a constant, or producing a design with the wrong number of columns.
    """

    @pytest.mark.parametrize("formula", [
        "y ~ center(x)",
        "y ~ standardize(x)",
        "y ~ center(x) + I(x ** 2)",
    ])
    def test_numeric_stateful_transforms_round_trip(self, formula):
        d = _missing_in_re_predictor()
        r = mlm.MixedLM.from_formula(formula, d, groups="g", re_formula="~z",
                                     missing="drop").fit()
        before = np.asarray(r.predict(NEW), float)
        after = np.asarray(pickle.loads(pickle.dumps(r)).predict(NEW), float)
        assert np.allclose(before, after, rtol=0, atol=0), (
            f"{formula}: predictions moved by "
            f"{np.max(np.abs(before - after)):.6f} across a pickle")

    def test_categorical_levels_round_trip(self):
        d = _missing_in_re_predictor()
        d["c"] = np.where(np.arange(len(d)) % 7 == 0, "rare", "common")
        d.loc[d["c"].eq("rare"), "z"] = np.nan   # every rare row drops out
        new = NEW.copy()
        new["c"] = ["common"] * len(new)

        r = mlm.MixedLM.from_formula("y ~ x + C(c)", d, groups="g",
                                     re_formula="~z", missing="drop").fit()
        before = np.asarray(r.predict(new), float)
        after = np.asarray(pickle.loads(pickle.dumps(r)).predict(new), float)
        assert np.allclose(before, after, rtol=0, atol=0)

    def test_the_fit_used_fewer_rows_than_the_retained_frame(self):
        """Guards the premise: without this the tests above prove nothing."""
        d = _missing_in_re_predictor()
        r = mlm.MixedLM.from_formula("y ~ center(x)", d, groups="g",
                                     re_formula="~z", missing="drop").fit()
        assert r.model.data_frame is not None
        assert r.model._design_rows is not None
        assert len(r.model._design_rows) < len(r.model.data_frame), (
            "the fixture no longer drops rows, so it cannot detect the bug")

    def test_save_and_load_round_trips_too(self, tmp_path):
        """Public API, not only direct pickle."""
        d = _missing_in_re_predictor()
        r = mlm.MixedLM.from_formula("y ~ center(x)", d, groups="g",
                                     re_formula="~z", missing="drop").fit()
        path = tmp_path / "m.pkl"
        r.save(path)
        with open(path, "rb") as fh:
            restored = pickle.load(fh)
        assert np.allclose(np.asarray(r.predict(NEW), float),
                           np.asarray(restored.predict(NEW), float),
                           rtol=0, atol=0)

    def test_a_mismatched_rebuild_is_discarded_not_used(self):
        """A rebuild that disagrees with the fit must not be trusted."""
        d = _missing_in_re_predictor()
        r = mlm.MixedLM.from_formula("y ~ center(x)", d, groups="g",
                                     re_formula="~z", missing="drop").fit()
        restored = pickle.loads(pickle.dumps(r))
        model = restored.model
        model._design_info = None
        # Point the rebuild at the *unfiltered* frame, which is what the bug
        # did. The check must reject the result rather than predict with it.
        model._design_rows = np.arange(len(model.data_frame), dtype=np.intp)
        with pytest.raises(ValueError, match="rebuilding the design"):
            model._check_rebuilt(
                __import__("patsy").dmatrix("center(x)", model.data_frame,
                                            NA_action="drop").design_info,
                model.exog.shape[0] + 1)


def module_level_square(v):
    """A module-level transform: picklable by qualified name."""
    return np.asarray(v, float) ** 2


class TestCallerDefinedTransforms:
    def test_a_module_level_transform_survives(self):
        d = clustered(n=200, m=20).reset_index(drop=True)
        r = mlm.MixedLM.from_formula("y ~ module_level_square(x)", d,
                                     groups="g").fit()
        new = pd.DataFrame({"x": [0.1, 0.5]})
        before = np.asarray(r.predict(new), float)
        after = np.asarray(pickle.loads(pickle.dumps(r)).predict(new), float)
        assert np.allclose(before, after, rtol=0, atol=0)

    def test_an_unpicklable_transform_fails_naming_itself(self):
        d = clustered(n=200, m=20).reset_index(drop=True)
        cube = lambda v: np.asarray(v, float) ** 3          # noqa: E731
        r = mlm.MixedLM.from_formula("y ~ cube(x)", d, groups="g").fit()
        restored = pickle.loads(pickle.dumps(r))
        with pytest.raises(ValueError) as excinfo:
            restored.predict(pd.DataFrame({"x": [0.1, 0.5]}))
        message = str(excinfo.value)
        assert "cube" in message, "the missing transform is not named"
        assert "were dropped" not in message, (
            "the error still blames discarded training data")
        assert "lambda" in message or "closure" in message

    def test_a_failed_rebuild_leaves_no_partial_design(self):
        d = clustered(n=200, m=20).reset_index(drop=True)
        cube = lambda v: np.asarray(v, float) ** 3          # noqa: E731
        r = mlm.MixedLM.from_formula("y ~ cube(x)", d, groups="g",
                                     re_formula="~x").fit()
        restored = pickle.loads(pickle.dumps(r))
        assert restored.model._ensure_design_info() is False
        assert restored.model._design_info is None
        assert restored.model._re_design_info is None


# ============================================ 4. dependency floors (P1)
class TestDependencyFloors:
    """The floors have to be executable, and stated identically everywhere.

    patsy 0.5.0 and 0.5.1 do `from collections import Mapping`, removed in
    Python 3.10; 0.5.2 fails its own version comparison. All three raise on
    import under every Python this package supports, so `patsy>=0.5` was a
    promise that could not be kept. 0.5.3 is the lowest that imports.
    """

    FLOORS: ClassVar[dict[str, str]] = {
        "numpy": "1.23", "scipy": "1.9", "pandas": "1.5", "patsy": "0.5.3"}

    def test_pyproject_declares_the_verified_floors(self):
        # tomllib arrived in 3.11 and this package supports 3.10, which is
        # also the interpreter the floors job runs on -- so read the metadata
        # only where the standard library can.
        tomllib = pytest.importorskip("tomllib")
        data = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
        declared = dict(
            spec.split(">=") for spec in data["project"]["dependencies"])
        assert declared == self.FLOORS

    def test_runtime_requirements_agree_with_pyproject(self):
        text = (ROOT / "requirements-runtime.txt").read_text("utf-8")
        pinned = dict(line.split(">=") for line in text.splitlines()
                      if ">=" in line and not line.startswith("#"))
        assert pinned == self.FLOORS

    def test_ci_pins_the_exact_declared_floors(self):
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
        for name, floor in self.FLOORS.items():
            exact = floor if floor.count(".") == 2 else floor + ".0"
            assert f'"{name}=={exact}"' in ci, (
                f"CI does not install {name}=={exact}, so the declared floor "
                "for it is untested")

    def test_the_installed_patsy_is_at_or_above_the_verified_floor(self):
        import patsy
        parts = tuple(int(p) for p in patsy.__version__.split(".")[:3])
        assert parts >= (0, 5, 3), (
            f"patsy {patsy.__version__} is below the verified floor 0.5.3, "
            "which cannot be imported on Python 3.10+")


# ================================= 5. the checker's CI prerequisites (P2)
class TestCheckerPrerequisites:
    def test_the_job_running_the_checker_supplies_them(self):
        """Jobs share no filesystem, so each needs its own tools.

        The checker inspects `dist/*.tar.gz` and shells out to `cargo audit`.
        Building an sdist in the `sdist-contents` job and installing
        cargo-audit in the `audit` job does nothing for the job that runs the
        checker.
        """
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
        job = ci[ci.index("  native-safety:"):ci.index("  install-smoke:")]
        assert "maturin sdist" in job, (
            "the checker's F37 reads dist/*.tar.gz; this job builds no sdist")
        assert "cargo install cargo-audit" in job, (
            "the checker's R1 runs `cargo audit`; this job does not install it")
        assert "verify_review_findings.py" in job

    def test_the_guard_is_reachable_and_names_what_is_missing(self):
        source = (ROOT / "scripts" / "verify_review_findings.py").read_text(
            "utf-8")
        assert "def require_prerequisites()" in source
        # A top-level call, at column zero. Searching the whole file finds
        # this test's own source too, and splitting on "GROUPS = [" finds
        # the checker's own s5, whose body mentions that string.
        assert any(line == "require_prerequisites()"
                   for line in source.splitlines()), (
            "the guard is defined but never called")


# ================================= 7. redistribution notices (P2)
class TestRedistributionNotices:
    """Static linking makes the wheel a binary redistribution.

    MIT, BSD-2-Clause and Apache-2.0 each require the copyright notice, the
    conditions and the disclaimer to travel with it. An SBOM naming the
    licences satisfies none of them.
    """

    def test_the_notice_file_carries_actual_licence_text(self):
        path = ROOT / "THIRD-PARTY-LICENSES.md"
        assert path.is_file(), "run `python scripts/collect_notices.py`"
        text = path.read_text("utf-8")
        assert "Redistribution and use in source and binary forms" in text, (
            "the BSD-2-Clause conditions are not reproduced")
        assert text.upper().count("THE SOFTWARE IS PROVIDED") >= 1
        assert text.count("Copyright") >= 5

    @pytest.mark.parametrize("crate", ["numpy", "pyo3", "rayon", "ndarray"])
    def test_every_linked_crate_has_its_text(self, crate):
        text = (ROOT / "THIRD-PARTY-LICENSES.md").read_text("utf-8")
        assert f"## {crate} " in text

    def test_the_generator_is_reproducible(self):
        """A committed notice file that nothing regenerates goes stale."""
        result = subprocess.run(
            [sys.executable, "scripts/collect_notices.py", "--check"],
            cwd=ROOT, capture_output=True, text=True)
        if "cargo" in result.stderr and result.returncode not in (0, 1):
            pytest.skip("cargo unavailable")
        assert result.returncode == 0, result.stdout + result.stderr

    def test_the_built_wheel_ships_them(self):
        wheels = sorted((ROOT / "dist").glob("*.whl"),
                        key=lambda p: p.stat().st_mtime)
        if not wheels:
            pytest.skip("no wheel built")
        with zipfile.ZipFile(wheels[-1]) as z:
            names = z.namelist()
            notice = next(
                (n for n in names if n.endswith("THIRD-PARTY-LICENSES.md")),
                None)
            assert notice, (
                "the wheel ships only our own LICENSE; the upstream notices "
                f"are absent. Files: {[n for n in names if 'licenses' in n]}")
            body = z.read(notice).decode("utf-8")
        assert "Redistribution and use in source and binary forms" in body


# ==================================== 8. benchmark attribution (P2)
class TestBenchmarkClaims:
    DOCS: ClassVar[list[str]] = ["README.md", "docs/BENCHMARKS.md"]

    @pytest.mark.parametrize("name", DOCS)
    def test_no_document_calls_the_gap_pure_bridge_overhead(self, name):
        """`bench/time_pymer4.py` says the measured path also runs lmerTest
        inference and result extraction. The published claim has to agree."""
        text = (ROOT / name).read_text("utf-8").lower()
        for line in text.splitlines():
            for phrase in ("pure bridge overhead", "pure rpy2 bridge",
                           "the bridge costs"):
                if phrase not in line:
                    continue
                # The phrase may appear where the document withdraws it. It
                # may not appear as a claim.
                assert "was wrong" in line or "called it" in line, (
                    f"{name} attributes the pymer4 gap to marshalling alone: "
                    f"{line.strip()!r}. The benchmark measures marshalling "
                    "plus lmerTest inference plus extraction and does not "
                    "separate them.")

    @pytest.mark.parametrize("name", DOCS)
    def test_the_disclosure_is_present_where_the_number_is(self, name):
        text = (ROOT / name).read_text("utf-8")
        if "744" not in text:
            pytest.skip(f"{name} does not quote the figure")
        assert "lmerTest" in text and "Satterthwaite" in text, (
            f"{name} quotes 744 s without saying what work it includes")

    def test_evaluation_count_ranges_agree_across_documents(self):
        """README quoted 44-80/11-16 where the measured table gives 44-64/11-13."""
        table = _bench_evaluation_table()
        numeric = f"{min(table['numeric'])}–{max(table['numeric'])}"
        analytic = f"{min(table['analytic'])}–{max(table['analytic'])}"
        for name in ("README.md", "CHANGELOG.md", "docs/DESIGN.md"):
            text = (ROOT / name).read_text("utf-8")
            assert numeric in text and analytic in text, (
                f"{name} should quote {numeric} to {analytic} from the "
                "measured table in docs/BENCHMARKS.md")
            for stale in ("44–80", "11–16"):
                assert stale not in text, f"{name} still quotes {stale}"


def _bench_evaluation_table():
    """Parse the S3/S4 columns out of the BENCHMARKS table."""
    text = (ROOT / "docs" / "BENCHMARKS.md").read_text("utf-8")
    start = text.index("## What the analytic gradient buys")
    end = text.index("\n## ", start + 1)
    section = text[start:end]
    numeric, analytic = [], []
    for line in section.splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
        if len(cells) != 5 or not cells[0].replace(",", "").isdigit():
            continue
        numeric.append(int(cells[2].replace(",", "")))
        analytic.append(int(cells[3].replace(",", "")))
    assert numeric and analytic, "could not parse the evaluation-count table"
    return {"numeric": numeric, "analytic": analytic}


# ==================================== 10. compatibility claims (P2/P3)
class TestCompatibilityClaims:
    def test_use_sparse_warns_as_documented(self):
        d = clustered(n=100, m=10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mlm.MixedLM.from_formula("y ~ x", d, groups="g", use_sparse=True)
        assert any("use_sparse" in str(w.message) for w in caught), (
            "COMPATIBILITY.md promises a warning for non-default use_sparse")

    def test_the_default_does_not_warn(self):
        d = clustered(n=100, m=10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mlm.MixedLM.from_formula("y ~ x", d, groups="g", use_sparse=False)
        assert not [w for w in caught if "use_sparse" in str(w.message)]

    def test_the_documented_portable_cov_re_accessor_works_here(self):
        d = clustered(n=100, m=10)
        r = mlm.MixedLM.from_formula("y ~ x", d, groups="g").fit()
        assert np.asarray(r.cov_re)[0, 0] == pytest.approx(
            float(r.cov_re[0, 0]))

    def test_the_document_no_longer_claims_bare_subscription_is_portable(self):
        """`cov_re[0, 0]` is a column-key lookup on a DataFrame, so it raises
        on statsmodels."""
        text = (ROOT / "docs" / "COMPATIBILITY.md").read_text("utf-8")
        assert "`np.asarray(cov_re)[0, 0]` works on both" in text
        assert "`cov_re[0, 0]` works on both" not in text

    def test_the_readme_qualifies_the_same_arguments_claim(self):
        text = (ROOT / "README.md").read_text("utf-8")
        assert "Same classes, same arguments" not in text
        assert "COMPATIBILITY.md" in text

    def test_the_readme_does_not_present_unrun_wheels_as_verified(self):
        text = (ROOT / "README.md").read_text("utf-8")
        assert "Currently verified" in text, (
            "the install section should say which artifacts have actually "
            "been built and installed")

    def test_limitations_narrows_the_persistence_promise(self):
        text = (ROOT / "docs" / "LIMITATIONS.md").read_text("utf-8")
        assert "no universal formula round-trip guarantee" in text
        assert "the exact rows the model was fitted on" in text

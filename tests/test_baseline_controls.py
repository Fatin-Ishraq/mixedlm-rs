"""Negative controls for the baseline document checks.

A validator that never fails is indistinguishable from one that always passes.
These mutate a copy of a document, run the real check against the mutation, and
require it to fail -- so "the tables agree with the recorded run" means the
check could have said otherwise.

The three mutations are the ones that actually happened or were demonstrated:

* **cross-experiment substitution.** A review replaced "42 wins out of 120"
  with 131, the 400-case sweep's win count, and both validators accepted it,
  because the old check asked only whether a number appeared *somewhere* in
  the recorded runs.
* **a missing row.** Dropping "we found a worse optimum" leaves every
  remaining number correct while making the result look unambiguous.
* **swapped outcomes.** Exchanging two values keeps the multiset of numbers
  intact, so any check that compares sets rather than pairs passes.

Nothing here writes to a tracked document; each mutation goes to a temporary
file and the check is pointed at that.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASELINE = ROOT / "bench" / "baseline.json"


def _load(name, relative):
    """Import by path, not by package name.

    `import tests.test_baseline` depends on pytest's rootdir and sys.path
    happening to make `tests` a package, which differs between a local run and
    a CI runner. Loading from an explicit path does not.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline_tests = _load("_baseline_tests", "tests/test_baseline.py")


@pytest.fixture(scope="module")
def record():
    if not BASELINE.is_file():
        pytest.skip("bench/baseline.json absent; run `python bench/baseline.py`")
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.fixture
def mutated(tmp_path, monkeypatch):
    """Run a check against a mutated copy of a document."""
    def run(document, mutate, check):
        source = ROOT / document
        text = source.read_text(encoding="utf-8")
        changed = mutate(text)
        assert changed != text, (
            "the mutation did not change the document; this control is stale "
            "and is no longer testing anything")
        target = tmp_path / source.name
        target.write_text(changed, encoding="utf-8")
        monkeypatch.setitem(baseline_tests.DOCS, document, target)
        return check
    return run


def _caught(check, document, payload):
    try:
        check(document, payload)
    except AssertionError as exc:
        return str(exc)
    return None


# --------------------------------------------------------- cross-experiment
def test_a_count_from_the_other_experiment_is_rejected(record, mutated):
    """The exact substitution a review demonstrated getting through."""
    wrong = record["stress"]["counts"]["we found a better optimum"]
    right = record["differential"]["counts"]["we found a better optimum"]
    assert wrong != right, "the two runs happen to agree; pick another outcome"

    check = mutated(
        "README.md",
        lambda t: t.replace(
            f"| mixedlm-rs found a strictly **better** optimum | **{right}** |",
            f"| mixedlm-rs found a strictly **better** optimum | **{wrong}** |"),
        baseline_tests.test_documented_outcome_counts_match_their_own_experiment)
    message = _caught(check, "README.md", record)
    assert message is not None, (
        f"a 120-case table reporting {wrong} wins -- the 400-case sweep's "
        "figure -- was accepted")
    assert str(wrong) in message and "differential" in message


def test_a_missing_outcome_row_is_rejected(record, mutated):
    worse = record["differential"]["counts"]["we found a worse optimum"]
    check = mutated(
        "README.md",
        lambda t: t.replace(
            f"| mixedlm-rs found a **worse** optimum | **{worse}** |\n", ""),
        baseline_tests.test_every_outcome_table_reports_every_outcome)
    message = _caught(check, "README.md", record)
    assert message is not None, "a table missing an outcome row was accepted"
    assert "we found a worse optimum" in message


def test_swapped_outcome_values_are_rejected(record, mutated):
    counts = record["differential"]["counts"]
    better, same = counts["we found a better optimum"], counts["same optimum"]
    assert better != same, "the two outcomes agree; a swap is undetectable"

    def swap(text):
        return (text
                .replace(f"| mixedlm-rs found a strictly **better** optimum | **{better}** |",
                         f"| mixedlm-rs found a strictly **better** optimum | **{same}** |")
                .replace(f"| same optimum | {same} |",
                         f"| same optimum | {better} |"))

    check = mutated(
        "README.md", swap,
        baseline_tests.test_documented_outcome_counts_match_their_own_experiment)
    message = _caught(check, "README.md", record)
    assert message is not None, (
        "swapping two outcome values kept the multiset of numbers intact and "
        "was accepted")


def test_the_unmutated_document_passes(record):
    """The other half of every control above: the check is not simply broken."""
    for name in sorted(baseline_tests.DOCS):
        baseline_tests.test_documented_outcome_counts_match_their_own_experiment(
            name, record)
        baseline_tests.test_every_outcome_table_reports_every_outcome(
            name, record)


# ------------------------------------------------------- freshness controls
def test_the_generator_files_are_watched_for_changes():
    """The differential fixtures come from a generator, and a change to it
    changes the numbers without touching the package."""
    watched = set(baseline_tests.NUMERIC_PATHS)
    for required in ("bench", "src", "python"):
        assert required in watched, f"{required} is not watched for freshness"
    assert any("test_fuzz" in extra for extra in baseline_tests.NUMERIC_FILES), (
        "tests/test_fuzz.py supplies the differential fixtures; a change to it "
        "changes the recorded counts, so it belongs in freshness validation")


def test_a_changed_generator_would_invalidate_the_baseline(record):
    """Directly: the freshness rule must name the generator files."""
    for path in baseline_tests.NUMERIC_FILES:
        assert (ROOT / path).is_file(), (
            f"freshness watches {path}, which does not exist -- the rule is "
            "watching nothing")


# -------------------------------------------------- installed-artifact proof
def test_the_recorded_artifact_covers_the_python_sources(record):
    """An extension hash alone does not identify the installed package.

    The changes in these reviews were Python changes. An old and a new Python
    wrapper can sit on top of the same compiled extension and report the same
    version, so the extension's hash says nothing about them.
    """
    env = record["environment"]
    if "python_sources_sha256" not in env:
        pytest.skip("baseline predates source identification; re-record")
    assert env["python_sources_sha256"], "the Python sources were not hashed"
    assert env.get("python_sources_files"), "no Python sources were listed"


def test_the_installed_sources_match_the_recorded_ones(record):
    """And the correspondence is checked, not merely recorded."""
    env = record["environment"]
    if "python_sources_sha256" not in env:
        pytest.skip("baseline predates source identification; re-record")
    if shutil.which("git") is None:
        pytest.skip("git unavailable")

    recorder = _load("_baseline_recorder", "bench/baseline.py")

    live = recorder._python_sources_digest()
    assert live["python_sources_files"] == env["python_sources_files"], (
        "the installed package contains a different set of Python modules "
        f"than the baseline recorded: {live['python_sources_files']} against "
        f"{env['python_sources_files']}")
    assert live["python_sources_sha256"] == env["python_sources_sha256"], (
        "the installed Python sources differ from the ones the baseline was "
        "recorded against. Rebuild and reinstall, then re-record: the numbers "
        "in bench/baseline.json describe different code from what is loaded.")

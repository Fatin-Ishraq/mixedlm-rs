"""Every headline timing must be traceable to a recorded experiment.

`bench/baseline.json` does this for the correctness sweeps. This is the same
discipline for the performance claims: a speedup in a README is a claim until
a reader can find the commit, the machine, the repetition count and the thread
settings behind it.

Two kinds of number live in the documents and they are not interchangeable:

* **measured** -- this package against statsmodels, re-run on the candidate
  build by `bench/performance.py`;
* **historical** -- the lme4 and pymer4 columns, which need R, rpy2 and a
  writable R library. A machine without them produces no number rather than a
  wrong one, so those are carried forward with the environment that produced
  them and must stay labelled as such.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECORD = ROOT / "bench" / "performance.json"
DOCS = {
    "README.md": ROOT / "README.md",
    "docs/BENCHMARKS.md": ROOT / "docs" / "BENCHMARKS.md",
}


@pytest.fixture(scope="module")
def record():
    if not RECORD.is_file():
        pytest.skip("bench/performance.json absent; "
                    "run `python bench/performance.py`")
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_the_record_identifies_its_experiment(record):
    """Commit, machine, repetitions, threads. Any missing one makes the
    numbers unreproducible in a way no reader could detect."""
    env = record["environment"]
    for field in ("commit", "python", "platform", "logical_cpus",
                  "rayon_num_threads", "repetitions", "statistic",
                  "extension_sha256", "mixedlm_rs"):
        assert env.get(field) not in (None, ""), f"{field} was not recorded"
    assert env["working_tree_dirty"] is False, (
        "the timings were recorded with uncommitted changes, so the commit "
        "they name does not describe the build that produced them")


def test_the_record_names_its_dependency_versions(record):
    deps = record["environment"]["dependencies"]
    for name in ("numpy", "scipy", "pandas", "statsmodels"):
        assert deps.get(name), f"{name} version not recorded"


def test_it_was_a_full_run(record):
    assert record["quick"] is False, (
        "bench/performance.json came from a --quick run, which measures only "
        "the small cases; the documents quote the large ones")


def test_every_measured_fit_agreed_before_it_was_timed(record):
    """A fast wrong answer is not a benchmark result."""
    import sys
    sys.path.insert(0, str(ROOT / "bench"))
    import tolerances

    for row in record["measured"]["rows"]:
        gap = row.get("agreement")
        if gap is None or not row.get("statsmodels_converged"):
            continue
        assert gap["fixed_effects_in_standard_errors"] <= tolerances.FIXED_EFFECT_SE, row
        assert gap["cov_re_relative"] <= tolerances.COV_RE_REL, row
        assert gap["delta_loglik"] >= -tolerances.DEVIANCE_ABS / 2, row


def test_every_repetition_is_kept(record):
    """The minimum is reported; the spread has to remain inspectable."""
    for row in record["measured"]["rows"]:
        reps = row["ours_repetitions"]
        assert len(reps) == record["environment"]["repetitions"], row
        assert min(reps) == pytest.approx(row["ours_seconds"]), (
            "the reported figure is not the minimum of the repetitions")


def test_the_historical_block_is_separate_and_labelled(record):
    """It must be impossible to read an lme4 number as freshly measured."""
    historical = record["historical"]
    assert "source" in historical and "environment" in historical
    assert historical["rows"], "no historical rows recorded"
    measured_keys = {k for row in record["measured"]["rows"] for k in row}
    assert "lme4" not in measured_keys and "pymer4" not in measured_keys, (
        "an lme4 or pymer4 timing appears in the measured block; those need R "
        "and cannot be measured here")


def test_the_pymer4_caveat_travels_with_the_numbers(record):
    caveat = record["historical"]["caveat"].lower()
    assert "lmertest" in caveat and "not a measurement" in caveat, (
        "the pymer4 column is the end-to-end cost of fitting through pymer4, "
        "which also runs lmerTest and extracts results; that has to travel "
        "with the number")


# ------------------------------------------------------------- the documents
def _quoted_speedups(text):
    """Speedup factors written as `123x` in a table row."""
    return {int(m.replace(",", ""))
            for m in re.findall(r"\|\s*\*{0,2}([\d,]+)x\*{0,2}\s*\|", text)}


@pytest.mark.parametrize("name", sorted(DOCS))
def test_documented_speedups_are_in_the_recorded_range(name, record):
    """Not exact equality: timings move between machines, and pinning a
    document to one machine's milliseconds would make it wrong everywhere
    else. What must hold is that a quoted factor is one this experiment
    actually produced, for a case it actually ran."""
    text = DOCS[name].read_text(encoding="utf-8")
    quoted = _quoted_speedups(text)
    if not quoted:
        pytest.skip(f"{name} quotes no speedup factors")

    measured = [row["speedup"] for row in record["measured"]["rows"]
                if row.get("speedup")]
    historical = [row["lme4"] / row["ours"]
                  for row in record["historical"]["rows"] if row.get("lme4")]
    historical += [row["pymer4"] / row["ours"]
                   for row in record["historical"]["rows"] if row.get("pymer4")]
    if not measured:
        pytest.skip("no measured speedups recorded")

    ceiling = max(measured + historical) * 1.5
    outsized = sorted(v for v in quoted if v > ceiling)
    assert not outsized, (
        f"{name} quotes speedup(s) {outsized} larger than anything the "
        f"recorded experiments produced (largest {max(measured + historical):.0f}x). "
        "Re-run bench/performance.py and update the table, or say which "
        "experiment the number comes from.")


@pytest.mark.parametrize("name", sorted(DOCS))
def test_the_documents_say_where_the_numbers_came_from(name, record):
    text = DOCS[name].read_text(encoding="utf-8")
    if not _quoted_speedups(text):
        pytest.skip(f"{name} quotes no speedup factors")
    assert "bench/performance.json" in text or "performance.json" in text, (
        f"{name} quotes timings without pointing at the recorded experiment; "
        "a reader cannot trace the number to a machine or a commit")


def test_the_documents_label_the_r_comparisons_as_historical():
    """lme4 and pymer4 columns are not from the candidate build."""
    text = DOCS["docs/BENCHMARKS.md"].read_text(encoding="utf-8")
    if "lme4" not in text:
        pytest.skip("no lme4 comparison documented")
    assert "historical" in text.lower() or "recorded on" in text.lower(), (
        "the lme4/pymer4 comparison needs R and was not re-measured on this "
        "build; say so where the numbers are")

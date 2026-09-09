"""Every count quoted in the documents must match the recorded baseline.

The 120-fixture differential comparison appeared in three documents and
disagreed: BENCHMARKS.md said 19 wins and 75 ties where README.md and
CORRECTNESS.md said 42 and 52. Both had been true at some commit. Nothing tied
either to a run, so nothing caught the drift.

`bench/baseline.json` is now the single record -- commit, environment, seeds,
counts, individual losses, warning categories -- written by
`python bench/baseline.py`. These tests read the documents and check the
numbers in them against that file, so a stale table fails the suite.

They are deliberately strict about *where* a number may appear: a count that is
not in the baseline cannot be quoted at all.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASELINE = ROOT / "bench" / "baseline.json"

DOCS = {
    "README.md": ROOT / "README.md",
    "docs/CORRECTNESS.md": ROOT / "docs" / "CORRECTNESS.md",
    "docs/BENCHMARKS.md": ROOT / "docs" / "BENCHMARKS.md",
}


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE.is_file():
        pytest.skip("bench/baseline.json absent; run `python bench/baseline.py`")
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_the_baseline_records_its_environment(baseline):
    """A count without an environment is not reproducible.

    Optimiser outcomes can move with the SciPy build, so the environment is
    part of the claim rather than a footnote to it.
    """
    env = baseline["environment"]
    assert env["commit"], "the baseline must record the commit it was run at"
    assert env["python"] and env["platform"]
    for dep in ("numpy", "scipy", "pandas", "statsmodels"):
        assert env["dependencies"][dep], f"{dep} version not recorded"


def test_the_baseline_was_recorded_on_a_clean_tree(baseline):
    assert baseline["environment"]["working_tree_dirty"] is False, (
        "the baseline was recorded with uncommitted changes, so the commit it "
        "names does not describe the code that produced it. Re-run "
        "`python bench/baseline.py` on a clean tree.")


def test_the_baseline_is_a_full_run(baseline):
    assert baseline["quick"] is False, (
        "bench/baseline.json came from a --quick run; the documents quote the "
        "full 120 and 400 case counts")
    assert baseline["differential"]["cases"] == 120
    assert baseline["stress"]["cases"] == 400


# ------------------------------------------------------ differential counts
def _differential_table_rows(text):
    """Rows of a markdown table whose left cell mentions an outcome."""
    wanted = {
        "did not converge": "reference did not converge",
        "better": "we found a better optimum",
        "same optimum": "same optimum",
        "worse": "we found a worse optimum",
    }
    found = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2:
            continue
        label, value = cells
        digits = re.findall(r"\d+", value)
        if not digits:
            continue
        for needle, key in wanted.items():
            if needle in label.lower():
                found.setdefault(key, []).append(int(digits[-1]))
    return found


@pytest.mark.parametrize("name", sorted(DOCS))
def test_documented_differential_counts_match_the_baseline(name, baseline):
    """Every outcome count quoted anywhere must be one the baseline records.

    Deliberately permissive about *which* table a number is in -- the stress
    sweep and the differential comparison share row labels -- but strict that
    it has to come from a recorded run. A number matching neither is stale.
    """
    text = DOCS[name].read_text(encoding="utf-8")
    allowed = set(baseline["differential"]["counts"].values())
    allowed |= set(baseline["stress"]["counts"].values())
    rows = _differential_table_rows(text)

    for key, values in rows.items():
        for value in values:
            assert value in allowed, (
                f"{name} quotes {value} for '{key}', which appears in neither "
                "the recorded differential nor the recorded stress counts "
                f"({sorted(allowed)}). Re-run `python bench/baseline.py` and "
                "update the document.")


def test_no_document_quotes_the_old_numbers(baseline):
    """The specific drift that motivated this file.

    19 wins / 75 ties was the count at an earlier commit and survived in
    BENCHMARKS.md long after README and CORRECTNESS had moved to 42 / 52.
    """
    counts = baseline["differential"]["counts"]
    stale = {19, 75} - {counts["we found a better optimum"],
                        counts["same optimum"]}
    for name, path in DOCS.items():
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if not line.startswith("|") or "|" not in line[1:]:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) != 2:
                continue
            label, value = cells
            if "better" not in label.lower() and "same optimum" not in label.lower():
                continue
            digits = {int(d) for d in re.findall(r"\d+", value)}
            assert not (digits & stale), (
                f"{name} still quotes a superseded count in {line.strip()!r}")


# ------------------------------------------------------------- stress counts
def _section(text, heading):
    """The lines under one `## heading`, up to the next `## `.

    Scoping matters: CORRECTNESS.md carries two outcome tables with identical
    row labels -- the 120-case differential comparison and the 400-case stress
    sweep. Matching on the label alone finds the first one and compares it
    against the other's counts, which is a test failing for its own reasons.
    """
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == heading)
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end])


def _table_counts(section):
    """`| label | 123 |` rows, as {label: value}."""
    out = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2:
            continue
        digits = re.findall(r"\d+", cells[1])
        if digits:
            out[cells[0].lower()] = int(digits[-1])
    return out


STRESS_ROWS = {
    "statsmodels did not converge": "reference did not converge",
    "we found a strictly **better** optimum": "we found a better optimum",
    "same optimum": "same optimum",
    "we found a **worse** optimum": "we found a worse optimum",
}


def test_documented_stress_counts_match_the_baseline(baseline):
    text = DOCS["docs/CORRECTNESS.md"].read_text(encoding="utf-8")
    section = _section(text, "## Adversarial stress sweep")
    rows = _table_counts(section)
    counts = baseline["stress"]["counts"]

    for label, key in STRESS_ROWS.items():
        if label.lower() not in rows:
            continue
        assert rows[label.lower()] == counts.get(key, 0), (
            f"the stress table says {rows[label.lower()]} for {key!r}; the "
            f"baseline records {counts.get(key, 0)}")


def test_every_recorded_loss_is_written_down(baseline):
    """A loss that is not in the document is a loss being quietly dropped."""
    section = _section(
        DOCS["docs/CORRECTNESS.md"].read_text(encoding="utf-8"),
        "## Adversarial stress sweep")
    # Compare parsed numbers, not string forms: the document may write
    # 5.619e-05 where the baseline holds 5.6190123e-05, and a substring match
    # would fail on the formatting rather than on the content.
    written = [float(tok) for tok in
               re.findall(r"\d+\.\d+e-\d+", section)]
    for loss in baseline["stress"]["losses"]:
        gap = loss["deviance_worse_by"]
        assert any(abs(w - gap) <= 0.02 * gap for w in written), (
            f"stress case {loss['case']} ({loss['shape']}) is worse by "
            f"{gap:.4e} deviance and is not recorded in the stress section of "
            f"CORRECTNESS.md (which lists {written})")


def test_the_number_of_losses_matches(baseline):
    section = _section(
        DOCS["docs/CORRECTNESS.md"].read_text(encoding="utf-8"),
        "## Adversarial stress sweep")
    rows = _table_counts(section)
    n = len(baseline["stress"]["losses"])
    assert rows.get("we found a **worse** optimum".lower()) == n


# -------------------------------------------------------------- suite counts
def test_documented_test_counts_match_the_recorded_run(baseline):
    if "suites" not in baseline:
        pytest.skip("baseline recorded with --skip-suites")
    summary = baseline["suites"]["pytest"]["summary"]
    match = re.search(r"(\d+) passed", summary)
    assert match, f"could not read a pass count from {summary!r}"
    passed = int(match.group(1))

    for name, path in DOCS.items():
        text = path.read_text(encoding="utf-8")
        for quoted in re.findall(r"\*\*(\d+) Python tests", text):
            assert int(quoted) == passed, (
                f"{name} says {quoted} Python tests; the recorded run has "
                f"{passed}")
        for quoted in re.findall(r"pytest tests/ -q\s+# (\d+) tests", text):
            assert int(quoted) == passed, (
                f"{name} says {quoted} tests; the recorded run has {passed}")


def test_the_recorded_suites_passed(baseline):
    if "suites" not in baseline:
        pytest.skip("baseline recorded with --skip-suites")
    assert baseline["suites"]["pytest"]["returncode"] == 0
    assert baseline["suites"]["cargo"]["returncode"] == 0

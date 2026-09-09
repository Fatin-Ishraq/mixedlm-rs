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
import os
import pathlib
import re
import sys

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


# Files whose contents can change a fitted number. Deliberately not "any file
# the diff touches": the gate fired on a Ruff configuration edit, which cannot
# move a deviance by any mechanism, and a gate that cries wolf gets silenced by
# re-recording rather than by thinking. pyproject.toml is included only through
# the sections that reach the build -- dependencies, build-system, tool.maturin
# -- not tool.ruff or tool.mypy.
# bench/ is included because the generators decide which cases are run: a
# change to differential.py or stress_sweep.py changes the numbers without
# touching a line of the package.
NUMERIC_PATHS = ("src", "python", "Cargo.toml", "Cargo.lock", "bench")
BUILD_SECTIONS = ("[project]", "[build-system]", "[tool.maturin]")


def _pyproject_build_sections(text):
    """The build-relevant part of a pyproject, as a string.

    Comparing this across two commits answers "could the built artifact have
    changed?" without answering "did any byte of the file change?".
    """
    out, keep = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            keep = any(stripped.startswith(sec[:-1]) for sec in BUILD_SECTIONS)
        if keep and stripped and not stripped.startswith("#"):
            out.append(stripped)
    return "\n".join(out)


def test_no_code_has_changed_since_the_baseline_was_recorded(baseline):
    """Doc and lint-config edits after a baseline are fine. Code edits are not.

    The clean-tree check above only says the tree was clean *at record time*.
    It says nothing about what happened afterwards, and a numerical baseline
    recorded before a change to the optimiser describes the wrong code.
    """
    import subprocess

    commit = baseline["environment"]["commit"]
    if commit is None:
        pytest.skip("baseline has no commit recorded")

    probe = subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"],
                           cwd=ROOT, capture_output=True, text=True)
    if probe.returncode != 0:
        # A shallow clone -- GitHub's default checkout depth is 1 -- does not
        # contain the recorded commit, and skipping there turns this gate off
        # in exactly the environment it is meant to guard. CI sets
        # fetch-depth: 0; if the history is missing there, that is a
        # configuration failure and has to be loud.
        if os.environ.get("CI"):
            pytest.fail(
                f"the baseline commit {commit[:10]} is not in this "
                "repository, so freshness cannot be checked. CI must check "
                "out full history (`fetch-depth: 0`).")
        pytest.skip("the recorded commit is not in this repository")

    # `commit..HEAD` misses uncommitted edits, which is the state a developer
    # is actually in when they run the suite. Diffing the working tree against
    # the recorded commit covers both.
    diff = subprocess.run(
        ["git", "diff", "--name-only", commit, "--", *NUMERIC_PATHS],
        cwd=ROOT, capture_output=True, text=True)
    if diff.returncode != 0:
        pytest.skip("git diff unavailable")
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--",
         *NUMERIC_PATHS],
        cwd=ROOT, capture_output=True, text=True)
    # The baseline file itself is the output, not an input to it.
    ignore = {"bench/baseline.json"}
    changed = [f for f in diff.stdout.split() if f.strip() and f not in ignore]
    changed += [f + " (untracked)" for f in untracked.stdout.split()
                if f.strip() and f not in ignore]

    was = subprocess.run(["git", "show", f"{commit}:pyproject.toml"],
                         cwd=ROOT, capture_output=True, text=True)
    if was.returncode == 0:
        now = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        if _pyproject_build_sections(was.stdout) != _pyproject_build_sections(now):
            changed.append("pyproject.toml (build sections)")

    assert not changed, (
        "code has changed since bench/baseline.json was recorded at "
        f"{commit[:10]}, so its numbers may no longer describe this tree: "
        + ", ".join(changed)
        + ". Re-run `python bench/baseline.py`.")


def test_the_baseline_names_the_binary_that_produced_it(baseline):
    """A version string does not identify a build.

    The Python sources ship inside the wheel, so a checkout and an installed
    wheel report the same ``__version__`` while containing different code. The
    baseline records the extension's path and content hash so the numbers can
    be traced to an actual artifact.
    """
    env = baseline["environment"]
    if "extension_sha256" not in env:
        pytest.skip("baseline predates binary identification; re-record")
    assert env["extension_sha256"], (
        "the baseline does not identify the compiled extension it ran "
        "against. Re-run `python bench/baseline.py`.")
    assert env["extension_bytes"], "extension size not recorded"


def test_the_baseline_publishes_no_machine_paths(baseline):
    """This file is committed, so it must not carry a home directory.

    The hash identifies the build; the absolute path only identifies the
    developer's account, and the benchmark report was already held to this.
    """
    blob = json.dumps(baseline)
    for marker in ("C:\\Users", "/home/", "/Users/", "AppData"):
        assert marker not in blob, (
            f"bench/baseline.json contains {marker!r}, which publishes a "
            "machine path. Re-run `python bench/baseline.py`.")


def test_every_recorded_suite_passed(baseline):
    """A baseline is only a gate if the run it records was green.

    The recorder used to check pytest's exit code alone, so a failing
    ``cargo test --lib`` was written into the file and reported as a success.
    """
    for name, run in baseline.get("suites", {}).items():
        assert run.get("returncode", 0) == 0, (
            f"bench/baseline.json records a failing {name} run "
            f"({run.get('summary')!r}); it is a record of a broken build, not "
            "a baseline. Fix the failure and re-record.")


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
    """Every outcome count quoted anywhere must be the one recorded *for that
    outcome*.

    Deliberately permissive about *which* table a number is in -- the stress
    sweep and the differential comparison share row labels, and a document may
    legitimately quote either. Strict that the number has to be the recorded
    value for the label it is written against: checking membership in the set
    of all recorded counts, as this once did, accepts a table with "better"
    and "same optimum" swapped, which is a wrong table made of right numbers.
    """
    text = DOCS[name].read_text(encoding="utf-8")
    differential = baseline["differential"]["counts"]
    stress = baseline["stress"]["counts"]
    rows = _differential_table_rows(text)

    for key, values in rows.items():
        allowed = {c[key] for c in (differential, stress) if key in c}
        if not allowed:
            continue
        for value in values:
            assert value in allowed, (
                f"{name} quotes {value} for '{key}', but the recorded runs "
                f"give {sorted(allowed)} for that outcome "
                f"(differential {differential.get(key)}, "
                f"stress {stress.get(key)}). A number recorded for a "
                "*different* outcome is not a defence. Re-run "
                "`python bench/baseline.py` and update the document.")


@pytest.mark.parametrize("name", sorted(DOCS))
def test_documented_outcome_tables_are_complete(name, baseline):
    """A table that simply omits a row cannot be caught by checking the rows
    that are present.

    Dropping "we found a worse optimum" from a table makes the results look
    unambiguous while every number left behind is correct, so per-row value
    checking passes. Any table that reports outcomes has to report all four.
    """
    text = DOCS[name].read_text(encoding="utf-8")
    rows = _differential_table_rows(text)
    if not rows:
        pytest.skip(f"{name} quotes no outcome table")
    required = {"reference did not converge", "we found a better optimum",
                "same optimum", "we found a worse optimum"}
    missing = sorted(required - set(rows))
    assert not missing, (
        f"{name} has an outcome table that omits {missing}. A partial table "
        "reads as a complete one; every outcome the baseline records has to "
        "appear.")


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
def test_documented_test_counts_match_what_the_suite_collects():
    """The documented figure is the *collected* count in a complete
    environment, checked against a live collection.

    Not the pass count: how many pass depends on how many skip, and that
    depends on what is installed.

    Not read from bench/baseline.json either: adding a test changes the
    collected count without changing any fitted number, so it does not
    invalidate the numerical baseline, and the documents and a stale baseline
    could sit at the same wrong number and still agree.

    But *collected* is only environment-independent when the optional
    dependencies are present. `tests/test_vs_statsmodels.py` and
    `tests/test_fuzz.py` call `pytest.importorskip` at module scope, so
    without statsmodels those files contribute zero collected tests rather
    than skipped ones -- the declared-floors CI job collects 398 against 576.
    So this runs only where the full toolchain is installed, and the documents
    say which environment the number describes.
    """
    import importlib.util
    import subprocess

    for optional in ("statsmodels", "mypy"):
        if importlib.util.find_spec(optional) is None:
            pytest.skip(f"{optional} absent; collection is not comparable")
    if not (ROOT / "data" / "sleepstudy.csv").is_file():
        pytest.skip("lme4 fixtures absent; collection is not comparable")

    run = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q",
         "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True)
    match = re.search(r"(\d+)\s+tests? collected", run.stdout)
    if not match:
        pytest.skip(f"could not collect: {run.stdout[-200:]}")
    collected = int(match.group(1))

    for name, path in DOCS.items():
        text = path.read_text(encoding="utf-8")
        for quoted in re.findall(r"\*\*(\d+) Python tests", text):
            assert int(quoted) == collected, (
                f"{name} says {quoted} Python tests; the suite collects "
                f"{collected}")
        for quoted in re.findall(r"pytest tests/ -q\s+# (\d+) tests", text):
            assert int(quoted) == collected, (
                f"{name} says {quoted} tests; the suite collects {collected}")

def test_the_recorded_run_had_no_failures(baseline):
    if "suites" not in baseline:
        pytest.skip("baseline recorded with --skip-suites")
    tally = baseline["suites"]["pytest"]
    assert tally.get("failed", 0) == 0, (
        f"the recorded run had {tally['failed']} failure(s); a baseline from "
        "a failing run is not a baseline")
    assert tally.get("error", 0) == 0


def test_the_recorded_suites_passed(baseline):
    if "suites" not in baseline:
        pytest.skip("baseline recorded with --skip-suites")
    assert baseline["suites"]["pytest"]["returncode"] == 0
    assert baseline["suites"]["cargo"]["returncode"] == 0

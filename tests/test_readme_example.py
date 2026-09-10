"""The README's example must run for someone who just installed the package.

It used to open `sleepstudy.csv`, which is not shipped: those fixtures are
GPL-2 and are excluded from both the wheel and the sdist. So the first thing a
new user copied out of the README failed on line four, with a FileNotFoundError
for a file the package never had.

The example is extracted from the README and executed here, in a subprocess
started outside the checkout, so what is tested is the text a reader actually
copies rather than a paraphrase of it that can drift.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
BLOCK = re.compile(r"<!-- readme-example -->\s*```python\n(.*?)```", re.S)


@pytest.fixture(scope="module")
def example() -> str:
    text = README.read_text(encoding="utf-8")
    found = BLOCK.search(text)
    assert found, (
        "the README no longer has a block marked `<!-- readme-example -->`; "
        "the markers are what lets this test check the real text")
    return found.group(1)


def test_the_example_reads_no_unshipped_file(example):
    for forbidden in ("read_csv", "read_table", "open(", "sleepstudy.csv"):
        assert forbidden not in example, (
            f"the example calls {forbidden!r}, so it depends on a file that "
            "pip install does not provide")


def test_the_example_runs_after_a_plain_install(example):
    """Run it from outside the checkout, as an installed user would.

    The working directory matters: inside the repository, `import mixedlm_rs`
    could resolve to `python/mixedlm_rs/` rather than the installed package,
    and the example would pass without the installed artifact working at all.
    """
    with tempfile.TemporaryDirectory() as elsewhere:
        done = subprocess.run(
            [sys.executable, "-c", example], cwd=elsewhere,
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300)
    assert done.returncode == 0, (
        f"the README example failed:\n{done.stdout}\n{done.stderr}")

    out = done.stdout
    assert "Mixed Linear Model Regression Results" in out, out[-500:]
    assert "converged: True" in out, "the example's model did not converge"


def test_the_example_recovers_the_effect_it_simulates(example):
    """A runnable example that returns nonsense is not much of an example.

    The data is generated with a slope of 10; the fit has to find it. This
    also pins the example's seed: an unseeded example that occasionally prints
    a wild estimate is worse than no example.
    """
    assert "default_rng(0)" in example, "the example must be deterministic"

    with tempfile.TemporaryDirectory() as elsewhere:
        done = subprocess.run(
            [sys.executable, "-c", example], cwd=elsewhere,
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300)
    assert done.returncode == 0, done.stderr

    line = next(ln for ln in done.stdout.splitlines()
                if ln.startswith("fixed effects:"))
    values = [float(v) for v in re.findall(r"-?\d+\.\d+", line)]
    assert len(values) == 2, line
    intercept, slope = values
    assert 200 < intercept < 300, f"intercept {intercept} is not plausible"
    assert 5 < slope < 15, (
        f"the example simulates a slope of 10 and the fit returned {slope}")


def test_the_readme_says_the_example_needs_no_data_file(example):
    text = README.read_text(encoding="utf-8")
    assert "no data file to fetch" in text, (
        "say that the example is self-contained; a reader who was previously "
        "sent looking for sleepstudy.csv has no reason to assume otherwise")


def test_the_sleepstudy_reference_names_its_licence(example):
    """The dataset is still referenced. It must stay labelled as unshipped."""
    text = README.read_text(encoding="utf-8")
    if "sleepstudy" not in text:
        pytest.skip("the README no longer mentions sleepstudy")
    window = text[text.index("sleepstudy"):][:900]
    assert "GPL-2" in window and "not shipped" in window, (
        "the sleepstudy reference must say the file is GPL-2 and is not "
        "shipped with the package")

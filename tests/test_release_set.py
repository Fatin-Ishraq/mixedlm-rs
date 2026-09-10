"""What gets published must be what was verified.

The release workflow builds wheels, verifies each on a runner of its own
architecture, and uploads. The gap was in between: it downloaded *every*
artifact in the run and merged them into one directory. The run also contains
the called CI workflow's wheels, whose filenames are identical to the release
wheels -- so a CI build, produced under different conditions and never
installed or run by any verification job, would silently replace the release
build and be published in its place.

Two defences, both tested here: the workflow selects artifacts by an explicit
pattern, and `scripts/check_release_set.py` compares the upload directory
against hashes recorded by the jobs that actually installed and ran each file.

Each case below is a real failure mode, and the gate must reject it. The
"clean" case is the negative control: a gate that rejects everything is as
useless as one that rejects nothing.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "check_release_set.py"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def artifacts():
    wheels = sorted((ROOT / "dist").glob("*.whl"))
    sdists = sorted((ROOT / "dist").glob("*.tar.gz"))
    if not wheels or not sdists:
        pytest.skip("no built artifacts in dist/; run maturin build && sdist")
    return wheels[-1], sdists[-1]


@pytest.fixture
def release(tmp_path, artifacts):
    """A verified release set, which individual tests then damage."""
    wheel, sdist = artifacts
    dist = tmp_path / "dist"
    manifests = tmp_path / "manifests"
    dist.mkdir()
    manifests.mkdir()
    for source in (wheel, sdist):
        shutil.copy(source, dist / source.name)
        (manifests / f"{source.name}.json").write_text(json.dumps({
            "filename": source.name,
            "sha256": sha256(source),
            "job": "verify",
            "runner": "test",
        }), encoding="utf-8")
    return dist, manifests


def run_gate(dist, manifests, expect=2):
    return subprocess.run(
        [sys.executable, str(GATE), "--dist", str(dist),
         "--manifests", str(manifests), "--expect", str(expect)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")


def test_a_verified_set_is_accepted(release):
    """The negative control for every rejection below."""
    dist, manifests = release
    result = run_gate(dist, manifests)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "each matching the artifact" in result.stdout


def test_a_ci_wheel_with_the_same_name_is_rejected(release, artifacts):
    """The finding itself.

    `merge-multiple` flattens artifacts into one directory. A CI wheel and a
    release wheel for the same platform have the same filename, so one
    overwrites the other and the survivor may be the unverified build.
    """
    wheel, _sdist = artifacts
    dist, manifests = release
    intruder = dist / "wheels-windows-latest"
    intruder.mkdir()
    (intruder / wheel.name).write_bytes(wheel.read_bytes() + b"a CI build")

    result = run_gate(dist, manifests)
    assert result.returncode != 0, "a colliding CI wheel was accepted"
    assert "appears 2 times" in result.stderr
    assert "distinct contents" in result.stderr


def test_an_unverified_extra_distribution_is_rejected(release, artifacts):
    wheel, _sdist = artifacts
    dist, manifests = release
    shutil.copy(wheel, dist / "mixedlm_rs-0.1.0-py3-none-any.whl")
    result = run_gate(dist, manifests, expect=2)
    assert result.returncode != 0
    assert "no verification job recorded it" in result.stderr


def test_a_missing_distribution_is_rejected(release):
    dist, manifests = release
    for path in dist.glob("*.tar.gz"):
        path.unlink()
    result = run_gate(dist, manifests)
    assert result.returncode != 0
    assert "absent from the upload directory" in result.stderr


def test_a_changed_file_is_rejected(release):
    dist, manifests = release
    target = next(dist.glob("*.whl"))
    target.write_bytes(target.read_bytes() + b"tampered")
    result = run_gate(dist, manifests)
    assert result.returncode != 0
    assert "changed after it was verified" in result.stderr


def test_a_stray_file_is_rejected(release):
    dist, manifests = release
    (dist / "build.log").write_text("noise", encoding="utf-8")
    result = run_gate(dist, manifests)
    assert result.returncode != 0
    assert "not distributions" in result.stderr


def test_an_unexpected_count_is_rejected(release):
    """A platform that failed to build must stop the release, not ship short."""
    dist, manifests = release
    result = run_gate(dist, manifests, expect=6)
    assert result.returncode != 0
    assert "expected exactly 6" in result.stderr


def test_no_manifests_at_all_is_rejected(release):
    dist, manifests = release
    for path in manifests.glob("*.json"):
        path.unlink()
    result = run_gate(dist, manifests)
    assert result.returncode != 0
    assert "no verification manifests" in result.stderr


# ------------------------------------------------------- workflow wiring
class TestTheWorkflowSelectsExplicitly:
    @pytest.fixture(scope="class")
    def workflow(self):
        yaml = pytest.importorskip("yaml")
        return yaml.safe_load(
            (ROOT / ".github" / "workflows" / "release.yml").read_text("utf-8"))

    def test_every_download_is_filtered(self, workflow):
        """An unfiltered download takes the called CI workflow's wheels too."""
        unfiltered = []
        for name, job in workflow["jobs"].items():
            for step in job.get("steps", []) or []:
                uses = step.get("uses", "")
                if not uses.startswith("actions/download-artifact"):
                    continue
                using = step.get("with", {}) or {}
                if not using.get("pattern") and not using.get("name"):
                    unfiltered.append(f"{name}: {step.get('name', uses)}")
        assert not unfiltered, (
            "these steps download every artifact in the run, including the "
            f"called CI workflow's wheels: {unfiltered}")

    def test_release_artifacts_are_named_apart_from_ci(self, workflow):
        names = []
        for job in workflow["jobs"].values():
            for step in job.get("steps", []) or []:
                if step.get("uses", "").startswith("actions/upload-artifact"):
                    names.append((step.get("with", {}) or {}).get("name", ""))
        published = [n for n in names if "wheel" in n or "sdist" in n]
        assert published, "the release uploads no wheels or sdist"
        assert all(n.startswith(("release-", "verified-")) for n in published), (
            f"release artifacts must be distinguishable from CI's: {published}")

    def test_publish_runs_the_gate_before_uploading(self, workflow):
        steps = workflow["jobs"]["publish"]["steps"]
        order = [step.get("run", "") + step.get("uses", "") for step in steps]
        gate = next(i for i, text in enumerate(order)
                    if "check_release_set.py" in text)
        upload = next(i for i, text in enumerate(order)
                      if "gh-action-pypi-publish" in text)
        assert gate < upload, "the release set is checked after uploading"

    def test_twine_check_runs_before_uploading(self, workflow):
        steps = workflow["jobs"]["publish"]["steps"]
        order = [step.get("run", "") + step.get("uses", "") for step in steps]
        assert any("twine check" in text for text in order)
        check = next(i for i, t in enumerate(order) if "twine check" in t)
        upload = next(i for i, t in enumerate(order)
                      if "gh-action-pypi-publish" in t)
        assert check < upload


class TestEveryWheelIsExecuted:
    """No advertised wheel ships without a runtime test on its architecture."""

    @pytest.fixture(scope="class")
    def workflow(self):
        yaml = pytest.importorskip("yaml")
        return yaml.safe_load(
            (ROOT / ".github" / "workflows" / "release.yml").read_text("utf-8"))

    def test_built_and_executed_platforms_agree(self, workflow):
        jobs = workflow["jobs"]
        built = {(e["platform"], e["target"])
                 for e in jobs["wheels"]["strategy"]["matrix"]["include"]}
        ran = {(e["platform"], e["target"])
               for e in jobs["verify-wheels"]["strategy"]["matrix"]["include"]}
        gap = sorted(built - ran)
        assert not gap, (
            "these wheels would be published without ever being run: "
            + ", ".join(f"{p}/{t}" for p, t in gap))

    def test_the_retired_macos_runner_is_gone(self, workflow):
        """GitHub retired macos-13 in December 2025."""
        text = (ROOT / ".github" / "workflows" / "release.yml").read_text("utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue                      # the note explaining the change
            assert "macos-13" not in stripped, (
                f"a retired runner image is still selected: {stripped}")

    def test_ci_proves_the_unusual_runner_labels(self):
        """A label in a release matrix is a claim until something runs on it.

        Neither label is used by the ordinary OS matrix, so without this job
        the Intel macOS and ARM64 Linux wheels would first meet their runners
        during a release.
        """
        yaml = pytest.importorskip("yaml")
        ci = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8"))
        probe = ci["jobs"]["architectures"]["strategy"]["matrix"]["include"]
        labels = {entry["label"] for entry in probe}

        release = yaml.safe_load(
            (ROOT / ".github" / "workflows" / "release.yml").read_text("utf-8"))
        used = {e["runner"] for e in
                release["jobs"]["verify-wheels"]["strategy"]["matrix"]["include"]}
        common = {"ubuntu-latest", "macos-latest", "windows-latest"}
        unusual = used - common
        assert unusual <= labels, (
            f"the release uses {sorted(unusual - labels)}, which no CI job "
            "ever runs on, so the labels are unverified")

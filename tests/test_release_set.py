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

    @staticmethod
    def _job_running(workflow, needle):
        for name, job in workflow["jobs"].items():
            for step in job.get("steps") or []:
                if needle in (step.get("run", "") + step.get("uses", "")):
                    return name
        return None

    def test_the_gate_runs_in_a_job_publish_depends_on(self, workflow):
        """Not merely "before the upload step" -- in a *separate* job.

        The gate used to sit inside `publish`, which is skipped on a
        `publish_to=nowhere` rehearsal. So the one defence against publishing
        an unverified file was the one thing a rehearsal could not exercise.
        """
        gate_job = self._job_running(workflow, "check_release_set.py")
        assert gate_job, "nothing runs the release-set gate"
        assert gate_job != "publish", (
            "the gate is inside the publish job, so a rehearsal skips it")
        assert gate_job in workflow["jobs"]["publish"]["needs"], (
            f"publish does not depend on {gate_job}, so it could upload "
            "without the gate having run")

    def test_the_gate_job_is_not_conditional(self, workflow):
        """It must run on a rehearsal too, or rehearsing proves less."""
        gate_job = self._job_running(workflow, "check_release_set.py")
        assert "if" not in workflow["jobs"][gate_job], (
            f"{gate_job} is conditional, so a rehearsal may skip the gate")

    def test_twine_check_runs_before_any_upload(self, workflow):
        twine_job = self._job_running(workflow, "twine check")
        assert twine_job, "nothing runs twine check"
        assert (twine_job == "publish"
                or twine_job in workflow["jobs"]["publish"]["needs"]), (
            "twine check does not run before the upload")

    def test_every_downloaded_artifact_is_uploaded_somewhere(self, workflow):
        """A rename that misses one side fails only at release time.

        Renaming the sdist upload to `release-sdist` left `verify-sdist`
        downloading `sdist`, and the rehearsal died with "Artifact not found
        for name: sdist" after building every wheel. Nothing local caught it,
        because both halves are valid YAML on their own.
        """
        uploaded, downloaded = set(), {}
        for name, job in workflow["jobs"].items():
            for step in job.get("steps") or []:
                uses = step.get("uses", "")
                using = step.get("with") or {}
                if uses.startswith("actions/upload-artifact") and using.get("name"):
                    uploaded.add(using["name"])
                elif uses.startswith("actions/download-artifact") and using.get("name"):
                    downloaded[using["name"]] = name

        def matches(wanted):
            # Matrix names contain ${{ }} on both sides; compare the literal
            # prefix before the first expression.
            head = wanted.split("${{")[0]
            return any(candidate.split("${{")[0] == head
                       for candidate in uploaded)

        missing = {n: j for n, j in downloaded.items() if not matches(n)}
        assert not missing, (
            "these jobs download an artifact no job uploads, which fails only "
            f"once a release is under way: {missing}. Uploaded: "
            f"{sorted(uploaded)}")

    def test_publish_uploads_only_the_checked_set(self, workflow):
        """One named artifact, not a pattern it could widen."""
        downloads = [
            (step.get("with") or {}) for step in workflow["jobs"]["publish"]["steps"]
            if step.get("uses", "").startswith("actions/download-artifact")]
        assert downloads, "publish downloads nothing"
        for using in downloads:
            assert using.get("name"), (
                "publish downloads by pattern; it must take the single "
                f"artifact the gate blessed, got {using}")
            assert not using.get("merge-multiple"), (
                "publish merges artifacts, so it can assemble a set the gate "
                "never saw")


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

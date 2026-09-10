"""Install one exact release artifact and run it.

Used by the release workflow on each runner whose platform matches a built
wheel, and runnable locally against `dist/`. It answers one question: does
*this specific file* install into a clean environment and produce correct
numbers there?

    python scripts/verify_release_artifacts.py --wheel dist/mixedlm_rs-...whl
    python scripts/verify_release_artifacts.py --sdist dist/mixedlm_rs-....tar.gz

Two things it does that the obvious version does not.

**It installs the artifact by path, with dependencies allowed to resolve.**
The release workflow used `pip install --no-index --find-links collected
mixedlm-rs`, which cannot work: `--no-index` applies to dependencies too, and
`collected/` holds only this project's own files, so the install died on
`No matching distribution found for numpy>=1.23`. Naming the file directly
installs that file; dependencies come from the index as they would for a real
user.

**It proves the installed distribution is the file it was given.** pip records
a `direct_url.json` for a path install, containing the archive's SHA-256. That
hash is compared against the file on disk, so "installed from the intended
artifact" is verified rather than assumed -- a resolver fallback to a same-named
distribution from elsewhere would change it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Fitted in a clean environment against a known-good answer. Deliberately not
# an import-and-print: an artifact that imports but computes the wrong numbers
# is the failure worth catching here.
SMOKE = r"""
import json, sys
import numpy as np, pandas as pd
import mixedlm_rs as mlm
import importlib.metadata as md

rng = np.random.default_rng(0)
g = np.repeat(np.arange(20), 10)
x = rng.normal(size=200)
y = 1 + 0.5 * x + rng.normal(size=20)[g] + rng.normal(size=200) * 0.5
d = pd.DataFrame({"y": y, "x": x, "g": g})
r = mlm.mixedlm("y ~ x", d, groups=d["g"], re_formula="~x").fit()
assert r.converged, "the documented example did not converge"
assert abs(float(r.fe_params[1]) - 0.5) < 0.15, r.fe_params

# Round-trips through the persistence path too: the last three reviews all
# found defects there, and none of them would show up in an import check.
import pickle
new = pd.DataFrame({"x": [0.1, 0.5]})
before = np.asarray(r.predict(new), float)
after = np.asarray(pickle.loads(pickle.dumps(r)).predict(new), float)
assert np.allclose(before, after, rtol=0, atol=0), (before, after)

dist = md.distribution("mixedlm-rs")
raw = dist.read_text("direct_url.json")
print(json.dumps({
    "package_file": mlm.__file__,
    "version": dist.version,
    "direct_url": json.loads(raw) if raw else None,
    "llf": float(r.llf),
}))
"""


def readme_example():
    """The example a reader copies, extracted from the README itself.

    Taken from the file rather than duplicated here, so the thing verified is
    the text that is published.
    """
    import re
    readme = ROOT / "README.md"
    if not readme.is_file():
        return None
    pattern = (r"<!-- readme-example -->\s*```python\n(.*?)```")
    found = re.search(pattern, readme.read_text(encoding="utf-8"), re.S)
    return found.group(1) if found else None


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd, **kwargs):
    return subprocess.run(cmd, text=True, encoding="utf-8",
                          errors="replace", **kwargs)


def verify(artifact: pathlib.Path, kind: str, keep: bool) -> int:
    artifact = artifact.resolve()
    if not artifact.is_file():
        print(f"no such artifact: {artifact}", file=sys.stderr)
        return 1
    digest = sha256(artifact)
    print(f"verifying {kind}: {artifact.name}")
    print(f"  sha256 {digest}")

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="mixedlm-verify-"))
    failures: list[str] = []
    try:
        venv = workdir / "venv"
        run([sys.executable, "-m", "venv", str(venv)], check=True)
        python = venv / ("Scripts" if sys.platform == "win32" else "bin") / (
            "python.exe" if sys.platform == "win32" else "python")
        run([str(python), "-m", "pip", "install", "--quiet", "--upgrade",
             "pip"], check=True)

        install_target = artifact
        if kind == "sdist":
            # Rebuild a wheel from *this* archive rather than installing the
            # sdist directly, so what gets exercised is the wheel a user
            # building from source would obtain.
            unpacked = workdir / "src"
            shutil.unpack_archive(str(artifact), str(unpacked))
            [tree] = [p for p in unpacked.iterdir() if p.is_dir()]
            run([str(python), "-m", "pip", "install", "--quiet", "maturin"],
                check=True)
            built = run([str(python), "-m", "maturin", "build", "--release",
                         "--out", str(workdir / "rebuilt")], cwd=tree,
                        capture_output=True)
            if built.returncode != 0:
                print(built.stdout or "", built.stderr or "", file=sys.stderr)
                return 1
            wheels = sorted((workdir / "rebuilt").glob("*.whl"))
            if not wheels:
                print("the sdist produced no wheel", file=sys.stderr)
                return 1
            install_target = wheels[-1]
            print(f"  rebuilt {install_target.name} from the sdist")

        # The corrected install: an exact path, dependencies resolving.
        install = run([str(python), "-m", "pip", "install", "--quiet",
                       str(install_target)], capture_output=True)
        if install.returncode != 0:
            print(install.stdout or "", install.stderr or "", file=sys.stderr)
            failures.append("install failed")
            return 1

        # Run from outside the checkout: inside it, Python would import
        # python/mixedlm_rs/ and the artifact would never be touched.
        smoke = run([str(python), "-c", SMOKE], cwd=str(workdir),
                    capture_output=True)
        if smoke.returncode != 0:
            print(smoke.stdout or "", smoke.stderr or "", file=sys.stderr)
            failures.append("smoke test failed")
            return 1

        # The documented example, verbatim from the README, against this
        # artifact. It is the first thing a new user runs, and it used to open
        # a CSV that neither distribution ships -- so it failed on line four
        # for everyone who installed from PyPI while passing every test here.
        example = readme_example()
        if example is None:
            failures.append("no `<!-- readme-example -->` block in README.md")
        else:
            shown = run([str(python), "-c", example], cwd=str(workdir),
                        capture_output=True)
            if shown.returncode != 0:
                print(shown.stdout or "", shown.stderr or "", file=sys.stderr)
                failures.append("the README example failed against this "
                                "artifact")
            elif "Mixed Linear Model Regression Results" not in shown.stdout:
                failures.append("the README example produced no summary")
            else:
                print("  the README example runs against the installed package")

        report = json.loads(smoke.stdout.strip().splitlines()[-1])
        print(f"  imported from {report['package_file']}")
        print(f"  version {report['version']}, llf {report['llf']:.9f}")

        if str(ROOT) in report["package_file"]:
            failures.append(
                f"imported from the checkout ({report['package_file']}), not "
                "from the installed artifact")

        direct = report.get("direct_url") or {}
        recorded = (direct.get("archive_info") or {}).get("hash", "")
        expected = sha256(install_target)
        if not recorded:
            failures.append("pip recorded no direct_url.json; provenance "
                            "cannot be established")
        elif recorded.removeprefix("sha256=") != expected:
            failures.append(
                f"the installed distribution came from {recorded}, not from "
                f"the artifact under test (sha256={expected})")
        else:
            print(f"  provenance ok: installed from {install_target.name} "
                  f"({expected[:16]}...)")
    finally:
        if keep:
            print(f"  kept {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    if failures:
        for line in failures:
            print(f"FAIL: {line}", file=sys.stderr)
        return 1
    print(f"  {kind} ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", type=pathlib.Path)
    ap.add_argument("--sdist", type=pathlib.Path)
    ap.add_argument("--keep", action="store_true",
                    help="leave the temporary environment in place")
    args = ap.parse_args()
    if not args.wheel and not args.sdist:
        ap.error("pass --wheel and/or --sdist")

    status = 0
    if args.wheel:
        status |= verify(args.wheel, "wheel", args.keep)
    if args.sdist:
        status |= verify(args.sdist, "sdist", args.keep)
    return status


if __name__ == "__main__":
    raise SystemExit(main())

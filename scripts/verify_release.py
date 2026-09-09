"""Prove the artifacts install and work, outside the checkout.

Building a wheel is not evidence that it installs; installing it inside the
source tree is not evidence either, because Python will happily import
`python/mixedlm_rs/` and the compiled module from `target/` and never touch the
wheel at all. Every check here runs from a temporary directory, in a fresh
virtual environment, against the artifact.

What it does:

1. builds a wheel and an sdist from the current tree;
2. installs the **exact wheel** into a clean venv outside the checkout, and
   runs the documented example, numerical smoke tests, and save/load there;
3. unpacks the sdist somewhere else, builds a **second wheel from it**, and
   runs the same checks against that -- which is what catches a file the sdist
   forgot to include;
4. checks version consistency, metadata, licence files, declared dependencies
   and the typing files in both artifacts.

    python scripts/verify_release.py [--keep] [--skip-sdist]

Exit code 0 means every check passed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(f"{label}: {detail}" if detail else label)
    return ok


def run(cmd, cwd=None, env=None):
    # utf-8 with replacement: maturin prints emoji, and the Windows console
    # default (cp1252) cannot decode them, which crashes the reader thread
    # and loses the output we are trying to check.
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def expect(cmd, cwd=None, label=None):
    p = run(cmd, cwd=cwd)
    ok = p.returncode == 0
    check(label or " ".join(map(str, cmd[:3])), ok,
          "" if ok else (p.stderr or p.stdout).strip()[-400:])
    return p


# ------------------------------------------------------------------- building
def build(out: pathlib.Path, skip_sdist: bool) -> tuple[pathlib.Path, pathlib.Path | None]:
    print("\nbuilding artifacts")
    out.mkdir(parents=True, exist_ok=True)
    expect([sys.executable, "-m", "maturin", "build", "--release",
            "--out", str(out)], cwd=ROOT, label="wheel builds")
    wheels = sorted(out.glob("*.whl"))
    check("exactly one wheel produced", len(wheels) == 1,
          f"{[w.name for w in wheels]}")
    sdist = None
    if not skip_sdist:
        expect([sys.executable, "-m", "maturin", "sdist", "--out", str(out)],
               cwd=ROOT, label="sdist builds")
        sdists = sorted(out.glob("*.tar.gz"))
        check("exactly one sdist produced", len(sdists) == 1,
              f"{[s.name for s in sdists]}")
        sdist = sdists[-1] if sdists else None
    return (wheels[-1] if wheels else None), sdist


# ----------------------------------------------------------------- inspection
def inspect_wheel(wheel: pathlib.Path, version: str) -> None:
    print(f"\ninspecting {wheel.name}")
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        meta = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
        text = z.read(meta).decode("utf-8") if meta else ""
        licences = [n for n in names if ".dist-info/licenses/" in n
                    or n.endswith(".dist-info/LICENSE")]
        notices = next((n for n in names
                        if n.endswith("THIRD-PARTY-LICENSES.md")), None)
        notice_text = z.read(notices).decode("utf-8") if notices else ""

    check("wheel filename carries the version", version in wheel.name, wheel.name)
    check("METADATA present", bool(meta))
    check("METADATA version matches", f"Version: {version}" in text)
    check("licence file included", bool(licences), str(licences))
    # "some licence file exists" is satisfied by our own MIT LICENSE alone,
    # which is exactly the state that shipped a wheel missing every upstream
    # notice. The extension statically links BSD-2-Clause and MIT crates, and
    # both require their copyright, conditions and disclaimer to accompany a
    # binary redistribution -- so check for the texts, by content.
    check("upstream licence texts included", bool(notices), str(notices))
    check("upstream notices carry the BSD-2-Clause disclaimer",
          "Redistribution and use in source and binary forms" in notice_text
          and "THE SOFTWARE IS PROVIDED" in notice_text.upper())
    check("upstream notices carry a copyright line",
          notice_text.count("Copyright") >= 5,
          f"{notice_text.count('Copyright')} copyright notices")
    for crate in ("numpy", "pyo3", "rayon", "ndarray"):
        check(f"notice text present for linked crate `{crate}`",
              f"## {crate} " in notice_text)
    check("py.typed included",
          any(n.endswith("mixedlm_rs/py.typed") for n in names))
    check("native stub included",
          any(n.endswith("mixedlm_rs/_mixedlm_rs.pyi") for n in names))
    check("compiled module included",
          any(n.endswith((".so", ".pyd", ".dylib")) for n in names))
    check("abi3 tagged", "abi3" in wheel.name, wheel.name)
    check("no GPL test fixtures", not any("/data/" in n for n in names))

    for dep in ("numpy", "scipy", "pandas", "patsy"):
        check(f"declares {dep}", f"Requires-Dist: {dep}" in text)
    check("declares a Python floor", "Requires-Python:" in text)
    check("project URLs present", "Project-URL:" in text)


def inspect_sdist(sdist: pathlib.Path, version: str) -> None:
    print(f"\ninspecting {sdist.name}")
    with tarfile.open(sdist) as t:
        names = t.getnames()
    check("sdist filename carries the version", version in sdist.name)
    check("no GPL test fixtures", not any("/data/" in n for n in names),
          str([n for n in names if "/data/" in n][:3]))
    for needed in ("Cargo.toml", "pyproject.toml", "LICENSE",
                   "src/lib.rs", "python/mixedlm_rs/__init__.py",
                   "python/mixedlm_rs/py.typed",
                   "python/mixedlm_rs/_mixedlm_rs.pyi"):
        check(f"sdist contains {needed}",
              any(n.endswith("/" + needed) for n in names))
    check("sdist contains the tests",
          any("/tests/test_" in n for n in names))


# ------------------------------------------------------------------- smoke
SMOKE = r'''
import json, pathlib, pickle, sys, tempfile, warnings
import numpy as np, pandas as pd
import mixedlm_rs as mlm

out = {"version": mlm.__version__, "file": mlm.__file__}

# The documented example, from the README.
rng = np.random.default_rng(0)
g = np.repeat(np.arange(20), 10)
x = rng.normal(size=200)
y = 1 + 0.5 * x + rng.normal(size=20)[g] + rng.normal(size=200) * 0.5
d = pd.DataFrame({"y": y, "x": x, "g": g})
r = mlm.mixedlm("y ~ x", d, groups=d["g"], re_formula="~x").fit()
assert r.converged, "the documented example did not converge"
assert "Mixed Linear Model" in str(r.summary())
out["llf"] = float(r.llf)

# Numerical smoke: response translation must not move the fit.
moved = mlm.mixedlm("y ~ x", d.assign(y=d["y"] + 1e8), groups=d["g"],
                    re_formula="~x").fit()
assert abs(moved.llf - r.llf) < 1e-6, (moved.llf, r.llf)
out["translation_gap"] = abs(moved.llf - r.llf)

# The array API, with no formula involved.
endog = d["y"].to_numpy()
exog = np.column_stack([np.ones(len(d)), d["x"].to_numpy()])
r2 = mlm.MixedLM(endog, exog, d["g"].to_numpy()).fit()
assert np.isfinite(r2.llf)

# Malformed native input must raise, not abort the interpreter.
try:
    mlm.LmmCore(endog, exog, np.ones((len(d), 1)),
                d["g"].to_numpy().astype(np.int64), 20).deviance(np.array([]))
    raise AssertionError("empty theta did not raise")
except ValueError:
    pass

# save / load, including formula prediction on raw new data.
new = pd.DataFrame({"x": [0.5, -1.0]})
before = r.predict(new)
with tempfile.TemporaryDirectory() as tmp:
    path = pathlib.Path(tmp) / "fit.pkl"
    r.save(path)
    back = mlm.MixedLMResults.load(path)
    assert np.allclose(back.predict(new), before), "formula prediction lost"
    assert abs(back.llf - r.llf) < 1e-12
    r.save(path, with_data=False)
    lean = mlm.MixedLMResults.load(path)
    assert abs(lean.llf - r.llf) < 1e-12
assert np.allclose(pickle.loads(pickle.dumps(r)).fe_params, r.fe_params)

# The typing marker has to be in the installed tree, not just the repo.
pkg = pathlib.Path(mlm.__file__).parent
assert (pkg / "py.typed").is_file(), "py.typed missing from the install"
assert (pkg / "_mixedlm_rs.pyi").is_file(), "stub missing from the install"

print(json.dumps(out))
'''


def venv_python(venv: pathlib.Path) -> pathlib.Path:
    return venv / ("Scripts" if sys.platform == "win32" else "bin") / (
        "python.exe" if sys.platform == "win32" else "python")


def install_and_smoke(wheel: pathlib.Path, workdir: pathlib.Path,
                      version: str, label: str) -> None:
    print(f"\n{label}")
    venv = workdir / "venv"
    expect([sys.executable, "-m", "venv", str(venv)], label="venv created")
    py = venv_python(venv)

    expect([str(py), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
           label="pip upgraded")
    # The wheel by exact path, so nothing can silently come from an index.
    expect([str(py), "-m", "pip", "install", "--quiet", str(wheel)],
           label="wheel installs (with its declared dependencies)")

    script = workdir / "smoke.py"
    script.write_text(SMOKE, encoding="utf-8")
    # Run from the temp directory: inside the checkout, Python would import
    # python/mixedlm_rs/ and never touch the installed artifact.
    p = run([str(py), str(script)], cwd=str(workdir))
    ok = p.returncode == 0
    check("smoke tests pass against the installed artifact", ok,
          "" if ok else (p.stderr or p.stdout).strip()[-800:])
    if ok:
        info = json.loads(p.stdout.strip().splitlines()[-1])
        check("installed version matches", info["version"] == version,
              info["version"])
        check("imported from the venv, not the checkout",
              str(venv) in info["file"], info["file"])
        check("response translation is exact",
              info["translation_gap"] < 1e-6, f"{info['translation_gap']:.2e}")

    # Runtime requirements and extras must be told apart: `requires()` returns
    # both, and the extras carry a `; extra == "..."` marker. Stripping that
    # marker -- as this first did -- makes ruff and pytest look like runtime
    # dependencies of a numerical library, which would be a real packaging bug
    # if it were true.
    q = run([str(py), "-c",
             "import importlib.metadata as m, json;"
             "d=m.distribution('mixedlm-rs');"
             "reqs=[r for r in (d.requires or [])];"
             "runtime=[r for r in reqs if 'extra ==' not in r];"
             "extras=[r for r in reqs if 'extra ==' in r];"
             "print(json.dumps({'version': d.version,"
             " 'runtime': runtime, 'extras': extras}))"],
            cwd=str(workdir))
    check("metadata readable from the install", q.returncode == 0,
          q.stderr.strip()[-200:])
    if q.returncode == 0:
        info = json.loads(q.stdout.strip().splitlines()[-1])
        check("installed metadata version matches",
              info["version"] == version, info["version"])
        runtime = {r.split(">")[0].split("=")[0].split(";")[0].strip()
                   for r in info["runtime"]}
        check("runtime dependencies are exactly the four declared",
              runtime == {"numpy", "scipy", "pandas", "patsy"},
              str(sorted(runtime)))
        check("developer tooling is behind an extra, not required",
              not (runtime & {"pytest", "ruff", "mypy", "statsmodels"}),
              str(sorted(runtime)))
        check("extras are declared", bool(info["extras"]),
              str(len(info["extras"])) + " entries")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true",
                    help="leave the temporary directory in place")
    ap.add_argument("--skip-sdist", action="store_true")
    args = ap.parse_args()

    import tomllib
    version = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    print(f"verifying mixedlm-rs {version} from {ROOT}")

    work = pathlib.Path(tempfile.mkdtemp(prefix="mixedlm-release-"))
    print(f"working outside the checkout in {work}")
    try:
        wheel, sdist = build(work / "artifacts", args.skip_sdist)
        if wheel is None:
            print("\nno wheel built; stopping")
            return 1

        inspect_wheel(wheel, version)
        install_and_smoke(wheel, work / "from-wheel", version,
                          "installing the built wheel into a clean venv")

        if sdist is not None:
            inspect_sdist(sdist, version)
            print("\nrebuilding from the unpacked sdist")
            unpacked = work / "unpacked"
            unpacked.mkdir()
            with tarfile.open(sdist) as t:
                t.extractall(unpacked)
            [tree] = list(unpacked.iterdir())
            out2 = work / "artifacts-from-sdist"
            expect([sys.executable, "-m", "maturin", "build", "--release",
                    "--out", str(out2)], cwd=tree,
                   label="wheel builds from the sdist")
            rebuilt = sorted(out2.glob("*.whl"))
            if rebuilt:
                inspect_wheel(rebuilt[-1], version)
                install_and_smoke(rebuilt[-1], work / "from-sdist", version,
                                  "installing the sdist-built wheel")
            else:
                check("sdist produced a wheel", False)
    finally:
        if args.keep:
            print(f"\nkept {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all release checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

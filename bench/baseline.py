"""Record one numerical baseline, so the documents quote a run instead of memory.

The counts for the 120-fixture differential comparison appeared in three
documents and disagreed: BENCHMARKS.md said 19 wins and 75 ties where README
and CORRECTNESS.md said 42 and 52. Both had been true at different commits.
Nothing tied either to a run.

This writes `bench/baseline.json`: the commit, the exact environment, the
seeds, the counts, every individual loss, and every warning category raised.
`tests/test_baseline.py` then checks the documents against that file, so a
stale number fails the suite rather than sitting in a table.

    python bench/baseline.py                 # run everything, write the file
    python bench/baseline.py --quick         # 30 differential / 60 stress cases

Outcomes can move with the SciPy build -- the optimiser is theirs -- so the
recorded environment is part of the claim, not a footnote to it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "baseline.json"
sys.path.insert(0, str(ROOT / "tests"))

warnings.simplefilter("always")


def _python_sources_digest() -> dict:
    """Hash the installed *Python* sources, not only the extension.

    The extension's hash was the whole provenance record, and it does not
    establish what it was claimed to. Every change in the recent reviews was a
    Python change: an old and a new wrapper import the same compiled
    extension, report the same version, and produce different numbers. Hashing
    the modules that are actually imported closes that.

    The digest is over (relative name, content) pairs in sorted order, so it
    is stable across machines and directories and changes when any module
    does.
    """
    import hashlib

    import mixedlm_rs

    package = pathlib.Path(mixedlm_rs.__file__).resolve().parent
    files = sorted(q for q in package.glob("*.py"))
    digest = hashlib.sha256()
    names = []
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        names.append(path.name)
    return {
        "python_sources_sha256": digest.hexdigest() if names else None,
        "python_sources_files": names,
    }


def _extension_identity() -> dict:
    """Which compiled binary actually produced these numbers.

    The Python sources live inside the wheel, so an editable checkout and an
    installed wheel can disagree while both reporting version 0.1.0. A
    baseline that records only the version string cannot tell you whether its
    numbers came from the tree you are looking at. The extension's path and a
    content hash can.
    """
    import hashlib

    import mixedlm_rs
    from mixedlm_rs import _mixedlm_rs

    path = getattr(_mixedlm_rs, "__file__", None)
    digest, size = None, None
    if path and pathlib.Path(path).is_file():
        raw = pathlib.Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        size = len(raw)
    package = pathlib.Path(mixedlm_rs.__file__).resolve().parent
    return {
        "extension_name": _redact(path),
        "extension_sha256": digest,
        "extension_bytes": size,
        "package_location": _redact(str(package)),
        "imported_from_checkout": str(ROOT.resolve()) in str(package),
    }


def _redact(path) -> str | None:
    """A path stripped of everything specific to this machine.

    This file is committed and published, so a full path publishes the
    developer's home directory and account name. None of that identifies the
    build -- the sha256 above does -- and `imported_from_checkout` already
    answers the question a reader actually has, which is whether the numbers
    came from an installed artifact or from the source tree.

    What is kept is the last two components, enough to tell a site-packages
    install from a checkout at a glance.
    """
    if not path:
        return None
    parts = pathlib.Path(path).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else str(path)


def environment() -> dict:
    import mixedlm_rs

    versions = {}
    for name in ("numpy", "scipy", "pandas", "patsy", "statsmodels"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT,
            capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        commit, dirty = None, None

    return {
        "commit": commit,
        "working_tree_dirty": dirty,
        "mixedlm_rs": mixedlm_rs.__version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "dependencies": versions,
        **_extension_identity(),
        **_python_sources_digest(),
    }


def _record(warns) -> dict[str, int]:
    counts: dict[str, int] = {}
    for w in warns:
        key = w.category.__name__
        counts[key] = counts.get(key, 0) + 1
    return counts


def differential(cases: int) -> dict:
    """The 120-fixture comparison against statsmodels, on the criterion."""
    import mixedlm_rs as mlm
    import statsmodels.formula.api as smf
    from test_fuzz import random_case

    counts = {"reference did not converge": 0, "we found a better optimum": 0,
              "same optimum": 0, "we found a worse optimum": 0}
    losses, warn_totals, uncertified = [], {}, []

    t0 = time.perf_counter()
    for seed in range(cases):
        df, re_formula, reml = random_case(np.random.default_rng(10_000 + seed))
        kw = {"groups": df["g"], "re_formula": re_formula}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ours = mlm.mixedlm("y ~ x1 + x2", df, **kw).fit(reml=reml)
            theirs = smf.mixedlm("y ~ x1 + x2", df, **kw).fit(reml=reml)
        for k, v in _record(w).items():
            warn_totals[k] = warn_totals.get(k, 0) + v
        if not ours.converged:
            uncertified.append(seed)

        if not theirs.converged:
            counts["reference did not converge"] += 1
            continue
        gap = 2.0 * (float(ours.llf) - float(theirs.llf))
        if gap > 1e-6:
            counts["we found a better optimum"] += 1
        elif gap < -1e-6:
            counts["we found a worse optimum"] += 1
            losses.append({"seed": seed, "deviance_worse_by": -gap,
                           "n": len(df), "groups": int(df["g"].nunique())})
        else:
            counts["same optimum"] += 1

    return {
        "cases": cases,
        "seed_base": 10_000,
        "generator": "tests/test_fuzz.py::random_case",
        "seconds": time.perf_counter() - t0,
        "counts": counts,
        "losses": losses,
        "uncertified_seeds": uncertified,
        "warnings": warn_totals,
    }


def stress(cases: int, seed: int) -> dict:
    """The adversarial sweep, reusing the committed generator."""
    from stress_sweep import SHAPES, classify, make_case

    counts, losses, uncertified, warn_totals = {}, [], [], {}
    t0 = time.perf_counter()
    for i in range(cases):
        shape = SHAPES[i % len(SHAPES)]
        rng = np.random.default_rng(seed * 1_000_003 + i)
        df, re_formula, reml = make_case(rng, shape)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            row = classify(df, re_formula, reml)
        for k, v in _record(w).items():
            warn_totals[k] = warn_totals.get(k, 0) + v

        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
        if row["outcome"] == "we found a worse optimum":
            losses.append({"case": i, "shape": shape,
                           "deviance_worse_by": -row["deviance_gap"],
                           "n": len(df),
                           "groups": int(df["g"].nunique())})
        if not row.get("our_converged", True):
            uncertified.append({"case": i, "shape": shape})

    return {
        "cases": cases,
        "seed": seed,
        "generator": "bench/stress_sweep.py::make_case",
        "seconds": time.perf_counter() - t0,
        "counts": counts,
        "losses": losses,
        "uncertified": uncertified,
        "warnings": warn_totals,
    }


def suites() -> dict:
    """pytest and cargo, so the recorded counts come from a run too.

    The *collected* count is what gets recorded and quoted, not the number
    that passed. How many pass depends on how many skip, and that depends on
    the environment -- statsmodels present or not, the lme4 fixtures present
    or not, mypy installed or not. A documented pass count therefore drifts
    for reasons that have nothing to do with the code, which is exactly the
    kind of unfalsifiable number this file exists to eliminate.
    """
    import re

    out = {}
    p = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q",
                        "--no-header"], cwd=ROOT, capture_output=True,
                       text=True)
    summary = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    tally = {kind: int(n) for n, kind in
             re.findall(r"(\d+) (passed|failed|skipped|xfailed|xpassed|error)",
                        summary)}
    out["pytest"] = {
        "returncode": p.returncode,
        "summary": summary,
        "collected": sum(tally.values()),
        **tally,
    }

    c = subprocess.run(["cargo", "test", "--lib", "-q"], cwd=ROOT,
                       capture_output=True, text=True)
    line = [ln for ln in c.stdout.splitlines() if "test result" in ln]
    csum = line[-1].strip() if line else ""
    match = re.search(r"(\d+) passed", csum)
    out["cargo"] = {"returncode": c.returncode, "summary": csum,
                    "collected": int(match.group(1)) if match else 0}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fewer cases, for a smoke run")
    ap.add_argument("--skip-suites", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    n_diff = 30 if args.quick else 120
    n_stress = 60 if args.quick else 400

    print("environment ...", flush=True)
    env = environment()
    print(f"  commit {env['commit']} (dirty={env['working_tree_dirty']})")

    print(f"differential comparison, {n_diff} cases ...", flush=True)
    diff = differential(n_diff)
    print("  " + ", ".join(f"{k}: {v}" for k, v in diff["counts"].items()))

    print(f"stress sweep, {n_stress} cases ...", flush=True)
    st = stress(n_stress, seed=0)
    print("  " + ", ".join(f"{k}: {v}" for k, v in st["counts"].items()))

    payload = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": env,
        "differential": diff,
        "stress": st,
        "quick": args.quick,
    }
    out_path = pathlib.Path(args.out)

    # Written *before* the suites run, and rewritten after them.
    # tests/test_baseline.py reads this file, so running the suites first has
    # them check the previous run's numbers and then records their failure as
    # this run's result -- which is what happened the first time.
    out_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    if not args.skip_suites:
        print("test suites ...", flush=True)
        payload["suites"] = suites()
        for k, v in payload["suites"].items():
            print(f"  {k}: {v['summary']}")
        out_path.write_text(json.dumps(payload, indent=1) + "\n",
                            encoding="utf-8")

    print(f"\nwrote {args.out}")
    # Every recorded suite has to have passed. Checking pytest alone let a
    # failing `cargo test` be written into the file and reported as success,
    # which makes the baseline a record of a broken build rather than a gate.
    failed = [name for name, run in payload.get("suites", {}).items()
              if run.get("returncode", 0) != 0]
    if failed:
        print(f"the recorded {' and '.join(failed)} run(s) FAILED; "
              "this baseline is not usable")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

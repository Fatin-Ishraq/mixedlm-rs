# Release candidate: 0.1.0

What was verified, how, and what the release procedure is. Written so someone
who was not here can check every claim without asking, and so the next person to
publish does not have to reconstruct the steps.

Nothing in this file is a summary of intent. Every figure below points at a
recorded artifact or a CI run.

---

## The candidate

| | |
|---|---|
| Version | `0.1.0` (`pyproject.toml` and `Cargo.toml` agree; the release workflow refuses to build if they do not) |
| Status | **Beta.** One grouping factor only — no crossed or nested random effects, no variance components, no GLMMs, no Kenward–Roger or Satterthwaite corrections. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md). |
| Commit | see `bench/baseline.json` → `environment.commit`, which is recorded on a clean tree |
| Licence | MIT. The compiled extension statically links BSD-2-Clause and MIT crates; their notice texts ship inside every wheel. |

## Intended platforms

Five wheels, `abi3-py310`, so one per platform covers Python 3.10 through 3.14.

| wheel | built on | executed on |
|---|---|---|
| `linux` / `x86_64` | `ubuntu-latest` (manylinux container) | `ubuntu-latest` |
| `linux` / `aarch64` | `ubuntu-latest` (cross, QEMU) | `ubuntu-24.04-arm` |
| `macos` / `x86_64` | `macos-latest` | `macos-15-intel` |
| `macos` / `aarch64` | `macos-latest` | `macos-latest` |
| `windows` / `x64` | `windows-latest` | `windows-latest` |

Every published wheel is installed into a clean environment on a runner of its
own architecture and used to fit a model there. The `platform-coverage` job
fails the release if a built wheel has no matching runtime test, so this table
cannot silently drift.

Both unusual runner labels are exercised by CI's `architectures` job, which
asserts `platform.machine()` before building — `macos-13` was pinned here until
it was found to have been retired in December 2025, and nothing caught it
because ordinary CI only ever used `macos-latest`.

**Not covered:** 32-bit, musl, and any architecture outside those five.

## Numerical evidence

Recorded in **`bench/baseline.json`**, written by `python bench/baseline.py` on
a clean tree. It carries the commit, the extension's SHA-256, a hash of the
installed Python sources, the dependency versions, the seeds, the individual
per-case losses, and the exit status of both suites.

| sweep | cases | outcome |
|---|---:|---|
| differential vs statsmodels | 120 | 26 reference non-convergences, 42 better optima, 52 ties, **0 worse** |
| adversarial stress sweep | 400 | 99 non-convergences, 131 better, 167 ties, 3 worse |

Those counts did not move across the release fixes. The three worse cases in
the stress sweep are deviance gaps of ~6.3e-05, ~4.4e-05 and ~2.7e-06 and are
disclosed in `docs/CORRECTNESS.md`; they are not a newly found defect.

The primary oracle is **lme4's published fits**, not statsmodels: the
log-likelihood agrees with lme4 to six decimals on `sleepstudy`, `Dyestuff` and
`Dyestuff2`.

## Performance evidence

Recorded in **`bench/performance.json`**, written by
`python bench/performance.py`. It carries the commit, extension hash, machine,
dependency versions, repetition count, thread settings, and every individual
repetition — not only the reported minimum.

Agreement is checked **before** any timing is reported, against the thresholds
in `bench/tolerances.py`. The worst fixed-effect gap in the recorded run is
3.8e-05 standard errors against a 0.05 limit.

> **The published figures were corrected for this release.** Earlier revisions
> claimed 218x / 870x / 1454x / 1836x for the four middle rows. Those do not
> reproduce. Run on 2026-09-10, the original `bench/scaling.py` gives
> 161x / 598x / 1078x / 1370x and the recorded run gives 183x / 661x / 1092x /
> 1404x — the two agree within 12% of each other, and both disagree with what
> was published. What changed between the recordings is not established; the
> old numbers are superseded, not explained.

The **lme4** and **pymer4** comparisons are **historical**: they need R, rpy2
and a writable R library, which the recording machine does not have. They sit
in a separate `historical` block in the JSON, tagged with the environment that
produced them, and both documents label them as not re-measured on this build.
To refresh them, run `bench/vs_lme4.py` on a machine with the R toolchain.

The README's charts are drawn from those two JSON files by
`python bench/make_charts.py`, so a chart cannot drift from the run it depicts.

## Continuous integration

The green run for the final commit is linked from the CI badge in the README.
It covers, for every push:

- the full suite on Linux, macOS and Windows against Python 3.10–3.14;
- the **exact** declared dependency floors (`numpy==1.23.0`, `scipy==1.9.0`,
  `pandas==1.5.0`, `patsy==0.5.3`) on Python 3.10 — pinned to the exact minimum,
  not the release series;
- the minimum supported Rust version, read from `Cargo.toml`;
- `ruff`, `mypy --strict`, `cargo fmt --check`, `clippy -D warnings`;
- `cargo audit --deny warnings` and `pip-audit --strict` against
  `requirements-runtime.txt`;
- the two architecture probes described above;
- `scripts/verify_review_findings.py` — every finding from four external
  reviews, re-checked against the built artifact;
- artifact verification on all three ordinary platforms, plus a clean-install
  smoke test that runs the documented example with no test extra installed.

## Packaging checks

| check | how |
|---|---|
| Metadata | `python -m twine check --strict dist/*` — run locally, and in the publish job before upload |
| Contents and notices | `scripts/verify_release.py`; every wheel carries `THIRD-PARTY-LICENSES.md` with the BSD-2-Clause conditions and disclaimer |
| Exact-wheel install | `scripts/verify_release_artifacts.py --wheel <path>` — installs by path outside the checkout, then compares pip's recorded `direct_url.json` hash against the file |
| Exact-sdist rebuild | `scripts/verify_release_artifacts.py --sdist <path>` — unpacks, rebuilds a wheel, installs and runs that |
| Documented example | the README block is extracted and executed against each installed artifact; it is self-contained and opens no data file |
| No GPL in the sdist | `sdist-contents` CI job — the lme4 fixtures under `data/` are GPL-2 and this project is MIT |

## What gets published

The publish job downloads **only** `release-*` artifacts by pattern, then
`scripts/check_release_set.py` compares the upload directory against SHA-256
records written by the jobs that actually installed and ran each file. It
refuses an unexpected file, a missing one, a duplicate filename, or a file whose
hash changed after verification, and requires exactly six distributions.

This exists because the release workflow calls the full CI workflow, which
uploads wheels of its own with **filenames identical to the release wheels**. An
unfiltered download plus `merge-multiple` would let a CI build silently replace
the verified one.

## Release procedure

Publication is gated on `tests` (the full CI suite for this exact commit),
`verify-wheels`, `verify-sdist`, `notices` and `platform-coverage`. A tag on an
untested or failing commit stops at the first of those.

**Before anything is published, one thing must be confirmed by hand.** PyPI's
Trusted Publisher registration is private and cannot be read from here. Check at
<https://pypi.org/manage/account/publishing/> that it reads exactly:

| field | value |
|---|---|
| Owner | `Fatin-Ishraq` |
| Repository name | `mixedlm-rs` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The `pypi` and `testpypi` GitHub environments exist. Neither has protection
rules; adding yourself as a required reviewer on `pypi` makes the run pause for
approval before upload.

Then:

```bash
# 1. Rehearse without publishing. Builds everything, runs every gate, uploads
#    nothing. This is the first execution of the release workflow.
gh workflow run release.yml -f publish_to=nowhere

# 2. Publish to TestPyPI. Exercises the upload path without spending 0.1.0,
#    which can never be reused on real PyPI even after a delete.
gh workflow run release.yml -f publish_to=testpypi

# 3. Release.
git tag v0.1.0 && git push origin v0.1.0
```

The tag may be `v0.1` or `v0.1.0`: the version check compares PEP 440 versions,
and those are the same release. Cargo requires all three components, so the
declared version stays `0.1.0`.

## The completion check

**A successful upload does not establish that the package works.** The release
is complete when, from a machine that has never built this project:

```bash
pip install mixedlm-rs
```

installs, and the README example runs against it and prints a summary. Until
that has been done from public PyPI, the release is uploaded, not verified.

## Known limitations carried into this release

- One grouping factor. Crossed and nested random effects are the largest gap.
- No Kenward–Roger or Satterthwaite corrections; `pvalues` are normal-based.
- `method="rust"` is experimental, warns, and on two known seeds returns a worse
  stationary point with `converged=False` and diagnostics attached.
- Pickling reproduces formula prediction for patsy's transforms, module-level
  functions and module aliases. A formula closing over a lambda or a local
  function cannot be restored; `predict` then names the missing transform.
- `groups` Series alignment is by **label** where statsmodels aligns
  positionally. Documented in `docs/COMPATIBILITY.md`; pass `np.asarray(groups)`
  for the old behaviour.

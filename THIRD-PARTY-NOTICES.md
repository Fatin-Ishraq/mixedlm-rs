# Third-party notices

`mixedlm-rs` is MIT licensed (see [LICENSE](LICENSE)). This file records what
else it builds on, what ships in the distributed artifacts, and what does not.

## What is in the wheel

The wheel contains the Python package, the compiled extension, `py.typed` and
the type stub. The compiled extension statically links the Rust crates below.

Because the extension is statically linked, the wheel *is* a binary
redistribution of those crates, and MIT, BSD-2-Clause and Apache-2.0 each
require the copyright notice, conditions and disclaimer to accompany it. Naming
the licences here and in the SBOM does not satisfy that. The full upstream text
of every linked crate is therefore reproduced in
[THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md), which is generated from the
crate sources by `scripts/collect_notices.py` and ships inside the wheel and the
sdist at `mixedlm_rs-<version>.dist-info/licenses/`. An earlier release candidate
shipped only our own LICENSE, which was a licence violation rather than an
oversight in documentation.

| crate | licence | role |
|---|---|---|
| [`pyo3`](https://github.com/PyO3/pyo3) | MIT OR Apache-2.0 | Python bindings |
| [`rust-numpy`](https://github.com/PyO3/rust-numpy) (`numpy`) | BSD-2-Clause | NumPy array interop |
| [`rayon`](https://github.com/rayon-rs/rayon) | MIT OR Apache-2.0 | parallel per-group factorisation |

`rust-numpy` additionally pulls in `ndarray`, `matrixmultiply`, `num-complex`,
`num-integer`, `num-traits` and `rawpointer`; `pyo3` pulls in `pyo3-ffi`,
`pyo3-macros`, `pyo3-macros-backend`, `proc-macro2`, `quote`, `syn`,
`unicode-ident`, `heck`, `indoc`, `unindent`, `target-lexicon`, `memoffset`,
`once_cell`, `portable-atomic` and `libc`; `rayon` pulls in `rayon-core`,
`crossbeam-deque`, `crossbeam-epoch`, `crossbeam-utils` and `either`.

Every one is MIT, Apache-2.0, BSD-2-Clause, or dual-licensed under compatible
terms. Verified with `cargo tree --format "{p} {l}" -e normal`, which finds no
GPL, LGPL or AGPL crate anywhere in the graph. `Cargo.lock` pins the exact
set.

No GPL or LGPL code is linked into the extension.

## Runtime dependencies

Installed by pip alongside this package, not vendored into it:

| package | licence |
|---|---|
| [NumPy](https://numpy.org/) | BSD-3-Clause |
| [SciPy](https://scipy.org/) | BSD-3-Clause |
| [pandas](https://pandas.pydata.org/) | BSD-3-Clause |
| [patsy](https://github.com/pydata/patsy) | BSD-2-Clause |

## Attribution: the algorithm

The formulation implemented here is Douglas Bates and colleagues', published as:

> Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). *Fitting Linear
> Mixed-Effects Models Using lme4.* Journal of Statistical Software, 67(1),
> 1–48. <https://doi.org/10.18637/jss.v067.i01>

The profiled REML/ML criterion, the relative covariance factor `Lambda`, the
spherical random effects `u` with `b = Lambda u`, and the penalised least
squares system are all from that paper. This is an independent implementation
of a published method, not a translation of `lme4`'s source: no code from
`lme4` (GPL-2+) was read or reused, and none is linked.

The profiled ML gradient in that paper (equations 46–48) is the basis for the
REML gradient in `src/lmm.rs`; the specialisation to the single-grouping-factor
block structure is this package's.

## Attribution: the API

The public API mirrors
[`statsmodels`](https://www.statsmodels.org/) (BSD-3-Clause), specifically
`statsmodels.regression.mixed_linear_model`, so that changing an import is a
viable migration for a large share of code. Class names, argument names,
parameter packing and the `summary()` layout follow it deliberately.
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) states exactly where the two
agree and where they do not. No statsmodels code is vendored.

## Test fixtures: NOT distributed

The CSVs under `data/` are `lme4`'s example datasets — `sleepstudy`,
`Dyestuff`, `Dyestuff2`, `Penicillin`, `Pastes`, `cbpp`, `InstEval` — retrieved
via the [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) mirror.

**They are GPL-2, and this project is MIT.** They are therefore excluded from
both distributed artifacts:

- the **wheel** has never contained them;
- the **source archive** excludes them via `exclude` in `Cargo.toml`, which is
  the lever that governs an sdist built by maturin. An earlier release
  candidate did ship them there while the README said otherwise; the
  `sdist-contents` CI job now fails the build if a `data/` entry reappears.

They exist in the git repository so the oracle tests can run against lme4's
published numbers. `tests/test_lme4_oracles.py` skips cleanly when the
directory is absent, so the suite still runs from a distribution.

## Benchmark comparisons

`bench/` compares against `statsmodels` (BSD-3-Clause), and optionally against
`lme4` (GPL-2+) through R and [`pymer4`](https://github.com/ejolly/pymer4)
(MIT) through `rpy2` (GPL-2+). None of those is a dependency of this package;
they are invoked as external programs for measurement only, and only when you
run the benchmarks yourself.

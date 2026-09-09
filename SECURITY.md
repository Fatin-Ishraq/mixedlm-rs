# Security

## Reporting

Open a GitHub issue for anything that is not itself sensitive. For a report you
would rather not file publicly, use GitHub's private vulnerability reporting on
this repository ("Security" → "Report a vulnerability").

There is no separate security release channel: fixes go out as ordinary
releases, and the changelog says what changed.

## What this package is, from a security point of view

A numerical library with a compiled extension. It reads arrays and fits a
model. It opens no sockets, spawns no processes, reads and writes no files
except through `MixedLMResults.save` / `.load`, which you call explicitly, and
executes no code it was not given.

Two things are worth knowing anyway.

**Formula strings are evaluated.** `mixedlm("y ~ x", ...)` hands the formula to
patsy, which evaluates it as a Python expression against your data and the
calling frame. A formula from an untrusted source is therefore arbitrary code
execution, exactly as it is in `statsmodels` and `patsy` themselves. If you
accept formulas from users, either validate them or use the array API
(`MixedLM(endog, exog, groups, exog_re=...)`), which evaluates nothing.

**Unpickling a saved fit executes code.** `MixedLMResults.load` is `pickle`.
Load only files you produced. This is Python's usual pickle caveat and not
specific to this package.

## The compiled core

`src/` is built with `panic = "abort"`, so a panic terminates the process
rather than raising a catchable Python exception. Every argument crossing the
boundary is therefore validated in `src/lib.rs` before it reaches any indexing:
dimensions, `theta` length, finiteness, group codes in range, and the
allocation arithmetic. `tests/test_native_safety.py` drives each of those cases
in a **child process** and asserts the exit code, because an in-process test
cannot observe an abort — it dies with it.

That suite covers a review finding: an empty or oversized `theta`, and a
zero-width design, previously reached unchecked indexing and terminated the
interpreter with `0xC0000409` on Windows. Those are denial of service against
a process that feeds untrusted shapes into the low-level `LmmCore` interface;
no memory-corruption or code-execution path was demonstrated. They now raise
`ValueError`.

There is no `unsafe` block in this crate.

## Dependency advisories

Both ecosystems are audited on every CI run — Rust with `cargo audit`, Python
with `pip-audit` — and the workflow fails on a finding. Auditing only when
someone remembers to look is not a control.

Run them locally with:

```bash
cargo audit
python -m pip_audit --requirement requirements-runtime.txt
```

### Current status

Recorded against commit-time state, PyO3 0.29.2 / rust-numpy 0.29.0.

**Rust: clean.** `cargo audit` reports no advisories across the 31 crates in
`Cargo.lock`.

**Python: clean for this package's dependency closure.** That closure is
`numpy`, `scipy`, `pandas`, `patsy`, and their own transitive requirements
(`packaging`, `python-dateutil`, `six`, `tzdata`). None carries an open
advisory.

A reachability note, because "an affected version is installed" is not the same
claim as "this package is exploitable": running `pip-audit` against a *developer*
environment here reports `cryptography` (PYSEC-2026-3552) and `nltk`
(PYSEC-2026-3740). Neither is in the closure above — both arrive through
unrelated tooling that happens to share the interpreter. They are not
reachable from any code path in this package, and upgrading them changes
nothing about it. This is why the CI job audits the declared dependency set
rather than whatever is installed alongside it.

If a future advisory does land on a real dependency, the same standard applies:
record which entry point reaches the affected code, or state that none does.

## Supported versions

Only the latest release is supported. Before 1.0 there are no backports.

| version | supported |
|---|---|
| 0.1.x | yes, latest patch |
| < 0.1 | no |

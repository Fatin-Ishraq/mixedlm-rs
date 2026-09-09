# Test fixtures

These CSVs are **test fixtures only**, and they are **GPL-2**, while this
project is MIT. They are therefore excluded from *both* distributed artifacts:

- the **wheel** contains nothing but the Python package and the compiled
  extension;
- the **source archive** excludes this directory too, via `exclude` in
  `Cargo.toml` — which is the lever that matters, because maturin builds the
  sdist from Cargo's file list, not from `[tool.maturin]`.

An earlier release candidate did ship them in the sdist while the README said
otherwise. The `sdist-contents` CI job now fails the build if a `data/` entry
reappears in the archive.

They live in the git repository only. `tests/test_lme4_oracles.py` skips
cleanly when the directory is absent, so the suite still runs from a
distribution — it just checks less.

| file | source | licence |
|---|---|---|
| `sleepstudy.csv` | lme4 | GPL-2 |
| `Dyestuff.csv`, `Dyestuff2.csv` | lme4 | GPL-2 |
| `Penicillin.csv`, `Pastes.csv`, `cbpp.csv`, `InstEval.csv` | lme4 | GPL-2 |

Retrieved from the [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/)
mirror of R packages.

They are here because correctness for this package is anchored on **lme4's
published fits**, which requires lme4's data. `sleepstudy`, `Dyestuff` and
`Dyestuff2` have a single grouping factor and are used in the oracle tests;
`Penicillin` (crossed), `Pastes` (nested), `cbpp` (binomial) and `InstEval`
(large crossed) are outside this release's scope and are retained for the
crossed-random-effects work.

To rebuild the directory from scratch:

```bash
for f in sleepstudy Dyestuff Dyestuff2 Penicillin Pastes cbpp InstEval; do
  curl -sL -o "data/$f.csv" \
    "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/$f.csv"
done
```

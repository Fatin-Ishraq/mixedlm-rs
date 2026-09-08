# Test fixtures

These CSVs are **test fixtures only**. They are not bundled in the wheel — the
built distribution contains nothing but the Python package and the compiled
extension (verified: no `.csv` and no `data/` entries in the wheel).

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

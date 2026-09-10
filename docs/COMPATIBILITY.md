# Compatibility with `statsmodels.MixedLM`

The README used to say "change the import, that is the whole migration". That
is true for a lot of code and false for some, and which half you are in matters
more than the slogan. This page is the actual contract.

It was produced by diffing the two results objects attribute by attribute, not
from memory. Where a row says a type differs, that difference was observed.

**Scope first.** This package fits linear mixed models with **one grouping
factor**. If your model has two `(...|...)` terms — crossed subject and item
effects, three-level nesting — it is not supported at all, and no amount of API
compatibility helps. See [LIMITATIONS.md](LIMITATIONS.md).

---

## 1. Compatible: same call, same meaning, same numbers

Verified against `statsmodels` on every fixture in the test suite, and against
`lme4`'s published fits for `sleepstudy`, `Dyestuff` and `Dyestuff2`.

| what | notes |
|---|---|
| `MixedLM(endog, exog, groups, exog_re=...)` | Same positional order. |
| `MixedLM.from_formula(...)`, `mixedlm(...)` | Same signature, including `re_formula`, `subset`, `missing`, `eval_env`. |
| `fit(reml=...)` | REML default, ML on request. |
| `fe_params`, `cov_re`, `cov_re_unscaled`, `scale`, `vcomp` | Same quantities. |
| `params`, `bse`, `bse_fe`, `bse_re` | Same packing, same parameterisation — including the reference's own `sqrt(scale)` quirk in `bse_re`, reproduced deliberately. |
| `tvalues`, `pvalues`, `conf_int()` | Normal-based, as in the reference. |
| `llf`, `aic`, `bic` | `aic`/`bic` are `nan` under REML, as in the reference. |
| `random_effects` | Dict of per-group Series, conditional modes. |
| `random_effects_cov` | Conditional covariance given each group's data. |
| `fittedvalues` (conditional), `resid`, `predict()` (marginal) | Same distinction the reference draws. |
| `summary()` | Same layout and the same displayed quantities. |
| `MixedLMParams` — `from_packed`, `get_packed`, `from_components`, `copy` | Same defaults, including `has_fe=False`. |
| `install()` / `uninstall()` | Aliases the mixed-model entry points only. |
| `nobs`, `k_fe`, `k_re`, `k_re2`, `k_vc`, `method`, `converged`, `df_resid`, `df_modelwc` | |

Two intentional differences inside this section, both because the reference is
wrong rather than merely different:

- **Convergence.** A boundary optimum is reported as `converged=True` with
  `singular=True` beside it, where the reference emits a convergence warning.
  Deleting random-effect terms to silence that warning is a documented source
  of anti-conservative *p*-values.
- **`__version__`** reports this package's version. The statsmodels API level
  being emulated is `__statsmodels_version__`.

---

## 2. Different: same name, different type or different coordinates

Code that reads values works. Code that relies on the *container* does not.

| what | statsmodels | here | why |
|---|---|---|---|
| `fe_params`, `params`, `bse`, `bse_fe`, `bse_re`, `tvalues`, `pvalues`, `fittedvalues`, `resid` | `pandas.Series` | `numpy.ndarray` | Name-based access (`result.fe_params["x"]`) raises. Use `fe_params_labelled`, `params_labelled`, or `param_names`. |
| `cov_re`, `cov_re_unscaled` | `pandas.DataFrame` | `numpy.ndarray` | `cov_re.loc["Group", "Group"]` raises here. `cov_re[0, 0]` raises on *statsmodels* -- `[0, 0]` is column selection on a DataFrame, not element access -- so it is not a portable form either. `np.asarray(cov_re)[0, 0]` works on both. |
| `llf`, `scale` | `numpy.float64` | `float` | Equal numerically; `isinstance(x, np.float64)` is False. |
| `df_resid` | `numpy.int64` | `int` | Same. |
| `summary()` | `Summary` with `.tables`, `.as_html()`, `.as_latex()` | text-only object with `.as_text()` | `str()` and `print()` are identical in shape. |
| `t_test`, `wald_test`, `f_test` | `ContrastResults` | minimal object with `effect`, `sd`, `statistic`, `pvalue`, `df_denom`, `df_num` | Fixed effects only; the variance parameters sit on a bounded space where Wald statistics are not chi-square. |
| `score(params)`, `hessian(params)` | packed covariance parameters | **internal `theta` coordinates** — the entries of the relative covariance factor | Not comparable term by term. `hessian` is `(k_re2, k_re2)`, not the reference's full square. |
| `cov_params()` | full covariance of the packed vector | fixed-effect block is the exact conditional GLS covariance `scale * (X'V^-1 X)^-1` **at the estimated covariance parameters** -- it does not account for their estimation, exactly as statsmodels' does not; variance-component rows carry delta-method variances on the diagonal and `NaN` off-diagonal | No cross-block terms; see below. |
| `loglike(params)` | accepts its packing | also accepts a covariance-only vector and `MixedLMParams` | `profile_fe=False` raises: this criterion has no un-profiled surface. |
| `group_list` | method splitting an array | same | It was a property returning labels here until 0.1.0; that was a bug. |

### Covariance and inference limitations

- The fixed-effect block is `sigma^2 (X' V(theta_hat)^-1 X)^-1`, the GLS
  covariance **conditional on the fitted variance parameters**. That is what
  both `lme4` and `statsmodels` report, but it does not propagate uncertainty
  in `theta`, so it is not in general the fixed-effect block of the inverse
  full observed information, and it is mildly anti-conservative in small
  samples.
- **Kenward–Roger and Satterthwaite corrections are not implemented.** If your
  design is small and the *p*-values matter, that is a real gap; `lmerTest`
  does this in R and `pymer4` exposes it.
- Joint inference across fixed effects and variance parameters is unavailable.
- `bse_re` is `NaN` at a singular fit, where the profiled Hessian need not be
  positive definite and inverting it would fabricate standard errors.

---

## 3. Unsupported: raises rather than pretending

Everything here raises. Nothing is silently ignored.

| what | behaviour |
|---|---|
| `vc_formula`, `exog_vc`, `VCSpec` passed to a model | `NotImplementedError`. Crossed and nested random effects are the single biggest gap. |
| `fe_pen`, `cov_pen`, `free` | `NotImplementedError` — they change the estimator. |
| `MixedLM.fit_regularized` | `NotImplementedError`. |
| `profile_re`, `bootstrap`, `get_distribution` | `NotImplementedError`. |
| `MixedLM.get_scale` | `NotImplementedError`; use `results.scale`. |
| `loglike(..., profile_fe=False)` | `NotImplementedError`. |
| GLMMs | Out of scope; this is `MixedLM`, not `MixedGLM`. |

### Public methods and attributes that simply do not exist

Accessing these raises `AttributeError`. Listed rather than summarised, because
"mostly complete" is not a contract.

**On results:** `bsejac`, `bsejhj`, `covjac`, `covjhj`, `hessv`, `hist`,
`score_obsv`, `normalized_cov_params`, `k_constant`, `use_sqrt`, `freepat`,
`cov_pen`, `remove_data`, `get_nlfun`, `t_test_pairwise`, `wald_test_terms`,
`initialize`.

**On the model:** `get_fe_params`, `score_full`, `score_sqrt`, `k_params`,
`k_constant`, `n_totobs`, `fe_pen`, `re_pen`, `row_indices`, and the
per-group split views `endog_li`, `exog_li`, `exog_re_li`, `exog_re2_li`.

### Accepted, ignored, and warned about

`niter_sa`, `do_cg`, `full_output`, `use_sparse`, and statsmodels' optimiser
names (`"bfgs"`, `"lbfgs"`, `"cg"`, …). A non-default value warns; an
unrecognised optimiser name raises. Unknown keyword arguments to `MixedLM(...)`
and `fit(...)` raise `TypeError` **before** the optimiser runs.

---

### How `groups` and `subset` are aligned

This is stated explicitly because getting it wrong is silent. A group vector
that does not correspond to the rows it is fitted against produces a plausible
number, not an error.

`subset` selects **by index label**, or by position if you pass a boolean mask
of `len(data)`. `groups` is resolved against the *unsubset* frame and subset
alongside it, so you never have to subset it yourself and cannot half-subset it
by accident.

| `groups` is | aligned by | notes |
|---|---|---|
| a column name | -- | read from `data`, then subset |
| a Series indexed like `data` | position and label agree | |
| a Series with any other meaningful index | **index label** | may be pre-subset, reordered, or carry the full index; every selected row must have a label |
| a Series with a default `RangeIndex` | **position** | a bare `pd.Series(array)` says nothing beyond row order; length must match `data` or the selection |
| an array | **position** | length must match `data` or the selection |

Anything else raises: duplicate labels in `groups`, labels absent from
`data.index`, a non-unique `data.index` with a labelled Series, or a length
matching neither the frame nor the selection. There is deliberately no
best-effort fallback -- a `groups` argument that cannot be resolved
unambiguously is a question, not a default.

statsmodels aligns a `groups` Series positionally in all cases. If you relied
on that with a reordered, meaningfully-indexed Series, you were fitting a
different model than you thought; pass `np.asarray(groups)` to keep the old
behaviour explicitly.

## 4. Additions

Not in `statsmodels`, so nothing depends on them, but they are the reason some
of the above is safe to rely on.

| what | why |
|---|---|
| `results.singular` | Boundary fit, reported separately from convergence, as `lme4` does with `isSingular`. |
| `results.diagnostics` | Which optimiser ran, the projected gradient it stopped at, the tolerance required, evaluations, iterations, starts, message. |
| `results.bse_cov_re` | Standard errors on the same scale as `cov_re`, which `bse_re` is not. |
| `results.param_names`, `params_labelled`, `fe_params_labelled` | Name-based access, replacing what the Series return types would have given. |
| `results.save(path, with_data=False)` | Smaller file, at the cost of formula prediction after loading. |
| `method="rust"` | **Experimental.** See LIMITATIONS.md. |

---

## Deciding whether to switch

Change the import and run your tests. The checklist below is a *filter*, not a
warranty: it puts the most common blockers first, and the sections above remain
the contract. Three checks cannot certify a migration, and an earlier version of
this page promised exactly that.

**Stops the migration outright**

1. **More than one grouping factor**, crossed or nested random effects, or
   variance components (`vc_formula` / `exog_vc`). Not implemented; they raise.
2. **Anything other than a linear mixed model** — GLMM families, and
   `fit_regularized` for L1-penalised fixed effects. Absent.

**Needs a code change before it will run**

3. **Indexing results by name** — `result.fe_params["x"]`,
   `result.cov_re.loc[...]`. Those raise; use the `_labelled` views or
   positional indexing.
4. **`summary().tables`, `as_html()`, `wald_test_terms`, `t_test_pairwise`,
   or the `bsejac` family.** Absent.
5. **`free`** (fixing individual covariance parameters), **`fe_pen` / `cov_pen`**
   (penalties), and **`profile_re`** (profile-likelihood intervals for a
   variance). Not implemented; they raise rather than being ignored.
6. **Arguments accepted and warned about rather than honoured** — `niter_sa`,
   `do_cg`, `full_output`, `use_sparse`, and statsmodels' optimiser names. Your
   code keeps running; the option does not do what its name says.

**Runs, but returns something different**

7. **Return types.** `cov_re`, `cov_re_unscaled`, `random_effects` and friends
   are arrays and dicts where statsmodels returns DataFrames and Series. Code
   that only computes with them is fine; code that calls DataFrame methods on
   them is not. See section 2.
8. **`groups` alignment.** A `groups` Series with a meaningful index is aligned
   to `data` **by label** here, where statsmodels aligns positionally. If you
   pass a reordered or pre-subset Series, the two libraries fit different
   models — and statsmodels' answer is the wrong one. Pass
   `np.asarray(groups)` to keep positional behaviour explicitly. See "How
   `groups` and `subset` are aligned".
9. **Persistence.** Pickling reproduces formula prediction for patsy's own
   transforms, module-level functions and module aliases, but a formula that
   closes over a lambda or a local function cannot be restored. See
   docs/LIMITATIONS.md.
10. **`score` and `hessian` coordinates**, and `pvalues` always normal-based.
    Numerically incomparable term-by-term with statsmodels'.

If none of these touch your code, the import swap is the whole migration. If
any do, this page says what changes — and the tables above, not this list, are
the complete statement.

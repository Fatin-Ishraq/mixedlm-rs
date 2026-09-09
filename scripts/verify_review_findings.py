"""Re-check every finding from the two external reviews.

39 findings from the first review and 8 from the re-audit. Each check
reproduces the reviewer's own case and asserts the fixed behaviour: the
numerical ones by fitting, the safety ones by driving the compiled core in a
child process (`panic = "abort"` means an unfixed case kills the interpreter
rather than raising), and the documentation ones by reading what the files
actually say. Nothing here trusts a claim made elsewhere in the repository.

    python scripts/verify_review_findings.py

Exit code 0 means all 47 are still fixed. The review documents themselves live
outside the repository -- they were an external deliverable -- so this file is
the durable record of what they said and what closing each one meant.

Three of these checks were wrong on first writing, in the direction that
matters least but is still worth recording: they searched for wording the
documents did not use (bold markers around "not a measurement of BOBYQA", a
different phrasing of the #9097 caveat) and one flagged an "or True" that
appears only inside a comment explaining the assertion that was removed. The
repository was right and the checker was wrong in all three.
"""
from __future__ import annotations

import json
import pathlib
import pickle
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
import tomllib

warnings.filterwarnings("ignore")
ROOT = pathlib.Path.cwd()
while not (ROOT / "pyproject.toml").exists():
    if ROOT == ROOT.parent:
        raise SystemExit("run from inside the repository")
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT / "tests"))
import mixedlm_rs as mlm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from mixedlm_rs import ExperimentalWarning  # noqa: E402

RESULTS: list[tuple[str, str, bool, str]] = []


def check(group: str, label: str, fn):
    try:
        ok, detail = fn()
    except Exception as exc:  # a check that errors is a check that failed
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    RESULTS.append((group, label, bool(ok), str(detail)[:88]))


# ---------------------------------------------------------------- fixtures
def frame(n=200, m=20, seed=7):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(m), n // m)
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.6, size=m)[g] + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x": x, "g": g})


DF = frame()
R = mlm.mixedlm("y ~ x", DF, groups=DF["g"]).fit()
SM = smf.mixedlm("y ~ x", DF, groups=DF["g"]).fit()


def child(expr):
    pre = textwrap.dedent("""
        import numpy as np
        from mixedlm_rs._mixedlm_rs import LmmCore
        n, m = 40, 10
        y = np.random.default_rng(0).normal(size=n)
        X = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
        Z = np.ones((n, 1)); codes = np.repeat(np.arange(m), 4).astype(np.int64)
        core = LmmCore(y, X, Z, codes, m)
    """)
    code = pre + f"\ntry:\n    {expr}\n    print('NOERR')\nexcept ValueError:\n    print('VE')"
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


# ===================================================== REVIEW.md, findings 1-39
def f01():
    cases = ["core.deviance(np.array([]))", "core.deviance_grad(np.array([]))",
             "core.solution(np.array([]))", "core.deviance(np.array([1.,2.,3.]))",
             "core.deviance(np.array([np.nan]))",
             "LmmCore(y, X, np.zeros((n,0)), codes, m)",
             "LmmCore(y, np.zeros((n,0)), Z, codes, m)"]
    bad = [c for c in cases if child(c) != (0, "VE")]
    return not bad, f"{len(cases)-len(bad)}/{len(cases)} raise ValueError, none abort"


def f02():
    llfs = []
    for shift in (0.0, 1e4, 1e6, 1e8):
        d = DF.assign(y=DF["y"] + shift)
        llfs.append(mlm.mixedlm("y ~ x", d, groups=d["g"]).fit().llf)
    spread = max(llfs) - min(llfs)
    return spread < 1e-6, f"llf spread {spread:.2e} across shifts to 1e8"


def f03():
    r = mlm.mixedlm("y ~ x", DF, groups=DF["g"]).fit(ftol=0.1, gtol=0.1)
    d = r.diagnostics
    agree = d["converged"] == (d["max_abs_projected_gradient"] <= d["gradient_tolerance"])
    return agree, f"flag={d['converged']} == (grad<=tol) at loose ftol"


def f04():
    ths, convs = [], []
    for cy in (1.0, 1e3, 1e6):
        for cx in (1.0, 1e4):
            r = mlm.mixedlm("y ~ x", DF.assign(y=DF["y"] * cy, x=DF["x"] * cx),
                            groups=DF["g"]).fit()
            ths.append(float(np.asarray(r._res["theta"])[0]))
            convs.append(r.converged)
    return all(convs) and (max(ths) - min(ths) < 1e-7), \
        f"theta spread {max(ths)-min(ths):.2e} over 6 unit systems"


def f05():
    from test_fuzz import random_case
    bad = []
    for seed in (32, 54, 3, 17, 42, 6, 10):
        df, ref, reml = random_case(np.random.default_rng(10000 + seed))
        kw = {"groups": df["g"], "re_formula": ref}
        a = mlm.mixedlm("y ~ x1 + x2", df, **kw).fit(reml=reml)
        b = mlm.mixedlm("y ~ x1 + x2", df, **kw).fit(reml=reml, method="rust")
        if 2 * (a.llf - b.llf) > 1e-6 and b.converged:
            bad.append(seed)
    return not bad, f"no seed returns a worse optimum marked converged ({bad or 'none'})"


def f06():
    from mixedlm_rs._fit import _starts

    class Core:
        def default_theta(self): return [1.0, 0.0, 1.0]
        def lower_bounds(self): return [0.0, -np.inf, 0.0]
    offs = [s[1] for s in _starts(Core(), None, 5)[1:]]
    return any(o < 0 for o in offs) and any(o > 0 for o in offs), \
        f"off-diagonal starts reach both signs: {np.round(offs, 3)}"


def f07():
    tau2, s2 = float(R.cov_re[0, 0]), float(R.scale)
    sizes = DF["g"].value_counts().sort_index().to_numpy()
    worst = max(
        abs(float(np.asarray(R.random_effects_cov[lab])[0, 0]) - 1 / (1 / tau2 + nn / s2))
        for lab, nn in zip(sorted(DF["g"].unique()), sizes, strict=True))
    ours = float(np.asarray(R.random_effects_cov[0])[0, 0])
    theirs = float(np.asarray(SM.random_effects_cov[0])[0, 0])
    return worst < 1e-9 and abs(ours - theirs) < 1e-6, \
        f"matches closed form to {worst:.1e}; sm {theirs:.6f} vs {ours:.6f}"


def f08():
    k = list(R.random_effects_cov)
    return R.random_effects_cov[k[0]] is not R.random_effects_cov[k[1]], \
        "each group gets its own frame"


def f09():
    line = next(x for x in str(R.summary()).splitlines() if x.startswith("Group Var"))
    shown = float(line.split()[2])
    return abs(shown - R.cov_re[0, 0]) < 5e-3, \
        f"summary shows {shown:.3f} = cov_re {R.cov_re[0,0]:.3f}, not the ratio"


def f10():
    ours, theirs = float(R.bse_re[0]), float(np.asarray(SM.bse_re)[0])
    packed = np.asarray(R.bse)[R.k_fe]
    return abs(ours - theirs) < 0.05 * theirs and abs(ours - np.sqrt(R.scale) * packed) < 1e-9, \
        f"bse_re {ours:.5f} vs statsmodels {theirs:.5f}"


def f11():
    cp = R.cov_params()
    p = R.k_fe
    return np.all(np.isnan(cp[p:, :p])) and "conditional" in mlm.MixedLMResults.cov_params.__doc__, \
        "no cross-block terms; docstring says conditional on theta_hat"


def f12():
    core = mlm.mixedlm("y ~ x", pd.DataFrame(
        {"y": np.random.default_rng(0).normal(size=40),
         "x": np.random.default_rng(1).normal(size=40),
         "g": np.arange(40)}), groups=np.arange(40))._core()
    vals = [core.deviance([t], True) for t in (0.0, 1.0, 10.0, 100.0)]
    flat = (max(vals) - min(vals)) / max(1.0, abs(vals[0])) < 1e-9
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d1 = frame()
        mlm.mixedlm("y ~ x", d1.assign(g=0.0), groups=np.zeros(len(d1))).fit()
    single = any("not identifiable" in str(x.message) for x in w)
    return flat and single, f"criterion flat (not divergent); single group caught: {single}"


def f13():
    b = json.loads((ROOT / "bench" / "baseline.json").read_text())
    losses = b["stress"]["losses"]
    txt = (ROOT / "docs" / "CORRECTNESS.md").read_text(encoding="utf-8")
    listed = all(f"{lo['deviance_worse_by']:.3e}"[:3] in txt or
                 f"{lo['deviance_worse_by']:.4e}"[:4] in txt for lo in losses)
    return (ROOT / "bench" / "stress_sweep.py").exists() and listed, \
        f"sweep committed; {len(losses)} losses each recorded with its gap"


def f14():
    d2 = pd.read_csv(ROOT / "data" / "Dyestuff2.csv")
    s = mlm.mixedlm("Yield ~ 1", d2, groups=d2["Batch"]).fit()
    return s.singular and s.converged and np.all(np.isnan(np.asarray(s.bse_re))), \
        "singular=True, converged=True, bse_re=nan"


def f15():
    d = DF.assign(xd=DF["x"])
    try:
        mlm.mixedlm("y ~ x + xd", d, groups=d["g"]).fit()
        return False, "duplicate column accepted"
    except ValueError as e:
        return "rank deficient" in str(e), "names the design, not theta"


def f16():
    d = DF.assign(y2=DF["y"] * 2)
    try:
        mlm.mixedlm("y + y2 ~ x", d, groups=d["g"]).fit()
        return False, "multi-column response accepted"
    except ValueError:
        return True, "refused, as statsmodels does"


def f17():
    d = DF.copy()
    d.loc[d.index[:5], "g"] = np.nan
    r = mlm.mixedlm("y ~ x", d, groups=d["g"], missing="drop").fit()
    ok1 = r.nobs == 195 and not any(pd.isna(x) for x in r.model.group_labels)
    d2 = DF.copy()
    d2.loc[d2.index[:7], "x"] = np.nan
    r2 = mlm.mixedlm("y ~ 1", d2, groups=d2["g"], re_formula="~x", missing="drop").fit()
    return ok1 and r2.nobs == 193, "joint drop; no NaN level; re-only predictor ok"


def f18():
    d = pd.concat([DF, DF]).set_index(np.repeat(np.arange(len(DF)), 2)[:2 * len(DF)])
    r = mlm.mixedlm("y ~ x", d, groups=d["g"]).fit()
    mask = np.arange(len(DF)) < 150
    r2 = mlm.mixedlm("y ~ x", DF, groups=DF["g"].to_numpy(), subset=mask).fit()
    return r.nobs == len(d) and r2.nobs == 150, \
        f"duplicate index {r.nobs}=={len(d)}; subset+array groups {r2.nobs}"


def f19():
    p = mlm.MixedLMParams.from_components(fe_params=np.array([1., .5]),
                                          cov_re=np.array([[0.4]]))
    mlm.mixedlm("y ~ x", DF, groups=DF["g"]).fit(start_params=p)
    return True, "MixedLMParams accepted by fit"


def f20():
    try:
        mlm.mixedlm("y ~ x", DF, groups=DF["g"]).fit(start_params=np.zeros(7))
        return False, "mis-sized start silently defaulted"
    except ValueError:
        return True, "mis-sized start refused"


def f21():
    m = mlm.mixedlm("y ~ x", DF, groups=DF["g"])
    v = m.loglike(np.array([1.0]))
    ml = mlm.mixedlm("y ~ x", DF, groups=DF["g"])
    rml = ml.fit(reml=False)
    follows = abs(ml.loglike(rml.params_object) - rml.llf) < 1e-6
    try:
        m.loglike(np.array([1.0]), profile_fe=False)
        refused = False
    except NotImplementedError:
        refused = True
    return np.isfinite(v) and follows and refused, \
        "covariance-only accepted; follows model criterion; profile_fe=False raises"


def f22():
    pr = R.predict(pd.DataFrame({"x": [1.0, 2.0]}))
    return len(pr) == 2, f"formula predict on new data -> {np.round(pr, 4)}"


def f23():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mlm.mixedlm("y ~ x", DF, groups=DF["g"]).fit(do_cg=False, full_output=True)
    warned = len(w) >= 2
    try:
        mlm.mixedlm("y ~ x", DF, groups=DF["g"]).fit(method="TYPO")
        m1 = False
    except ValueError:
        m1 = True
    try:
        mlm.mixedlm("y ~ x", DF, groups=DF["g"], nonsense=1)
        m2 = False
    except TypeError:
        m2 = True
    return warned and m1 and m2, "no-op args warn; unknown method and kwargs raise"


def f24():
    p = mlm.MixedLMParams.from_components(fe_params=np.array([1., 2.]),
                                          cov_re=np.array([[2.0]]))
    default_no_fe = p.get_packed(use_sqrt=True).size == 1
    sing = np.isfinite(mlm.MixedLMParams.from_components(
        fe_params=np.array([1.]), cov_re=np.zeros((2, 2))).get_packed(True)).all()
    v = mlm.VCSpec(["a"], [["a0"]], [np.ones((3, 1))])
    return default_no_fe and sing and v.colnames == [["a0"]], \
        "has_fe=False default; singular packs; VCSpec order"


def f25():
    have = all(hasattr(R, a) for a in
               ("t_test", "wald_test", "f_test", "df_resid", "params_object",
                "param_names", "params_labelled"))
    gl = len(R.model.group_list(R.resid)) == 20
    cp = np.isfinite(R.cov_params(r_matrix=np.eye(2, R.params.size))).all()
    return have and gl and cp, "tests, df_resid, params_object, labels, group_list, cov_params args"


def f26():
    back = pickle.loads(pickle.dumps(R))
    return abs(back.llf - R.llf) < 1e-12, "results pickle round-trip"


def f27():
    m = mlm.mixedlm("y ~ x", DF, groups=DF["g"])
    n = 0
    for name in ("fit_regularized", "get_distribution"):
        try:
            getattr(m, name)()
        except NotImplementedError:
            n += 1
    return n == 2, "fit_regularized and get_distribution refuse"


def f28():
    t = (ROOT / "docs" / "DESIGN.md").read_text(encoding="utf-8")
    return "Both claims are false" in t and "profiled over both the scale" in t, \
        "DESIGN.md corrected against statsmodels' own docstring"


def f29():
    t = (ROOT / "bench" / "stages.py").read_text(encoding="utf-8")
    b = (ROOT / "docs" / "BENCHMARKS.md").read_text(encoding="utf-8")
    return ("core = LmmCore(y, X, Z, codes, m)" in t and "S6" in t
            and "182.5x" in b and "1844x" in b), \
        "core built inside timed region; S6 like-for-like; old 1844x disclosed"


def f30():
    t = (ROOT / "bench" / "time_pymer4.py").read_text(encoding="utf-8")
    return "NOT a measurement of serialisation alone" in t and "lmerTest" in t, \
        "pymer4 overhead no longer called pure bridge"


def f31():
    s = (ROOT / "bench" / "scaling.py").read_text(encoding="utf-8")
    return "raise SystemExit" in s and "MAX_FE_SE" in s, \
        "scaling aborts rather than printing an unchecked timing"


def f32():
    b = (ROOT / "docs" / "BENCHMARKS.md").read_text(encoding="utf-8")
    # Strip bold markers and normalise en dashes: the claim is in the prose,
    # not in its typography, and matching the typography is how three of these
    # checks failed on a repository that was already correct.
    plain = b.replace("**", "").replace("–", "-")
    return ("not a measurement of BOBYQA" in plain and "44-64" in plain
            and "not novel" in plain), \
        "S3 is not BOBYQA; 44-64 vs 11-13; gradient not claimed novel"


def f33():
    t = (ROOT / "tests" / "test_fuzz.py").read_text(encoding="utf-8")
    return "assert_local_optimum" in t and "finite-difference" in t, \
        "non-convergence branch certified independently"


def f34():
    g = (ROOT / "tests" / "test_gradient.py").read_text(encoding="utf-8")
    f = (ROOT / "tests" / "test_fuzz.py").read_text(encoding="utf-8")
    # The only surviving "or True" is a comment recording what was removed;
    # what matters is that no *executable* assertion still carries it.
    live = [ln for ln in f.splitlines()
            if "or True" in ln and not ln.strip().startswith("#")]
    return ("test_gradient_across_shapes" in g
            and "test_gradient_at_exactly_zero" in g and not live), \
        f"full q x p product; zero boundary; {len(live)} live `or True`"


def f35():
    v = (ROOT / "bench" / "vs_lme4.py").read_text(encoding="utf-8")
    t = (ROOT / "bench" / "time_pymer4.py").read_text(encoding="utf-8")
    b = (ROOT / "docs" / "BENCHMARKS.md").read_text(encoding="utf-8")
    return ("pymer4_results.csv" in v and "if rep:" in t and "2.0.6" in b
            and "C:/Users/Fatin" not in v), \
        "report reads stored csv; warm-up excluded; lme4 2.0.6; no machine paths"


def f36():
    r = (ROOT / "README.md").read_text(encoding="utf-8")
    return ("not a measurement made here" in r
            and "not a reproduction of their" in r
            and "should not be read as a measured" in r), \
        "#9097 is their report on their data, not a measurement here"


def f37():
    import glob
    import tarfile
    paths = sorted(glob.glob(str(ROOT / "dist" / "*.tar.gz")))
    if not paths:
        return False, "no sdist built"
    with tarfile.open(paths[-1]) as t:
        names = t.getnames()
    return not any("/data/" in n for n in names), \
        f"sdist has {len(names)} entries, no data/"


def f38():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    needed = ["3.10", "3.11", "3.12", "3.13", "3.14", "msrv", "audit",
              "artifacts", "mypy", "ruff"]
    missing = [n for n in needed if n not in ci]
    return not missing, "CI covers all 5 pythons, msrv, audits, artifacts, lint"


def f39():
    d = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return "patsy>=0.5" in d["project"]["dependencies"], \
        "patsy is a runtime dependency, not an extra"


# ================================================== RECHECK.md, findings 1-8
def r1():
    lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    audit = subprocess.run(["cargo", "audit"], cwd=ROOT, capture_output=True,
                           text=True)
    return ('name = "pyo3"' in lock and "0.29" in lock.split('name = "pyo3"')[1][:80]
            and audit.returncode == 0), "PyO3 0.29.x; cargo audit exits 0"


def r2():
    d = frame().set_index(np.arange(100, 300))
    r = mlm.mixedlm("y ~ x", d, groups=d["g"], subset=np.arange(100, 250)).fit()
    try:
        mlm.mixedlm("y ~ x", d, groups=d["g"], subset=np.arange(150)).fit()
        raised = False
    except ValueError:
        raised = True
    return r.nobs == 150 and raised, \
        f"labels 100..249 select {r.nobs} rows; absent labels raise"


def r3():
    p = ROOT / "docs" / "COMPATIBILITY.md"
    t = p.read_text(encoding="utf-8")
    named = all(n in t for n in ("bsejac", "wald_test_terms", "t_test_pairwise",
                                 "score_full", "get_fe_params", "remove_data"))
    return p.exists() and named, "COMPATIBILITY.md names every absent method"


def r4():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mlm.mixedlm("y ~ x", DF, groups=DF["g"]).fit(method="rust")
    exp = any(issubclass(x.category, ExperimentalWarning) for x in w)
    lim = (ROOT / "docs" / "LIMITATIONS.md").read_text(encoding="utf-8")
    return exp and "experimental" in lim and "returns the bad estimate" in lim, \
        "ExperimentalWarning raised; behaviour documented"


def r5():
    b = json.loads((ROOT / "bench" / "baseline.json").read_text())
    bm = (ROOT / "docs" / "BENCHMARKS.md").read_text(encoding="utf-8")
    lim = (ROOT / "docs" / "LIMITATIONS.md").read_text(encoding="utf-8")
    wins = b["differential"]["counts"]["we found a better optimum"]
    consistent = f"| {wins} |" in bm or f"**{wins}**" in bm
    typing_fixed = "no type annotations" not in lim
    return consistent and typing_fixed and "environment" in bm.lower(), \
        f"BENCHMARKS quotes {wins} wins; typing text corrected; env recorded"


def r6():
    import inspect
    import typing
    broken = []
    for name in mlm.__all__:
        obj = getattr(mlm, name)
        members = [(name, obj)] if inspect.isfunction(obj) else []
        if inspect.isclass(obj):
            members = [(f"{name}.{n}", m.fget if isinstance(m, property) else m)
                       for n, m in vars(obj).items()
                       if isinstance(m, property) or inspect.isfunction(m)]
        for label, fn in members:
            if fn is None:
                continue
            try:
                typing.get_type_hints(fn)
            except Exception:
                broken.append(label)
    return not broken, f"get_type_hints resolves on every public member ({len(broken)} broken)"


def r7():
    import time
    t0 = time.perf_counter()
    try:
        mlm.mixedlm("y ~ x", DF, groups=DF["g"]).fit(nonsense=1)
    except TypeError:
        pass
    quick = time.perf_counter() - t0
    t0 = time.perf_counter()
    mlm.mixedlm("y ~ x", DF, groups=DF["g"]).fit()
    full = time.perf_counter() - t0
    return quick < max(full * 0.5, 0.05), \
        f"rejected in {quick*1000:.1f}ms against a {full*1000:.0f}ms fit"


def r8():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    return ('rust-version = "1.83"' in cargo and "numpy==1.23.0" in ci
            and (ROOT / "scripts" / "verify_release.py").exists()), \
        "MSRV 1.83 declared; exact dependency floors; artifact verification"


GROUPS = [
    ("REVIEW", [(f"F{i:02d}", fn) for i, fn in enumerate(
        [f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11, f12, f13,
         f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26,
         f27, f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38, f39], 1)]),
    ("RECHECK", [(f"R{i}", fn) for i, fn in enumerate(
        [r1, r2, r3, r4, r5, r6, r7, r8], 1)]),
]

for group, items in GROUPS:
    for label, fn in items:
        check(group, label, fn)

width = max(len(d) for *_, d in RESULTS)
print()
for group in ("REVIEW", "RECHECK"):
    rows = [r for r in RESULTS if r[0] == group]
    print(f"--- .review/{group}.md: {sum(1 for r in rows if r[2])}/{len(rows)} ---")
    for _, label, ok, detail in rows:
        print(f"  {label}  {'PASS' if ok else '**FAIL**':9s} {detail}")
    print()

failed = [r for r in RESULTS if not r[2]]
print(f"TOTAL: {len(RESULTS) - len(failed)}/{len(RESULTS)} findings verified fixed")
sys.exit(1 if failed else 0)

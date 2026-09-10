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

import importlib.util
import json
import pathlib
import pickle
import re
import shutil
import subprocess
import sys
import textwrap
import warnings
import zipfile

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
for _needed in ("mixedlm_rs", "statsmodels", "patsy", "yaml"):
    if importlib.util.find_spec(_needed) is None:
        # A bare ModuleNotFoundError here reads as a broken script. It is a
        # missing prerequisite, and the differential findings compare against
        # statsmodels directly, so there is no reduced mode to fall back to.
        raise SystemExit(
            f"cannot verify the findings: {_needed} is not installed. The "
            "checker fits both implementations and compares them, and reads "
            "the workflow files to check the release wiring.")

import mixedlm_rs as mlm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from mixedlm_rs import ExperimentalWarning  # noqa: E402

# Document checks read through this rather than touching the filesystem
# directly, so a negative control can hand one a mutated copy of a file and
# prove the check actually fails when the claim is missing. A check that
# passes against a file with the claim deleted is not checking anything.
_OVERRIDE: dict[str, str] = {}


def read(path) -> str:
    key = str(pathlib.Path(path).resolve())
    if key in _OVERRIDE:
        return _OVERRIDE[key]
    return pathlib.Path(path).read_text(encoding="utf-8")


class without:
    """Temporarily remove a phrase from a file, as seen by :func:`read`."""

    def __init__(self, path, phrase, replacement=""):
        self.path = pathlib.Path(path).resolve()
        self.phrase = phrase
        self.replacement = replacement

    def __enter__(self):
        original = self.path.read_text(encoding="utf-8")
        if self.phrase not in original:
            raise AssertionError(
                f"negative control is stale: {self.path.name} does not "
                f"contain {self.phrase!r}")
        _OVERRIDE[str(self.path)] = original.replace(
            self.phrase, self.replacement)
        return self

    def __exit__(self, *exc):
        _OVERRIDE.pop(str(self.path), None)
        return False


def _module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def require_prerequisites() -> None:
    """Refuse to run without what the checks need, and say what is missing.

    Two checks reach outside the interpreter: one inspects a built sdist, one
    shells out to ``cargo audit``. A CI job that has neither is not running a
    weaker version of this script -- it is running a script that cannot make
    the claim in its own summary line. Failing here names the missing step
    instead of surfacing as "no sdist built" or a bare FileNotFoundError from
    deep inside a check.
    """
    missing = []
    if shutil.which("maturin") is None and not _module("maturin"):
        missing.append(
            "maturin is not available -- run `pip install maturin` "
            "(finding 37 builds an sdist from this tree and inspects it)")
    if shutil.which("cargo") is None:
        missing.append("cargo is not on PATH (finding R1 runs `cargo audit`)")
    elif subprocess.run(["cargo", "audit", "--version"],
                        capture_output=True).returncode != 0:
        missing.append(
            "cargo-audit is not installed -- run "
            "`cargo install cargo-audit --locked` (finding R1 runs it)")
    if missing:
        raise SystemExit(
            "cannot verify the findings; prerequisites missing:\n  - "
            + "\n  - ".join(missing))


EN_DASH = "\u2013"
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
    txt = read(ROOT / "docs" / "CORRECTNESS.md")
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
    t = read(ROOT / "docs" / "DESIGN.md")
    return "Both claims are false" in t and "profiled over both the scale" in t, \
        "DESIGN.md corrected against statsmodels' own docstring"


def f29():
    t = read(ROOT / "bench" / "stages.py")
    b = read(ROOT / "docs" / "BENCHMARKS.md")
    return ("core = LmmCore(y, X, Z, codes, m)" in t and "S6" in t
            and "182.5x" in b and "1844x" in b), \
        "core built inside timed region; S6 like-for-like; old 1844x disclosed"


def f30():
    """The *published* attribution, not only the script's own comment.

    This check used to read `bench/time_pymer4.py` alone and report the
    finding closed, while README.md and docs/BENCHMARKS.md both still called
    the pymer4 gap pure bridge overhead. Correcting a comment in the benchmark
    does not correct the claim a reader sees.
    """
    t = read(ROOT / "bench" / "time_pymer4.py")
    script_ok = ("NOT a measurement of serialisation alone" in t
                 and "lmerTest" in t)
    published = []
    for name in ("README.md", "docs/BENCHMARKS.md"):
        text = read(ROOT / name)
        for line in text.splitlines():
            low = line.lower()
            if ("pure bridge overhead" in low or "pure rpy2 bridge" in low
                    or "the bridge costs" in low):
                # Allowed only where the document withdraws the claim.
                if "was wrong" not in low and "called it" not in low:
                    published.append(f"{name}: {line.strip()[:50]}")
        if "744" in text and not ("lmerTest" in text
                                  and "Satterthwaite" in text):
            published.append(f"{name}: 744 s quoted without its inclusions")
    return script_ok and not published, \
        ("script and both documents attribute the gap to the measured "
         "end-to-end path" if script_ok and not published
         else "; ".join(published) or "benchmark script comment missing")


def f31():
    s = read(ROOT / "bench" / "scaling.py")
    return "raise SystemExit" in s and "MAX_FE_SE" in s, \
        "scaling aborts rather than printing an unchecked timing"


def f32():
    b = read(ROOT / "docs" / "BENCHMARKS.md")
    # Strip bold markers and normalise en dashes: the claim is in the prose,
    # not in its typography, and matching the typography is how three of these
    # checks failed on a repository that was already correct.
    plain = b.replace("**", "").replace("–", "-")
    return ("not a measurement of BOBYQA" in plain and "44-64" in plain
            and "not novel" in plain), \
        "S3 is not BOBYQA; 44-64 vs 11-13; gradient not claimed novel"


def f33():
    t = read(ROOT / "tests" / "test_fuzz.py")
    return "assert_local_optimum" in t and "finite-difference" in t, \
        "non-convergence branch certified independently"


def f34():
    g = read(ROOT / "tests" / "test_gradient.py")
    f = read(ROOT / "tests" / "test_fuzz.py")
    # The only surviving "or True" is a comment recording what was removed;
    # what matters is that no *executable* assertion still carries it.
    live = [ln for ln in f.splitlines()
            if "or True" in ln and not ln.strip().startswith("#")]
    return ("test_gradient_across_shapes" in g
            and "test_gradient_at_exactly_zero" in g and not live), \
        f"full q x p product; zero boundary; {len(live)} live `or True`"


# Any absolute home directory, not one developer's. Hard-coding a
# username makes the check pass on every machine except the one it was
# written on, which is the wrong way round.
_MACHINE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]Users[\\/]|/home/|/Users/)\S+")


def f35():
    v = read(ROOT / "bench" / "vs_lme4.py")
    t = read(ROOT / "bench" / "time_pymer4.py")
    b = read(ROOT / "docs" / "BENCHMARKS.md")
    leaked = [m for f in (v, t, b) for m in _MACHINE_PATH.findall(f)]
    return ("pymer4_results.csv" in v and "if rep:" in t and "2.0.6" in b
            and not leaked), \
        ("report reads stored csv; warm-up excluded; lme4 2.0.6; no "
         "machine paths" if not leaked else f"machine paths: {leaked[:2]}")


def f36():
    r = read(ROOT / "README.md")
    return ("not a measurement made here" in r
            and "not a reproduction of their" in r
            and "should not be read as a measured" in r), \
        "#9097 is their report on their data, not a measurement here"


def f37():
    """The sdist ships no GPL-2 fixtures -- checked against *this* tree.

    Picking the newest `dist/*.tar.gz` and reading it proves nothing about
    the current checkout: a stale archive from an earlier commit satisfies it
    just as well. The archive is therefore built here, into a scratch
    directory, and its contents are compared against the working tree.
    """
    import tarfile
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        built = subprocess.run(
            [sys.executable, "-m", "maturin", "sdist", "--out", tmp],
            cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        paths = sorted(pathlib.Path(tmp).glob("*.tar.gz"))
        if built.returncode != 0 or not paths:
            return False, f"could not build an sdist: {built.stderr[-70:]}"
        with tarfile.open(paths[-1]) as t:
            names = t.getnames()

    leaked = [n for n in names if "/data/" in n or n.endswith("/data")]
    # Source identity: the archive has to contain this tree's own sources.
    version = tomllib.loads(read(ROOT / "pyproject.toml"))["project"]["version"]
    expected = f"mixedlm_rs-{version}/python/mixedlm_rs/mixed_linear_model.py"
    current = any(n.endswith("python/mixedlm_rs/mixed_linear_model.py")
                  for n in names)
    return (not leaked and current), \
        (f"freshly built sdist: {len(names)} entries, no data/"
         if not leaked and current
         else f"leaked {leaked[:3]} / missing {expected}")


def f38():
    ci = read(ROOT / ".github" / "workflows" / "ci.yml")
    needed = ["3.10", "3.11", "3.12", "3.13", "3.14", "msrv", "audit",
              "artifacts", "mypy", "ruff"]
    missing = [n for n in needed if n not in ci]
    return not missing, "CI covers all 5 pythons, msrv, audits, artifacts, lint"


def f39():
    d = tomllib.loads(read(ROOT / "pyproject.toml"))
    return "patsy>=0.5.3" in d["project"]["dependencies"], \
        "patsy is a runtime dependency, not an extra"


# ================================================== RECHECK.md, findings 1-8
def r1():
    lock = read(ROOT / "Cargo.lock")
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
    lim = read(ROOT / "docs" / "LIMITATIONS.md")
    return exp and "experimental" in lim and "returns the bad estimate" in lim, \
        "ExperimentalWarning raised; behaviour documented"


def r5():
    b = json.loads((ROOT / "bench" / "baseline.json").read_text())
    bm = read(ROOT / "docs" / "BENCHMARKS.md")
    lim = read(ROOT / "docs" / "LIMITATIONS.md")
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
    """Bad arguments are rejected before the optimiser runs.

    Asserted by spying on the fitting core rather than by timing. The timing
    version compared a validation against a fit on a shared runner and failed
    a CI job at 35ms against a 20ms allowance, with the comparison fit taking
    7ms -- a scheduling artefact, not a regression.
    """
    module = mlm.mixed_linear_model
    original = module.fit_core
    entered = []

    def spy(*args, **kwargs):
        entered.append(True)
        return original(*args, **kwargs)

    module.fit_core = spy
    try:
        try:
            mlm.mixedlm("y ~ x", DF, groups=DF["g"]).fit(nonsense=1)
            raised = False
        except TypeError:
            raised = True
        rejected_first = raised and not entered
        # Negative control in the same breath: a real fit must trip the spy,
        # or the check above would pass even with a spy that sees nothing.
        mlm.mixedlm("y ~ x", DF, groups=DF["g"]).fit()
        spy_works = bool(entered)
    finally:
        module.fit_core = original
    return rejected_first and spy_works, (
        "rejected without entering the fitting core; a real fit does enter it"
        if rejected_first and spy_works
        else f"raised={raised} entered_before_reject={not rejected_first} "
             f"spy_observes_a_real_fit={spy_works}")



def r8():
    ci = read(ROOT / ".github" / "workflows" / "ci.yml")
    cargo = read(ROOT / "Cargo.toml")
    return ('rust-version = "1.83"' in cargo and "numpy==1.23.0" in ci
            and (ROOT / "scripts" / "verify_release.py").exists()), \
        "MSRV 1.83 declared; exact dependency floors; artifact verification"



# ============================================ STAGE13-REVIEW.md, findings 1-10
def s1():
    """A labelled, shuffled, already-subset group Series fits the same model.

    Behavioural. 137 of 150 observations were assigned to the wrong cluster,
    collapsing the random-intercept variance from 0.8776 to 0.0092.
    """
    rng = np.random.default_rng(0)
    n, m = 300, 20
    g = np.repeat(np.arange(m), n // m)
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(0, 1.0, m)[g] + rng.normal(0, 0.4, n)
    d = pd.DataFrame({"y": y, "x": x, "g": g},
                     index=[f"r{i}" for i in range(n)])
    labels = d.index[:150]
    want = mlm.MixedLM.from_formula("y ~ x", d, groups="g",
                                    subset=labels).fit()
    shuffled = d.loc[labels, "g"].sample(frac=1.0, random_state=1)
    got = mlm.MixedLM.from_formula("y ~ x", d, groups=shuffled,
                                   subset=labels).fit()
    gap = abs(float(got.cov_re[0, 0]) - float(want.cov_re[0, 0]))
    return gap == 0.0, f"shuffled pre-subset groups: cov_re gap {gap:.3e}"


def s1b():
    """Ambiguous group input is rejected rather than guessed at."""
    d = pd.DataFrame({"y": np.arange(40.0), "x": np.arange(40.0),
                      "g": np.repeat(np.arange(4), 10)})
    pos = np.arange(20, dtype=np.intp)
    raised = 0
    for series in (pd.Series(np.arange(20), index=["a"] * 20),
                   pd.Series(np.arange(20),
                             index=[f"z{i}" for i in range(20)])):
        try:
            mlm.mixed_linear_model._align_groups(series, d, pos)
        except ValueError:
            raised += 1
    return raised == 2, f"{raised}/2 ambiguous group Series rejected"


def _na_in_re_frame():
    rng = np.random.default_rng(3)
    n, m = 200, 20
    g = np.repeat(np.arange(m), n // m)
    x, z = rng.normal(size=n), rng.normal(size=n)
    y = 1 + 0.5 * x + 0.3 * z + rng.normal(0, 0.5, n)
    d = pd.DataFrame({"y": y, "x": x, "z": z, "g": g})
    d.loc[d.index[::4], "z"] = np.nan
    return d


def s2():
    """Predictions do not move across a pickle when transforms are stateful.

    Behavioural. Every prediction shifted by 4.274472 because the design was
    rebuilt from the retained frame rather than the fitted rows.
    """
    d = _na_in_re_frame()
    new = pd.DataFrame({"x": [0.1, 0.5, 1.2], "z": [0.0, 1.0, -0.5],
                        "g": [0, 1, 2]})
    r = mlm.MixedLM.from_formula("y ~ center(x)", d, groups="g",
                                 re_formula="~z", missing="drop").fit()
    before = np.asarray(r.predict(new), float)
    after = np.asarray(pickle.loads(pickle.dumps(r)).predict(new), float)
    gap = float(np.max(np.abs(before - after)))
    return gap == 0.0, f"center() across pickle: max shift {gap:.3e}"


def s2b():
    """The categorical variant rebuilt an extra column and raised."""
    d = _na_in_re_frame()
    d["c"] = np.where(np.arange(len(d)) % 7 == 0, "rare", "common")
    d.loc[d["c"].eq("rare"), "z"] = np.nan
    new = pd.DataFrame({"x": [0.1], "z": [0.0], "g": [0], "c": ["common"]})
    r = mlm.MixedLM.from_formula("y ~ x + C(c)", d, groups="g",
                                 re_formula="~z", missing="drop").fit()
    before = np.asarray(r.predict(new), float)
    after = np.asarray(pickle.loads(pickle.dumps(r)).predict(new), float)
    return float(np.max(np.abs(before - after))) == 0.0, \
        "C() levels rebuilt from the fitted rows, no column mismatch"


def _square(v):
    return np.asarray(v, float) ** 2


def s3():
    """A module-level transform survives; an unpicklable one is named."""
    d = pd.DataFrame({"y": np.random.default_rng(1).normal(size=200),
                      "x": np.random.default_rng(2).normal(size=200),
                      "g": np.repeat(np.arange(20), 10)})
    globals()["_square"] = _square
    r = mlm.MixedLM.from_formula("y ~ _square(x)", d, groups="g").fit()
    new = pd.DataFrame({"x": [0.1, 0.5]})
    before = np.asarray(r.predict(new), float)
    after = np.asarray(pickle.loads(pickle.dumps(r)).predict(new), float)
    survives = bool(np.allclose(before, after, rtol=0, atol=0))

    def local(v):
        return np.asarray(v, float) ** 3

    r2 = mlm.MixedLM.from_formula("y ~ local(x)", d, groups="g").fit()
    try:
        pickle.loads(pickle.dumps(r2)).predict(new)
        named = False
    except ValueError as exc:
        text = str(exc)
        named = "local" in text and "were dropped" not in text
    return survives and named, \
        "module-level transform round-trips; a closure fails naming itself"


def s3b():
    """A failed rebuild leaves no half-built design behind."""
    d = pd.DataFrame({"y": np.random.default_rng(1).normal(size=200),
                      "x": np.random.default_rng(2).normal(size=200),
                      "g": np.repeat(np.arange(20), 10)})

    def local(v):
        return np.asarray(v, float) ** 3

    r = mlm.MixedLM.from_formula("y ~ local(x)", d, groups="g",
                                 re_formula="~x").fit()
    m = pickle.loads(pickle.dumps(r)).model
    ok = m._ensure_design_info() is False
    return ok and m._design_info is None and m._re_design_info is None, \
        "no partial design metadata after a failed reconstruction"


def s4():
    """The declared floors are the ones execution supports.

    patsy 0.5.0/0.5.1 raise `ImportError: cannot import name 'Mapping'` and
    0.5.2 fails its own version comparison, on every supported Python.
    """
    floors = {"numpy": "1.23", "scipy": "1.9", "pandas": "1.5",
              "patsy": "0.5.3"}
    data = tomllib.loads(read(ROOT / "pyproject.toml"))
    declared = dict(spec.split(">=")
                    for spec in data["project"]["dependencies"])
    req = dict(line.split(">=") for line in
               read(ROOT / "requirements-runtime.txt").splitlines()
               if ">=" in line and not line.startswith("#"))
    ci = read(ROOT / ".github" / "workflows" / "ci.yml")
    pinned = all(f'"{n}=={v if v.count(".") == 2 else v + ".0"}"' in ci
                 for n, v in floors.items())
    return declared == floors and req == floors and pinned, \
        f"metadata, requirements and CI all at {floors}"


def s5():
    """The job that runs this checker supplies what it needs."""
    ci = read(ROOT / ".github" / "workflows" / "ci.yml")
    job = ci[ci.index("  native-safety:"):ci.index("  install-smoke:")]
    have = ("maturin sdist" in job and "cargo install cargo-audit" in job
            and "verify_review_findings.py" in job)
    src = read(ROOT / "scripts" / "verify_review_findings.py")
    # A top-level call, at column zero. Searching the whole file finds this
    # check's own source, and splitting on the group-list assignment finds
    # it too.
    guarded = any(line == "require_prerequisites()"
                  for line in src.splitlines())
    return have and guarded, \
        "sdist built and cargo-audit installed in-job; guard is called"


def s6():
    """The document checks fail when the claim is removed.

    A check that passes against a file with its claim deleted verifies
    nothing. Each control mutates one file, re-runs one check, and requires it
    to report failure.
    """
    controls = [
        ("README.md", "Currently verified", s10c),
        ("docs/COMPATIBILITY.md", "`np.asarray(cov_re)[0, 0]` works on both",
         s10b),
        ("docs/LIMITATIONS.md", "no universal formula round-trip guarantee",
         s10d),
    ]
    survived = []
    for filename, phrase, check_fn in controls:
        with without(ROOT / filename, phrase):
            if check_fn()[0]:
                survived.append(f"{check_fn.__name__} passes without {phrase!r}")
    return not survived, \
        (f"{len(controls)}/{len(controls)} document checks fail when their "
         "claim is removed" if not survived else "; ".join(survived))


def s7():
    """The wheel carries the upstream notices, by content."""
    wheels = sorted((ROOT / "dist").glob("*.whl"),
                    key=lambda q: q.stat().st_mtime)
    if not wheels:
        return False, "no wheel built"
    with zipfile.ZipFile(wheels[-1]) as z:
        names = z.namelist()
        name = next((n for n in names
                     if n.endswith("THIRD-PARTY-LICENSES.md")), None)
        body = z.read(name).decode("utf-8") if name else ""
    crates = [c for c in ("numpy", "pyo3", "rayon", "ndarray")
              if f"## {c} " in body]
    ok = (bool(name)
          and "Redistribution and use in source and binary forms" in body
          and body.count("Copyright") >= 5
          and len(crates) == 4)
    return ok, (f"wheel ships BSD-2/MIT texts for {len(crates)}/4 crates, "
                f"{body.count('Copyright')} copyright notices")


def s8():
    """The evaluation-count ranges agree with the measured table."""
    text = read(ROOT / "docs" / "BENCHMARKS.md")
    start = text.index("## What the analytic gradient buys")
    section = text[start:text.index("\n## ", start + 1)]
    numeric, analytic = [], []
    for line in section.splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
        if len(cells) != 5 or not cells[0].replace(",", "").isdigit():
            continue
        numeric.append(int(cells[2]))
        analytic.append(int(cells[3]))
    want_n = f"{min(numeric)}\u2013{max(numeric)}"
    want_a = f"{min(analytic)}\u2013{max(analytic)}"
    stale = []
    for name in ("README.md", "CHANGELOG.md", "docs/DESIGN.md"):
        text = read(ROOT / name)
        if want_n not in text or want_a not in text:
            stale.append(f"{name} does not quote {want_n}/{want_a}")
        if EN_DASH.join(("44", "80")) in text or EN_DASH.join(("11", "16")) in text:
            stale.append(f"{name} still quotes the superseded range")
    return not stale, (f"3 documents quote the measured {want_n} to {want_a}"
                       if not stale else "; ".join(stale))


def s9():
    """The baseline gates catch what they claim to.

    Three separate defects: the freshness check fired on a lint-config edit,
    the recorder reported success with a failing cargo run, and the document
    validator accepted a count recorded for a different outcome.
    """
    src = read(ROOT / "tests" / "test_baseline.py")
    rec = read(ROOT / "bench" / "baseline.py")
    controls = read(ROOT / "tests" / "test_baseline_controls.py")
    narrowed = "_pyproject_build_sections" in src and '"bench"' in src
    paired = ("test_documented_outcome_counts_match_their_own_experiment" in src
              and "_tables_with_outcomes" in src)
    complete = "test_every_outcome_table_reports_every_outcome" in src
    negative = all(
        name in controls for name in
        ("test_a_count_from_the_other_experiment_is_rejected",
         "test_a_missing_outcome_row_is_rejected",
         "test_swapped_outcome_values_are_rejected"))
    working_tree = "--exclude-standard" in src
    shallow = "fetch-depth" in read(ROOT / ".github" / "workflows" / "ci.yml")
    binary = "_extension_identity" in rec and "extension_sha256" in rec
    sources = "_python_sources_digest" in rec and "python_sources_sha256" in rec
    generator = "NUMERIC_FILES" in src and "test_fuzz" in src
    suites = "failed = [name for name, run in" in rec
    checks = {"narrowed": narrowed, "experiment-paired": paired,
              "row-complete": complete, "negative-controls": negative,
              "working-tree": working_tree, "full-history": shallow,
              "binary-identity": binary, "source-identity": sources,
              "generator-watched": generator, "all-suites": suites}
    bad = [k for k, v in checks.items() if not v]
    return not bad, ("freshness, identity, pairing and propagation all fixed"
                     if not bad else f"missing: {bad}")


def s10():
    """use_sparse warns, as the compatibility contract says it does."""
    d = pd.DataFrame({"y": np.random.default_rng(0).normal(size=100),
                      "x": np.random.default_rng(1).normal(size=100),
                      "g": np.repeat(np.arange(10), 10)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mlm.MixedLM.from_formula("y ~ x", d, groups="g", use_sparse=True)
    warned = any("use_sparse" in str(w.message) for w in caught)
    with warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter("always")
        mlm.MixedLM.from_formula("y ~ x", d, groups="g")
    silent = not any("use_sparse" in str(w.message) for w in quiet)
    return warned and silent, "non-default use_sparse warns; the default does not"


def s10b():
    text = read(ROOT / "docs" / "COMPATIBILITY.md")
    return ("`np.asarray(cov_re)[0, 0]` works on both" in text
            and "`cov_re[0, 0]` works on both" not in text), \
        "the documented portable accessor is the one that works on both"


def s10c():
    text = read(ROOT / "README.md")
    return ("Currently verified" in text
            and "Same classes, same arguments" not in text), \
        "install and compatibility claims match what has been run"


def s10d():
    text = read(ROOT / "docs" / "LIMITATIONS.md")
    return ("no universal formula round-trip guarantee" in text
            and "the exact rows the model was fitted on" in text), \
        "the persistence promise is narrowed to what is delivered"



# ============================================ RECHECK2.md, findings 1-8
def t1():
    """A missing group label no longer breaks restored prediction.

    Row-selection bookkeeping is unified: patsy's retained rows and the
    group-present filter resolve together, before the design is built, so the
    fitted design and any rebuilt design are the same object.
    """
    d = DF.copy()
    d.loc[d.index[0], "g"] = np.nan
    r = mlm.mixedlm("y ~ center(x)", d, groups="g", missing="drop").fit()
    new = pd.DataFrame({"x": [0.1, 0.5, 1.2]})
    before = np.asarray(r.predict(new), float)
    after = np.asarray(pickle.loads(pickle.dumps(r)).predict(new), float)
    rows_agree = len(r.model._design_rows) == int(r.nobs)
    return bool(np.array_equal(before, after)) and rows_agree, \
        (f"missing group: {int(r.nobs)} rows fitted, "
         f"{len(r.model._design_rows)} recorded, prediction gap "
         f"{float(np.max(np.abs(before - after))):.3e}")


def t2():
    """A module alias in a formula survives a pickle."""
    import numpy as numeric  # noqa: F401  (patsy resolves it)
    # log() needs a positive column; the shared fixture's x is centred normal.
    d = DF.copy()
    d["x"] = np.abs(np.asarray(d["x"], float)) + 1.0
    results = []
    for kwargs in ({}, {"re_formula": "~numeric.log(x)"}):
        formula = "y ~ x" if kwargs else "y ~ numeric.log(x)"
        r = mlm.MixedLM.from_formula(formula, d, groups="g", **kwargs).fit()
        new = pd.DataFrame({"x": [0.4, 0.9]})
        before = np.asarray(r.predict(new), float)
        after = np.asarray(pickle.loads(pickle.dumps(r)).predict(new), float)
        results.append(bool(np.array_equal(before, after)))
    captured = mlm.MixedLM.from_formula(
        "y ~ numeric.log(x)", d, groups="g")._formula_namespace
    root_only = "numeric" in captured and "log" not in captured
    return all(results) and root_only, \
        "fixed and random formulas round-trip; only the alias root captured"


def t3():
    """Capture touches nothing the formula does not resolve externally."""
    calls = []

    class Tracked:
        def __reduce__(self):
            calls.append("reduce")
            return (str, ("tracked",))

    x = Tracked()                       # noqa: F841 - shadows a column
    unused = np.arange(1000.0)          # noqa: F841
    model = mlm.MixedLM.from_formula("y ~ x", DF, groups="g")
    captured = model._formula_namespace or {}
    arrays = [k for k, v in captured.items() if isinstance(v, np.ndarray)]
    return (not calls and "x" not in captured and not arrays), \
        (f"__reduce__ calls {len(calls)}, captured {sorted(captured)}, "
         f"retained arrays {arrays}")


def t4():
    """The release verify step installs an exact artifact, deps resolving."""
    text = read(ROOT / ".github" / "workflows" / "release.yml")
    broken = "--no-index --find-links collected mixedlm-rs" in text
    uses_script = "verify_release_artifacts.py" in text
    script = read(ROOT / "scripts" / "verify_release_artifacts.py")
    provenance = "direct_url" in script and "archive_info" in script
    return (not broken) and uses_script and provenance, \
        ("exact-path install with dependency resolution; provenance checked "
         "against pip's recorded archive hash")


def t5():
    """Publication is gated on tests for this commit, and on run artifacts."""
    release = read(ROOT / ".github" / "workflows" / "release.yml")
    ci = read(ROOT / ".github" / "workflows" / "ci.yml")
    import yaml  # named by the prerequisite check above if absent

    data = yaml.safe_load(release)

    def upstream(job, seen=None):
        """Everything `job` depends on, transitively.

        Direct dependencies are not the property: the verify jobs reach
        `publish` through `release-set`, and a check that only looked one
        level deep would call that a regression.
        """
        seen = seen if seen is not None else set()
        needs = data["jobs"][job].get("needs") or []
        if isinstance(needs, str):
            needs = [needs]
        for name in needs:
            if name not in seen:
                seen.add(name)
                upstream(name, seen)
        return seen

    publish = sorted(upstream("publish"))
    required = {"tests", "verify-wheels", "verify-sdist"}
    callable_ci = "workflow_call" in ci
    built = {(e["platform"], e["target"]) for e in
             data["jobs"]["wheels"]["strategy"]["matrix"]["include"]}
    ran = {(e["platform"], e["target"]) for e in
           data["jobs"]["verify-wheels"]["strategy"]["matrix"]["include"]}
    gap = sorted(built - ran)
    # Every published wheel is installed and run on its own architecture, so
    # there is no gap left to disclose -- an earlier version of this check
    # required the disclosure text, which is now simply false.
    retired = any("macos-13" in line and not line.strip().startswith("#")
                  for line in release.splitlines())
    selective = "pattern: release-*" in release
    gated = "check_release_set.py" in release
    ok = (required <= set(publish) and callable_ci and not gap
          and not retired and selective and gated)
    return ok, (
        f"publish needs {sorted(publish)}; all {len(built)} wheels executed "
        "on their own architecture; upload set selected and hash-checked"
        if ok else
        f"gap={gap} retired_runner={retired} selective={selective} "
        f"gated={gated}")


def t6():
    """No wall-clock assertion decides whether validation preceded fitting."""
    text = read(ROOT / "tests" / "test_fit_arguments.py")
    timing = [line.strip() for line in text.splitlines()
              if "perf_counter" in line or "quick <" in line]
    spy = "optimiser_must_not_run" in text and "fit_core" in text
    control = "test_the_spy_would_notice_a_real_fit" in text
    return (not timing) and spy and control, \
        ("argument rejection asserted by spying on fit_core, with a negative "
         "control" if not timing else f"timing assertions remain: {timing[:2]}")


def t7():
    """A count from the wrong experiment is rejected."""
    controls = read(ROOT / "tests" / "test_baseline_controls.py")
    src = read(ROOT / "tests" / "test_baseline.py")
    return (all(name in controls for name in (
                "test_a_count_from_the_other_experiment_is_rejected",
                "test_a_missing_outcome_row_is_rejected",
                "test_swapped_outcome_values_are_rejected"))
            and "test_documented_outcome_counts_match_their_own_experiment" in src
            and "NUMERIC_FILES" in src), \
        "cross-experiment, missing-row and swap controls all present"


def t8():
    """The migration conclusion defers to the full contract."""
    text = read(ROOT / "docs" / "COMPATIBILITY.md")
    section = text[text.index("## Deciding whether to switch"):]
    absolute = "the import swap really is the whole migration" in section
    named = [n for n in ("free", "fe_pen", "profile_re", "fit_regularized",
                         "vc_formula", "use_sparse", "by label")
             if n not in section]
    return (not absolute) and not named, \
        ("checklist covers penalties, free, profiling, alignment and "
         "persistence" if not named else f"still omits {named}")


# ============================================ RECHECK3.md, blockers 1-3
def u1():
    """Infinities are resolved before transform state is learned.

    Four routes to one root cause: an infinity in a fixed predictor, a random
    predictor, the response, and one produced by a transform rather than
    supplied. Each changed predictions by ~0.083 across a pickle, silently,
    because the reconstruction guard compared names and dimensions while the
    design values differed by 0.05.
    """
    rng = np.random.default_rng(16)
    n = 200
    g = np.repeat(np.arange(20), 10)
    x = np.arange(n, dtype=float) / 10 + 1
    y = 2 + 1.7 * x + rng.normal(size=20)[g] + rng.normal(scale=0.3, size=n)
    base = pd.DataFrame({"y": y, "x": x, "w": rng.normal(size=n), "g": g})
    new = pd.DataFrame({"x": [6.0, 10.0, 18.0], "w": [1.0, 2.0, 3.0]})

    gaps = {}
    for case in ("fixed_inf", "random_inf", "response_inf", "transform_inf"):
        d = base.copy()
        formula, re_formula = "y ~ center(x)", None
        if case == "fixed_inf":
            d.loc[0, "w"] = np.inf
            formula += " + w"
        elif case == "random_inf":
            d.loc[0, "w"] = np.inf
            re_formula = "~w"
        elif case == "response_inf":
            d.loc[0, "y"] = np.inf
        else:
            d["w"] = np.exp(d["w"])
            d.loc[0, "w"] = 0.0
            formula += " + np.log(w)"
        r = mlm.MixedLM.from_formula(formula, d, groups="g",
                                     re_formula=re_formula,
                                     missing="drop").fit()
        before = np.asarray(r.predict(new), float)
        after = np.asarray(
            pickle.loads(pickle.dumps(r)).predict(new), float)
        gaps[case] = float(np.max(np.abs(before - after)))
    worst = max(gaps.values())
    return worst == 0.0, f"4 infinity variants, largest prediction gap {worst:.1e}"


def u1b():
    """The rebuild guard compares design values, not just dimensions."""
    d = pd.DataFrame({"y": np.random.default_rng(1).normal(size=200),
                      "x": np.arange(200.0) / 10 + 1,
                      "w": np.random.default_rng(2).normal(size=200),
                      "g": np.repeat(np.arange(20), 10)})
    r = mlm.MixedLM.from_formula("y ~ center(x) + w", d, groups="g",
                                 missing="drop").fit()
    model = r.model
    shifted = model.exog.copy()
    shifted[0, 1] += 0.05                    # the observed magnitude
    try:
        model._compare_design("fixed-effect", shifted, model.exog)
        rejected = False
    except ValueError as exc:
        rejected = "same shape and column names" in str(exc)
    # And an identical design is still accepted.
    try:
        model._compare_design("fixed-effect", model.exog.copy(), model.exog)
        accepts = True
    except ValueError:
        accepts = False
    return rejected and accepts, \
        "a same-shape design differing by 0.05 is rejected; an equal one is not"


def u2():
    """Only verified release artifacts can be published."""
    release = read(ROOT / ".github" / "workflows" / "release.yml")
    import yaml
    data = yaml.safe_load(release)

    unfiltered = []
    for name, job in data["jobs"].items():
        for step in job.get("steps") or []:
            if not step.get("uses", "").startswith("actions/download-artifact"):
                continue
            using = step.get("with") or {}
            if not using.get("pattern") and not using.get("name"):
                unfiltered.append(name)
    gate = "check_release_set.py" in release
    twine = "twine check" in release
    named = "release-wheel-" in release and "verified-wheel-" in release
    script = (ROOT / "scripts" / "check_release_set.py").is_file()
    ok = not unfiltered and gate and twine and named and script
    return ok, ("artifacts selected by pattern, hashes checked against the "
                "verify jobs, twine before upload" if ok else
                f"unfiltered={unfiltered} gate={gate} twine={twine} "
                f"named={named}")


def u3():
    """Every published wheel runs on its own architecture."""
    release = read(ROOT / ".github" / "workflows" / "release.yml")
    ci = read(ROOT / ".github" / "workflows" / "ci.yml")
    import yaml
    data = yaml.safe_load(release)
    built = {(e["platform"], e["target"]) for e in
             data["jobs"]["wheels"]["strategy"]["matrix"]["include"]}
    ran = {(e["platform"], e["target"]): e["runner"] for e in
           data["jobs"]["verify-wheels"]["strategy"]["matrix"]["include"]}
    gap = sorted(built - set(ran))

    # macos-13 was retired in December 2025. A comment may mention it; a
    # runner selection may not.
    retired = [line.strip() for line in release.splitlines()
               if "macos-13" in line and not line.strip().startswith("#")]

    # A label is a claim until something runs on it. CI proves both unusual
    # ones and asserts the machine type.
    probe = yaml.safe_load(ci)["jobs"]["architectures"]["strategy"]["matrix"]
    proven = {e["label"] for e in probe["include"]}
    common = {"ubuntu-latest", "macos-latest", "windows-latest"}
    unproven = sorted((set(ran.values()) - common) - proven)

    ok = not gap and not retired and not unproven
    return ok, (f"all {len(built)} wheels executed; unusual labels "
                f"{sorted(proven)} proven in CI" if ok else
                f"gap={gap} retired={retired} unproven={unproven}")



GROUPS = [
    ("REVIEW", [(f"F{i:02d}", fn) for i, fn in enumerate(
        [f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11, f12, f13,
         f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26,
         f27, f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38, f39], 1)]),
    ("RECHECK", [(f"R{i}", fn) for i, fn in enumerate(
        [r1, r2, r3, r4, r5, r6, r7, r8], 1)]),
    ("STAGE13", [("S1", s1), ("S1b", s1b), ("S2", s2), ("S2b", s2b),
                 ("S3", s3), ("S3b", s3b), ("S4", s4), ("S5", s5),
                 ("S6", s6), ("S7", s7), ("S8", s8), ("S9", s9),
                 ("S10", s10), ("S10b", s10b), ("S10c", s10c),
                 ("S10d", s10d)]),
    ("RECHECK2", [("T1", t1), ("T2", t2), ("T3", t3), ("T4", t4),
                  ("T5", t5), ("T6", t6), ("T7", t7), ("T8", t8)]),
    ("RECHECK3", [("U1", u1), ("U1b", u1b), ("U2", u2), ("U3", u3)]),
]

# What each check is actually evidence of. Reporting a single total invited
# reading "47/47" as certification that the package is correct, which it is
# not: a documentation check establishes that a file says something, and a
# release check establishes that an artifact was built here, today, on this
# machine. Only the behavioural ones run the code and assert on the answer.
KIND = {
    "behaviour": {
        "T1", "T2", "T3", "T6", "U1", "U1b",
        "F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10",
        "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20",
        "F21", "F22", "F23", "F24", "F25", "F26", "F27", "F28",
        "R2", "R3", "R4", "R5", "R6", "R7",
        "S1", "S1b", "S2", "S2b", "S3", "S3b", "S10",
    },
    "release evidence": {"F37", "R1", "S7"},
    "workflow (not executed)": {"T4", "T5", "U2", "U3"},
    "negative control": {"S6"},
}


require_prerequisites()

for group, items in GROUPS:
    for label, fn in items:
        check(group, label, fn)

width = max(len(d) for *_, d in RESULTS)
print()
for group in ("REVIEW", "RECHECK", "STAGE13", "RECHECK2",
              "RECHECK3"):
    rows = [r for r in RESULTS if r[0] == group]
    # Named, not pathed: these documents were an external deliverable and are
    # not in the repository, so printing a repo-relative path would send a
    # reader looking for a file that is not there.
    doc = {"REVIEW": "review 1 (REVIEW.md)",
           "RECHECK": "review 2 (RECHECK.md)",
           "STAGE13": "review 3 (STAGE13-REVIEW.md)",
           "RECHECK2": "review 4 (RECHECK2.md)",
           "RECHECK3": "review 5 (RECHECK3.md)"}[group]
    print(f"--- {doc}: {sum(1 for r in rows if r[2])}/{len(rows)} ---")
    for _, label, ok, detail in rows:
        print(f"  {label}  {'PASS' if ok else '**FAIL**':9s} {detail}")
    print()

failed = [r for r in RESULTS if not r[2]]


def kind_of(label):
    for name, members in KIND.items():
        if label in members:
            return name
    return "documentation"


print(f"TOTAL: {len(RESULTS) - len(failed)}/{len(RESULTS)} checks pass")
for name in ("behaviour", "release evidence", "negative control",
             "workflow (not executed)", "documentation"):
    rows = [r for r in RESULTS if kind_of(r[1]) == name]
    if rows:
        print(f"  {name:18s} {sum(1 for r in rows if r[2])}/{len(rows)}")
print()
print(textwrap.fill(
    "What this establishes: the behavioural checks re-run each reviewer's "
    "own case and assert on the answer. The documentation checks establish "
    "that a file says a particular thing -- the negative controls confirm "
    "they fail when it does not -- and the release checks establish that an "
    "artifact built on this machine has the expected contents. Passing is "
    "evidence that these specific defects are fixed. It is not a claim that "
    "the package is free of defects nobody has looked for.", 78))
sys.exit(1 if failed else 0)

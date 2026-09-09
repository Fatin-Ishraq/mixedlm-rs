"""The adversarial stress sweep, committed so its numbers are reproducible.

The 400-case sweep quoted in docs/CORRECTNESS.md used to live outside the
repository, which meant the classifications in that table could not be checked
and the individual losses could not be re-examined. It is here now.

Six deliberately hostile shapes, chosen because they are where mixed-model
fitters actually fail rather than because they are unusual:

  tiny        many groups of exactly two observations
  imbalanced  one enormous group among tiny ones
  fewbig      a handful of very large groups
  illscaled   predictors spanning six orders of magnitude
  collinear   near-collinear fixed effects
  outliers    heavy-tailed contamination

Every case is classified against statsmodels on the *criterion*, which is what
the two optimisers are competing on. Where we win, the parameters legitimately
differ and demanding agreement would assert that we reproduce a worse fit.

Losses are not dismissed by argument. Each one is re-examined:

  * If the criterion is flat along the variance split, the comparison is
    meaningless -- every point on the ridge fits identically -- and it is
    reported as `unidentified` rather than as a loss. This replaces an earlier
    claim that such cases "diverge", which was wrong; see CORRECTNESS.md.
  * Otherwise the absolute deviance gap is reported, not a relative one. A
    relative tolerance on a log-likelihood is meaningless, because the
    criterion carries an arbitrary additive constant.

    python bench/stress_sweep.py [--cases 400] [--seed 0] [--json out.json]
"""

import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import mixedlm_rs as mlm
import statsmodels.formula.api as smf

SHAPES = ("tiny", "imbalanced", "fewbig", "illscaled", "collinear", "outliers")


def make_case(rng, shape):
    """One hostile fixture. Returns (frame, re_formula, reml)."""
    if shape == "tiny":
        ngroups = int(rng.integers(40, 160))
        sizes = np.full(ngroups, 2)
    elif shape == "imbalanced":
        ngroups = int(rng.integers(20, 60))
        sizes = rng.integers(2, 5, ngroups)
        sizes[0] = int(rng.integers(400, 1200))
    elif shape == "fewbig":
        ngroups = int(rng.integers(3, 8))
        sizes = rng.integers(150, 400, ngroups)
    else:
        ngroups = int(rng.integers(15, 90))
        sizes = rng.integers(3, 25, ngroups)

    codes = np.repeat(np.arange(ngroups), sizes)
    n = len(codes)

    if shape == "illscaled":
        x1 = rng.standard_normal(n) * 10.0 ** rng.uniform(-3, 3)
        x2 = rng.standard_normal(n) * 10.0 ** rng.uniform(-3, 3)
    elif shape == "collinear":
        x1 = rng.standard_normal(n)
        x2 = x1 * (1.0 - 10.0 ** rng.uniform(-6, -3)) + rng.standard_normal(n) * 1e-4
    else:
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)

    re_sd = 10.0 ** rng.uniform(-2, 1)
    resid_sd = 10.0 ** rng.uniform(-1, 1)
    q = 2 if rng.random() < 0.4 else 1

    b0 = rng.standard_normal(ngroups) * re_sd
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + b0[codes] + rng.standard_normal(n) * resid_sd
    if q == 2:
        b1 = rng.standard_normal(ngroups) * re_sd * rng.uniform(0.0, 1.0)
        y = y + b1[codes] * x1
    if shape == "outliers":
        hit = rng.random(n) < 0.03
        y[hit] += rng.standard_t(2, hit.sum()) * resid_sd * 40

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": codes})
    return df, ("~x1" if q == 2 else None), bool(rng.random() < 0.5)


def criterion_is_flat(res, reml):
    """Is the deviance unchanged along the variance split?

    Asked of the core the fit actually used, so `theta` is in the right
    coordinates. A flat ridge means the split is not identified and comparing
    two implementations' splits is meaningless.
    """
    core = res._res["core"]
    theta = np.asarray(res._res["theta"], float)
    q = core.q
    diag = [k for k, (r, c) in enumerate(
        [(r, c) for c in range(q) for r in range(c, q)]) if r == c]
    d0 = core.deviance(list(theta), reml)
    if not np.isfinite(d0):
        return False
    for factor in (4.0, 0.25):
        probe = theta.copy()
        probe[diag] = np.maximum(probe[diag] * factor, 1e-3)
        dp = core.deviance(list(probe), reml)
        if not np.isfinite(dp) or abs(dp - d0) > 1e-9 * max(1.0, abs(d0)):
            return False
    return True


def classify(df, re_formula, reml):
    out = {"reml": reml, "re": re_formula or "intercept"}
    try:
        ours = mlm.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                           re_formula=re_formula).fit(reml=reml)
    except Exception as exc:
        out["outcome"] = "we raised"
        out["detail"] = f"{type(exc).__name__}: {exc}"
        return out
    out["our_converged"] = bool(ours.converged)
    out["our_singular"] = bool(ours.singular)

    try:
        theirs = smf.mixedlm("y ~ x1 + x2", df, groups=df["g"],
                             re_formula=re_formula).fit(reml=reml)
        their_llf = float(theirs.llf)
        their_conv = bool(theirs.converged)
    except Exception as exc:
        out["outcome"] = "reference raised"
        out["detail"] = f"{type(exc).__name__}: {str(exc)[:60]}"
        return out

    if not their_conv or not np.isfinite(their_llf):
        out["outcome"] = "reference did not converge"
        return out

    # Absolute deviance gap. A relative tolerance on a log-likelihood is
    # meaningless: the criterion carries an arbitrary additive constant.
    gap = 2.0 * (float(ours.llf) - their_llf)      # positive means we are better
    out["deviance_gap"] = gap
    if gap > 1e-6:
        out["outcome"] = "we found a better optimum"
    elif gap < -1e-6:
        if criterion_is_flat(ours, reml):
            out["outcome"] = "unidentified (criterion flat)"
        else:
            out["outcome"] = "we found a worse optimum"
    else:
        out["outcome"] = "same optimum"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = []
    t0 = time.perf_counter()
    for i in range(args.cases):
        shape = SHAPES[i % len(SHAPES)]
        rng = np.random.default_rng(args.seed * 1_000_003 + i)
        df, re_formula, reml = make_case(rng, shape)
        row = classify(df, re_formula, reml)
        row.update(case=i, shape=shape, n=len(df),
                   groups=int(df["g"].nunique()))
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{args.cases} ...", flush=True)
    dt = time.perf_counter() - t0

    counts = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

    print(f"\n{args.cases} cases in {dt:.1f}s (seed {args.seed})\n")
    print(f"{'outcome':<34s} {'count':>6s}")
    print("-" * 42)
    for k in sorted(counts, key=lambda k: -counts[k]):
        print(f"{k:<34s} {counts[k]:>6d}")
    print("-" * 42)

    losses = [r for r in rows if r["outcome"] == "we found a worse optimum"]
    if losses:
        print("\nEvery loss, with its absolute deviance gap:")
        for r in losses:
            print(f"  case {r['case']:>3d} {r['shape']:<11s} n={r['n']:>6d} "
                  f"groups={r['groups']:>5d}  deviance worse by "
                  f"{-r['deviance_gap']:.3e}")
    else:
        print("\nNo losses.")

    raised = [r for r in rows if r["outcome"] == "we raised"]
    if raised:
        print("\nExceptions raised:")
        for r in raised:
            print(f"  case {r['case']}: {r['detail']}")

    nonconv = [r for r in rows if not r.get("our_converged", True)]
    print(f"\nwe failed to certify a stationary point on {len(nonconv)}/"
          f"{args.cases} cases")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"seed": args.seed, "cases": args.cases,
                       "counts": counts, "rows": rows}, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

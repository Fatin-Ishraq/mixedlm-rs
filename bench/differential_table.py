"""Reproduce the differential-testing table quoted in docs/CORRECTNESS.md.

The fuzz suite in `tests/test_fuzz.py` asserts that we are never worse; this
script produces the *counts* that the documentation reports, from the same
fixture generator and the same seeds, so the table is checkable rather than
quoted.

Classification is on the criterion, in deviance units, which is what the two
optimisers are competing on. Parameters are not compared here: where we find a
better optimum they legitimately differ, and where the surface is flat two
optimisers stop at different points without disagreeing about anything.

    python bench/differential_table.py [--cases 120] [--json out.json]
"""

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))

import mixedlm_rs as mlm
import statsmodels.formula.api as smf
from test_fuzz import random_case

ORDER = [
    "statsmodels did not converge",
    "we found a strictly better optimum",
    "same optimum",
    "we found a worse optimum",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=120)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    counts = dict.fromkeys(ORDER, 0)
    worse = []
    for seed in range(args.cases):
        df, re_formula, reml = random_case(np.random.default_rng(10_000 + seed))
        kw = {"groups": df["g"], "re_formula": re_formula}
        ours = mlm.mixedlm("y ~ x1 + x2", df, **kw).fit(reml=reml)
        theirs = smf.mixedlm("y ~ x1 + x2", df, **kw).fit(reml=reml)
        if not theirs.converged:
            key = "statsmodels did not converge"
        else:
            gap = 2.0 * (float(ours.llf) - float(theirs.llf))
            if gap > 1e-6:
                key = "we found a strictly better optimum"
            elif gap < -1e-6:
                key = "we found a worse optimum"
                worse.append((seed, gap))
            else:
                key = "same optimum"
        counts[key] += 1

    print(f"\n{args.cases} randomised fixtures\n")
    print(f"| {'outcome':<38s} | {'count':>5s} |")
    print(f"|{'-' * 40}|{'-' * 7}|")
    for key in ORDER:
        print(f"| {key:<38s} | {counts[key]:>5d} |")
    failed = counts["statsmodels did not converge"] \
        + counts["we found a strictly better optimum"]
    print(f"\nreference failed or landed worse on {failed}/{args.cases} "
          f"({100 * failed / args.cases:.1f}%)")
    for seed, gap in worse:
        print(f"  worse on seed {seed}: deviance gap {gap:.3e}")

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"cases": args.cases, "counts": counts}, indent=1),
            encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

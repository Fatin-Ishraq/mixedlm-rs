"""One set of agreement tolerances, with the reasoning for each.

Every benchmark in this directory compares two fits before it is allowed to
report a timing. They used to disagree about how: `stress_sweep.py` and
`differential_table.py` compared deviances absolutely while `stages.py` used
`1e-6 * abs(deviance)` -- a *relative* tolerance on a quantity that carries an
arbitrary additive constant, which is the thing the documentation criticises
elsewhere. Scaling a tolerance by a constant that can be shifted at will makes
the same difference pass or fail depending on the units.

So the numbers live here, once, with what they mean.
"""

# ---------------------------------------------------------------- criterion
#
# Deviance is -2 log-likelihood. Differences are on a fixed, interpretable
# scale: a likelihood-ratio test at one degree of freedom needs 3.84 to reach
# 5%, so 1e-3 is more than three orders of magnitude below anything that could
# change a conclusion.
#
# It is also comfortably above the floating-point noise. The largest criterion
# measured here is around 1.3e6 (500,264 rows), and double precision resolves
# that to roughly 1e-10 relative, i.e. about 1e-4 absolute. 1e-3 clears that
# with an order of magnitude to spare, so a failure means a real difference in
# the fit rather than accumulated rounding.
DEVIANCE_ABS = 1e-3

# ------------------------------------------------------------ fixed effects
#
# Compared on the scale of their own standard errors, which is the only scale
# that means anything: a coefficient differing by 2% of one SE is two
# optimisers stopping at slightly different points on a flat likelihood, not a
# disagreement between implementations. 0.05 SE is still far below the
# resolution of any inference drawn from the coefficient.
FIXED_EFFECT_SE = 0.05

# --------------------------------------------------------- variance components
#
# Relative to their own size, because a variance has no meaningful absolute
# scale and no additive constant to be confused by. 2% is loose enough for two
# different optimisers on a flat ridge and tight enough that a genuinely
# different variance decomposition fails.
COV_RE_REL = 0.02

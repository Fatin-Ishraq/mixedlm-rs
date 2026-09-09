"""Malformed input to the compiled core must raise, never kill the process.

The crate is built with `panic = "abort"`, so a missing bounds check is not a
Python exception -- it is an immediate process termination that no `except` can
catch. On Windows these showed up as exit status 3221226505 (0xC0000409, the
stack-buffer-security check); on Linux as SIGABRT.

That makes this untestable in-process: a test that triggers one takes the whole
pytest run down with it. Every case therefore runs in its own child process and
the assertion is on the exit code. A test that merely called these functions
and asserted `pytest.raises` would have passed by killing the runner.

The cases below were all real: an empty or oversized `theta`, a non-finite
`theta`, and a zero-column design each reached unchecked indexing.
"""

import subprocess
import sys
import textwrap

import pytest

PREAMBLE = textwrap.dedent("""
    import numpy as np
    from mixedlm_rs._mixedlm_rs import LmmCore

    n, m = 40, 10
    y = np.random.default_rng(0).normal(size=n)
    X = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
    Z = np.ones((n, 1))
    codes = np.repeat(np.arange(m), 4).astype(np.int64)
    core = LmmCore(y, X, Z, codes, m)
""")

CASES = [
    ("empty theta to deviance", "core.deviance(np.array([]))"),
    ("empty theta to deviance_grad", "core.deviance_grad(np.array([]))"),
    ("empty theta to solution", "core.solution(np.array([]))"),
    ("oversized theta", "core.deviance(np.array([1.0, 2.0, 3.0]))"),
    ("nan theta", "core.deviance(np.array([np.nan]))"),
    ("inf theta", "core.deviance(np.array([np.inf]))"),
    ("zero-column Z", "LmmCore(y, X, np.zeros((n, 0)), codes, m)"),
    ("zero-column X", "LmmCore(y, np.zeros((n, 0)), Z, codes, m)"),
    ("empty y", "LmmCore(np.zeros(0), np.zeros((0, 2)), np.zeros((0, 1)),"
                " np.zeros(0, dtype=np.int64), 1)"),
    ("zero groups", "LmmCore(y, X, Z, codes, 0)"),
    ("group code out of range",
     "LmmCore(y, X, Z, np.full(n, 99, dtype=np.int64), m)"),
    ("nan in a start vector", 'core.fit([float("nan")], True, 10, 1e-8, 1e-12)'),
    ("empty starts", "core.fit([], True, 10, 1e-8, 1e-12)"),
]


def run_in_child(expr):
    """Evaluate `expr` in a fresh interpreter; return (returncode, stdout)."""
    code = PREAMBLE + textwrap.dedent(f"""
        try:
            {expr}
        except ValueError as exc:
            print("RAISED", type(exc).__name__)
        except Exception as exc:            # any other exception is still a bug
            print("WRONG_TYPE", type(exc).__name__, exc)
        else:
            print("NO_ERROR")
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=120)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


@pytest.mark.parametrize("name,expr", CASES, ids=[c[0] for c in CASES])
def test_malformed_input_raises_without_aborting(name, expr):
    rc, out, err = run_in_child(expr)
    assert rc == 0, (
        f"{name}: the interpreter exited with {rc} instead of raising. "
        f"panic=abort turns a missing check into a process kill.\n"
        f"stderr tail: {err[-400:]}")
    assert out.startswith("RAISED"), f"{name}: expected a ValueError, got {out!r}"

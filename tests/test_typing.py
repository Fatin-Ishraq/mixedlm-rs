"""The `py.typed` marker has to be backed by annotations that actually work.

The first attempt at this shipped `py.typed` against 0 of 42 annotated public
functions, which is worse than shipping nothing: the marker suppresses mypy's
"missing library stubs" warning, so callers got silent `Any` instead of a
message telling them the package is untyped.

So the claim is checked three ways, from cheapest to strongest:

1. every public annotation *resolves* at runtime via ``typing.get_type_hints``;
2. every public function is annotated at all;
3. ``mypy --strict`` accepts ``tests/typing_usage.py``, which exercises the
   public surface for real. That is the one that would catch a wrong
   annotation rather than a missing one, and it runs in CI.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import subprocess
import sys
import typing

import mixedlm_rs as mlm
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
USAGE = pathlib.Path(__file__).with_name("typing_usage.py")


def public_callables():
    """Every public function and property reachable from the package root."""
    for name in mlm.__all__:
        obj = getattr(mlm, name)
        if inspect.isfunction(obj):
            yield name, obj
        elif inspect.isclass(obj):
            for attr, member in vars(obj).items():
                if attr.startswith("_") and attr != "__init__":
                    continue
                if isinstance(member, property):
                    if member.fget is not None:
                        yield f"{name}.{attr}", member.fget
                elif inspect.isfunction(member):
                    yield f"{name}.{attr}", member


# ------------------------------------------------------------------ marker
def test_py_typed_marker_ships_in_the_installed_package():
    marker = pathlib.Path(mlm.__file__).parent / "py.typed"
    assert marker.is_file(), (
        "py.typed is missing from the installed package, so type checkers "
        "will ignore the annotations entirely")


def test_the_native_stub_ships_too():
    stub = pathlib.Path(mlm.__file__).parent / "_mixedlm_rs.pyi"
    assert stub.is_file(), (
        "the compiled module has no stub, so LmmCore is Any to a checker "
        "while py.typed claims the package is typed")


# ------------------------------------------------------------- resolvability
def test_every_public_annotation_resolves():
    """`get_type_hints` is what a runtime consumer calls.

    This failed on 18 members while the file still looked fully annotated:
    `ArrayLike` was imported under `if TYPE_CHECKING`, so the strings in
    `__annotations__` referred to a name that did not exist at runtime.
    """
    broken = []
    for label, fn in public_callables():
        try:
            typing.get_type_hints(fn)
        except Exception as exc:                       # noqa: BLE001
            broken.append(f"{label}: {type(exc).__name__}: {exc}")
    assert not broken, "annotations that do not resolve:\n  " + "\n  ".join(broken)


def test_module_level_annotations_resolve():
    for mod in ("mixedlm_rs", "mixedlm_rs.mixed_linear_model",
                "mixedlm_rs._fit", "mixedlm_rs._install"):
        typing.get_type_hints(importlib.import_module(mod))


# ---------------------------------------------------------------- completeness
def test_every_public_function_is_annotated():
    missing = []
    for label, fn in public_callables():
        sig = inspect.signature(fn)
        if sig.return_annotation is inspect.Signature.empty:
            missing.append(f"{label} (no return annotation)")
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if param.annotation is inspect.Parameter.empty:
                missing.append(f"{label} (parameter {pname})")
    assert not missing, "unannotated public API:\n  " + "\n  ".join(missing)


def test_the_usage_file_imports_and_runs():
    """A checked file that no longer matches the API is worthless.

    Importing it catches an attribute that was renamed or removed, which mypy
    would also catch -- but this runs in the ordinary suite, where mypy does
    not.
    """
    sys.path.insert(0, str(USAGE.parent))
    try:
        usage = importlib.import_module("typing_usage")
    finally:
        sys.path.pop(0)
    df = usage.build_frame()
    usage.results_surface(df)
    usage.hypothesis_tests(df)
    usage.likelihood_surface(df)
    usage.model_metadata(df)
    usage.parameter_containers()
    usage.native_core(df)


# -------------------------------------------------------------------- mypy
@pytest.mark.slow
def test_mypy_strict_accepts_the_usage_file():
    """The strongest of the three, and the slowest.

    Skipped when mypy is absent so the ordinary suite does not require it; the
    lint CI job installs it and this then runs there.
    """
    pytest.importorskip("mypy")
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", str(USAGE), "--strict",
         "--ignore-missing-imports", "--no-error-summary"],
        capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert proc.returncode == 0, (
        "mypy --strict rejected the representative usage:\n"
        + proc.stdout + proc.stderr)

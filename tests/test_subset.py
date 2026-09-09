"""`subset=` selection, and its alignment with an external `groups`.

`subset` selects by **index label**, which is pandas' and statsmodels'
meaning. This used to treat a non-boolean `subset` as row *positions*:
``subset=[10, 20]`` on a frame indexed 100..199 quietly selected rows 10 and
20 instead of raising, and on a string-labelled frame it selected nothing at
all. Both are silent wrong answers rather than errors, which is why they are
pinned here across integer labels, string labels, duplicate labels, and frames
with rows missing.

The second half of each test is the part that actually matters: the selected
rows have to stay aligned with an externally supplied `groups`. A selection
that quietly reorders or duplicates rows relative to `groups` produces a fit
on data nobody has.
"""

import warnings

import mixedlm_rs as mlm
import numpy as np
import pandas as pd
import pytest


def frame(n=200, m=20, index=None, seed=5):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(m), n // m)
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.6, size=m)[g] + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "x": x, "g": g})
    if index is not None:
        df.index = index
    return df


def fit(df, groups, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mlm.mixedlm("y ~ x", df, groups=groups, **kw).fit()


def assert_same_fit(a, b):
    assert a.nobs == b.nobs
    assert a.llf == pytest.approx(b.llf, abs=1e-9)
    assert np.allclose(a.fe_params, b.fe_params, rtol=1e-9, atol=1e-12)


# ------------------------------------------------------------ boolean masks
def test_boolean_mask_array():
    df = frame()
    mask = np.zeros(len(df), dtype=bool)
    mask[:150] = True
    got = fit(df, df["g"], subset=mask)
    want = fit(df.iloc[:150], df["g"].iloc[:150])
    assert_same_fit(got, want)


def test_boolean_mask_series_must_share_the_index():
    df = frame(index=[f"r{i}" for i in range(200)])
    mask = pd.Series(np.arange(200) < 150, index=df.index)
    got = fit(df, df["g"], subset=mask)
    assert got.nobs == 150

    wrong = pd.Series(np.arange(200) < 150, index=range(200))
    with pytest.raises(ValueError, match="same index"):
        fit(df, df["g"], subset=wrong)


def test_boolean_mask_of_the_wrong_length_is_refused():
    df = frame()
    with pytest.raises(ValueError, match="one per row"):
        fit(df, df["g"], subset=np.ones(37, dtype=bool))


# ------------------------------------------------------- integer index labels
def test_integer_labels_are_labels_not_positions():
    """The regression. A frame indexed 100..299, selecting labels 100..249."""
    df = frame(index=np.arange(100, 300))
    labels = np.arange(100, 250)
    got = fit(df, df["g"], subset=labels)
    want = fit(df.loc[labels], df["g"].loc[labels])

    assert got.nobs == 150
    assert_same_fit(got, want)

    # And the old positional reading is now an error rather than a wrong answer:
    # labels 0..149 do not exist in this frame.
    with pytest.raises(ValueError, match="not in the index"):
        fit(df, df["g"], subset=np.arange(150))


def test_integer_labels_out_of_order_select_the_same_rows():
    df = frame(index=np.arange(100, 300))
    labels = np.arange(100, 150)
    a = fit(df, df["g"], subset=labels)
    b = fit(df, df["g"], subset=labels[::-1])
    assert_same_fit(a, b)


# -------------------------------------------------------- string index labels
def test_string_labels():
    df = frame(index=[f"row{i:03d}" for i in range(200)])
    labels = [f"row{i:03d}" for i in range(150)]
    got = fit(df, df["g"], subset=labels)
    want = fit(df.loc[labels], df["g"].loc[labels])
    assert got.nobs == 150
    assert_same_fit(got, want)


def test_unknown_string_label_names_itself():
    df = frame(index=[f"row{i:03d}" for i in range(200)])
    with pytest.raises(ValueError) as exc:
        fit(df, df["g"], subset=["row000", "nope", "also-nope"])
    msg = str(exc.value)
    assert "not in the index" in msg
    assert "nope" in msg


# ------------------------------------------------------------ duplicate index
def test_duplicate_index_labels_select_every_matching_row_once():
    """Each matching row appears once, in the frame's own order.

    `.loc` would return one copy per occurrence *in the subset*, multiplying
    rows and breaking alignment with a positional `groups`.
    """
    df = frame(index=np.repeat(np.arange(100), 2))
    got = fit(df, df["g"], subset=[0, 1, 2])
    assert got.nobs == 6

    # A repeated label in the subset must not multiply the selection.
    again = fit(df, df["g"], subset=[0, 0, 1, 1, 2])
    assert again.nobs == 6
    assert_same_fit(got, again)


def test_duplicate_index_keeps_external_groups_aligned():
    df = frame(index=np.repeat(np.arange(100), 2))
    external = df["g"].to_numpy()          # positional, not indexed
    labels = list(range(60))

    got = fit(df, external, subset=labels)
    # Same rows, subset by hand, groups subset by hand: must agree exactly.
    pos = np.flatnonzero(df.index.isin(labels))
    want = fit(df.iloc[pos].reset_index(drop=True), external[pos])
    assert_same_fit(got, want)


# ---------------------------------------------------------------- missing rows
def test_subset_then_drop_missing():
    df = frame()
    df.loc[df.index[:10], "x"] = np.nan
    got = fit(df, df["g"], subset=np.arange(100), missing="drop")
    assert got.nobs == 90


def test_missing_group_label_inside_a_subset():
    df = frame()
    df.loc[df.index[:5], "g"] = np.nan
    got = fit(df, df["g"], subset=np.arange(100), missing="drop")
    assert got.nobs == 95
    assert got.model.n_groups == 10          # 100 rows / 10 per group, 5 dropped


# ------------------------------------------------------ external group forms
def test_groups_as_a_series_with_a_matching_index():
    df = frame(index=[f"r{i}" for i in range(200)])
    labels = [f"r{i}" for i in range(150)]
    got = fit(df, df["g"], subset=labels)
    want = fit(df.loc[labels], df["g"].loc[labels])
    assert_same_fit(got, want)


def test_groups_as_a_bare_array_is_positional_against_the_unsubset_frame():
    df = frame(index=np.arange(100, 300))
    external = df["g"].to_numpy()
    got = fit(df, external, subset=np.arange(100, 250))
    want = fit(df.iloc[:150].reset_index(drop=True), external[:150])
    assert_same_fit(got, want)


def test_groups_already_subset_by_the_caller_is_accepted():
    df = frame()
    sub = np.arange(150)
    got = fit(df, df["g"].to_numpy()[sub], subset=sub)
    assert got.nobs == 150


def test_groups_of_an_unrelated_length_is_refused():
    df = frame()
    with pytest.raises(ValueError, match="matches neither"):
        fit(df, np.zeros(37), subset=np.arange(150))


def test_unknown_group_column_names_the_columns():
    df = frame()
    with pytest.raises(ValueError, match="not a column"):
        fit(df, "no_such_column")


# ------------------------------------------------------------- empty / shapes
def test_empty_subset_is_refused():
    df = frame()
    with pytest.raises(ValueError, match="selected no rows"):
        fit(df, df["g"], subset=np.zeros(len(df), dtype=bool))
    with pytest.raises(ValueError, match="selected no rows"):
        fit(df, df["g"], subset=[])


def test_two_dimensional_subset_is_refused():
    df = frame()
    with pytest.raises(ValueError, match="one-dimensional"):
        fit(df, df["g"], subset=np.arange(100).reshape(50, 2))


def test_subset_matches_a_hand_built_model():
    """End to end: the subset path and the array path must agree exactly."""
    df = frame(index=[f"k{i}" for i in range(200)])
    labels = [f"k{i}" for i in range(0, 200, 2)]
    got = fit(df, df["g"], subset=labels)

    sub = df.loc[labels]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = mlm.MixedLM(
            sub["y"].to_numpy(),
            np.column_stack([np.ones(len(sub)), sub["x"].to_numpy()]),
            sub["g"].to_numpy(),
        ).fit()
    assert_same_fit(got, want)

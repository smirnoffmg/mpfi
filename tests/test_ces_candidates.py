"""Candidate column disturbances for the CES catalogue, screened in E73.

Each one is a data error from the data-quality literature (the taxonomy note, L14),
turned into a disturbance of one column: what it corrupts, how many rows, and when it
does not apply are pinned down here before any run.
"""

import numpy as np
import pandas as pd
import pytest

from mpfi.ces import (
    CANDIDATE_SHARE,
    CANDIDATES,
    PERTURBATIONS,
    PerturbationNotApplicable,
    category_typo,
    column_swap,
    monotone_scale,
    outliers,
    partial_missing,
    partial_missing_mnar,
    precision_loss,
    sentinel,
    slice_offset,
    unit_error,
)

N = 1000


@pytest.fixture
def frame():
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "x": rng.normal(50.0, 10.0, N),
            "y": rng.normal(0.0, 1.0, N),
            "brand": rng.choice(["acme", "Bolt", "corp"], N),
            "tag": rng.choice(["p", "q"], N),
        }
    )


def changed(before: pd.Series, after: pd.Series) -> int:
    both_missing = before.isna() & after.isna()
    return int(((before != after) & ~both_missing).sum())


def rng():
    return np.random.default_rng(0)


def test_the_candidates_are_the_preregistered_list_in_its_order():
    names = [perturb.__name__ for perturb in CANDIDATES]

    assert names == [
        "partial_missing",
        "partial_missing_mnar",
        "sentinel",
        "outliers",
        "unit_error",
        "slice_offset",
        "precision_loss",
        "column_swap",
        "category_typo",
        "monotone_scale",
    ]
    assert CANDIDATE_SHARE == 0.1


def test_the_protocol_catalogue_is_untouched():
    assert [p.__name__ for p in PERTURBATIONS] == [
        "mean_imputation",
        "zero_fill",
        "gaussian_noise",
    ]


@pytest.mark.parametrize("disturb", CANDIDATES)
def test_no_candidate_touches_the_original_or_another_column(frame, disturb):
    original = frame.copy()
    column = "brand" if disturb is category_typo else "x"

    result = disturb(frame, column, rng(), others=["x", "y", "brand", "tag"])

    pd.testing.assert_frame_equal(frame, original)
    untouched = [c for c in frame.columns if c not in (column, "y")]
    pd.testing.assert_frame_equal(result[untouched], frame[untouched])


@pytest.mark.parametrize("disturb", CANDIDATES)
def test_the_same_generator_gives_the_same_corruption(frame, disturb):
    column = "brand" if disturb is category_typo else "x"
    others = ["x", "y", "brand", "tag"]

    first = disturb(frame, column, np.random.default_rng(3), others=others)
    again = disturb(frame, column, np.random.default_rng(3), others=others)

    pd.testing.assert_frame_equal(first, again)


def test_partial_missing_empties_a_tenth_of_rows_at_random(frame):
    result = partial_missing(frame, "x", rng())

    assert result["x"].isna().sum() == 100
    gone = result["x"].isna()
    assert frame.loc[gone, "x"].mean() == pytest.approx(frame["x"].mean(), abs=3.0)


def test_partial_missing_also_empties_a_categorical_column(frame):
    result = partial_missing(frame, "brand", rng())

    assert result["brand"].isna().sum() == 100


def test_mnar_missingness_takes_the_largest_values(frame):
    result = partial_missing_mnar(frame, "x", rng())

    gone = result["x"].isna()
    assert gone.sum() == 100
    assert frame.loc[gone, "x"].min() >= frame.loc[~gone, "x"].max()


def test_mnar_missingness_needs_a_number(frame):
    with pytest.raises(PerturbationNotApplicable):
        partial_missing_mnar(frame, "brand", rng())


def test_a_sentinel_stands_in_for_a_tenth_of_the_values(frame):
    numeric = sentinel(frame, "x", rng())
    categorical = sentinel(frame, "brand", rng())

    assert (numeric["x"] == -999).sum() == 100
    assert (categorical["brand"] == "?").sum() == 100


def test_a_sentinel_joins_the_categories_of_a_pandas_categorical(frame):
    typed = frame.assign(brand=frame["brand"].astype("category"))

    result = sentinel(typed, "brand", rng())

    assert isinstance(result["brand"].dtype, pd.CategoricalDtype)
    assert (result["brand"] == "?").sum() == 100


def test_outliers_sit_ten_spreads_from_the_median(frame):
    result = outliers(frame, "x", rng())

    moved = result["x"] != frame["x"]
    assert moved.sum() == 100
    distance = (result.loc[moved, "x"] - frame["x"].median()).abs()
    np.testing.assert_allclose(distance, 10 * frame["x"].std())
    assert (result.loc[moved, "x"] > frame["x"].median()).sum() not in (0, 100)


def test_a_unit_error_multiplies_a_tenth_of_rows_by_a_power_of_ten(frame):
    result = unit_error(frame, "x", rng())

    moved = result["x"] != frame["x"]
    assert moved.sum() == 100
    factors = (result.loc[moved, "x"] / frame.loc[moved, "x"]).round(6)
    assert set(factors) == {10.0, 100.0, 1000.0}


def test_a_slice_offset_shifts_rows_where_the_next_column_is_in_its_top_decile(frame):
    result = slice_offset(frame, "x", rng(), others=["x", "y", "brand"])

    moved = result["x"] != frame["x"]
    assert moved.equals(frame["y"] >= frame["y"].quantile(0.9))
    np.testing.assert_allclose(
        result.loc[moved, "x"] - frame.loc[moved, "x"], frame["x"].std()
    )


def test_a_slice_offset_needs_another_numeric_column(frame):
    with pytest.raises(PerturbationNotApplicable):
        slice_offset(frame, "x", rng(), others=["x", "brand"])


def test_precision_loss_keeps_two_significant_digits(frame):
    values = pd.DataFrame({"v": [1234.5, 0.012345, -98.76, 0.0, 7.0]})

    result = precision_loss(values, "v", rng())

    np.testing.assert_allclose(result["v"], [1200.0, 0.012, -99.0, 0.0, 7.0])


def test_a_column_swap_exchanges_a_tenth_of_rows_with_the_next_like_column(frame):
    result = column_swap(frame, "x", rng(), others=["x", "brand", "y"])

    moved = result["x"] != frame["x"]
    assert moved.sum() == 100
    pd.testing.assert_series_equal(
        result.loc[moved, "x"], frame.loc[moved, "y"], check_names=False
    )
    pd.testing.assert_series_equal(
        result.loc[moved, "y"], frame.loc[moved, "x"], check_names=False
    )


def test_a_column_swap_wraps_around_to_find_a_partner(frame):
    result = column_swap(frame, "tag", rng(), others=["brand", "x", "tag"])

    assert changed(frame["brand"], result["brand"]) > 0


def test_a_column_swap_without_a_like_partner_does_not_apply(frame):
    with pytest.raises(PerturbationNotApplicable):
        column_swap(frame, "x", rng(), others=["x", "brand"])


def test_a_typo_turns_a_tenth_of_rows_into_categories_never_seen(frame):
    result = category_typo(frame, "brand", rng())

    moved = result["brand"] != frame["brand"]
    assert moved.sum() == 100
    assert not set(result.loc[moved, "brand"]) & set(frame["brand"])
    assert set(result.loc[moved, "brand"]) <= {"ACME~", "bOLT~", "CORP~"}


def test_a_typo_needs_a_categorical_column(frame):
    with pytest.raises(PerturbationNotApplicable):
        category_typo(frame, "x", rng())


def test_the_probe_rescales_the_whole_column(frame):
    result = monotone_scale(frame, "x", rng())

    np.testing.assert_allclose(result["x"], 1000 * frame["x"])


@pytest.mark.parametrize(
    "disturb", [outliers, unit_error, slice_offset, precision_loss, monotone_scale]
)
def test_number_only_candidates_refuse_a_categorical_column(frame, disturb):
    with pytest.raises(PerturbationNotApplicable):
        disturb(frame, "brand", rng(), others=["brand", "tag"])


def test_an_integer_column_stays_integer_unless_it_gains_a_gap_or_a_fraction():
    counts = pd.DataFrame({"n": np.arange(N, dtype="int64")})

    assert sentinel(counts, "n", rng())["n"].dtype == "int64"
    assert unit_error(counts, "n", rng())["n"].dtype == "int64"
    assert partial_missing(counts, "n", rng())["n"].dtype == "float64"
    assert outliers(counts, "n", rng())["n"].dtype == "float64"

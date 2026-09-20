"""CES measures what a local change does to the quality a pipeline reports.

The perturbation is the experiment; these tests pin down what counts as one and
how the shift is turned into a number.
"""

import numpy as np
import pandas as pd
import pytest

from mpfi.ces import (
    ces,
    gaussian_noise,
    mean_imputation,
    model_sensitivity,
    relative_shift,
    zero_fill,
)


@pytest.fixture
def frame():
    return pd.DataFrame({"signal": [1.0, 2.0, 3.0, 4.0], "noise": [0.0, 0.0, 1.0, 1.0]})


def test_no_change_means_no_shift():
    assert relative_shift(0.9, 0.9) == 0.0


def test_the_shift_is_relative_to_the_baseline():
    assert relative_shift(0.8, 0.4) == pytest.approx(0.5)


def test_a_baseline_of_zero_has_no_relative_shift():
    """Dividing by it would report infinite fragility for a useless model."""
    assert relative_shift(0.0, 0.3) == 0.0


def test_mean_imputation_flattens_a_column_to_its_mean(frame):
    result = mean_imputation(frame, "signal")

    assert result["signal"].tolist() == [2.5, 2.5, 2.5, 2.5]
    assert result["noise"].tolist() == frame["noise"].tolist()


def test_zero_fill_empties_a_column(frame):
    assert zero_fill(frame, "signal")["signal"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_noise_is_scaled_to_the_column_and_repeats_with_a_seed(frame):
    first = gaussian_noise(frame, "signal", np.random.default_rng(0))
    again = gaussian_noise(frame, "signal", np.random.default_rng(0))

    assert first["signal"].tolist() == again["signal"].tolist()
    assert first["signal"].tolist() != frame["signal"].tolist()
    assert first["noise"].tolist() == frame["noise"].tolist()


def test_the_original_frame_is_never_touched(frame):
    mean_imputation(frame, "signal")
    zero_fill(frame, "signal")

    assert frame["signal"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_sensitivity_is_zero_when_the_pipeline_ignores_the_data(frame):
    def constant(data, seed):
        return 0.75

    assert model_sensitivity(constant, frame, ["signal"], seeds=[0, 1]) == 0.0


def test_sensitivity_rises_with_a_column_the_pipeline_depends_on(frame):
    def reads_signal(data, seed):
        return float(data["signal"].mean()) / 4

    measured = model_sensitivity(reads_signal, frame, ["signal"], seeds=[0, 1])

    assert measured > 0


def test_ces_is_the_geometric_mean_of_its_halves():
    assert ces(0.4, 0.9) == pytest.approx((0.4 * 0.9) ** 0.5)


def test_ces_needs_both_halves_to_be_high():
    """Sensitivity on one level alone is not a cascade."""
    assert ces(0.9, 0.0) == 0.0


def test_a_real_pipeline_reacts_to_the_column_it_depends_on():
    """The mechanics checked where the answer is known in advance.

    The label is a function of `signal` alone, so disturbing `signal` has to
    cost accuracy and disturbing `decoy` has to leave it alone.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    size = 400
    signal = rng.normal(size=size)
    data = pd.DataFrame({"signal": signal, "decoy": rng.normal(size=size)})
    label = (signal > 0).astype(int)

    def run(frame, seed):
        x_train, x_test, y_train, y_test = train_test_split(
            frame, label, test_size=0.3, random_state=seed
        )
        model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])
        model.fit(x_train, y_train)
        return float(accuracy_score(y_test, model.predict(x_test)))

    seeds = [0, 1, 2]
    on_signal = model_sensitivity(run, data, ["signal"], seeds)
    on_decoy = model_sensitivity(run, data, ["decoy"], seeds)

    assert on_signal > 0.1
    assert on_decoy < 0.05
    assert on_signal > on_decoy * 5

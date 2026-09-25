"""Churn is the primary CES outcome: how many test predictions a disturbance flips.

A summary metric can stand still while predictions change underneath it, so the
harness compares the predictions themselves, a run against its paired baseline.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mpfi.ces import (
    Failure,
    Measurement,
    Prediction,
    Shift,
    label_churn,
    mean_absolute_change,
    measure_model_churn,
    measure_pipeline_churn,
    noise_floor,
    paired_churn,
    probability_churn,
    regression_churn,
    relative_shift,
)


def test_identical_labels_have_no_churn():
    assert label_churn([0, 1, 1, 0], [0, 1, 1, 0]) == 0.0


def test_churn_is_the_share_of_objects_whose_label_changed():
    assert label_churn(["a", "b", "c", "d"], ["a", "b", "x", "y"]) == 0.5


def test_churn_needs_the_same_test_objects():
    with pytest.raises(ValueError):
        label_churn([0, 1, 1], [0, 1])


def test_churn_sees_what_an_unchanged_metric_hides():
    """Errors that cancel leave accuracy still while every prediction flips."""
    truth = np.array([0, 1, 0, 1])
    before = np.array([0, 1, 1, 0])
    after = np.array([1, 0, 0, 1])
    accuracy_before = float((before == truth).mean())
    accuracy_after = float((after == truth).mean())

    assert relative_shift(accuracy_before, accuracy_after) == 0.0
    assert label_churn(before, after) == 1.0


def test_probability_churn_counts_risks_that_moved_past_the_threshold():
    before = [0.10, 0.50, 0.90, 0.30]
    after = [0.12, 0.70, 0.60, 0.30]

    assert probability_churn(before, after, delta=0.1) == 0.5


def test_probability_churn_takes_the_class_that_moved_most():
    before = [[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]]
    after = [[0.6, 0.35, 0.05], [0.2, 0.5, 0.3]]

    assert probability_churn(before, after, delta=0.1) == 0.5


def test_regression_churn_counts_moves_beyond_a_share_of_the_target_spread():
    before = [1.0, 2.0, 3.0, 4.0]
    after = [1.05, 2.5, 3.0, 2.0]

    assert regression_churn(before, after, scale=2.0, tolerance=0.1) == 0.5


def test_regression_churn_without_tolerance_counts_any_change():
    assert regression_churn([1.0, 2.0], [1.0, 2.001], scale=1.0, tolerance=0.0) == 0.5


def test_regression_churn_needs_a_target_that_varies():
    with pytest.raises(ValueError):
        regression_churn([1.0, 2.0], [1.0, 3.0], scale=0.0)


def test_mean_absolute_change_is_in_units_of_the_target_spread():
    assert mean_absolute_change([1.0, 2.0], [2.0, 4.0], scale=3.0) == pytest.approx(0.5)


def test_the_noise_floor_compares_model_seeds_on_one_split():
    split_a = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 1])]
    split_b = [np.array([1, 1, 1, 1]), np.array([0, 0, 1, 1])]

    assert noise_floor([split_a, split_b], label_churn) == pytest.approx(0.375)


def test_the_noise_floor_uses_every_pair_of_model_seeds():
    one_split = [np.array([0, 0]), np.array([0, 1]), np.array([1, 1])]

    assert noise_floor([one_split], label_churn) == 0.5


def test_the_noise_floor_needs_two_model_seeds():
    with pytest.raises(ValueError):
        noise_floor([[np.array([0, 1])]], label_churn)


def test_paired_churn_compares_runs_with_the_same_seeds_only():
    baseline = {(0, 0): np.array([0, 0]), (1, 0): np.array([1, 1])}
    perturbed = {(0, 0): np.array([0, 1]), (1, 0): np.array([1, 1])}

    assert paired_churn(baseline, perturbed, label_churn) == pytest.approx(0.25)


def test_paired_churn_refuses_runs_that_do_not_pair():
    with pytest.raises(ValueError):
        paired_churn({(0, 0): np.array([0])}, {(1, 0): np.array([0])}, label_churn)


def test_the_noise_floor_is_subtracted_unless_asked_for_raw():
    measured = Measurement(
        baseline=float("nan"),
        shifts=(Shift("a", 0.3), Shift("b", 0.5)),
        failures=(Failure("c", "boom"),),
        noise_floor=0.2,
    )

    assert measured.values() == pytest.approx([0.1, 0.3, 1.0])
    assert measured.values(noise="raw") == pytest.approx([0.3, 0.5, 1.0])
    assert measured.scores(noise="raw")["max"] == 1.0


def test_a_metric_measurement_has_no_floor_to_subtract():
    measured = Measurement(baseline=0.8, shifts=(Shift("a", 0.3),), failures=())

    assert measured.values() == [0.3]


@pytest.fixture
def frame():
    return pd.DataFrame({"signal": [1.0, 2.0, 3.0, 4.0], "decoy": [0.0, 0.0, 1.0, 1.0]})


def test_a_pipeline_that_ignores_a_column_does_not_churn_on_it(frame):
    def run(data, model_seed, split_seed):
        return Prediction(np.array([model_seed % 2, 1, 0, 1]))

    outcomes = measure_model_churn(
        run, frame, ["signal"], model_seeds=[0, 1], split_seeds=[0]
    )

    assert outcomes.churn.values(noise="raw") == [0.0, 0.0, 0.0]
    assert outcomes.churn.noise_floor == 0.25
    assert outcomes.metric is None


def test_baseline_and_disturbed_runs_are_paired_by_both_seeds(frame):
    calls = []

    def run(data, model_seed, split_seed):
        calls.append((model_seed, split_seed))
        return Prediction((data["signal"] > 2.5).to_numpy().astype(int))

    measure_model_churn(
        run,
        frame,
        ["signal"],
        model_seeds=[0, 1],
        split_seeds=[5, 6],
        perturbations=[lambda data, column, rng: data],
    )

    pairs = [(0, 5), (0, 6), (1, 5), (1, 6)]
    assert calls == pairs + pairs


def test_churn_rises_with_a_column_the_predictions_depend_on(frame):
    def run(data, model_seed, split_seed):
        return Prediction((data["signal"] > 2.5).to_numpy().astype(int))

    outcomes = measure_model_churn(
        run, frame, ["signal", "decoy"], model_seeds=[0, 1], split_seeds=[0]
    )
    by_name = {shift.perturbation: shift.shift for shift in outcomes.churn.shifts}

    assert by_name["zero_fill(signal)"] == 0.5
    assert by_name["zero_fill(decoy)"] == 0.0


def test_the_metric_stays_as_a_secondary_outcome(frame):
    def run(data, model_seed, split_seed):
        labels = (data["signal"] > 2.5).to_numpy().astype(int)
        return Prediction(labels, metric=0.5 + 0.1 * float(labels.sum()))

    outcomes = measure_model_churn(
        run,
        frame,
        ["signal"],
        model_seeds=[0, 1],
        split_seeds=[0],
        perturbations=[lambda data, column, rng: data.assign(signal=0.0)],
    )

    assert outcomes.metric is not None
    assert outcomes.metric.baseline == pytest.approx(0.7)
    assert outcomes.metric.shifts[0].shift == pytest.approx(0.2 / 0.7)


def test_a_disturbance_that_breaks_the_run_is_a_churn_failure(frame):
    def run(data, model_seed, split_seed):
        if (data["signal"] == 0).all():
            raise ValueError("no signal left")
        return Prediction(np.zeros(len(data)))

    outcomes = measure_model_churn(
        run, frame, ["signal"], model_seeds=[0, 1], split_seeds=[0]
    )

    assert [f.perturbation for f in outcomes.churn.failures] == ["zero_fill(signal)"]


def test_predictions_on_other_test_objects_are_skipped_not_scored(frame):
    def run(data, model_seed, split_seed):
        return Prediction(np.zeros(len(data.dropna())))

    outcomes = measure_model_churn(
        run,
        frame,
        ["signal"],
        model_seeds=[0, 1],
        split_seeds=[0],
        perturbations=[lambda data, column, rng: data.iloc[:2]],
    )

    assert outcomes.churn.shifts == ()
    assert outcomes.churn.failures == ()
    assert outcomes.churn.skipped == ("<lambda>(signal)",)


def test_pipeline_churn_is_measured_over_its_stages():
    def build():
        return Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(C=1.0))]
        )

    def run(pipe, data, model_seed, split_seed):
        flipped = "scale" not in pipe.named_steps
        return Prediction(np.array([1, 0, 1, 0]) if flipped else np.zeros(4))

    data = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    outcomes = measure_pipeline_churn(
        run, build, data, model_seeds=[0, 1], split_seeds=[0]
    )
    by_name = {shift.perturbation: shift.shift for shift in outcomes.churn.shifts}

    assert by_name["drop:scale"] == 0.5
    assert by_name["clf.C×0.8"] == 0.0
    assert outcomes.churn.noise_floor == 0.0


def test_the_floor_keeps_a_seed_sensitive_model_from_reading_as_fragile():
    """A forest retrained on a disturbed decoy draws different trees.

    Its predictions then differ about as much as two seeds of the baseline do,
    and only the excess over that floor speaks of the column.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    from mpfi.ces import gaussian_noise

    rng = np.random.default_rng(0)
    size = 600
    signal = rng.normal(size=size)
    data = pd.DataFrame({"signal": signal, "decoy": rng.normal(size=size)})
    label = (signal + rng.normal(scale=0.5, size=size) > 0).astype(int)

    def run(frame, model_seed, split_seed):
        x_train, x_test, y_train, _ = train_test_split(
            frame, label, test_size=0.3, random_state=split_seed
        )
        model = RandomForestClassifier(n_estimators=20, random_state=model_seed)
        return Prediction(model.fit(x_train, y_train).predict(x_test))

    seeds = {"model_seeds": [0, 1, 2], "split_seeds": [0, 1]}
    on_decoy = measure_model_churn(
        run, data, ["decoy"], perturbations=[gaussian_noise], **seeds
    ).churn
    on_signal = measure_model_churn(
        run, data, ["signal"], perturbations=[gaussian_noise], **seeds
    ).churn

    assert on_decoy.score(noise="raw") > 0.02
    assert abs(on_decoy.score()) < on_decoy.score(noise="raw") / 2
    assert on_signal.score() > abs(on_decoy.score()) * 3

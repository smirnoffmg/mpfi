"""PPS disturbs a stage of the pipeline rather than the data going into it.

Three families, following the design: the preprocessing, the feature
engineering, and the estimator's own settings.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from mpfi.ces import (
    drop_step,
    pipeline_propagation,
    replace_scaler,
    scale_hyperparameter,
    swap_steps,
)


@pytest.fixture
def pipeline():
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("shrink", MinMaxScaler()),
            ("clf", LogisticRegression(C=1.0)),
        ]
    )


def names(pipe):
    return [name for name, _ in pipe.steps]


def test_dropping_a_step_leaves_the_others(pipeline):
    result = drop_step(pipeline, "shrink")

    assert names(result) == ["scale", "clf"]
    assert names(pipeline) == ["scale", "shrink", "clf"]


def test_dropping_the_estimator_is_refused(pipeline):
    """Without the last step there is nothing to score."""
    assert drop_step(pipeline, "clf") is None


def test_swapping_reorders_two_neighbours(pipeline):
    result = swap_steps(pipeline, 0)

    assert names(result) == ["shrink", "scale", "clf"]
    assert names(pipeline) == ["scale", "shrink", "clf"]


def test_replacing_the_scaler_keeps_the_shape_and_changes_the_kind(pipeline):
    result = replace_scaler(pipeline)

    assert names(result) == names(pipeline)
    assert type(result.steps[0][1]) is not type(pipeline.steps[0][1])


def test_a_hyperparameter_moves_by_the_given_share(pipeline):
    result = scale_hyperparameter(pipeline, "clf", "C", 1.2)

    assert result.named_steps["clf"].C == pytest.approx(1.2)
    assert pipeline.named_steps["clf"].C == 1.0


def test_propagation_is_zero_when_the_stages_do_not_matter():
    def build():
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])

    def run(pipe, frame, seed):
        return 0.8

    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    assert pipeline_propagation(run, build, frame, seeds=[0]) == 0.0


def test_propagation_is_higher_for_a_pipeline_that_needs_its_stages():
    """PPS is a median over perturbations, so it is read by comparison.

    One pipeline has a feature a thousand times the other and a distance-based
    estimator, so its stages carry the result; the other works on features that
    are already comparable and barely notices.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(0)
    size = 300
    small = rng.normal(size=size)
    label = (small > 0).astype(int)
    lopsided = pd.DataFrame({"small": small, "huge": rng.normal(size=size) * 1000})
    comparable = pd.DataFrame({"small": small, "other": rng.normal(size=size)})

    def build():
        return Pipeline(
            [("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]
        )

    def run(pipe, data, seed):
        x_train, x_test, y_train, y_test = train_test_split(
            data, label, test_size=0.3, random_state=seed
        )
        pipe.fit(x_train, y_train)
        return float(accuracy_score(y_test, pipe.predict(x_test)))

    seeds = [0, 1]
    assert pipeline_propagation(run, build, lopsided, seeds) > pipeline_propagation(
        run, build, comparable, seeds
    )

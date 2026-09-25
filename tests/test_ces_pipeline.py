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


def test_every_variant_carries_a_name_and_the_steps_it_touches(pipeline):
    from mpfi.ces import named_pipeline_perturbations

    variants = named_pipeline_perturbations(pipeline)

    assert "drop:scale" in [variant.name for variant in variants]
    drop = next(v for v in variants if v.name == "drop:scale")
    assert drop.steps == ("scale",)
    assert names(drop.pipeline) == ["shrink", "clf"]


def test_trees_get_a_leaf_setting_moved_as_well():
    from sklearn.ensemble import GradientBoostingClassifier

    from mpfi.ces import named_pipeline_perturbations

    trees = Pipeline(
        [("scale", StandardScaler()), ("clf", GradientBoostingClassifier(max_depth=3))]
    )
    moved = [v.name for v in named_pipeline_perturbations(trees)]

    assert "clf.max_depth×0.8" in moved
    assert "clf.min_samples_leaf×1.2" in moved


def _booster(class_name: str, **defaults: object) -> type:
    """A stand-in with a booster's class name and settings; cloned, never fit."""
    from sklearn.base import BaseEstimator

    def __init__(self: object, **params: object) -> None:
        for key, value in {**defaults, **params}.items():
            setattr(self, key, value)

    def get_params(self: object, deep: bool = True) -> dict[str, object]:
        return {key: getattr(self, key) for key in defaults}

    return type(
        class_name,
        (BaseEstimator,),
        {"__init__": __init__, "get_params": get_params},
    )


def _leaf_variants(model: object) -> list[str]:
    from mpfi.ces import named_pipeline_perturbations

    trees = Pipeline([("clf", model)])
    return [
        v.name
        for v in named_pipeline_perturbations(trees)
        if v.pipeline is not None and "×" in v.name
    ]


def test_lightgbm_moves_min_child_samples_rather_than_the_hessian_floor():
    LGBMRegressor = _booster(
        "LGBMRegressor", max_depth=3, min_child_weight=0.001, min_child_samples=20
    )

    moved = _leaf_variants(LGBMRegressor())

    assert "clf.min_child_samples×0.8" in moved
    assert "clf.min_child_samples×1.2" in moved
    assert not any("min_child_weight" in name for name in moved)


def test_xgboost_moves_min_child_weight_from_its_library_default():
    from mpfi.ces import named_pipeline_perturbations

    XGBClassifier = _booster("XGBClassifier", max_depth=4, min_child_weight=None)
    variants = named_pipeline_perturbations(Pipeline([("clf", XGBClassifier())]))
    down = next(v for v in variants if v.name == "clf.min_child_weight×0.8")

    assert down.pipeline is not None
    assert down.pipeline.named_steps["clf"].min_child_weight == pytest.approx(0.8)


def test_catboost_leaf_size_is_moved_only_where_its_grow_policy_uses_it():
    CatBoostRegressor = _booster(
        "CatBoostRegressor", depth=6, min_data_in_leaf=None, grow_policy=None
    )

    assert not any("min_data_in_leaf" in n for n in _leaf_variants(CatBoostRegressor()))
    assert "clf.min_data_in_leaf×1.2" in _leaf_variants(
        CatBoostRegressor(grow_policy="Depthwise")
    )


def test_an_unknown_model_keeps_the_first_leaf_setting_it_has():
    Custom = _booster("CustomTrees", max_depth=3, min_child_weight=0.5)

    assert "clf.min_child_weight×1.2" in _leaf_variants(Custom())


def test_a_whole_number_setting_moves_by_at_least_one():
    """min_samples_leaf=1 times 1.2 rounds back to 1 and would disturb nothing."""
    from sklearn.ensemble import GradientBoostingClassifier

    trees = Pipeline([("clf", GradientBoostingClassifier(min_samples_leaf=1))])

    up = scale_hyperparameter(trees, "clf", "min_samples_leaf", 1.2)

    assert up.named_steps["clf"].min_samples_leaf == 2
    assert scale_hyperparameter(trees, "clf", "min_samples_leaf", 0.8) is None


def test_a_feature_can_be_dropped_right_before_the_model():
    from mpfi.ces import named_pipeline_perturbations

    variants = named_pipeline_perturbations(
        Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())]),
        drop_columns=["b"],
    )
    dropped = next(v for v in variants if v.name == "drop_feature:b")
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0, 0.0]})
    label = [0, 0, 1, 1]

    dropped.pipeline.fit(frame, label)

    assert list(dropped.pipeline.named_steps["clf"].feature_names_in_) == ["a"]


def test_a_feature_that_cannot_be_found_is_not_applicable_rather_than_fragile():
    from mpfi.ces import DropColumn, PerturbationNotApplicable

    with pytest.raises(PerturbationNotApplicable):
        DropColumn("b").fit(np.zeros((3, 2))).transform(np.zeros((3, 2)))


def test_the_encoding_of_categories_is_swapped():
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

    from mpfi.ces import replace_encoder

    one_hot = Pipeline([("encode", OneHotEncoder()), ("clf", LogisticRegression())])

    result = replace_encoder(one_hot)

    assert isinstance(result.named_steps["encode"], OrdinalEncoder)
    assert replace_encoder(Pipeline([("clf", LogisticRegression())])) is None


def test_a_scaler_inside_a_column_transformer_keeps_its_columns():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import RobustScaler

    scoped = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [("num", RobustScaler(), ["age", "bmi"])], remainder="passthrough"
                ),
            ),
            ("clf", LogisticRegression()),
        ]
    )

    result = replace_scaler(scoped)

    num = dict((n, (t, c)) for n, t, c in result.named_steps["prep"].transformers)
    assert type(num["num"][0]) is StandardScaler
    assert num["num"][1] == ["age", "bmi"]


def test_a_scaler_the_harness_cannot_see_into_is_left_alone():
    """A home-made step named *Scaler may be scoped to a few columns inside."""
    from sklearn.base import BaseEstimator, TransformerMixin

    class ColumnScaler(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    assert (
        replace_scaler(
            Pipeline([("scale", ColumnScaler()), ("clf", LogisticRegression())])
        )
        is None
    )


class Resampler:
    """Stands in for an imblearn sampler: all the harness looks at is fit_resample."""

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_resample(self, X, y):
        return X, y


def test_variants_that_touch_a_sampler_are_evaluation_not_pipeline():
    from mpfi.ces import named_pipeline_perturbations

    sampled = Pipeline(
        [
            ("scale", StandardScaler()),
            ("smote", Resampler()),
            ("clf", LogisticRegression()),
        ]
    )
    kinds = {v.name: v.kind for v in named_pipeline_perturbations(sampled)}

    assert kinds["drop:smote"] == "evaluation"
    assert kinds["drop:scale"] == "pipeline"


def test_evaluation_variants_stay_out_of_pps_but_are_reported():
    from mpfi.ces import measure_pipeline_propagation

    def build():
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("smote", Resampler()),
                ("clf", LogisticRegression()),
            ]
        )

    def run(pipe, frame, seed):
        return 0.1 if "smote" not in pipe.named_steps else 0.8

    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    measured = measure_pipeline_propagation(run, build, frame, seeds=[0])

    assert measured.score(max) == 0.0
    assert [s.perturbation for s in measured.evaluation] == ["drop:smote"]
    assert measured.evaluation[0].shift == pytest.approx(0.875)


def test_propagation_records_a_variant_that_breaks():
    from mpfi.ces import measure_pipeline_propagation

    def build():
        return Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(C=1.0))]
        )

    def run(pipe, frame, seed):
        if "scale" not in pipe.named_steps:
            raise RuntimeError("needs scaling")
        return 0.8

    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    measured = measure_pipeline_propagation(run, build, frame, seeds=[0])

    assert [f.perturbation for f in measured.failures] == ["drop:scale"]
    assert measured.score(max, failures="count_as_one") == 1.0

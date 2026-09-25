"""CES — Cascade Effect Score: what a local change does to a pipeline's behaviour.

An empirical proxy for CACE, "changing anything changes everything". A pipeline
is run once as it stands, then again with one thing disturbed, and the distance
between the two runs is the measurement. MSS disturbs the data a column at a
time; PPS disturbs a stage of the pipeline. CES is their geometric mean, so that
sensitivity on one level alone does not pass for a cascade.

The primary outcome is prediction churn: the share of test objects whose
prediction changed between a baseline run and a disturbed run with the same
model and split seeds, less the churn two baseline runs show when only the model
seed differs. A run is `run(frame, model_seed, split_seed) -> Prediction`
(`run(pipeline, frame, model_seed, split_seed)` for PPS), measured by
`measure_model_churn` and `measure_pipeline_churn`; `label_churn`,
`probability_churn`, `regression_churn`, `noise_floor` and `paired_churn` work
on stored predictions as well.

The secondary outcome is the relative shift of the quality the pipeline reports
— accuracy, F1, RMSE. It comes from the same runs when a `Prediction` carries a
`metric`, or from the original harness with `run(frame, seed) -> float`:
`measure_model_sensitivity`, `measure_pipeline_propagation`.
"""

import itertools
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline

# Noise stays small enough to be a disturbance rather than a different dataset.
NOISE_SCALE = 0.1

# Moving a knob that only changes how fast a tree is built says nothing about
# fragility, so the settings that carry the model's behaviour come first.
INFLUENTIAL_SETTINGS = (
    "n_neighbors",
    "C",
    "alpha",
    "max_depth",
    "n_estimators",
    "learning_rate",
    "min_samples_leaf",
    "num_leaves",
    "reg_lambda",
)

# Settings that decide how small a leaf may get. Trees are indifferent to scaling,
# so for boosted and random forests these are the knobs a pipeline author tunes.
# Known libraries are matched by type in `leaf_setting`; this order is only for
# a model none of them claims.
LEAF_SETTINGS = (
    "min_samples_leaf",
    "min_child_weight",
    "min_child_samples",
    "min_data_in_leaf",
)

# LightGBM prefers its primary name to any alias; the wrapper always carries
# min_child_samples, so an alias the author passed explicitly is looked at first.
# min_child_weight there is a Hessian floor of 1e-3 and moving it changes nothing.
# https://lightgbm.readthedocs.io/en/latest/Parameters.html#min_data_in_leaf
_LIGHTGBM_LEAF = (
    "min_data_in_leaf",
    "min_child_samples",
    "min_data_per_leaf",
    "min_data",
    "min_samples_leaf",
)
# The wrapper leaves min_child_weight at None and the library then uses 1.
# https://xgboost.readthedocs.io/en/stable/parameter.html
_XGBOOST_MIN_CHILD_WEIGHT = 1.0
# min_data_in_leaf (alias min_child_samples, default 1) works only under these
# grow policies; the default SymmetricTree ignores it.
# https://catboost.ai/docs/en/references/training-parameters/common#min_data_in_leaf
_CATBOOST_LEAF_POLICIES = ("Depthwise", "Lossguide")
_CATBOOST_MIN_DATA_IN_LEAF = 1

Run = Callable[["pd.DataFrame", int], float]
Aggregate = Callable[[list[float]], float]
Perturbation = Callable[["pd.DataFrame", str, "np.random.Generator"], "pd.DataFrame"]

AGGREGATES: dict[str, Aggregate] = {
    "median": statistics.median,
    "mean": statistics.fmean,
    "max": max,
}

FailureRule = Literal["count_as_one", "exclude"]
# Not agreed with the supervisor; to be decided 26.09.2026. A run that breaks under
# a disturbance has lost all its quality, which is what a shift of 1.0 says; leaving
# it out would read the most fragile pipelines as the soundest.
DEFAULT_FAILURE_RULE: FailureRule = "count_as_one"

EvaluationRule = Literal["separate", "exclude", "include"]
# Not agreed with the supervisor; to be decided 26.09.2026. A sampler changes what
# the model is trained and, when the author resamples before splitting, scored on:
# its disturbance measures the evaluation procedure, not the pipeline.
DEFAULT_EVALUATION_RULE: EvaluationRule = "separate"

NoiseRule = Literal["subtract", "raw"]
# Не согласовано, решение 26.09. Retraining a seed-sensitive model on disturbed data
# changes its random draws as well, so even an irrelevant disturbance flips about
# as many predictions as a second seed does; only the excess speaks of the pipeline.
# A ratio would be undefined for a deterministic model, whose floor is zero.
DEFAULT_NOISE_RULE: NoiseRule = "subtract"

# Не согласовано, решение 26.09. A regression prediction counts as changed when it
# moves by more than this share of the target's standard deviation.
REGRESSION_TOLERANCE = 0.1
# Не согласовано, решение 26.09. A predicted probability counts as changed when it
# moves by more than this, after the δ of Watson-Daniels et al. (2023).
PROBABILITY_DELTA = 0.1


class PerturbationNotApplicable(ValueError):
    """The disturbance cannot be made on this pipeline; it says nothing about it."""


@dataclass(frozen=True)
class Shift:
    perturbation: str
    shift: float


@dataclass(frozen=True)
class Failure:
    perturbation: str
    error: str


@dataclass(frozen=True)
class Measurement:
    """Everything one level of CES observed, before it is folded into a number.

    Kept whole so that a report can show how much the number depends on the
    aggregation and the failure rule — both still open in the protocol.
    """

    baseline: float
    shifts: tuple[Shift, ...]
    failures: tuple[Failure, ...]
    evaluation: tuple[Shift, ...] = ()
    evaluation_failures: tuple[Failure, ...] = ()
    skipped: tuple[str, ...] = ()
    noise_floor: float | None = None

    def values(
        self,
        failures: FailureRule = DEFAULT_FAILURE_RULE,
        noise: NoiseRule = DEFAULT_NOISE_RULE,
    ) -> list[float]:
        """Observed shifts, less the noise floor when there is one.

        A failure stays at 1.0 either way: a run that breaks has lost every
        prediction, whatever the seeds alone would have changed. The difference
        is not clipped at zero, so that null effects scatter around it instead
        of being pushed above it.
        """
        floor = self.noise_floor if noise == "subtract" and self.noise_floor else 0.0
        observed = [shift.shift - floor for shift in self.shifts]
        if failures == "count_as_one":
            observed += [1.0] * len(self.failures)
        return observed

    def score(
        self,
        aggregate: Aggregate = statistics.median,
        failures: FailureRule = DEFAULT_FAILURE_RULE,
        noise: NoiseRule = DEFAULT_NOISE_RULE,
    ) -> float:
        """The median by default, as the design specifies; not agreed as final."""
        observed = self.values(failures, noise)
        return aggregate(observed) if observed else 0.0

    def scores(
        self,
        failures: FailureRule = DEFAULT_FAILURE_RULE,
        noise: NoiseRule = DEFAULT_NOISE_RULE,
    ) -> dict[str, float]:
        return {
            name: self.score(aggregate, failures, noise)
            for name, aggregate in AGGREGATES.items()
        }


def fix_split(run: Callable[..., float], split_seed: int) -> Callable[..., float]:
    """Hold the split still so that the seed moves only the model.

    The protocol randomises the split and the row order along with the model,
    after Bouthillier et al. Fixing the split is a departure from it, useful where
    a shift is smaller than the spread the split alone produces. The wrapped run
    takes the split seed as its last argument.
    """

    def fixed(*args: Any) -> float:
        return run(*args, split_seed)

    return fixed


def _describe(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def relative_shift(baseline: float, perturbed: float) -> float:
    """How far quality moved, as a share of where it started.

    A baseline of zero has no share to speak of; reporting infinite fragility
    for a model that was already useless would say nothing about the pipeline.
    """
    if baseline == 0:
        return 0.0
    return abs(baseline - perturbed) / abs(baseline)


def mean_imputation(
    frame: "pd.DataFrame", column: str, rng: "np.random.Generator | None" = None
) -> "pd.DataFrame":
    """The column loses everything but its average."""
    disturbed = frame.copy()
    disturbed[column] = frame[column].mean()
    return disturbed


def zero_fill(
    frame: "pd.DataFrame", column: str, rng: "np.random.Generator | None" = None
) -> "pd.DataFrame":
    disturbed = frame.copy()
    disturbed[column] = 0
    return disturbed


def gaussian_noise(
    frame: "pd.DataFrame", column: str, rng: "np.random.Generator | None" = None
) -> "pd.DataFrame":
    """Noise scaled to the column's own spread, so every column is disturbed alike."""
    import numpy as np

    generator = rng if rng is not None else np.random.default_rng()
    spread = float(frame[column].std())
    if spread == 0:
        spread = 1.0
    disturbed = frame.copy()
    disturbed[column] = frame[column] + generator.normal(
        0, NOISE_SCALE * spread, len(frame)
    )
    return disturbed


PERTURBATIONS: tuple[Perturbation, ...] = (mean_imputation, zero_fill, gaussian_noise)


def measure_model_sensitivity(
    run: Run,
    frame: "pd.DataFrame",
    columns: Iterable[str],
    seeds: Iterable[int],
    perturbations: Iterable[Perturbation] = PERTURBATIONS,
) -> Measurement:
    """MSS observations: every column disturbed every way, each over every seed.

    A run that breaks under a disturbance is recorded, not raised: the break is
    the strongest fragility there is. A disturbance that cannot be made at all —
    the mean of a text column — is skipped, since it says nothing of the pipeline.
    """
    import numpy as np

    seeds = list(seeds)
    baseline = statistics.median(run(frame, seed) for seed in seeds)

    shifts: list[Shift] = []
    failures: list[Failure] = []
    skipped: list[str] = []
    for column in columns:
        for perturb in perturbations:
            name = f"{perturb.__name__}({column})"
            try:
                disturbed_frames = [
                    perturb(frame, column, np.random.default_rng(seed))
                    for seed in seeds
                ]
            except Exception:
                skipped.append(name)
                continue
            try:
                disturbed = statistics.median(
                    run(data, seed) for data, seed in zip(disturbed_frames, seeds)
                )
            except PerturbationNotApplicable:
                skipped.append(name)
                continue
            except Exception as error:
                failures.append(Failure(name, _describe(error)))
                continue
            shifts.append(Shift(name, relative_shift(baseline, disturbed)))
    return Measurement(baseline, tuple(shifts), tuple(failures), skipped=tuple(skipped))


def model_sensitivity(
    run: Run,
    frame: "pd.DataFrame",
    columns: Iterable[str],
    seeds: Iterable[int],
    perturbations: Iterable[Perturbation] = PERTURBATIONS,
    aggregate: Aggregate = statistics.median,
    failures: FailureRule = DEFAULT_FAILURE_RULE,
) -> float:
    """MSS: the shift over every column disturbed every way, aggregated.

    The median is what the design specifies. It is robust to a single wild
    perturbation, which is also its cost: a pipeline that survives most
    disturbances and collapses under one reads as sound. Sensitivity analysis
    is the place to try the alternatives, hence the argument; the median and the
    failure rule are not agreed with the supervisor yet (decision 26.09.2026).
    """
    measured = measure_model_sensitivity(run, frame, columns, seeds, perturbations)
    return measured.score(aggregate, failures)


def ces(model_sensitivity_score: float, pipeline_propagation_score: float) -> float:
    """The geometric mean penalises a pipeline fragile on one level only."""
    product = model_sensitivity_score * pipeline_propagation_score
    return product**0.5 if product > 0 else 0.0


def drop_step(pipeline: "Pipeline", name: str) -> "Pipeline | None":
    """Remove a transformer. The last step is the estimator and has to stay."""
    from sklearn.base import clone

    steps = list(pipeline.steps)
    if name == steps[-1][0] or name not in {step for step, _ in steps}:
        return None
    kept = [(step, obj) for step, obj in steps if step != name]
    return clone(pipeline).set_params(steps=[(s, clone(o)) for s, o in kept])


def swap_steps(pipeline: "Pipeline", index: int) -> "Pipeline | None":
    """Exchange two neighbouring transformers, leaving the estimator last."""
    from sklearn.base import clone

    steps = list(pipeline.steps)
    if index < 0 or index + 1 >= len(steps) - 1:
        return None
    reordered = list(steps)
    reordered[index], reordered[index + 1] = reordered[index + 1], reordered[index]
    return clone(pipeline).set_params(steps=[(s, clone(o)) for s, o in reordered])


_SCALERS = ("StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler")


def _find_nested(pipeline: "Pipeline", kinds: tuple[type, ...]) -> str | None:
    """The parameter path of the first step of a kind, wherever it sits.

    Walking the deep parameters reaches a step inside a ColumnTransformer or a
    nested pipeline, and replacing it by its path keeps the columns it applies to.
    """
    for key, value in pipeline.get_params(deep=True).items():
        if key != "steps" and isinstance(value, kinds):
            return str(key)
    return None


def replace_scaler(pipeline: "Pipeline") -> "Pipeline | None":
    """Swap one way of putting features on a common scale for another.

    Only scikit-learn scalers the harness can reach are replaced, in place, so a
    scaler scoped to a few columns stays scoped to them. A home-made step that
    scales inside is left alone: replacing it wholesale would also change which
    columns get scaled, and that is a different disturbance.
    """
    from sklearn import preprocessing
    from sklearn.base import clone

    kinds = tuple(getattr(preprocessing, name) for name in _SCALERS)
    copy = clone(pipeline)
    key = _find_nested(copy, kinds)
    if key is None:
        return None
    current = copy.get_params(deep=True)[key]
    other = (
        preprocessing.MinMaxScaler()
        if isinstance(current, preprocessing.StandardScaler)
        else preprocessing.StandardScaler()
    )
    return copy.set_params(**{key: other})


def replace_encoder(pipeline: "Pipeline") -> "Pipeline | None":
    """One-hot categories become ordinal codes, and the other way round."""
    from sklearn.base import clone
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

    copy = clone(pipeline)
    key = _find_nested(copy, (OneHotEncoder, OrdinalEncoder))
    if key is None:
        return None
    current = copy.get_params(deep=True)[key]
    other = (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if isinstance(current, OneHotEncoder)
        else OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    return copy.set_params(**{key: other})


def scale_hyperparameter(
    pipeline: "Pipeline",
    step: str,
    parameter: str,
    factor: float,
    default: float | None = None,
) -> "Pipeline | None":
    """Move one numeric setting of a step, keeping whole numbers whole.

    A whole number moves by at least one: min_samples_leaf=1 times 1.2 rounds back
    to 1 and would disturb nothing. A setting that cannot move is refused.
    `default` stands in for a setting left unset (None) that the library fills.
    """
    from sklearn.base import clone

    copy = clone(pipeline)
    current = copy.named_steps[step].get_params().get(parameter)
    if current is None:
        current = default
    if not isinstance(current, int | float) or isinstance(current, bool):
        return None
    if isinstance(current, int):
        moved: float = round(current * factor)
        if moved == current:
            moved = current + (1 if factor > 1 else -1)
        if moved < 1:
            return None
    else:
        moved = current * factor
        if moved == current:
            return None
    copy.set_params(**{f"{step}__{parameter}": moved})
    return copy


class DropColumn:
    """Remove one feature from the data on its way to the model.

    Works on named columns; a feature named `age` is also found as `num__age`, the
    name a ColumnTransformer gives it. On an unnamed array the column cannot be
    told apart, and the disturbance is not applicable rather than fragile.
    """

    def __init__(self, column: str) -> None:
        self.column = column

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"column": self.column}

    def set_params(self, **params: Any) -> "DropColumn":
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def fit(self, X: Any, y: Any = None) -> "DropColumn":
        return self

    def transform(self, X: Any) -> Any:
        columns = getattr(X, "columns", None)
        if columns is None:
            raise PerturbationNotApplicable(
                f"cannot find column {self.column!r} in data without names"
            )
        matches = [
            name
            for name in columns
            if name == self.column or str(name).endswith(f"__{self.column}")
        ]
        if len(matches) != 1:
            raise PerturbationNotApplicable(
                f"column {self.column!r} matches {len(matches)} columns"
            )
        return X.drop(columns=matches)

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        return self.fit(X, y).transform(X)


def drop_feature(pipeline: "Pipeline", column: str) -> "Pipeline | None":
    """The model loses one input; everything before it is left as it was."""
    from sklearn.base import clone

    copy = clone(pipeline)
    try:
        copy.set_output(transform="pandas")
    except ValueError:
        pass  # a home-made step without set_output keeps its own output
    steps = list(copy.steps)
    steps.insert(len(steps) - 1, (f"drop_{column}", DropColumn(column)))
    return copy.set_params(steps=steps)


@dataclass(frozen=True)
class PipelineVariant:
    """One disturbed copy of the pipeline and what it disturbed.

    `kind` is "evaluation" when the variant removes or replaces a step that
    changes which rows the model is trained on — a sampler — since that
    measures the evaluation procedure rather than the pipeline.
    """

    name: str
    pipeline: "Pipeline"
    steps: tuple[str, ...]
    kind: Literal["pipeline", "evaluation"] = "pipeline"


def _samplers(pipeline: "Pipeline") -> set[str]:
    return {name for name, step in pipeline.steps if hasattr(step, "fit_resample")}


def _numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def leaf_setting(model: Any) -> tuple[str, float | None] | None:
    """The setting that decides how small a leaf may get, by the model's library.

    Returns its name and the value the library uses when the setting is left
    None, or None when the model has no leaf size that applies.
    """
    settings = model.get_params()
    kind = type(model).__name__
    if kind.startswith("LGBM"):
        name = next((n for n in _LIGHTGBM_LEAF if _numeric(settings.get(n))), None)
        return None if name is None else (name, None)
    if kind.startswith("XGB"):
        return "min_child_weight", _XGBOOST_MIN_CHILD_WEIGHT
    if kind.startswith("CatBoost"):
        if settings.get("grow_policy") not in _CATBOOST_LEAF_POLICIES:
            return None
        name = next(
            (
                n
                for n in ("min_data_in_leaf", "min_child_samples")
                if settings.get(n) is not None
            ),
            "min_data_in_leaf",
        )
        return name, _CATBOOST_MIN_DATA_IN_LEAF
    if type(model).__module__.startswith("sklearn."):
        return ("min_samples_leaf", None) if "min_samples_leaf" in settings else None
    name = next((n for n in LEAF_SETTINGS if _numeric(settings.get(n))), None)
    return None if name is None else (name, None)


def named_pipeline_perturbations(
    pipeline: "Pipeline",
    drop_columns: Iterable[str] = (),
    evaluation_steps: Iterable[str] = (),
) -> list[PipelineVariant]:
    """Every stage disturbed once: preprocessing, ordering, and the estimator.

    The estimator gets its most influential setting moved, and its leaf size
    as chosen by `leaf_setting`: min_samples_leaf for scikit-learn,
    min_child_samples (or an alias) for LightGBM, min_child_weight for XGBoost,
    min_data_in_leaf for CatBoost under Depthwise or Lossguide growth only.

    `drop_columns` names features to take away right before the model.
    `evaluation_steps` adds steps to treat like samplers, beyond those found by
    their `fit_resample`.
    """
    evaluation = _samplers(pipeline) | set(evaluation_steps)
    names = [name for name, _ in pipeline.steps]
    estimator = names[-1]
    found: list[tuple[str, Pipeline | None, tuple[str, ...], bool]] = []

    for name in names[:-1]:
        found.append((f"drop:{name}", drop_step(pipeline, name), (name,), True))
    for index in range(max(0, len(names) - 2)):
        pair = (names[index], names[index + 1])
        found.append(
            (f"swap:{pair[0]}<->{pair[1]}", swap_steps(pipeline, index), pair, False)
        )

    scaled = replace_scaler(pipeline)
    if scaled is not None:
        from sklearn import preprocessing

        kinds = tuple(getattr(preprocessing, n) for n in _SCALERS)
        top = (_find_nested(pipeline, kinds) or "").split("__")[0]
        found.append((f"replace_scaler:{top}", scaled, (top,), True))

    model = pipeline.named_steps[estimator]
    settings = model.get_params()
    moved = [
        name for name in INFLUENTIAL_SETTINGS if _numeric(settings.get(name))
    ] or sorted(name for name, value in settings.items() if _numeric(value))
    chosen: list[tuple[str, float | None]] = [(name, None) for name in moved[:1]]
    leaf = leaf_setting(model)
    if leaf is not None and leaf[0] not in moved[:1]:
        chosen.append(leaf)
    for parameter, default in chosen:
        for factor in (0.8, 1.2):
            found.append(
                (
                    f"{estimator}.{parameter}×{factor}",
                    scale_hyperparameter(
                        pipeline, estimator, parameter, factor, default
                    ),
                    (estimator,),
                    False,
                )
            )

    encoded = replace_encoder(pipeline)
    if encoded is not None:
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

        top = (_find_nested(pipeline, (OneHotEncoder, OrdinalEncoder)) or "").split(
            "__"
        )[0]
        found.append((f"replace_encoder:{top}", encoded, (top,), True))

    for column in drop_columns:
        found.append(
            (f"drop_feature:{column}", drop_feature(pipeline, column), (), False)
        )

    return [
        PipelineVariant(
            name,
            variant,
            touched,
            "evaluation" if removes and evaluation & set(touched) else "pipeline",
        )
        for name, variant, touched, removes in found
        if variant is not None
    ]


def pipeline_perturbations(
    pipeline: "Pipeline", drop_columns: Iterable[str] = ()
) -> list["Pipeline"]:
    """Every stage disturbed once: preprocessing, ordering, and the estimator."""
    return [
        variant.pipeline
        for variant in named_pipeline_perturbations(pipeline, drop_columns)
    ]


def measure_pipeline_propagation(
    run: "Callable[[Pipeline, pd.DataFrame, int], float]",
    build: "Callable[[], Pipeline]",
    frame: "pd.DataFrame",
    seeds: Iterable[int],
    drop_columns: Sequence[str] = (),
    evaluation: EvaluationRule = DEFAULT_EVALUATION_RULE,
    evaluation_steps: Iterable[str] = (),
) -> Measurement:
    """PPS observations over the pipeline's own stages.

    Variants that disturb the evaluation procedure (a sampler) are, by the
    `evaluation` rule: reported apart and kept out of PPS ("separate", the
    default, not agreed), not run at all ("exclude"), or counted as any other
    ("include").
    """
    seeds = list(seeds)
    baseline = statistics.median(run(build(), frame, seed) for seed in seeds)

    shifts: list[Shift] = []
    failures: list[Failure] = []
    apart: list[Shift] = []
    apart_failures: list[Failure] = []
    skipped: list[str] = []
    variants = named_pipeline_perturbations(build(), drop_columns, evaluation_steps)
    for variant in variants:
        separate = variant.kind == "evaluation" and evaluation != "include"
        if separate and evaluation == "exclude":
            continue
        try:
            disturbed = statistics.median(
                run(variant.pipeline, frame, seed) for seed in seeds
            )
        except PerturbationNotApplicable:
            skipped.append(variant.name)
            continue
        except Exception as error:
            failure = Failure(variant.name, _describe(error))
            (apart_failures if separate else failures).append(failure)
            continue
        shift = Shift(variant.name, relative_shift(baseline, disturbed))
        (apart if separate else shifts).append(shift)
    return Measurement(
        baseline,
        tuple(shifts),
        tuple(failures),
        tuple(apart),
        tuple(apart_failures),
        tuple(skipped),
    )


def pipeline_propagation(
    run: "Callable[[Pipeline, pd.DataFrame, int], float]",
    build: "Callable[[], Pipeline]",
    frame: "pd.DataFrame",
    seeds: Iterable[int],
    aggregate: Aggregate = statistics.median,
    failures: FailureRule = DEFAULT_FAILURE_RULE,
    drop_columns: Sequence[str] = (),
    evaluation: EvaluationRule = DEFAULT_EVALUATION_RULE,
) -> float:
    """PPS: the shift over the pipeline's own stages, aggregated."""
    measured = measure_pipeline_propagation(
        run, build, frame, seeds, drop_columns, evaluation
    )
    return measured.score(aggregate, failures)


# Prediction churn — the primary outcome.
#
# A shift of the summary metric can be zero while many predictions flip, their
# errors cancelling; CACE is about behaviour, so the primary outcome compares the
# predictions of a disturbed run with those of its baseline on the same test
# objects. The metric shift is kept as a secondary outcome from the same runs.

Seeds = tuple[int, int]
Churn = Callable[[Any, Any], float]


@dataclass(frozen=True)
class Prediction:
    """What one run predicted on its test objects, in their order.

    Labels for a classifier, numbers for a regressor, or class probabilities for
    `probability_churn`. `metric` is the quality the run reports, if any; it
    feeds the secondary outcome.
    """

    values: Any
    metric: float | None = None


PairedRun = Callable[["pd.DataFrame", int, int], Prediction]
PairedPipelineRun = Callable[["Pipeline", "pd.DataFrame", int, int], Prediction]


@dataclass(frozen=True)
class Outcomes:
    """Churn, the primary outcome, and the metric shift where runs report one.

    `churn.baseline` is NaN: churn has no baseline quality, its reference level
    is `churn.noise_floor`.
    """

    churn: Measurement
    metric: Measurement | None = None


def _aligned(baseline: Any, perturbed: Any) -> "tuple[np.ndarray, np.ndarray]":
    import numpy as np

    before, after = np.asarray(baseline), np.asarray(perturbed)
    if before.shape != after.shape:
        raise ValueError(
            f"predictions of shape {before.shape} and {after.shape} do not pair"
        )
    return before, after


def label_churn(baseline: Any, perturbed: Any) -> float:
    """The share of test objects whose predicted label changed.

    The churn of Milani Fard et al. (2016), eq. (1), and the disagreement rate
    of Jiang et al. (2021).
    """
    before, after = _aligned(baseline, perturbed)
    return float((before != after).mean()) if before.size else 0.0


def probability_churn(
    baseline: Any, perturbed: Any, delta: float = PROBABILITY_DELTA
) -> float:
    """The share of objects whose predicted probability moved by more than delta.

    After Watson-Daniels et al. (2023), Definition 4. For several classes an
    object counts when any class moved that far. Sees a risk that drifts without
    crossing the decision threshold, which label churn misses.
    """
    before, after = _aligned(baseline, perturbed)
    moved = abs(after - before)
    if moved.ndim > 1:
        moved = moved.max(axis=1)
    return float((moved > delta).mean()) if moved.size else 0.0


def regression_churn(
    baseline: Any,
    perturbed: Any,
    scale: float,
    tolerance: float = REGRESSION_TOLERANCE,
) -> float:
    """The share of objects whose prediction moved by more than tolerance × scale.

    `scale` is the standard deviation of the target: fixed by the dataset, it
    keeps pipelines on the same data comparable, whereas the spread of the
    baseline's own predictions would shrink with a weak model. Retraining moves
    every real-valued prediction a little, hence a tolerance; with none, the
    count is of any change, as for labels.
    """
    if scale <= 0:
        raise ValueError("the target does not vary; there is no scale to move by")
    before, after = _aligned(baseline, perturbed)
    moved = abs(after - before)
    return float((moved > tolerance * scale).mean()) if moved.size else 0.0


def mean_absolute_change(baseline: Any, perturbed: Any, scale: float) -> float:
    """How far regression predictions moved on average, in target deviations.

    Free of a tolerance, so it checks that `regression_churn` does not hinge on
    the one chosen; unbounded, so a failure's 1.0 is not its maximum.
    """
    if scale <= 0:
        raise ValueError("the target does not vary; there is no scale to move by")
    before, after = _aligned(baseline, perturbed)
    return float(abs(after - before).mean()) / scale if before.size else 0.0


def noise_floor(runs_by_split: Iterable[Sequence[Any]], churn: Churn) -> float:
    """The churn two undisturbed runs show when only the model seed differs.

    Each group holds the baseline predictions of one split, one per model seed;
    every pair within a group is compared, and the median taken over all of them.
    """
    churns = [
        churn(first, second)
        for group in runs_by_split
        for first, second in itertools.combinations(group, 2)
    ]
    if not churns:
        raise ValueError("the noise floor needs at least two model seeds on a split")
    return statistics.median(churns)


def paired_churn(
    baseline: Mapping[Seeds, Any], perturbed: Mapping[Seeds, Any], churn: Churn
) -> float:
    """The median churn over runs paired by (model_seed, split_seed).

    Pairing marginalises out the variance the seeds bring, after Bouthillier et
    al. (2021), appendix C.2; the shared split is also what puts both runs on
    the same test objects.
    """
    if set(baseline) != set(perturbed):
        raise ValueError("baseline and disturbed runs were made with different seeds")
    return statistics.median(churn(baseline[key], perturbed[key]) for key in baseline)


def _seed_pairs(model_seeds: Sequence[int], split_seeds: Sequence[int]) -> list[Seeds]:
    return list(itertools.product(model_seeds, split_seeds))


Observation = tuple[str, "dict[Seeds, Prediction] | Failure", bool]


def _fold(
    observed: list[Observation],
    shift_of: Callable[[dict[Seeds, Prediction]], float | None],
    baseline: float,
    skipped: Sequence[str],
    floor: float | None = None,
) -> Measurement:
    shifts: list[Shift] = []
    failures: list[Failure] = []
    apart: list[Shift] = []
    apart_failures: list[Failure] = []
    left_out = list(skipped)
    for name, result, separate in observed:
        if isinstance(result, Failure):
            (apart_failures if separate else failures).append(result)
            continue
        shift = shift_of(result)
        if shift is None:
            left_out.append(name)
            continue
        (apart if separate else shifts).append(Shift(name, shift))
    return Measurement(
        baseline,
        tuple(shifts),
        tuple(failures),
        tuple(apart),
        tuple(apart_failures),
        tuple(left_out),
        floor,
    )


def _outcomes(
    baseline: dict[Seeds, Prediction],
    observed: list[Observation],
    skipped: Sequence[str],
    churn: Churn,
    split_seeds: Sequence[int],
) -> Outcomes:
    def churn_of(runs: dict[Seeds, Prediction]) -> float | None:
        shapes = {
            key: (_shape(baseline[key].values), _shape(run.values))
            for key, run in runs.items()
        }
        if any(before != after for before, after in shapes.values()):
            return None  # other test objects: churn is undefined, not large
        return paired_churn(
            {key: baseline[key].values for key in runs},
            {key: run.values for key, run in runs.items()},
            churn,
        )

    floor = noise_floor(
        (
            [prediction.values for key, prediction in baseline.items() if key[1] == s]
            for s in split_seeds
        ),
        churn,
    )
    churned = _fold(observed, churn_of, float("nan"), skipped, floor)

    completed = [r for _, r, _ in observed if not isinstance(r, Failure)]
    metrics = [[run.metric for run in runs.values()] for runs in [baseline, *completed]]
    if any(metric is None for group in metrics for metric in group):
        return Outcomes(churned)

    def metric_of(runs: dict[Seeds, Prediction]) -> float:
        return statistics.median(_metric(run) for run in runs.values())

    reference = metric_of(baseline)
    measured = _fold(
        observed,
        lambda runs: relative_shift(reference, metric_of(runs)),
        reference,
        skipped,
    )
    return Outcomes(churned, measured)


def _shape(values: Any) -> tuple[int, ...]:
    import numpy as np

    return tuple(np.shape(values))


def _metric(prediction: Prediction) -> float:
    if prediction.metric is None:
        raise ValueError("the run reported no metric")
    return prediction.metric


def measure_model_churn(
    run: PairedRun,
    frame: "pd.DataFrame",
    columns: Iterable[str],
    model_seeds: Sequence[int],
    split_seeds: Sequence[int],
    perturbations: Iterable[Perturbation] = PERTURBATIONS,
    churn: Churn = label_churn,
) -> Outcomes:
    """MSS by churn: every column disturbed every way, each run paired by seeds.

    `run(frame, model_seed, split_seed)` returns a `Prediction` on the test part
    of the split. Every (model_seed, split_seed) is run once undisturbed and once
    per disturbance, and the disturbance's churn is the median over the pairs.
    Two model seeds or more are needed for the noise floor. Failures and
    disturbances that cannot be made are handled as in
    `measure_model_sensitivity`; a disturbance that leaves the run predicting on
    other test objects is skipped for churn.

    `churn` defaults to `label_churn`; for a regressor pass
    `functools.partial(regression_churn, scale=float(y.std()))`.
    """
    import numpy as np

    pairs = _seed_pairs(model_seeds, split_seeds)
    baseline = {pair: run(frame, *pair) for pair in pairs}

    observed: list[Observation] = []
    skipped: list[str] = []
    for column in columns:
        for perturb in perturbations:
            name = f"{perturb.__name__}({column})"
            try:
                disturbed = {
                    pair: perturb(frame, column, np.random.default_rng(list(pair)))
                    for pair in pairs
                }
            except Exception:
                skipped.append(name)
                continue
            try:
                runs = {pair: run(data, *pair) for pair, data in disturbed.items()}
            except PerturbationNotApplicable:
                skipped.append(name)
                continue
            except Exception as error:
                observed.append((name, Failure(name, _describe(error)), False))
                continue
            observed.append((name, runs, False))
    return _outcomes(baseline, observed, skipped, churn, split_seeds)


def measure_pipeline_churn(
    run: PairedPipelineRun,
    build: "Callable[[], Pipeline]",
    frame: "pd.DataFrame",
    model_seeds: Sequence[int],
    split_seeds: Sequence[int],
    churn: Churn = label_churn,
    drop_columns: Sequence[str] = (),
    evaluation: EvaluationRule = DEFAULT_EVALUATION_RULE,
    evaluation_steps: Iterable[str] = (),
) -> Outcomes:
    """PPS by churn over the pipeline's own stages, runs paired by seeds.

    `run(pipeline, frame, model_seed, split_seed)` returns a `Prediction`. The
    `evaluation` rule and failures are as in `measure_pipeline_propagation`.
    """
    pairs = _seed_pairs(model_seeds, split_seeds)
    baseline = {pair: run(build(), frame, *pair) for pair in pairs}

    observed: list[Observation] = []
    skipped: list[str] = []
    variants = named_pipeline_perturbations(build(), drop_columns, evaluation_steps)
    for variant in variants:
        separate = variant.kind == "evaluation" and evaluation != "include"
        if separate and evaluation == "exclude":
            continue
        try:
            runs = {pair: run(variant.pipeline, frame, *pair) for pair in pairs}
        except PerturbationNotApplicable:
            skipped.append(variant.name)
            continue
        except Exception as error:
            failure = Failure(variant.name, _describe(error))
            observed.append((variant.name, failure, separate))
            continue
        observed.append((variant.name, runs, separate))
    return _outcomes(baseline, observed, skipped, churn, split_seeds)

"""CES — Cascade Effect Score: what a local change does to reported quality.

An empirical proxy for CACE, "changing anything changes everything". A pipeline
is run once as it stands, then again with one thing disturbed, and the distance
between the two qualities is the measurement. MSS disturbs the data a column at
a time; PPS disturbs a stage of the pipeline. CES is their geometric mean, so
that sensitivity on one level alone does not pass for a cascade.

Quality is whatever the pipeline reports — accuracy, F1, RMSE. The harness only
needs a callable that runs it: `run(frame, seed) -> float`.
"""

import statistics
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

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

Run = Callable[["pd.DataFrame", int], float]
Aggregate = Callable[[list[float]], float]
Perturbation = Callable[["pd.DataFrame", str, "np.random.Generator"], "pd.DataFrame"]


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


def model_sensitivity(
    run: Run,
    frame: "pd.DataFrame",
    columns: Iterable[str],
    seeds: Iterable[int],
    perturbations: Iterable[Perturbation] = PERTURBATIONS,
    aggregate: Aggregate = statistics.median,
) -> float:
    """MSS: the shift over every column disturbed every way, aggregated.

    The median is what the design specifies. It is robust to a single wild
    perturbation, which is also its cost: a pipeline that survives most
    disturbances and collapses under one reads as sound. Sensitivity analysis
    is the place to try the alternatives, hence the argument.
    """
    import numpy as np

    seeds = list(seeds)
    baseline = statistics.median(run(frame, seed) for seed in seeds)

    shifts = []
    for column in columns:
        for perturb in perturbations:
            disturbed = statistics.median(
                run(perturb(frame, column, np.random.default_rng(seed)), seed)
                for seed in seeds
            )
            shifts.append(relative_shift(baseline, disturbed))
    return aggregate(shifts) if shifts else 0.0


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


def replace_scaler(pipeline: "Pipeline") -> "Pipeline | None":
    """Swap one way of putting features on a common scale for another."""
    from sklearn.base import clone
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    steps = list(pipeline.steps)
    for position, (name, step) in enumerate(steps):
        if not type(step).__name__.endswith("Scaler"):
            continue
        other = MinMaxScaler() if isinstance(step, StandardScaler) else StandardScaler()
        replaced = list(steps)
        replaced[position] = (name, other)
        return clone(pipeline).set_params(steps=[(s, clone(o)) for s, o in replaced])
    return None


def scale_hyperparameter(
    pipeline: "Pipeline", step: str, parameter: str, factor: float
) -> "Pipeline | None":
    """Move one numeric setting of a step, keeping whole numbers whole."""
    from sklearn.base import clone

    copy = clone(pipeline)
    current = copy.named_steps[step].get_params().get(parameter)
    if not isinstance(current, int | float) or isinstance(current, bool):
        return None
    moved = (
        max(1, round(current * factor))
        if isinstance(current, int)
        else (current * factor)
    )
    copy.set_params(**{f"{step}__{parameter}": moved})
    return copy


def pipeline_perturbations(pipeline: "Pipeline") -> list["Pipeline"]:
    """Every stage disturbed once: preprocessing, ordering, and the estimator."""
    variants = []
    names = [name for name, _ in pipeline.steps]
    for name in names[:-1]:
        variants.append(drop_step(pipeline, name))
    for index in range(max(0, len(names) - 2)):
        variants.append(swap_steps(pipeline, index))
    variants.append(replace_scaler(pipeline))

    estimator = names[-1]
    settings = pipeline.named_steps[estimator].get_params()
    numeric = [
        name
        for name in INFLUENTIAL_SETTINGS
        if isinstance(settings.get(name), int | float)
        and not isinstance(settings.get(name), bool)
    ] or sorted(
        name
        for name, value in settings.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    )
    if numeric:
        for factor in (0.8, 1.2):
            variants.append(
                scale_hyperparameter(pipeline, estimator, numeric[0], factor)
            )
    return [variant for variant in variants if variant is not None]


def pipeline_propagation(
    run: "Callable[[Pipeline, pd.DataFrame, int], float]",
    build: "Callable[[], Pipeline]",
    frame: "pd.DataFrame",
    seeds: Iterable[int],
    aggregate: Aggregate = statistics.median,
) -> float:
    """PPS: the shift over the pipeline's own stages, aggregated."""
    seeds = list(seeds)
    baseline = statistics.median(run(build(), frame, seed) for seed in seeds)

    shifts = []
    for variant in pipeline_perturbations(build()):
        disturbed = statistics.median(run(variant, frame, seed) for seed in seeds)
        shifts.append(relative_shift(baseline, disturbed))
    return aggregate(shifts) if shifts else 0.0

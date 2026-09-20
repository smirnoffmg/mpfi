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

# Noise stays small enough to be a disturbance rather than a different dataset.
NOISE_SCALE = 0.1

Run = Callable[["pd.DataFrame", int], float]
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
) -> float:
    """MSS: the median shift over every column disturbed every way."""
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
    return statistics.median(shifts) if shifts else 0.0


def ces(model_sensitivity_score: float, pipeline_propagation_score: float) -> float:
    """The geometric mean penalises a pipeline fragile on one level only."""
    product = model_sensitivity_score * pipeline_propagation_score
    return product**0.5 if product > 0 else 0.0

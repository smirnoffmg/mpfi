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
import math
import random
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
# Protocol v2 measures noise as a ladder of doses and takes the area under the
# churn curve: at one dose the order of pipelines depends on the dose chosen
# (E57: rank correlation 0.47 between 0.1σ and 1σ, 0.8–0.9 between neighbours).
NOISE_DOSES = (0.01, 0.03, 0.1, 0.3, 1.0)
# A categorical column has no spread to scale noise to; this share of its rows is
# redrawn. Jenga corrupts a chosen fraction of a column's rows the same way
# (Schelter et al., EDBT 2021, section 2.2).
CATEGORY_NOISE_SHARE = 0.1
MISSING_CATEGORY = "__ces_missing__"

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
# Decided 03.10.2026 (protocol v2, decision 8): "count_as_one" is primary, "exclude"
# a sensitivity check. A run that breaks under a disturbance has lost all its quality,
# which is what a shift of 1.0 says; leaving it out would read the most fragile
# pipelines as the soundest.
DEFAULT_FAILURE_RULE: FailureRule = "count_as_one"

EvaluationRule = Literal["separate", "exclude", "include"]
# Decided 03.10.2026 (protocol v2, decision 7): "separate". A sampler changes what
# the model is trained and, when the author resamples before splitting, scored on:
# its disturbance measures the evaluation procedure, not the pipeline.
DEFAULT_EVALUATION_RULE: EvaluationRule = "separate"

NoiseRule = Literal["subtract", "raw"]
# Не согласовано, решение 26.09. Retraining a seed-sensitive model on disturbed data
# changes its random draws as well, so even an irrelevant disturbance flips about
# as many predictions as a second seed does; only the excess speaks of the pipeline.
# A ratio would be undefined for a deterministic model, whose floor is zero.
DEFAULT_NOISE_RULE: NoiseRule = "subtract"

# Decided 03.10.2026 (protocol v2, decision 4); 0.05 and 0.2 are sensitivity checks.
# A regression prediction counts as changed when it
# moves by more than this share of the target's standard deviation.
REGRESSION_TOLERANCE = 0.1
# Decided 03.10.2026 (protocol v2, decision 4): a secondary outcome; the primary churn
# of a classifier is the label one. A predicted probability counts as changed when it
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
        """The median by default.

        Protocol v2 pools perturbation types across the sample instead (decision 5),
        so this per-pipeline score is a sensitivity check.
        """
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


def _categorical(values: "pd.Series") -> bool:
    import pandas as pd

    return not pd.api.types.is_numeric_dtype(values)


def _with_category(values: "pd.Series", category: str) -> "pd.Series":
    import pandas as pd

    if (
        isinstance(values.dtype, pd.CategoricalDtype)
        and category not in values.cat.categories
    ):
        return values.cat.add_categories([category])
    return values


def mean_imputation(
    frame: "pd.DataFrame", column: str, rng: "np.random.Generator | None" = None
) -> "pd.DataFrame":
    """The column loses everything but its average; a categorical one, its mode."""
    import pandas as pd

    disturbed = frame.copy()
    values = frame[column]
    if _categorical(values):
        disturbed[column] = pd.Series(
            values.mode().iloc[0], index=values.index, dtype=values.dtype
        )
    else:
        disturbed[column] = values.mean()
    return disturbed


def zero_fill(
    frame: "pd.DataFrame", column: str, rng: "np.random.Generator | None" = None
) -> "pd.DataFrame":
    """A constant outside the data: zero, or a category the column never had."""
    import pandas as pd

    disturbed = frame.copy()
    values = frame[column]
    if _categorical(values):
        dtype = _with_category(values, MISSING_CATEGORY).dtype
        disturbed[column] = pd.Series(MISSING_CATEGORY, index=values.index, dtype=dtype)
    else:
        disturbed[column] = 0
    return disturbed


def gaussian_noise(
    frame: "pd.DataFrame",
    column: str,
    rng: "np.random.Generator | None" = None,
    scale: float = NOISE_SCALE,
) -> "pd.DataFrame":
    """Noise of `scale` times the column's spread, so every column is disturbed alike.

    A categorical column has no spread; there a share of the rows draws its value
    anew from the column itself, so the distribution stays and the rows lose theirs.
    The share grows with the dose as the spread does: CATEGORY_NOISE_SHARE at the
    default dose, every row at a dose of one spread.

    One generator seed gives every dose the same draws, only scaled, so the churn
    of a column changes smoothly along the ladder of NOISE_DOSES.
    """
    import numpy as np
    import pandas as pd

    generator = rng if rng is not None else np.random.default_rng()
    disturbed = frame.copy()
    values = frame[column]
    if _categorical(values):
        share = min(1.0, CATEGORY_NOISE_SHARE * scale / NOISE_SCALE)
        rows = generator.choice(len(frame), round(share * len(frame)), replace=False)
        redrawn = values.to_numpy(copy=True)
        redrawn[rows] = generator.choice(values.to_numpy(), len(rows))
        disturbed[column] = pd.Series(redrawn, index=values.index, dtype=values.dtype)
        return disturbed
    spread = float(values.std())
    if spread == 0:
        spread = 1.0
    disturbed[column] = values + generator.normal(0, scale * spread, len(frame))
    return disturbed


PERTURBATIONS: tuple[Perturbation, ...] = (mean_imputation, zero_fill, gaussian_noise)

# Candidates for the catalogue, screened against the rule preregistered in E73
# (masters/experiments/2026-10-29-catalog-screening/hypothesis.md). They stay out of
# PERTURBATIONS until a type passes, so a queue planned before keeps its spec. Each
# corrupts a share of the rows, as JENGA does, except precision loss (the literature
# rounds the whole column) and the probe. `others` are the MSS columns in order of
# mutual information; a candidate that needs a partner takes the next like column.
CANDIDATE_SHARE = 0.1
SENTINEL_NUMBER = -999
SENTINEL_CATEGORY = "?"
OUTLIER_SPREADS = 10.0
UNIT_FACTORS = (10, 100, 1000)
SLICE_QUANTILE = 0.9
SIGNIFICANT_DIGITS = 2
PROBE_FACTOR = 1000


def _rows(
    frame: "pd.DataFrame", rng: "np.random.Generator", share: float
) -> "np.ndarray":
    return rng.choice(len(frame), round(share * len(frame)), replace=False)


def _number(frame: "pd.DataFrame", column: str) -> "pd.Series":
    values = frame[column]
    if _categorical(values):
        raise PerturbationNotApplicable(f"{column} is not numeric")
    return values


def _partner(frame: "pd.DataFrame", column: str, others: Sequence[str]) -> str:
    """The next of `others` after `column`, wrapping around, of the same kind."""
    kind = _categorical(frame[column])
    order = list(others)
    start = order.index(column) + 1 if column in order else 0
    for candidate in order[start:] + order[:start]:
        if candidate != column and _categorical(frame[candidate]) == kind:
            return candidate
    raise PerturbationNotApplicable(f"no MSS column like {column} to pair with")


def _spread(values: "pd.Series") -> float:
    spread = float(values.std())
    return spread if spread > 0 else 1.0


def _put(
    frame: "pd.DataFrame", column: str, rows: "np.ndarray", value: Any
) -> "pd.DataFrame":
    """`value` into `rows` of a copy of the column.

    The column widens only as far as the value needs: an integer column that still
    holds whole numbers and no gaps stays integer, since adapters pick columns by dtype.
    """
    import pandas as pd

    disturbed = frame.copy()
    values = frame[column]
    if isinstance(values.dtype, pd.CategoricalDtype):
        incoming = pd.unique(pd.Series(value).dropna())
        new = [v for v in incoming if v not in values.cat.categories]
        disturbed[column] = values.cat.add_categories(new) if new else values
    elif _categorical(values):
        disturbed[column] = values.astype(object)
    else:
        disturbed[column] = values.astype(float)
    position = disturbed.columns.get_loc(column)
    assert isinstance(position, int), f"{column} is not a unique column"
    disturbed.iloc[rows, position] = value
    result = disturbed[column]
    if pd.api.types.is_integer_dtype(values) and result.notna().all():
        if (result == result.round()).all():
            disturbed[column] = result.astype(values.dtype)
    return disturbed


def partial_missing(
    frame: "pd.DataFrame",
    column: str,
    rng: "np.random.Generator",
    others: Sequence[str] = (),
    share: float = CANDIDATE_SHARE,
) -> "pd.DataFrame":
    """A share of the rows, drawn at random, lose their value (MCAR)."""
    return _put(frame, column, _rows(frame, rng, share), None)


def partial_missing_mnar(
    frame: "pd.DataFrame",
    column: str,
    rng: "np.random.Generator",
    others: Sequence[str] = (),
    share: float = CANDIDATE_SHARE,
) -> "pd.DataFrame":
    """The largest values go missing: missingness that depends on the value (MNAR)."""
    import numpy as np

    values = _number(frame, column).to_numpy(dtype=float)
    order = np.argsort(-values, kind="stable")
    return _put(frame, column, order[: round(share * len(frame))], np.nan)


def sentinel(
    frame: "pd.DataFrame",
    column: str,
    rng: "np.random.Generator",
    others: Sequence[str] = (),
    share: float = CANDIDATE_SHARE,
) -> "pd.DataFrame":
    """A placeholder stands in for the value: −999, or "?" in a categorical column."""
    value = SENTINEL_CATEGORY if _categorical(frame[column]) else SENTINEL_NUMBER
    return _put(frame, column, _rows(frame, rng, share), value)


def outliers(
    frame: "pd.DataFrame",
    column: str,
    rng: "np.random.Generator",
    others: Sequence[str] = (),
    share: float = CANDIDATE_SHARE,
) -> "pd.DataFrame":
    """A share of the rows jump OUTLIER_SPREADS spreads off the median, either way."""
    values = _number(frame, column)
    rows = _rows(frame, rng, share)
    signs = rng.choice([-1.0, 1.0], len(rows))
    far = float(values.median()) + signs * OUTLIER_SPREADS * _spread(values)
    return _put(frame, column, rows, far)


def unit_error(
    frame: "pd.DataFrame",
    column: str,
    rng: "np.random.Generator",
    others: Sequence[str] = (),
    share: float = CANDIDATE_SHARE,
) -> "pd.DataFrame":
    """A share of the rows recorded in another unit: ×10, ×100 or ×1000 by row."""
    values = _number(frame, column)
    rows = _rows(frame, rng, share)
    factors = rng.choice(UNIT_FACTORS, len(rows))
    return _put(frame, column, rows, values.to_numpy()[rows] * factors)


def slice_offset(
    frame: "pd.DataFrame",
    column: str,
    rng: "np.random.Generator",
    others: Sequence[str] = (),
    share: float = CANDIDATE_SHARE,
) -> "pd.DataFrame":
    """A systematic error on a slice: +1 spread where the partner is in its top decile.

    The slice is fixed by the partner's distribution, so `share` does not apply.
    """
    import numpy as np

    values = _number(frame, column)
    partner = frame[_partner(frame, column, others)]
    rows = np.flatnonzero(partner >= partner.quantile(SLICE_QUANTILE))
    return _put(frame, column, rows, values.to_numpy()[rows] + _spread(values))


def precision_loss(
    frame: "pd.DataFrame",
    column: str,
    rng: "np.random.Generator",
    others: Sequence[str] = (),
    share: float = CANDIDATE_SHARE,
) -> "pd.DataFrame":
    """The whole column keeps SIGNIFICANT_DIGITS significant digits."""
    import numpy as np

    values = _number(frame, column).to_numpy(dtype=float)
    magnitude = np.floor(np.log10(np.abs(np.where(values == 0, 1.0, values))))
    factor = 10.0 ** (SIGNIFICANT_DIGITS - 1 - magnitude)
    disturbed = frame.copy()
    disturbed[column] = np.round(values * factor) / factor
    return disturbed


def column_swap(
    frame: "pd.DataFrame",
    column: str,
    rng: "np.random.Generator",
    others: Sequence[str] = (),
    share: float = CANDIDATE_SHARE,
) -> "pd.DataFrame":
    """A share of the rows have the column's value and its partner's exchanged."""
    partner = _partner(frame, column, others)
    rows = _rows(frame, rng, share)
    mine = frame[column].to_numpy()[rows]
    theirs = frame[partner].to_numpy()[rows]
    return _put(_put(frame, column, rows, theirs), partner, rows, mine)


def category_typo(
    frame: "pd.DataFrame",
    column: str,
    rng: "np.random.Generator",
    others: Sequence[str] = (),
    share: float = CANDIDATE_SHARE,
) -> "pd.DataFrame":
    """A share of the rows misspelt, case swapped and a "~" added: new categories."""
    values = frame[column]
    if not _categorical(values):
        raise PerturbationNotApplicable(f"{column} is not categorical")
    rows = _rows(frame, rng, share)
    typos = [f"{value}".swapcase() + "~" for value in values.to_numpy()[rows]]
    return _put(frame, column, rows, typos)


def monotone_scale(
    frame: "pd.DataFrame",
    column: str,
    rng: "np.random.Generator",
    others: Sequence[str] = (),
    share: float = CANDIDATE_SHARE,
) -> "pd.DataFrame":
    """The probe: the whole column ×PROBE_FACTOR.

    A tree split on the column alone does not move; predictions change only where the
    pipeline mixes the column with others (sums, ratios, a shared scaler, PCA).
    """
    disturbed = frame.copy()
    disturbed[column] = _number(frame, column) * PROBE_FACTOR
    return disturbed


CANDIDATES: tuple[Callable[..., "pd.DataFrame"], ...] = (
    partial_missing,
    partial_missing_mnar,
    sentinel,
    outliers,
    unit_error,
    slice_offset,
    precision_loss,
    column_swap,
    category_typo,
    monotone_scale,
)


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
    failure rule is fixed by protocol v2 (decision 8, 03.10.2026).
    """
    measured = measure_model_sensitivity(run, frame, columns, seeds, perturbations)
    return measured.score(aggregate, failures)


def ces(model_sensitivity_score: float, pipeline_propagation_score: float) -> float:
    """The geometric mean penalises a pipeline fragile on one level only."""
    product = model_sensitivity_score * pipeline_propagation_score
    return product**0.5 if product > 0 else 0.0


def drop_step(pipeline: "Pipeline", name: str) -> "Pipeline | None":
    """Remove a transformer. The last step is the estimator and has to stay."""
    return drop_steps(pipeline, (name,))


def drop_steps(pipeline: "Pipeline", names: Iterable[str]) -> "Pipeline | None":
    """Remove several transformers at once; the estimator has to stay."""
    from sklearn.base import clone

    steps = list(pipeline.steps)
    dropped = set(names)
    if not dropped or steps[-1][0] in dropped or not dropped <= {s for s, _ in steps}:
        return None
    kept = [(step, obj) for step, obj in steps if step not in dropped]
    return clone(pipeline).set_params(steps=[(s, clone(o)) for s, o in kept])


StepScheme = Literal["one_at_a_time", "scaled"]
# Schema S of protocol v2 (E23): one step at a time lets the share of a long
# chain that is disturbed shrink with its length, and the median over such
# variants then falls with length whatever the pipeline (E21, E42, E46).
HALF_SUBSETS = 4


def _scaled_drops(names: Sequence[str]) -> list[tuple[str, ...]]:
    """K different halves of the transformers, or all there are, then all of them.

    The draw is seeded by the step names, so planning and running a pipeline in
    two processes list the same subsets.
    """
    if not names:
        return []
    whole = tuple(names)
    halves = [
        h
        for h in itertools.combinations(names, math.ceil(len(names) / 2))
        if h != whole
    ]
    if len(halves) > HALF_SUBSETS:
        drawn = random.Random(",".join(names)).sample(range(len(halves)), HALF_SUBSETS)
        halves = [halves[i] for i in sorted(drawn)]
    return [*halves, whole]


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
    current = copy.named_steps[step].get_params(deep=True).get(parameter)
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


def _searched(
    settings: Mapping[str, Any], owners: Sequence[tuple[str, Any]]
) -> set[str]:
    """Settings a hyperparameter search sets itself; moving them changes nothing."""
    tuned: set[str] = set()
    for prefix, owner in owners:
        grid = getattr(owner, "param_grid", None) or getattr(
            owner, "param_distributions", None
        )
        for part in grid if isinstance(grid, list) else [grid] if grid else []:
            tuned |= {f"{prefix}estimator__{name}" for name in part}
    return tuned & set(settings)


def model_settings(model: Any) -> list[tuple[str, float | None]]:
    """The settings of the estimator to move, looked for inside wrappers as well.

    The first INFLUENTIAL_SETTINGS found among the deep parameters — those of a
    model inside a fold average, a stack or a search — and the leaf size of the
    first model `leaf_setting` knows. A setting a search tunes is left out, and so
    is everything else: a seed or a fold count moved for want of a setting would
    be reseeding, which the noise floor already measures. Returns paths with the
    value the library uses for an unset one.
    """
    settings = model.get_params(deep=True)
    owners = [("", model)] + [
        (f"{key}__", value)
        for key, value in settings.items()
        if hasattr(value, "get_params") and not isinstance(value, type)
    ]
    tuned = _searched(settings, owners)
    influential = [
        key
        for name in INFLUENTIAL_SETTINGS
        for key, value in settings.items()
        if key.split("__")[-1] == name and _numeric(value) and key not in tuned
    ]
    chosen: list[tuple[str, float | None]] = [(key, None) for key in influential[:1]]
    for prefix, owner in owners:
        leaf = leaf_setting(owner)
        if leaf is None or f"{prefix}{leaf[0]}" in tuned:
            continue
        if f"{prefix}{leaf[0]}" not in influential[:1]:
            chosen.append((f"{prefix}{leaf[0]}", leaf[1]))
        break
    return chosen


def named_pipeline_perturbations(
    pipeline: "Pipeline",
    drop_columns: Iterable[str] = (),
    evaluation_steps: Iterable[str] = (),
    steps: StepScheme = "one_at_a_time",
    model_params: Sequence[str] = (),
) -> list[PipelineVariant]:
    """Every stage disturbed once: preprocessing, ordering, and the estimator.

    The estimator gets its most influential setting moved, and its leaf size
    as chosen by `leaf_setting`: min_samples_leaf for scikit-learn,
    min_child_samples (or an alias) for LightGBM, min_child_weight for XGBoost,
    min_data_in_leaf for CatBoost under Depthwise or Lossguide growth only.

    `drop_columns` names features to take away right before the model.
    `evaluation_steps` adds steps to treat like samplers, beyond those found by
    their `fit_resample`.
    `model_params` names the estimator's settings to move instead of those
    `model_settings` finds, for a model whose settings it cannot see.
    `steps="scaled"` replaces dropping one transformer at a time by schema S:
    `drop_half:` random halves of the transformers, and `drop_all`.
    """
    evaluation = _samplers(pipeline) | set(evaluation_steps)
    names = [name for name, _ in pipeline.steps]
    estimator = names[-1]
    found: list[tuple[str, Pipeline | None, tuple[str, ...], bool]] = []

    if steps == "scaled":
        for subset in _scaled_drops(names[:-1]):
            label = (
                "drop_all"
                if subset == tuple(names[:-1])
                else ("drop_half:" + "+".join(subset))
            )
            found.append((label, drop_steps(pipeline, subset), subset, True))
    else:
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
    chosen = (
        [(name, None) for name in model_params]
        if model_params
        else model_settings(model)
    )
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
    default, protocol v2 decision 7), not run at all ("exclude"), or counted as
    any other ("include").
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

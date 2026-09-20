"""MPFI turns a call graph into three numbers about structural fragility.

The operationalisation is what these tests pin down: which nodes count as
feature engineering, what couples them, how long a transformation chain is, and
which data boundaries carry an explicit contract.
"""

import textwrap

import pytest

from mpfi.components import components_for_source

PIPELINE = """
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    def add_ratio(df):
        return df.assign(ratio=df["a"] / df["b"])

    def drop_outliers(df):
        return df[df["a"] < 100]

    def prepare(df):
        return df.pipe(add_ratio).pipe(drop_outliers)

    def build():
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])
"""

PLAIN = """
    def helper(x):
        return x + 1

    def main():
        return helper(41)
"""


@pytest.fixture
def pipeline(tmp_path):
    return components_for_source(tmp_path, textwrap.dedent(PIPELINE))


def test_code_without_data_work_scores_zero(tmp_path):
    result = components_for_source(tmp_path, textwrap.dedent(PLAIN))

    assert result.fci == 0
    assert result.pdd == 0


def test_feature_engineering_functions_are_found(pipeline):
    assert {"main.add_ratio", "main.drop_outliers"} <= pipeline.feature_nodes


def test_a_pipe_chain_counts_as_a_chain_of_transformations(pipeline):
    # prepare -> add_ratio and prepare -> drop_outliers: depth two.
    assert pipeline.pdd >= 2


def test_coupling_counts_the_edges_between_feature_nodes(pipeline):
    assert pipeline.fci > 0


def test_a_contract_on_the_boundary_raises_coverage(tmp_path):
    without = components_for_source(
        tmp_path / "a",
        textwrap.dedent("""
        import pandas as pd

        def load(path):
            return pd.read_csv(path)
    """),
    )
    with_hint = components_for_source(
        tmp_path / "b",
        textwrap.dedent("""
        import pandas as pd

        def load(path: str) -> pd.DataFrame:
            frame = pd.read_csv(path)
            assert not frame.empty
            return frame
    """),
    )

    assert with_hint.scc > without.scc


def test_coverage_is_a_share(pipeline):
    assert 0.0 <= pipeline.scc <= 1.0

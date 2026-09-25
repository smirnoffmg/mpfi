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
    """add_ratio hands its table to drop_outliers; prepare only routes it."""
    assert pipeline.pdd == 2
    assert "main.prepare" not in pipeline.feature_nodes


def test_coupling_counts_the_edges_between_feature_nodes(pipeline):
    # One hand-off, add_ratio -> drop_outliers, over two feature nodes.
    assert pipeline.fci == pytest.approx(0.5)


FIVE_STEPS = """
    def a(df):
        return df.assign(x=1)

    def b(df):
        return df.assign(y=1)

    def cc(df):
        return df.assign(z=1)

    def d(df):
        return df.assign(w=1)

    def e(df):
        return df.assign(v=1)
"""


def measure(tmp_path, source):
    return components_for_source(tmp_path, textwrap.dedent(source))


@pytest.mark.parametrize(
    "prepare",
    [
        "df.pipe(a).pipe(b).pipe(cc).pipe(d).pipe(e)",
        "e(d(cc(b(a(df)))))",
    ],
    ids=["pipe", "composition"],
)
def test_a_chain_of_five_steps_has_depth_five_whatever_the_style(tmp_path, prepare):
    """PDD counts the steps a table passes through, not how the call is written.

    FCI counts the hand-offs between consecutive steps: four among five nodes.
    """
    result = measure(
        tmp_path,
        FIVE_STEPS
        + f"""
    def prepare(df):
        return {prepare}
""",
    )

    assert result.pdd == 5
    assert result.fci == pytest.approx(4 / 5)


def test_a_chain_through_a_reassigned_variable_has_depth_five(tmp_path):
    result = measure(
        tmp_path,
        FIVE_STEPS
        + """
    def prepare(df):
        df = a(df)
        df = df.pipe(b)
        df = cc(df.copy())
        df = d(df)
        return e(df)
""",
    )

    assert result.pdd == 5
    assert result.fci == pytest.approx(4 / 5)


def test_nested_calls_of_five_steps_still_have_depth_five(tmp_path):
    result = measure(
        tmp_path,
        """
    def e(df):
        return df.assign(v=1)

    def d(df):
        return e(df.assign(w=1))

    def cc(df):
        return d(df.assign(z=1))

    def b(df):
        return cc(df.assign(y=1))

    def a(df):
        return b(df.assign(x=1))
""",
    )

    assert result.pdd == 5
    assert result.fci == pytest.approx(4 / 5)


def test_two_steps_reading_the_same_table_are_not_a_chain(tmp_path):
    """A plain call graph keeps its shape: prepare calls both, one never feeds the
    other."""
    result = measure(
        tmp_path,
        """
    def left(df):
        return df.assign(x=1)

    def right(df):
        return df.assign(y=1)

    def prepare(df):
        x = left(df)
        y = right(df)
        return x.merge(y)
""",
    )

    assert result.pdd == 2
    assert result.fci == pytest.approx(2 / 3)


TRANSFORMERS = """
    from sklearn.base import TransformerMixin
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    class A(TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X.assign(a=1)

    class B(A):
        def transform(self, X):
            return X.assign(b=1)

    class C(A):
        def transform(self, X):
            return X.assign(c=1)

    class D(A):
        def transform(self, X):
            return X.assign(d=1)
"""

STEP_NODES = {"main.A.transform", "main.B.transform", "main.C.transform"}


@pytest.mark.parametrize(
    "build",
    [
        'Pipeline([("a", A()), ("b", B()), ("c", C())])',
        'Pipeline(steps=[("a", A()), ("b", B()), ("c", C())])',
        "make_pipeline(A(), B(), C())",
        "make_pipeline(A(), StandardScaler(), B(), C())",
    ],
    ids=["Pipeline", "steps-keyword", "make_pipeline", "library-step-between"],
)
def test_the_steps_of_an_sklearn_pipeline_form_a_chain(tmp_path, build):
    result = measure(
        tmp_path,
        TRANSFORMERS
        + f"""
    def build():
        return {build}
""",
    )

    # D reshapes a table too, but no pipeline hands it one.
    assert result.feature_nodes == STEP_NODES | {"main.D.transform"}
    assert result.pdd == 3
    assert result.fci == pytest.approx(2 / 4)


@pytest.mark.parametrize(
    "branches",
    [
        'ColumnTransformer([("num", make_pipeline(A(), B()), ["x"]), '
        '("cat", C(), ["y"])])',
        'FeatureUnion([("num", Pipeline([("a", A()), ("b", B())])), ("cat", C())])',
    ],
    ids=["ColumnTransformer", "FeatureUnion"],
)
def test_parallel_branches_count_the_longest_one_and_what_follows(tmp_path, branches):
    """num: A -> B, cat: C, then D after both: A -> B -> D is three steps."""
    result = measure(
        tmp_path,
        TRANSFORMERS
        + f"""
    def build():
        return Pipeline([("prep", {branches}), ("d", D())])
""",
    )

    assert result.pdd == 3
    assert result.feature_nodes == STEP_NODES | {"main.D.transform"}


def test_no_node_is_invented_for_a_method_the_class_inherits_from_a_library(
    tmp_path,
):
    result = measure(
        tmp_path,
        TRANSFORMERS
        + """
    def build():
        return Pipeline([("a", A()), ("b", B())])
""",
    )

    assert not any("fit_transform" in node for node in result.feature_nodes)


def test_a_function_transformer_step_is_its_function(tmp_path):
    result = measure(
        tmp_path,
        TRANSFORMERS
        + """
    def add_ratio(df):
        return df

    def build():
        return make_pipeline(FunctionTransformer(add_ratio), A())
""",
    )

    assert {"main.add_ratio", "main.A.transform"} <= result.feature_nodes
    assert result.pdd == 2


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


def test_the_test_suite_is_not_part_of_the_pipeline(tmp_path):
    """A test calling a transform is checking it, not coupling to it."""
    from mpfi.components import components

    package = tmp_path / "pkg"
    (package / "tests").mkdir(parents=True)
    (package / "steps.py").write_text(
        textwrap.dedent("""
        def add_ratio(df):
            return df.assign(ratio=1)
    """)
    )
    (package / "tests" / "test_steps.py").write_text(
        textwrap.dedent("""
        from steps import add_ratio

        def test_add_ratio(df):
            return add_ratio(df)
    """)
    )

    result = components(package)

    assert not any(node.startswith("tests.") for node in result.feature_nodes)


def test_a_function_handling_a_frame_is_a_boundary_even_without_a_contract(tmp_path):
    """Detecting a boundary must not depend on the contract.

    Read the contract here and every boundary is covered by definition.
    """
    from mpfi.components import components_for_source

    result = components_for_source(
        tmp_path / "bare",
        textwrap.dedent("""
        def prepare(df):
            return df.assign(ratio=1)
    """),
    )

    assert result.boundaries >= 1
    assert result.scc == 0.0


def test_an_annotated_frame_counts_as_a_contract(tmp_path):
    from mpfi.components import components_for_source

    result = components_for_source(
        tmp_path / "typed",
        textwrap.dedent("""
        import pandas as pd

        def prepare(df: pd.DataFrame) -> pd.DataFrame:
            return df.assign(ratio=1)
    """),
    )

    assert result.boundaries >= 1
    assert result.scc == 1.0


def test_code_that_never_touches_a_table_has_no_boundaries(tmp_path):
    from mpfi.components import components_for_source

    result = components_for_source(
        tmp_path / "plain",
        textwrap.dedent("""
        def add(a, b):
            return a + b
    """),
    )

    assert result.boundaries == 0

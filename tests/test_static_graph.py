"""A call graph from names alone: imports and scopes, no points-to analysis.

The question it answers is whether the fragility components need the expensive
machinery at all. What it cannot do is written down here as plainly as what it can.
"""

import textwrap

from mpfi.static_graph import call_graph_for_source


def graph(source):
    return call_graph_for_source(textwrap.dedent(source), "main")


def test_a_direct_call_is_an_edge():
    result = graph("""
        def helper(x):
            return x

        def run():
            return helper(1)
    """)

    assert result["main.run"] == {"main.helper"}


def test_a_call_into_an_imported_module_keeps_its_full_name():
    result = graph("""
        import pandas as pd

        def load(path):
            return pd.read_csv(path)
    """)

    assert "pandas.read_csv" in result["main.load"]


def test_a_method_on_an_imported_class_resolves_through_the_import():
    result = graph("""
        from sklearn.preprocessing import StandardScaler

        def build():
            return StandardScaler()
    """)

    assert "sklearn.preprocessing.StandardScaler" in result["main.build"]


def test_pipeline_patterns_are_included():
    result = graph("""
        def add_ratio(df):
            return df

        def prepare(df):
            return df.pipe(add_ratio)
    """)

    assert "main.add_ratio" in result["main.prepare"]


def test_nested_functions_get_their_own_scope():
    result = graph("""
        def outer():
            def inner():
                return 1
            return inner()
    """)

    assert result["main.outer"] == {"main.outer.inner"}


def test_a_call_through_a_variable_is_not_resolved():
    """The known price of dropping points-to: an alias stays unresolved."""
    result = graph("""
        def transform(df):
            return df

        def run(df):
            f = transform
            return f(df)
    """)

    assert "main.transform" not in result.get("main.run", set())

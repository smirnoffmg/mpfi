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


def test_a_method_calls_its_neighbour_through_self():
    result = graph("""
        class Model:
            def prepare(self, df):
                return df

            def fit(self, df):
                return self.prepare(df)
    """)

    assert result["main.Model.fit"] == {"main.Model.prepare"}


def test_self_reaches_a_method_inherited_from_a_base_class():
    result = graph("""
        class Base:
            def prepare(self, df):
                return df

        class Model(Base):
            def fit(self, df):
                return self.prepare(df)
    """)

    assert "main.Base.prepare" in result["main.Model.fit"]


def test_a_base_class_from_another_module_is_named_in_full():
    result = graph("""
        from library.core import Estimator

        class Model(Estimator):
            def fit(self, df):
                return self.prepare(df)
    """)

    assert "library.core.Estimator.prepare" in result["main.Model.fit"]


def test_a_variable_holding_a_fresh_instance_carries_its_class():
    result = graph("""
        class Scaler:
            def transform(self, df):
                return df

        def run(df):
            scaler = Scaler()
            return scaler.transform(df)
    """)

    assert "main.Scaler.transform" in result["main.run"]


def test_a_variable_assigned_twice_is_left_alone():
    """Two candidates mean guessing, and guessing is what this graph avoids."""
    result = graph("""
        class A:
            def run(self):
                return 1

        class B:
            def run(self):
                return 2

        def main(flag):
            x = A()
            x = B()
            return x.run()
    """)

    assert "main.A.run" not in result.get("main.main", set())
    assert "main.B.run" not in result.get("main.main", set())


def test_a_relative_import_is_resolved_against_its_package(tmp_path):
    from mpfi.static_graph import call_graph

    package = tmp_path / "prince"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "base.py").write_text(
        textwrap.dedent("""
        class Estimator:
            def fit(self, df):
                return df
    """)
    )
    (package / "model.py").write_text(
        textwrap.dedent("""
        from .base import Estimator

        class Model(Estimator):
            def run(self, df):
                return self.fit(df)
    """)
    )

    result = call_graph(package)

    assert "base.Estimator.fit" in result["model.Model.run"]


def test_a_class_from_a_sibling_module_is_known(tmp_path):
    from mpfi.static_graph import call_graph

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "steps.py").write_text(
        textwrap.dedent("""
        class Scaler:
            def transform(self, df):
                return df
    """)
    )
    (package / "run.py").write_text(
        textwrap.dedent("""
        from steps import Scaler

        def main(df):
            scaler = Scaler()
            return scaler.transform(df)
    """)
    )

    result = call_graph(package)

    assert "steps.Scaler.transform" in result["run.main"]


def test_super_reaches_the_base_class_method():
    result = graph("""
        class Base:
            def fit(self, df):
                return df

        class Model(Base):
            def fit(self, df):
                return super().fit(df)
    """)

    assert "main.Base.fit" in result["main.Model.fit"]


def test_an_attribute_holding_a_fresh_instance_is_followed_across_methods():
    """`self.scaler_ = StandardScaler()` in fit, used in transform."""
    result = graph("""
        class Scaler:
            def transform(self, df):
                return df

        class Model:
            def fit(self, df):
                self.scaler_ = Scaler()
                return df

            def apply(self, df):
                return self.scaler_.transform(df)
    """)

    assert "main.Scaler.transform" in result["main.Model.apply"]

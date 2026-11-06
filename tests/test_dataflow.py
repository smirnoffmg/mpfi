"""MPFI over data flow: the same two questions, asked of flat notebook code.

A notebook writes feature engineering as top-level statements, not functions, so
the function graph sees nothing. These tests pin down the data-flow reading: what
reaches the final model, which statements derive features, and how long and how
tangled the derivation is — once per table operation, once per column.
"""

import textwrap

from mpfi.dataflow import flow_components


def measure(source: str):
    return flow_components(textwrap.dedent(source))


CHAIN = """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    df = pd.read_csv("train.csv")
    df = df[df["a"] < 100]
    df["r"] = df["a"] / df["b"]
    df["s"] = df["r"] + df["c"]
    X = df.drop("target", axis=1)
    y = df["target"]
    model = RandomForestClassifier()
    model.fit(X, y)
"""


def test_operation_chain_counts_each_deriving_statement():
    result = measure(CHAIN)
    # filter -> r -> s; the projection to X is not feature engineering
    assert result.op_pdd == 3
    assert result.op_nodes == 3


def test_column_depth_follows_derivation_not_row_filters():
    result = measure(CHAIN)
    # s <- r <- a, b; the row filter selects rows and derives no column
    assert result.col_pdd == 2
    assert result.col_nodes == 2


def test_column_coupling_counts_edges_between_derived_columns():
    result = measure(CHAIN)
    # r -> s is the only edge between derived columns
    assert result.col_fci == 0.5
    # r has parents a, b; s has r, c
    assert result.col_fanin == 2.0


def test_code_that_does_not_reach_the_final_model_is_left_out():
    result = measure(
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        df = pd.read_csv("train.csv")
        df["log_a"] = df["a"].apply(lambda v: v + 1)
        plt.hist(df["log_a"])
        other = df.copy()
        other["z"] = other["a"] * 2
        other["w"] = other["z"] * 2
        first.fit(other[["z", "w"]], df["target"])
        df["r"] = df["a"] / df["b"]
        final.fit(df[["r", "c"]], df["target"])
        """
    )
    assert result.col_pdd == 1
    assert result.col_nodes == 1


def test_loop_over_a_literal_column_list_is_unrolled():
    result = measure(
        """
        import pandas as pd
        df = pd.read_csv("train.csv")
        for col in ["a", "b"]:
            df[col] = df[col].fillna(0)
        model.fit(df[["a", "b"]], df["target"])
        """
    )
    assert result.col_nodes == 2
    assert result.col_pdd == 1
    assert result.op_pdd == 2


def test_loop_over_a_named_literal_list_is_unrolled():
    result = measure(
        """
        import pandas as pd
        cols = ["a", "b"]
        df = pd.read_csv("train.csv")
        for col in cols:
            df[col + "_sq"] = df[col] ** 2
        model.fit(df[["a_sq", "b_sq"]], df["target"])
        """
    )
    assert result.col_nodes == 2


def test_frame_wide_transform_derives_every_column():
    result = measure(
        """
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        df = pd.read_csv("train.csv")
        df["r"] = df["a"] / df["b"]
        X = df[["a", "r"]]
        X = StandardScaler().fit_transform(X)
        model.fit(X, df["target"])
        """
    )
    # r (1) -> scaled r (2)
    assert result.col_pdd == 2
    assert result.op_pdd == 2


def test_inplace_method_redefines_its_frame():
    result = measure(
        """
        import pandas as pd
        df = pd.read_csv("train.csv")
        df.fillna(0, inplace=True)
        df["r"] = df["a"] / df["b"]
        model.fit(df[["r"]], df["target"])
        """
    )
    assert result.op_pdd == 2
    assert result.col_pdd == 2


def test_split_and_library_datasets_pass_the_frame_through():
    result = measure(
        """
        import pandas as pd
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        df = pd.read_csv("train.csv")
        df["r"] = df["a"] / df["b"]
        X = df.drop("target", axis=1)
        y = df["target"]
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2)
        booster = lgb.train(params, lgb.Dataset(X_tr, label=y_tr))
        """
    )
    assert result.col_pdd == 1
    assert result.op_pdd == 1


def test_label_lineage_does_not_count_as_features():
    result = measure(
        """
        import pandas as pd
        import numpy as np
        df = pd.read_csv("train.csv")
        y = np.log1p(df["price"])
        y = y.clip(0, 10)
        df["r"] = df["a"] / df["b"]
        model.fit(df[["r"]], y)
        """
    )
    assert result.col_pdd == 1


def test_fit_inside_a_user_function_is_the_sink():
    result = measure(
        """
        import pandas as pd
        import lightgbm as lgb
        def kfold_lightgbm(data, features):
            train_x = data[features]
            clf = lgb.LGBMClassifier()
            clf.fit(train_x, data["target"], eval_set=[(train_x, data["target"])])
        df = pd.read_csv("train.csv")
        df["r"] = df["a"] / df["b"]
        df["s"] = df["r"] * 2
        kfold_lightgbm(df, ["s"])
        """
    )
    assert result.col_pdd == 2
    assert result.sink_line == 7


def test_function_mutating_its_argument_in_place_changes_the_callers_frame():
    result = measure(
        """
        import pandas as pd
        def add_ratio(frame):
            frame["r"] = frame["a"] / frame["b"]
        df = pd.read_csv("train.csv")
        add_ratio(df)
        model.fit(df[["r"]], df["target"])
        """
    )
    assert result.col_pdd == 1


def test_column_names_built_with_format_from_parameters_resolve():
    result = measure(
        """
        import pandas as pd
        def add_price(frame, col, kind):
            frame["{}_{}_price".format(col, kind)] = frame[col] * 2
            frame["{}_{}_price".format(col, kind)] = frame[
                "{}_{}_price".format(col, kind)
            ].astype(int)
            return frame
        df = pd.read_csv("train.csv")
        for col in ["a", "b"]:
            for kind in ["mean", "max"]:
                df = add_price(df, col, kind)
        model.fit(df[["a_mean_price", "b_max_price"]], df["target"])
        """
    )
    # each price column: derived, then cast; the others are untouched
    assert result.col_pdd == 2
    assert result.col_nodes == 4


def test_recursive_function_does_not_loop():
    result = measure(
        """
        import pandas as pd
        def grow(frame):
            frame["r"] = frame["a"] * 2
            return grow(frame)
        df = grow(pd.read_csv("train.csv"))
        model.fit(df, df["target"])
        """
    )
    assert result.col_pdd >= 1


def test_user_function_on_a_frame_is_followed_inside():
    result = measure(
        """
        import pandas as pd
        def add_features(frame):
            frame["r"] = frame["a"] / frame["b"]
            return frame
        df = pd.read_csv("train.csv")
        df = add_features(df)
        model.fit(df, df["target"])
        """
    )
    assert result.op_pdd == 1
    assert result.col_pdd == 1


def test_unknown_loop_variable_touches_every_column():
    result = measure(
        """
        import pandas as pd
        df = pd.read_csv("train.csv")
        for col in df.columns:
            df[col] = df[col].fillna(0)
        model.fit(df, df["target"])
        """
    )
    assert result.col_pdd == 1
    assert result.op_pdd == 1


def test_groupby_aggregate_is_one_derivation():
    result = measure(
        """
        import pandas as pd
        df = pd.read_csv("train.csv")
        df["m"] = df.groupby("k")["v"].transform("mean")
        model.fit(df[["m"]], df["target"])
        """
    )
    assert result.col_pdd == 1
    assert result.col_fanin == 2.0


def test_last_model_fit_is_the_sink():
    result = measure(
        """
        import pandas as pd
        df = pd.read_csv("train.csv")
        df["r"] = df["a"] / df["b"]
        df["s"] = df["r"] * 2
        model.fit(df[["s"]], df["target"])
        model.fit(df[["r"]], df["target"])
        """
    )
    assert result.col_pdd == 1


def test_transformer_fit_with_one_argument_is_not_a_sink():
    result = measure(
        """
        import pandas as pd
        df = pd.read_csv("train.csv")
        df["r"] = df["a"] / df["b"]
        model.fit(df[["r"]], df["target"])
        scaler.fit(df[["a"]])
        """
    )
    assert result.col_pdd == 1


def test_no_model_fit_scores_zero():
    result = measure(
        """
        import pandas as pd
        df = pd.read_csv("train.csv")
        df["r"] = df["a"] / df["b"]
        """
    )
    assert result.op_pdd == 0
    assert result.col_pdd == 0
    assert result.op_fci == 0.0
    assert result.sink_line is None


def test_unparsable_source_scores_zero():
    assert measure("df = (").op_pdd == 0

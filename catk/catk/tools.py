"""Package catk Correspondence Analysis Toolkit:common tooling"""

import numpy as np
import pandas as pd

from .config import SUM_SYMBOL


def reindex_from(values: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """shape np.array using given DF, similar to pd.DataFrame().reindex_like(df)"""
    res = pd.DataFrame(values)
    res.index = df.index.copy()
    res.columns = df.columns.copy()
    return res


def add_margins(df: pd.DataFrame, *, cols_sum: bool = True, rows_sum: bool = True):
    """add margin sums a.k.a. masses to a DataFrame"""

    c, r, n = get_margins(df)

    # both margins
    if cols_sum and rows_sum:
        res = pd.concat([df, c], axis=0)
        # add now to avoid float conversion
        r.loc[SUM_SYMBOL] = n
        res = pd.concat([res, r], axis=1)
        # res.loc[SUM_SYMBOL, SUM_SYMBOL] = n

    # column-wise marginal sum: a new line to stack (vertically) at the end
    elif cols_sum:
        res = pd.concat([df, c], axis=0)

    # row-wise marginal sum: a new column to stack (horizontally) at the end
    elif rows_sum:
        res = pd.concat([df, r], axis=1)

    else:
        raise ValueError(f"add at least one rows or cols marginal sums {cols_sum = } {rows_sum = }")

    return res  # .astype(np.int64)


def get_margins(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, np.int64]:
    """extract row/cols margins and grand total from a DF without marginal sums."""

    # pd equiv of N @ np.ones((4,1), "int64")
    col_sum = pd.DataFrame(df.sum(axis=0)).T
    col_sum.index = [SUM_SYMBOL]

    # pd equiv of np.ones((1,4), "int64") @N
    row_sum = pd.DataFrame(df.sum(axis=1))
    row_sum.columns = [SUM_SYMBOL]

    assert np.isclose(col_sum.sum(axis=1), row_sum.sum(axis=0))

    return col_sum, row_sum, row_sum.sum().sum()

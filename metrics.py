"""Metrics for 2x2 coningency to R"""

from pathlib import Path
import pandas as pd
import numpy as np

import scopus.biblio_extractor as bex


def label(s):
    """Add label attribute to a function"""

    def wrapper(f):
        f.label = s
        return f

    return wrapper


@label("Simple projection")
def tt_projection_metric(arr):
    """The trivial one. Just keep the bottom right cell. Amount to compute on "the old matrix" """
    [_, _], [_, TT] = arr.reshape(2, 2)
    return TT


@label(f"% of compound with the activity")
def row_implication_metric(arr):
    """The % of paper that have both keywords among the first one. Values add to 1 row wise"""
    [_, _], [TF, TT] = arr.reshape(2, 2)
    return TT / (TF + TT)


@label(f"% of activity with the compound")
def col_implication_metric(arr):
    """The % of paper that have both keywords among the second one. Values add to 1 col wise"""
    [_, FT], [_, TT] = arr.reshape(2, 2)
    return TT / (FT + TT)


@label("Fowlkes-Mallows index")
def fowlkes_mallows_metric(arr):
    """Computes the Fowlkes-Mallows index: sqrt(the product of row and col implication)"""
    [_, FT], [TF, TT] = arr.reshape(2, 2)
    return TT / np.sqrt((FT + TT) * (TF + TT))


@label("Accuracy")
def accuracy_metric(arr):
    """Computes the accuracy"""
    [FF, FT], [TF, TT] = arr.reshape(2, 2)
    return (TT + FF) / (FF + FT + TF + TT)


@label("Custom metric #1")
def x_metric(arr):
    """Some another one by our own"""
    [FF, FT], [TF, TT] = arr.reshape(2, 2)
    return (FF + TT - FT - TF) / (FF + FT + TF + TT)


@label("Custom metric #2")
def fraction_metric(arr):
    """The % of paper that have both keywords among the ones having at least one."""
    [_, FT], [TF, TT] = arr.reshape(2, 2)
    return TT / (FT + TF - TT)


@label("The odds of having both the compound and the activity")
def odds_metric(arr):
    """The odds of having related keywords."""
    [FF, FT], [TF, TT] = arr.reshape(2, 2)
    # adds extra 0.5 to avoid division by 0.
    return (FF * TT + 0.5) / (FT * TF + 0.5)


@label("Logodds")
def logodds_metric(arr):
    """The log odds of having related keywords."""
    # BUG : on arrive à avoir du négatif ici
    return np.log(odds_metric(arr))


def apply_metric(data, func):
    """Apply the 2x2->R function on each submatrix"""
    C, A = len(data.index) // 2, len(data.columns) // 2
    values = np.moveaxis(data.values.reshape((C, 2, A, 2)), 1, -2).reshape((C * A, 4))
    matrix = np.apply_along_axis(func, 1, values).reshape((C, A))
    sub = data.xs(bex.SELECTORS[True], axis=0, level=2).xs(bex.SELECTORS[True], axis=1, level=2)
    df = pd.DataFrame(matrix, index=sub.index, columns=sub.columns)
    df.index.name = "Compounds"
    df.columns.name = "Activities"
    return df


metrics = [
    tt_projection_metric,
    row_implication_metric,
    col_implication_metric,
    fowlkes_mallows_metric,
    odds_metric,
    accuracy_metric,
]

if __name__ == "__main__":
    print(metrics)
    DATASET_FILENAME = Path("results/pharmaco_chemistry_2_cross_2022-10-24_15-57-13.csv")
    dataset, margin_rows, margin_cols, number_of_papers = bex.load_results(DATASET_FILENAME)
    df = apply_metric(dataset, odds_metric)
    print(tt_projection_metric)
    print(tt_projection_metric.label)

#%%
import pandas as pd
from pandas import DataFrame
import numpy as np


def row(dataset, seuil) -> DataFrame:
    """Remove all the row < seuil and return a new Dataframe
    Args:
        dataset (dataframe):
        seuil (int): the min value for remove the small one
    """

    good = dataset.xs("w/", axis = 0,  level=2).sum(axis=1) / (len(dataset.columns)/2) > seuil
    good = pd.concat([good]*2).sort_index()
    good.index = dataset.index
    return dataset[good]


def column(dataset : DataFrame, seuil) -> DataFrame:
    """Remove all the column < seuil and return a new Dataframe
    Args:
        dataset (dataframe):
        seuil (int): the min value for remove the small one
    """
    
    return row(dataset.T, seuil).T


def by_column_name(dataset, *categorie) -> DataFrame:
    """keep only the specific column

    Args:
        dataset (DataFrame):

    Returns:
        DataFrame: new dataframe
    """
    return dataset[(categorie)]

def by_row_name(dataset, *categorie) -> DataFrame:
    """keep only the specific row

    Args:
        dataset (DataFrame):

    Returns:
        DataFrame: new dataframe
    """
    return dataset.loc[(categorie)]


def keep_with_or_without(dataset, value) -> DataFrame:
    """Keep only the 3rd index choice

    Args:
        dataset (dataframe):
        value (str): the 3rd that you want to keep ("w/","w/o")

    Returns:
        DataFrame:
    """
    return dataset.loc[:, :, value].T.loc[:, :, value].T


# def intersection_metric(arr):
#     """The trivial one. Just keep the bottom right cell. Amount to compute on "the old matrix" """
#     [FF, FT], [TF, TT] = arr.reshape(2, 2)
#     return TT / (FF + FT + TF + TT)


def tt_projection_metric(arr):
    """The trivial one. Just keep the bottom right cell. Amount to compute on "the old matrix" """
    [FF, FT], [TF, TT] = arr.reshape(2, 2)
    return TT


def row_implication_metric(arr):
    """The % of paper that have both keywords among the first one. Values add to 1 row wise"""
    [FF, FT], [TF, TT] = arr.reshape(2, 2)
    return TT / (TF + TT)


def col_implication_metric(arr):
    """The % of paper that have both keywords among the second one. Values add to 1 col wise"""
    [FF, FT], [TF, TT] = arr.reshape(2, 2)
    return TT / (FT + TT)


def fowlkes_mallows_metric(arr):
    """Computes the Fowlkes-Mallows index: sqrt(the product of row and col implication)"""
    [FF, FT], [TF, TT] = arr.reshape(2, 2)
    return TT / np.sqrt((FT + TT) * (TF + TT))


def accuracy_metric(arr):
    """Computes the accuracy"""
    [FF, FT], [TF, TT] = arr.reshape(2, 2)
    return (TT + FF) / (FF + FT + TF + TT)


def x_metric(arr):
    """Some another one by our own"""
    [FF, FT], [TF, TT] = arr.reshape(2, 2)
    return (FF + TT - FT - TF) / (FF + FT + TF + TT)


def fraction_metric(arr):
    """The % of paper that have both keywords among the ones having at least one."""
    [FF, FT], [TF, TT] = arr.reshape(2, 2)
    return TT / (FT + TF - TT)


def odds_metric(arr):
    """The odds of having related keywords."""
    [FF, FT], [TF, TT] = arr.reshape(2, 2)
    return (FF * TT + 0.5) / (FT * TF + 0.5)


def logodds_metric(arr):
    """The log odds of having related keywords."""
    return np.log(odds_metric(arr))


metrics = [
    tt_projection_metric,
    row_implication_metric,
    col_implication_metric,
    fowlkes_mallows_metric,
    fraction_metric,
    odds_metric,
    accuracy_metric,
        # logodds_metric,
]




# %%

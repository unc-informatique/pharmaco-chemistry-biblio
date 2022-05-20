#%%
import pandas as pd
from pandas import DataFrame



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






# %%

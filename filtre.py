from pandas import DataFrame


def row(dataset, seuil) -> DataFrame:
    """_summary_
    Args:
        dataset (dataframe): _description_
        seuil (int): _description_
    """
    good = dataset.sum(1) > seuil
    for i in range(0, len(good), 2):
        res = good[i] and good[i+1]
        good[i] = res
        good[i+1] = res
    return dataset[good]


def column(dataset, seuil) -> DataFrame:
    """_summary_
    Args:
        dataset (dataframe): _description_
        seuil (int): _description_
    """
    good = dataset.sum(0) > seuil
    for i in range(0, len(good), 2):
        res = good[i] and good[i+1]
        good[i] = res
        good[i+1] = res

    return dataset.T[good.T].T


def ByColumnName(dataset, *categorie) -> DataFrame:
    """_summary_

    Args:
        dataset (_type_): _description_

    Returns:
        DataFrame: _description_
    """
    for arg in categorie:
        dataset = dataset[arg]
    return dataset

def ByRowName(dataset, *categorie) -> DataFrame:
    """_summary_

    Args:
        dataset (dataframe): _description_
        categorie (str): _description_
        activities (str): _description_

    Returns:
        dataframe: _description_
    """
    dataset = dataset.T
    for arg in categorie:
        dataset = dataset[arg]
    return dataset.T


def KeepWithOrWithOut(dataset, value) -> DataFrame:
    """_summary_

    Args:
        dataset (dataframe): _description_
        value (str): _description_

    Returns:
        DataFrame: _description_
    """
    return dataset.loc[:, :, value].T.loc[:, :, value].T
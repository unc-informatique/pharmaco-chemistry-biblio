"""Package catk Correspondence Analysis Toolkit: sample dataset module"""

from itertools import product
from pathlib import Path
import pandas as pd
from ..config import SUM_SYMBOL  # pylint: disable=unused-import

_ROOT = Path(__file__).parent.resolve()

# %%


def get_colours(with_margins: bool = False) -> pd.DataFrame:
    """classical example from Snee (1974)"""

    # contingencies in flat format : use pivot_table with or without margins
    # delim_whitespace=True
    colours = pd.read_csv(_ROOT / "colours.csv", delimiter=",").pivot_table(
        index="eyes",
        columns="hair",
        values="nb",
        margins=with_margins,
        margins_name=SUM_SYMBOL,
        aggfunc="sum",
    )

    # order used in
    pad = [SUM_SYMBOL] if with_margins else []
    colours = colours.reindex(
        columns=["black", "brown", "red", "blond"] + pad,
        index=["brown", "hazel", "green", "blue"] + pad,
    )
    return colours


def get_vitamin() -> pd.DataFrame:
    """Simple 2X2 contingency table"""

    # contingencies in 2D format, everything is fine (but col/rows names)
    vitamin = pd.read_csv(_ROOT / "vitamin_c.csv", delimiter=",", index_col=0)
    vitamin.index.name = "treatment"
    vitamin.columns.name = "sickness"  # pylint: disable=no-member

    return vitamin


def get_smokers() -> pd.DataFrame:
    """classical example from Greenacre 1984"""

    # contingencies in 2D format, everything is fine (but col/rows names)
    smokers = pd.read_csv(_ROOT / "smokers.csv", delimiter=",", index_col=0)
    smokers.index.name = "Staff group"
    smokers.columns.name = "Smoking category"  # pylint: disable=no-member

    return smokers


def get_wisconsin(with_margins: bool = False) -> pd.DataFrame:
    """Example from Fienberg (1977)"""

    # contingencies in flat format : use pivot_table with or without margins

    wisconsin = pd.read_csv(_ROOT / "wisconsin.csv", delimiter=",").pivot_table(
        index=["Social Stratum", "Parental Encouragement"],
        columns=["College Plans"],
        values="Count",
        aggfunc="sum",
        margins=with_margins,
        margins_name=SUM_SYMBOL,
    )

    # order used in paper
    stratums = ["Lower", "Lower Middle", "Upper Middle", "Upper"]
    encouragements = ["Low", "High"]
    pad = [(SUM_SYMBOL, "")] if with_margins else []
    idx = list(product(stratums, encouragements)) + pad
    wisconsin = wisconsin.reindex(index=pd.MultiIndex.from_tuples(idx))
    return wisconsin


print(f"'{__file__}' loaded")

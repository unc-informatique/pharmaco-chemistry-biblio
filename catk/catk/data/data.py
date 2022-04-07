"""Package catk Correspondence Analysis Toolkit: sample dataset module"""
from pathlib import Path
import pandas as pd
from ..config import SUM_SYMBOL  # pylint: disable=unused-import

_ROOT = Path(__file__).parent.resolve()

# %%


def get_colours(with_margins: bool = False) -> pd.DataFrame:
    """classical example from Snee (1974)"""

    # contingencies in flat format : use pivot_table with or without margins
    colours = pd.read_csv(_ROOT / "colours.csv", delim_whitespace=True).pivot_table(
        index="eyes",
        columns="hair",
        values="nb",
        margins=with_margins,
        margins_name=SUM_SYMBOL,
        aggfunc="sum",
    )

    # order used in
    colours = colours.reindex(
        columns=["black", "brown", "red", "blond"] + ([SUM_SYMBOL] if with_margins else []),
        index=["brown", "hazel", "green", "blue"] + ([SUM_SYMBOL] if with_margins else []),
    )
    return colours


def get_vitamin() -> pd.DataFrame:
    """Simple 2X2 contingency"""

    # contingencies in 2D format, everything is fine (but col/rows names)
    vitamin = pd.read_csv(_ROOT / "vitamin_c.csv", delimiter=",", index_col=0)
    vitamin.index.name = "treatment"
    vitamin.columns.name = "sickness"  # pylint: disable=no-member

    return vitamin


print(f"'{__file__}' loaded")

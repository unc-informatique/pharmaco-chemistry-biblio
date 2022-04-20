"""A demo for IPython"""
# pylint: disable=unused-import
# %%

from itertools import product
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import poisson
import statsmodels.api as sm

from IPython.display import display

from catk import CA, SUM_SYMBOL
from catk.data import get_colours, get_vitamin, get_smokers
from catk.tools import add_margins, reindex_from, get_margins

pd.set_option("display.precision", 2)

# # logging.basicConfig(level=logging.DEBUG)

print("Data: smokers")
df = get_smokers()
display(add_margins(df))

ca = CA()
ca.fit(df)

contribs = ca.contributions(2)
display(contribs)

# %%
fig, ax = plt.subplots()
ca.plot(ax=ax, coords=("standard", "principal"), legend=False)
plt.show()

# for coords in product(["standard", "principal"], repeat=2):
#     print(coords)
#     ca.plot(coords=coords, legend=False)
#     plt.show()
print("end")

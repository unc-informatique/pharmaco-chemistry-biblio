"""A demo for IPython"""
# pylint: disable=unused-import
# %%

import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display

from catk import CA
from catk.data import get_colours, get_vitamin
from catk.tools import add_margins, reindex_from, get_margins

pd.set_option("display.precision", 2)

# logging.basicConfig(level=logging.DEBUG)

print("Data: hair/eye colours")
df = get_colours(False)
display(add_margins(df))

ca = CA()
ca.fit(df)

print("Expectation: hair/eye colours")
display(add_margins(reindex_from(ca.n * ca.E, df)))

# display(100 * df.div(cols_margin, axis=1))
# display(100 * df.div(rows_margin, axis=0))

axes = ca.axes()
contribs = ca.contributions()

display(axes)
display(contribs)

ca.plot((1, 2))
print("End of demo")
# %%

# reconstruction

assert np.isclose(ca.S, ca.U @ ca.Da @ ca.Vt).all()


for k_rank in range(0, len(ca.Da)):
    print(f"Approximation at rank {k_rank}")
    mask = np.diag([1] * k_rank + [0] * (len(ca.Da) - k_rank))
    P = np.diag(ca.c ** 0.5) @ ca.U @ (ca.Da * mask) @ ca.Vt @ np.diag(ca.r ** 0.5) + ca.E
    display(reindex_from((ca.n * P), df).astype(int))
# %%

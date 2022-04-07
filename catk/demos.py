"""A demo for IPython"""
# pylint: disable=unused-import
# %%

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
from catk.data import get_colours, get_vitamin
from catk.tools import add_margins, reindex_from, get_margins

pd.set_option("display.precision", 2)

# logging.basicConfig(level=logging.DEBUG)

print("Data: hair/eyes colours")
df = get_colours(False)
display(add_margins(df))

ca = CA()
ca.fit(df)

assert np.isclose(ca.S, ca.U @ ca.Da @ ca.Vt).all()

print("Expectation: hair/eyes colours")
display(add_margins(reindex_from(ca.n * ca.E, df)))

# display(100 * df.div(cols_margin, axis=1))
# display(100 * df.div(rows_margin, axis=0))

axes = ca.axes()
contribs = ca.contributions(2)

display(axes)
display(contribs)

fig, ax = plt.subplots(1, 1)

ca.plot(ax=ax)
plt.show()
print("End of demo")
# %%
# reconstruction


for k_rank in range(0, len(ca.Da)):
    display(reindex_from(df - ca.n * ca.approx(k_rank), df).astype(int))


# %%

# Bootstrap : sampler k = 592 avec remise parmis ceux qu'on a déjà
# on tire au hasard dans I*J en suivant les probas de la table des frequence
# une multinomiale à I*J paramètres ?
I, J = df.shape

BOOTSRAPS = 10

for i in range(BOOTSRAPS):
    xs, ys = np.divmod(np.random.choice(I * J, size=ca.n, p=(ca.N / ca.n).flatten()), I)

    bs = reindex_from(pd.crosstab(xs, ys, margins=False, margins_name=SUM_SYMBOL), df)
    # display(bs)
    ax=CA().fit(bs).plot(legend=False, hue=ca.index.to_list()+ca.columns.to_list(), size=False)
plt.show()

# %%


# %%
# cas 2x2

df2 = get_vitamin()
ca2 = CA()
ca2.fit(df2)
display(ca2.axes())
display(ca2.contributions())


# %%
# cas chemotaxo

# DATASET_FILENAME = Path("../results/activities_2022-01-29_16-33-05.csv")
# dataset = pd.read_csv(DATASET_FILENAME, index_col=[0, 1, 2], header=[0, 1, 2])

# df_chemo = get_vitamin()
# ca2 = CA()
# ca2.fit(df2)
# display(ca2.axes())
# display(ca2.contributions())

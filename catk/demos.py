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

# # logging.basicConfig(level=logging.DEBUG)

# print("Data: hair/eyes colours")
# df = get_colours(False)
# # pour tester dans le cas I != J
# # df.drop(columns=["red"], inplace=True)
# # df = df.T

# display(add_margins(df))

# ca = CA()
# ca.fit(df)

# assert np.isclose(ca.S, ca.U @ ca.Da @ ca.Vt).all()

# print("Expectation: hair/eyes colours")
# display(add_margins(reindex_from(ca.n * ca.E, df)))

# # display(100 * df.div(cols_margin, axis=1))
# # display(100 * df.div(rows_margin, axis=0))

# axes = ca.axes()
# display(axes)
# contributions = ca.contributions(2)
# display(contributions)

# #       Coords (std)       Coords (princ.)       Contributions        Cosine²        Mass (%)  Kind
# #                  1     2               1     2             1      2       1      2               
# # brown        -1.10  1.44           -0.50  0.21         22.25  37.88   83.80  15.19    37.16  eyes
# # hazel        -0.32 -0.22           -0.15 -0.03          5.09   2.32   86.44   4.20    15.71  eyes
# # green        -0.28 -2.14           -0.13 -0.32          0.96  55.13   13.33  81.18    10.81  eyes
# # blue          1.83  0.47            0.84  0.07         71.70   4.67   99.27   0.69    36.32  eyes
# # black        -1.08  0.59           -0.49  0.09         43.12  13.04   96.70   3.11    18.24  hair
# # brown        -0.47 -1.12           -0.21 -0.17          3.40  19.80   54.24  33.63    48.31  hair
# # red           0.35 -2.27            0.16 -0.34          1.35  55.91   17.59  77.26    11.99  hair
# # blond         1.20  0.56            0.55  0.08         52.13  11.24   97.75   2.24    21.45  hair


# # sns.despine()
# # ca.plot()
# # plt.show()
# print("End of demo")
# # %%
# # reconstruction


# # for k_rank in range(0, len(ca.Da)):
# #     display(reindex_from(df - ca.n * ca.approx(k_rank), df).astype(int))


# # %%

# # Bootstrap : sampler k = 592 avec remise parmis ceux qu'on a déjà
# # on tire au hasard dans I*J en suivant les probas de la table des frequence
# # une multinomiale à I*J paramètres ?
# I, J = df.shape

# BOOTSRAPS = 0

# for i in range(BOOTSRAPS):
#     xs, ys = np.divmod(np.random.choice(I * J, size=ca.n, p=(ca.N / ca.n).flatten()), I)

#     bs = reindex_from(pd.crosstab(xs, ys, margins=False, margins_name=SUM_SYMBOL), df)
#     # display(bs)
#     ax = CA().fit(bs).plot(legend=False, hue=ca.index.to_list() + ca.columns.to_list(), size=False)
# plt.show()

# %%


# %%
# cas 2x2

df2 = get_vitamin()
ca2 = CA()
ca2.fit(df2)
display(ca2.axes())
display(ca2.contributions())
ca2.plot()
plt.show()

# %%
# DATASET_FILENAME = Path("../results/activities_2022-01-29_16-33-05.csv")
# dataset = pd.read_csv(DATASET_FILENAME, index_col=[0, 1, 2], header=[0, 1, 2])

# df_chemo = get_vitamin()
# ca2 = CA()
# ca2.fit(df2)
# display(ca2.axes())
# display(ca2.contributions())

"""Same with prince"""

# %%
# pylint: disable=unused-import
# pip install git+https://github.com/MaxHalford/Prince


import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import linalg
from scipy.stats import chi2_contingency


import prince

pd.set_option("display.precision", 2)
sns.set_theme(style="whitegrid", font_scale=1.10, rc={"figure.figsize": (8, 6)})

SUM_SYMBOL = "Σ"

# %%

# use margins from pivot_table
df = pd.read_csv("data/colours.csv", delim_whitespace=True).pivot_table(
    index="eyes",
    columns="hair",
    values="nb",
    # margins=True,
    # margins_name=SUM_SYMBOL,
    aggfunc=np.sum,
)

display(df)

# %%

ca = prince.CA(
    n_components=3,
    # n_iter=3,
    # copy=True,
    # check_input=True,
    engine="auto",
    random_state=42,
)

ca.fit(df)

# %%
ca.eigenvalues_
# %%
ca.explained_inertia_
# %%
ca.row_masses_
# %%
ca.row_coordinates(df)
# %%
ca.column_coordinates(df)
# %%
ca.row_contributions()
# %%
ca.column_contributions()
# %%
ca.row_cos2()
# %%
ca.column_cos2()

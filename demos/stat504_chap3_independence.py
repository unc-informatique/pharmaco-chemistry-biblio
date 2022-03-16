"""Stat504 @ UPenn

Applications de Two-Way Tables: Independence and Association

https://online.stat.psu.edu/stat504/lesson/3
"""

# pylint: disable=pointless-statement
#%%
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import power_divergence

FILENAME = Path("data/vitamin_c.csv")
# forcage du cas pour éviter que MyPy gueule
df = pd.DataFrame(pd.read_csv(FILENAME, index_col=0))
# .sort_index(axis=0).sort_index(axis=1)

df

# %%
margin_rows = df.sum(axis=1).values.reshape(2, 1)
margin_cols = df.sum(axis=0).values.reshape(1, 2)
N = df.sum().sum()
N

# %%
# https://online.stat.psu.edu/stat504/lesson/3/3.3

# effectifs observés
observed = df.values

# produit des marges : effectifs théoriques
expected = margin_rows @ margin_cols / N
expected

#%%
chi = power_divergence(
    observed.flatten(),
    expected.flatten(),
    ddof=2,  # attention : ici le delta ! dof = k - 1 - ddof
    lambda_="pearson",
)

# Pearson goodness-of-fit statistic
# coefficient du Chi2 à 1 = (J-1)(I-1) degrés des liberté
chi
#%%
div = power_divergence(
    observed.flatten(),
    expected.flatten(),
    ddof=2,  # attention : ici le delta ! dof = k - 1 - ddof
    lambda_="log-likelihood",
)
# likelihood-ratio chi-squared test statistic.
div


# %%
# https://online.stat.psu.edu/stat504/lesson/3/3.4

# difference in proportions, "row implication"
df.values / margin_rows

odds = (observed[0][0] * observed[1][1]) / (observed[1][0] * observed[0][1])
odds = 1 / odds

log_conf = 1.96 * np.sqrt((1 / observed).sum())
log_int = np.array([np.log(odds) + log_conf, np.log(odds) - log_conf])
print(odds, np.exp(log_int))

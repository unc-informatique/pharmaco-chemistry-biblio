"""Correspondence Analysis in Python

A custom alternative to <https://github.com/MaxHalford/prince>

We use the convention in <https://en.wikipedia.org/wiki/Correspondence_analysis>
"""
# %%
# pylint: disable=unused-import


from itertools import product

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import linalg
from scipy.stats import chi2_contingency


pd.set_option("display.precision", 2)
sns.set_theme(style="whitegrid", font_scale=1.10, rc={"figure.figsize": (8, 6)})

SUM_SYMBOL = "Σ"

# %%
# classical example from Snee (1974)

FILE = "../catk/catk/data/colours.csv"

# use margins from pivot_table
df_colours_margins = pd.read_csv(FILE, delim_whitespace=True).pivot_table(
    index="eyes",
    columns="hair",
    values="nb",
    margins=True,
    margins_name=SUM_SYMBOL,
    aggfunc=np.sum,
)

# pour être dans le même ordre que dans le livre
df_colours_margins = df_colours_margins.reindex(
    columns=["black", "brown", "red", "blond", SUM_SYMBOL],
    index=["brown", "hazel", "green", "blue", SUM_SYMBOL],
)


def reshape(values, df):
    """shape np.array using DF"""
    # res = pd.DataFrame().reindex_like(df)
    # res.loc[::, ::] = values
    res = pd.DataFrame(values)
    res.index = df.index.copy()
    res.columns = df.columns.copy()
    return res


def add_margins(df, *, col_sums=True, row_sums=True):
    """ajout des sommes marginales à un DF 2D"""
    # la somme des colonnes est une ligne à ajouter verticalement
    if col_sums:
        col_sum = pd.DataFrame(df.sum(axis=0)).T
        col_sum.index = [SUM_SYMBOL]
        ca_df = pd.concat([df, col_sum], axis=0)
    else:
        ca_df = df.copy()

    # la somme des lignes est une colonne à ajouter horizontalement
    if row_sums:
        row_sum = pd.DataFrame(ca_df.sum(axis=1))
        row_sum.columns = [SUM_SYMBOL]
        ca_df = pd.concat([ca_df, row_sum], axis=1)

    return ca_df


def split_margins(df) -> tuple[pd.DataFrame, pd.Series, pd.Series, np.int64]:
    """extract row/cols margins and grand total from a df"""
    return (
        df.drop(index=SUM_SYMBOL, columns=SUM_SYMBOL),
        df.loc[SUM_SYMBOL, ::],  # .drop(index=SUM_SYMBOL),
        df.loc[::, SUM_SYMBOL],  # .drop(index=SUM_SYMBOL),
        df.loc[SUM_SYMBOL, SUM_SYMBOL],
    )


display(df_colours_margins)
df_colours, cols_margin, rows_margin, N = split_margins(df_colours_margins)
# display(100 * df_colours_margins / N)
display(100 * df_colours_margins.div(cols_margin, axis=1))
display(100 * df_colours_margins.div(rows_margin, axis=0))


# %%

# "log-likelihood"
chi_score, p_value, dof, expected = chi2_contingency(df_colours, lambda_="pearson")
print(f"P-value = {p_value} with {dof} dof ({chi_score = })")
df_colours_expected = reshape(expected, df_colours)
display(df_colours_expected)


# %%

Z = df_colours.to_numpy() / N
display(100 * Z)

# %%

# 3 facons
# r = df_colours.sum(axis=1) / N
# r = rows_margin/N
r = np.sum(Z, axis=1)
c = np.sum(Z, axis=0)
E = r.reshape(-1, 1) @ c.reshape(1, -1)
# une autre façon de calculer expected
display(reshape(E * N, df_colours))

# %%
# centrage : différence entre % observés et attendus : la somme est nulle : par ligne ET par colonne
Zc = Z - E
display(reshape(100 * Zc, df_colours))
print(np.sum(Zc), Zc.sum(axis=1), Zc.sum(axis=0))
display(add_margins(100 * df_colours_expected / N))

# %%
# réduction par les poids lignes/colonnes
# - X @ np.diag([ci]) : revient à multiplier les colonnes par les facteur colonnes ci
# - np.diag([ri]) @ X : revient à multiplier les lignes par les facteur lignes ri
# in fine S_ij = Zc_ij / (sqrt(ri) * sqrt(ci))
#              = (M_ij/N - r_i*c_i)  / (sqrt(ri) * sqrt(ci))

# R = np.diag(1 / np.sqrt(r))
FACTOR = -0.5
R = np.diag(r ** FACTOR)
C = np.diag(c ** FACTOR)
# S = standardized residuals
S = R @ Zc @ C

# S = (df_colours - df_colours_expected)/N
# S = S.div(np.sqrt(c), axis = 1)
# S = S.div(np.sqrt(r), axis = 0)
display(reshape(100 * S, df_colours))


# %%
# decompose
I, J = S.shape
K = min(I, J)
U, D, Vt = linalg.svd(S)
# diagonal of sqrt of "eigenvalues"
Da = np.eye(K, K) * D
display(Da)
total_inertia = (Da ** 2).sum()


# SVD ensures that Vt  @ Vt.T == I == U.T @ U
assert np.allclose(Vt @ Vt.T, np.identity(J))
assert np.allclose(U.T @ U, np.identity(I))
# chi_score is the trace of (squared) eingenvalues times N
assert np.isclose(N * total_inertia, chi_score)
assert np.isclose(total_inertia, np.trace(S @ S.T))

# %%

inertias = D ** 2
inertias_pc = inertias / inertias.sum()
inertias_cum = np.cumsum(inertias_pc)

# tableau 4.3-1.
print(f"Inertie totale = {inertias.sum():.3f}. Contribution des VPs")
for i, j, k in zip(inertias, inertias_pc, inertias_cum):
    print(f"eiv = {i:.3f}, pc = {j:.3f}, cum = {k:.3f}")


# %%
# tableau 4.3 - 2
###############

NDIM = 2  # au max K - 1

# coordonnées dans le plan factoriel
# principal coordinates
c_coords = (np.diag(c ** (-0.5)) @ Vt.T @ Da)[::, :NDIM:]
# display(c_coords)
r_coords = (np.diag(r ** (-0.5)) @ U @ Da)[::, :NDIM:]
# display(r_coords)

# contribution des lignes/colonnes aux axes
c_contrib = c.reshape(-1, 1) * (c_coords ** 2) / inertias[:NDIM:]
r_contrib = r.reshape(-1, 1) * (r_coords ** 2) / inertias[:NDIM:]


# représentation par axe : cos²
c_psi2 = c_coords ** 2
c_cos2 = c_psi2 / c_psi2.sum(axis=1).reshape(-1, 1)
r_psi2 = r_coords ** 2
r_cos2 = r_psi2 / r_psi2.sum(axis=1).reshape(-1, 1)


# %%
#   return CA(r_coords, c_coords, principal_inertias)
# plot_ca = data_to_ca(df, ca_res.rows_coordinates, ca_res.cols_coordinates, axis=axis)
data = pd.DataFrame(
    np.hstack(
        [
            np.vstack((c_coords[::, :NDIM:], r_coords[::, :NDIM:])),
            100 * np.vstack((c_contrib[::, :NDIM:], r_contrib[::, :NDIM:])),
            100 * np.vstack((c_cos2[::, :NDIM:], r_cos2[::, :NDIM:])),
        ]
    )
)


data.columns = pd.MultiIndex.from_product([["Coords", "Contribs", "Cos²"], range(NDIM)])
data["Mass"] = cols_margin[:-1].to_list() + rows_margin[:-1].to_list()
# data.columns = ["1st axis", "2nd axis"]
data.index = df_colours.columns.to_list() + df_colours.index.to_list()
data["Kind"] = [df_colours.columns.name] * len(df_colours.columns) + [df_colours.index.name] * len(df_colours.index)


X_SHIFT, Y_SHIFT = 0.01, 0.01

ax = sns.scatterplot(data=data, x=("Coords", 0), y=("Coords", 1), hue="Kind", size="Mass", sizes=(16, 256))
ax.set_title(f"CA for Coulours, captured inertia={100*inertias_pc[:NDIM].sum():.2f}%")
ax.set_xlabel(f"1st axis ({100*inertias_pc[0]:.2f})")
ax.set_ylabel(f"2nd axis ({100*inertias_pc[1]:.2f})")
for index, row in data.iterrows():
    ax.annotate(index, (row[("Coords", 0)] + X_SHIFT, row[("Coords", 1)] + Y_SHIFT))

plt.show()
display(data.drop(columns="Coords", level=0))

# %%

# Bootstrap : sampler k = 592 avec remise parmis ceux qu'on a déjà
xs, ys = np.divmod(np.random.choice(I*J, size = N, p = Z.flatten()), I)
reshape(pd.crosstab(xs, ys, margins = True, margins_name= SUM_SYMBOL), df_colours_margins)

# %%

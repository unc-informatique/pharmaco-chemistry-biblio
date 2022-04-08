# type: ignore
"""Package catk Correspondence Analysis Toolkit: main class"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns

from scipy import linalg
from scipy.stats import chi2

# from .config import SUM_SYMBOL

logger = logging.getLogger(f"{__name__}")
CONTRIB_COLUMNS = ["Coords (std)", "Coords (princ.)", "Contributions (%)", "Cosine²"]


class CA:
    """Base class for Correspondance Analysis - CA

    Attributes
    ----------
    N : np.ndarray[I, J]
        the raw data matrix of dimension I x J
    I : int
        the number of rows
    J : int
        the number of columns
    index : pd.Index
        rows name/indices, from the dataframe if possible
    columns : pd.Index
        columns name/indices, from the dataframe if possible
    c : np.ndarray[J]
        marginal column vector sum of dimension J(x1)
    r : np.ndarray[I]
        marginal row vector sum of dimension (1x)I
    n : np.int64
        the grand total
    E : np.ndarray[I, J]
        relative expectations
    S : np.ndarray[I, J]
        the centered / reduced matrix to decompose S = U . D . Vt
    Dc_sq : np.ndarray[J, J]
        diagonal matrix from 1 / sqrt(c)
    Dr_sq : np.ndarray[I, I]
        diagonal matrix from 1 / sqrt(r)
    chi_score : np.float64
        the chi score, i.e., sum((O-E)² / E)
    U : np.ndarray[I, I]
        left eigen vectors
    eiv : np.ndarray[min(I, J)]
        eigenvalues of S, in decreasing order
    principal_inertias : np.ndarray[min(I, J)]
        eiv ** 2
    Da : np.ndarray[min(I, J), min(I, J)]
        eiv as a matrix
    Vt : np.ndarray[J, J]
        right eigen vectors

    Methods
    -------
    fit(data)
        fits

    Raises
    ------
    TypeError
        when init data is neither np.ndarray or pd.DataFrame
    """

    # pylint: disable=too-few-public-methods
    # pylint: disable=too-many-instance-attributes

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

        self.N = None
        self.I = None
        self.J = None
        self.index = None
        self.columns = None

        self.c = None
        self.r = None
        self.n = None

        self.E = None
        self.S = None

        self.Dr_sq = None
        self.Dc_sq = None

        self.chi_score = None

        self.U = None
        self.eiv = None
        self.Da = None
        self.principal_inertias = None
        self.Vt = None

    def fit(self, data) -> None:
        """Do computation"""

        if isinstance(data, pd.DataFrame):
            self.N = data.to_numpy()
            self.index = data.index.copy()
            self.columns = data.columns.copy()
        elif isinstance(data, np.ndarray):
            self.N = data
        else:
            raise TypeError(f"must use pandas's DataFrame or numpy's ndarray, not {type(data)}")

        # shape
        self.I, self.J = self.N.shape
        # self.K = min(self.I, self.J)
        logger.debug("fitting data of shape %i rows and %i columns", self.I, self.J)

        # grand total: the number of individuals
        self.n = self.N.sum()
        # observed frequencies/probability matrix
        P = self.N / self.n
        # column-wise sum a.k.a. column masses, i.e., average row profile
        self.c = P.sum(axis=0)
        # row-wise sum a.k.a. row masses, i.e, average column profile
        self.r = P.sum(axis=1)

        logger.debug("grand total %i observations", self.n)
        logger.debug("row masses\n%s", self.r)
        logger.debug("col masses\n%s", self.c)

        # expected probabilities under independence
        self.E = self.r.reshape(-1, 1) @ self.c.reshape(1, -1)
        logger.debug("expected values under independence\n%s", self.n * self.E)

        # manually compute chi_square
        self.chi_score = self.n * np.sum((P - self.E) ** 2 / self.E)
        # degrees of freedom
        dof = (self.I - 1) * (self.J - 1)
        logger.info("Chi square score %.2f with %i dof", self.chi_score, dof)
        logger.info("p-value = %s", chi2.sf(self.chi_score, dof))

        r_factor = -0.5
        c_factor = -0.5
        # if c_factor/r_factor are different from -0.5,
        # we have non scaled version, with average on axes != 1.0

        # diagonal matrices of row and column masses:
        self.Dr_sq = np.diag(self.r**r_factor)
        self.Dc_sq = np.diag(self.c**c_factor)

        # the centered matrix is P - E
        # then, we reduce/Scale it using masses to obtain S
        # that is, the matrix to diagonalize

        # in fine S_ij = Zc_ij / (sqrt(ri) * sqrt(ci))
        #              = (M_ij/N - r_i*c_i)  / sqrt(ri * ci)

        self.S = self.Dr_sq @ (P - self.E) @ self.Dc_sq

        logger.debug("S=\n%s", self.S)

        # SVD decomposition
        # eiv contains eigenvalue in decreasing order
        # the last one is 0 (up to float precision)
        self.U, self.eiv, self.Vt = linalg.svd(self.S)
        # self.Da = np.diag(self.eiv)
        if self.I == min(self.I, self.J):
            # (I, J) * (I, 1)
            self.Da = np.eye(self.I, self.J) * self.eiv.reshape(-1, 1)
        elif self.J == min(self.I, self.J):
            # (I, J) * (1, J)
            self.Da = np.eye(self.I, self.J) * self.eiv.reshape(1, -1)
        else:
            raise ValueError("that should not happen here")

        logger.debug("eiv=\n%s", self.eiv)
        logger.debug("D²=\n%s", self.Da**2)

        # SVD ensures that Vt  @ Vt.T == I == U.T @ U
        assert np.allclose(self.Vt @ self.Vt.T, np.identity(self.J))
        assert np.allclose(self.U.T @ self.U, np.identity(self.I))
        assert np.allclose(self.U @ self.Da @ self.Vt, self.S)

        # chi_score is the trace of (squared) eingenvalues times N
        assert np.isclose(self.chi_score, self.n * np.trace(self.Da**2))
        assert np.isclose(self.chi_score, self.n * np.trace(self.S @ self.S.T))

        self.principal_inertias = self.eiv**2

        # for chaining
        return self

    def axes(self) -> pd.DataFrame:
        """Describes axes"""

        K = min(self.I, self.J) - 1
        inertias = self.principal_inertias[:K].reshape(K, 1)
        chi_contribs = self.n * inertias
        inertias_pc = 100 * inertias / inertias.sum()
        inertias_cum = np.cumsum(inertias_pc).reshape(K, 1)

        res = pd.DataFrame(np.hstack((inertias, chi_contribs, inertias_pc, inertias_cum)))
        res.index = range(1, K + 1)
        res.index.name = "Axes"
        res.columns = ["Inertia (abs)", "Chi²", "Inertia (%)", "Cumulated (%)"]

        return res

    def contributions(self, K=None) -> pd.DataFrame:
        """Generate a report dataframe"""
        if self.N is None:
            raise ValueError("must call fit first")

        index = []
        rank = min(self.I, self.J) - 1
        if K is not None:
            if K > rank:
                raise ValueError(f"K must be at most {rank}")
        else:
            K = rank

        if self.columns is not None:
            index.extend(self.columns)
        else:
            index.extend(f"col-{j+1}" for j in range(self.J))

        if self.index is not None:
            index.extend(self.index)
        else:
            index.extend(f"row-{i+1}" for i in range(self.I))

        # standard coordinates
        c_coords_std = self.Dc_sq @ self.Vt.T
        r_coords_std = self.Dr_sq @ self.U

        # principal coordinates
        c_coords = (c_coords_std @ self.Da.T)[:, :K]
        r_coords = (r_coords_std @ self.Da)[:, :K]

        # print(self.I, self.J, K)
        # print("self.principal_inertias", self.principal_inertias.shape)
        # print("self.Da", self.Da.shape)
        # print("self.c", self.c.shape)
        # print("c_coords_std", c_coords_std.shape)
        # print("c_coords", c_coords.shape)
        # print("self.r", self.r.shape)
        # print("r_coords_std", r_coords_std.shape)
        # print("r_coords", r_coords.shape)

        # contribution (% of inertia due to r/c)
        #                           (J, 1) * (J, K)                 (1, K)
        c_contrib = self.c.reshape(-1, 1) * (c_coords**2) / self.principal_inertias[:K]
        #                           (I, 1) * (J, K)                 (1, K)
        r_contrib = self.r.reshape(-1, 1) * (r_coords**2) / self.principal_inertias[:K]

        # the sum of contributions to axes is 100% for rows and cols
        assert np.allclose(np.sum(c_contrib, axis=0), 1.0)
        assert np.allclose(np.sum(r_contrib, axis=0), 1.0)
        # print(np.sum(r_contrib, axis = 0))

        # c_contrib = (c_coords ** 2) / self.principal_inertias
        # r_contrib = (r_coords ** 2) / self.principal_inertias

        # cosine square (how much e/c fits to axis)
        c_cos2 = (c_coords**2) / (c_coords**2).sum(axis=1).reshape(-1, 1)
        r_cos2 = (r_coords**2) / (r_coords**2).sum(axis=1).reshape(-1, 1)

        res = pd.DataFrame(
            np.hstack(
                [
                    np.vstack((c_coords_std[:, :K], r_coords_std[:, :K])),
                    np.vstack((c_coords[:, :K], r_coords[:, :K])),
                    100 * np.vstack((c_contrib[:, :K], r_contrib[:, :K])),
                    100 * np.vstack((c_cos2[:, :K], r_cos2[:, :K])),
                ]
            )
        )
        res.index = index

        res.columns = pd.MultiIndex.from_product([CONTRIB_COLUMNS, range(1, K + 1)])

        res["Mass (%)"] = (100 * np.hstack((self.c, self.r))).tolist()
        res["Kind"] = [self.columns.name if self.columns is not None else "col"] * self.J + [
            self.index.name if self.index is not None else "row"
        ] * self.I

        return res

    def plot(self, axes=(1, 2), **kwargs):
        """Plot the diagram on two given axis using seaborn"""
        kwargs = {
            "sizes": (32, 512),
            "legend": "brief",
            "hue": "Kind",
            "size": "Mass (%)",
        } | kwargs
        X_SHIFT, Y_SHIFT = 0.01, 0.01
        x, y = axes

        # sns.set_theme(style="whitegrid", font_scale=1.05, rc={"figure.figsize": (8, 6)})
        data = self.contributions(K=None)
        x_col = (CONTRIB_COLUMNS[1], x)
        y_col = (CONTRIB_COLUMNS[1], y)
        if min(self.I - 1, self.J - 1) == 1:
            data[y_col] = np.zeros(len(data))
            rotation = 45
            inertias = np.array([self.principal_inertias[x - 1], 0])
        else:
            rotation = 0
            inertias = self.principal_inertias[[x - 1, y - 1]]

        inertias_pc = 100 * inertias / self.principal_inertias.sum()

        plot = sns.scatterplot(
            data=data,
            x=x_col,
            y=y_col,
            **kwargs,
        )
        plot.set_title(f"Total captured inertia={inertias_pc.sum():.2f}%")
        plot.set_xlabel(f"Axis #{x} ({inertias_pc[0]:.2f}%)")
        plot.set_ylabel(f"Axis #{y} ({inertias_pc[1]:.2f}%)")
        for index, row in data.iterrows():
            plot.annotate(index, (row[x_col] + X_SHIFT, row[y_col] + Y_SHIFT), rotation=rotation)
        return plot

    def approx(self, K=None):
        """Approximates the original frequency table up to order/rank K, with K the number of eigenvalues to use. K = 0 is expected frequencies under independence. K = min(I, J) - 1 is the complete model.

        Fixing rows and cols masses gives (I-1)+(J-1) parameters.
        Fixing the grand total gives 1 as well.
        If K = min(I, J) - 1, the system is saturated. Assume I <= J, so K = I - 1.

        I*J = (I-1)*(J-1) + I + J - 1
            = (I - 1) + (J - 1) + 1 + (I-1)*(J-1)
            = Dr + Dc + n + K*x

        So each additional order gives J-1 parameters.
        """
        # mask = np.diag([1] * K + [0] * (len(self.Da) - K))
        mask = np.eye(self.I, self.J) * ([1] * K + [0] * (len(self.eiv) - K))
        Dr_inv = np.diag(self.r**0.5)
        Dc_inv = np.diag(self.c**0.5)
        return (Dr_inv @ self.U @ (self.Da * mask) @ self.Vt @ Dc_inv) + (self.r.reshape(-1, 1) @ self.c.reshape(1, -1))


print(f"'{__file__}' loaded")

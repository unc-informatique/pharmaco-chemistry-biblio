"""Dataset generators"""

# %%
# pylint: disable=unused-import

import logging
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from pprint import pprint
from random import randint, sample

import numpy as np
import pandas as pd

from biblio_extractor import SELECTORS, Keyword, MARGIN_SYMB, CLASS_SYMB, finalize_results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(f"CHEMOTAXO.{__name__}")


def gen_independent_binomial_keywords(
    kws1: list[Keyword], kws2: list[Keyword], nb_papers: int, factor: float = 1.0
) -> pd.DataFrame:
    # pylint: disable=too-many-locals
    """Generates a plausible database of nb_papers papers at least 1 keyword from each list.

    The model is as follows :

    - sample nb_papers*2 from binomial distributions: the number of keywords in each category
    - for each paper, sample at random among keywords
    - compute all pairs of keyword
    - compute the contingency table and then fill the gaps

    In the end, the sum of (w/, w/) celle should amount to about factor*nb_papers
    """
    start_time = time.perf_counter()
    # the probability for the binomial law
    if factor < 1:
        raise ValueError(f"gen_independent_binomial_keywords() factor={factor:.4f} must be > 1")
    success_probability = np.sqrt(factor) - 1

    logger.debug("gen_independent_binomial_keywords() with %i compounds, %i activities", len(kws1), len(kws1))
    logger.debug(
        "gen_independent_binomial_keywords() generates %i papers with factor %f, p=%f",
        nb_papers,
        factor,
        success_probability,
    )

    # marginal sums
    margin_rows: dict[Keyword, int] = defaultdict(int)
    margin_cols: dict[Keyword, int] = defaultdict(int)

    # the list of all pairs of keywords in the database
    pairs: list[tuple[Keyword, Keyword]] = []
    # for each paper pick a number of keywords in each list
    # at random following a binomial law
    nbs1 = np.random.binomial(len(kws1), success_probability / len(kws1), nb_papers)
    nbs2 = np.random.binomial(len(kws2), success_probability / len(kws2), nb_papers)

    # for each paper
    for paper in range(nb_papers):
        # standardize the number of keywords
        nb1 = min(1 + nbs1[paper], len(kws1))
        nb2 = min(1 + nbs2[paper], len(kws2))
        # picks keywords at random and fill the margins
        kws1_paper = sample(kws1, nb1)
        for kw1 in kws1_paper:
            margin_rows[kw1] += 1
        kws2_paper = sample(kws2, nb2)
        for kw2 in kws2_paper:
            margin_cols[kw2] += 1
        # compute all pairs of keywords and store them
        pairs.extend(product(kws1_paper, kws2_paper))

    logger.info(
        "gen_independent_binomial_keywords() generates %i pairs, factor is %.2f ~ %.2f (given)",
        len(pairs),
        len(pairs) / nb_papers,
        factor,
    )

    logger.debug("gen_independent_binomial_keywords() margin_rows=%s", margin_rows)
    logger.debug("gen_independent_binomial_keywords() margin_cols=%s", margin_cols)

    elapsed = time.perf_counter() - start_time
    logger.info("gen_independent_binomial_keywords() generated %i papers in %.4f sec", nb_papers, elapsed)

    # BUG : may have a bug if some row or line is 0 everywhere
    # unzip the pairs
    s_compounds, s_activities = list(zip(*pairs))

    # TODO : voir comment le faire avec pivot_table ?
    # https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html
    # compute the number of papers having each pair of keywords, (w/, w/) cells
    contingency = pd.crosstab(index=[s_compounds], columns=[s_activities])

    # add an extra dimensions to rows and columns
    margin_idx = (CLASS_SYMB, MARGIN_SYMB, SELECTORS[True])
    rows_idx = [(c, n, k) for (c, n), k in product(kws1, SELECTORS)]
    cols_idx = [(c, n, k) for (c, n), k in product(kws2, SELECTORS)]

    # similary to extend_df
    res = pd.DataFrame(
        index=pd.MultiIndex.from_tuples([margin_idx] + rows_idx),
        columns=pd.MultiIndex.from_tuples([margin_idx] + cols_idx),
    )

    # fill with already known data
    # res.iloc[res.index.get_level_values(2) == "w/", res.columns.get_level_values(2) == "w/"] = contingency
    res.loc[
        [i for i in rows_idx if i[2] == SELECTORS[True]], [i for i in cols_idx if i[2] == SELECTORS[True]]
    ] = contingency.values

    res.loc[margin_idx, [i for i in cols_idx if i[2] == SELECTORS[True]]] = margin_cols
    res.loc[[i for i in rows_idx if i[2] == SELECTORS[True]], margin_idx] = margin_rows
    res.loc[margin_idx, margin_idx] = nb_papers

    # # now, compute the missing cells of confusion submatrixes
    # for kw1 in kws1:
    #     for kw2 in kws2:
    #         # logger.debug("res.loc[%s][%s]=%s", kw1, kw2, res.loc[kw1, kw2])
    #         # /!\ arr is A REFERENCE to the confusion submatrix
    #         arr = res.loc[kw1, kw2].values
    #         arr[1][0] = margin_rows[kw1] - arr[1][1]
    #         arr[0][1] = margin_cols[kw2] - arr[1][1]
    #         arr[0][0] = nb_papers - arr[0][1] - arr[1][0] - arr[1][1]

    elapsed = time.perf_counter() - start_time
    logger.info("gen_independent_binomial_keywords() generated whole db in %.4f sec", elapsed)
    return finalize_results(res)


SAMPLES_KW1 = [("humanities", "linguistics"), ("humanities", "sociology")]
SAMPLES_KW2 = [("cs", "databases"), ("cs", "web")]
# observed ratio
MAGIC = 1.571256414880884

if __name__ == "__main__":
    df = gen_independent_binomial_keywords(SAMPLES_KW1, SAMPLES_KW2, 1000, MAGIC)
    print(df)

"""Generate queries and summarizes number of articles from bibliographical DB (e.g., Scopus)"""
# pylint: disable=unused-import
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=line-too-long

# %%

import asyncio
from datetime import datetime
import logging
import ssl
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial, wraps
from itertools import combinations, product, chain
from os import environ
from pathlib import Path
from pprint import pprint
from random import randint, choice
from typing import Any, Awaitable, Callable, Iterator, Optional, Protocol

import certifi
import numpy as np
import pandas as pd
from aiohttp import ClientResponseError, ClientSession
from dotenv import load_dotenv

# take environment variables from .env.
# MUST DEFINE API_KEY with apy key from
load_dotenv()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(f"CHEMOTAXO.{__name__}")

# Web requests / API
API_KEY = {"X-ELS-APIKey": environ.get("API_KEY", "no-elsevier-api-key-defined")}
X_RATE_HEADERS = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# Input samples
INPUT_DATA = Path("data/activities.csv")
SAMPLE_DATA = Path("data/samples.csv")
TEST_DATA = Path("data/tests.csv")

# I/O and string configuration
CSV_PARAMS = {"sep": ";", "quotechar": '"'}
ALT_SEP = "/"
SELECTORS = ["w/o", "w/"]  # ordered as bools
MARGIN_SYMB = "Σ"
CLASS_SYMB = "*"
MARGIN_IDX = (CLASS_SYMB, MARGIN_SYMB, SELECTORS[1])
# sort multilevel indexes usging : ascending/ascending/descending
SORT_ORDER = [True, True, False]

# Default parameters
DEFAULT_PARALLEL_WORKERS = 8  # number of parallel jobs
DEFAULT_WORKER_DELAY = 1.0  # at most one req / sec
DEFAULT_SAMPLES = None  # no sampling

# Typing
# a keyword is a fully index row (or column) identifier made of a class and the keyword itself
Keyword = tuple[str, str]


@dataclass(frozen=True)
class Query:
    """an aliases for queries : KW1, KW2, POS_KW, NEG_KW, KIND
    where KIND defines the combination among {w/o, w/}x{w/o, w/}
    that is, a celle of the confusion matrix"""

    kws_1: list[Keyword]
    kws_2: list[Keyword]
    pos_kws: list[Keyword]
    neg_kws: list[Keyword]
    kind: tuple[Optional[bool], Optional[bool]]

    def short(self) -> str:
        """short representation of queries with first two fields omitted"""
        return f"+{self.pos_kws}, -{self.neg_kws}, k={self.kind}"


# type of searches
ResultAPI = tuple[Optional[int], float]
# SearchAPI = Callable[[ClientSession, Query, Optional[Any]], Awaitable[ResultAPI]]
class SearchAPI(Protocol):  # pylint: disable=too-few-public-methods
    """A class to describe callbacks to a web API"""

    __name__: str

    async def __call__(self, session: ClientSession, query: Query, delay: Optional[Any]) -> ResultAPI:
        pass


def load_input(filename: str | Path) -> pd.DataFrame:
    """loads an 'input' CSV dataset as a dataframe with two levels keywords"""
    # https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    # row/col dimension 0 is the class, row/col dimension 1 is the keyword
    dataset: pd.DataFrame = pd.read_csv(filename, index_col=[0, 1], header=[0, 1], sep=";").fillna(0)
    logger.debug("load_input(%s): input dataset read", filename)

    def normalize_names(expr: str) -> str:
        """convenience tool for normalizing strings"""
        return ALT_SEP.join(string.strip().lower() for string in expr.split(ALT_SEP))

    # normalize strings
    dataset.index = pd.MultiIndex.from_tuples([tuple(normalize_names(x) for x in t) for t in dataset.index])
    dataset.columns = pd.MultiIndex.from_tuples([tuple(normalize_names(x) for x in t) for t in dataset.columns])

    logger.info(
        "load_data(%s): %i compounds (with %i classes)",
        filename,
        len(dataset.index.levels[1]),
        len(dataset.index.levels[0]),
    )
    logger.info(
        "load_data(%s): %i activities (with %i classes)",
        filename,
        len(dataset.columns.levels[1]),
        len(dataset.columns.levels[0]),
    )

    return dataset


def load_results(filename: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """loads a CSV dataset as a dataframe with two levels keywords"""
    dataset_with_margin = pd.read_csv(filename, index_col=[0, 1, 2], header=[0, 1, 2], sep=";")
    logger.debug("load_results(%s): results dataset read", filename)
    dataset_with_margin.sort_index(axis=1, inplace=True, ascending=SORT_ORDER)
    dataset_with_margin.sort_index(axis=0, inplace=True, ascending=SORT_ORDER)
    logger.debug("load_results(): dataset\n%s", dataset_with_margin)

    grand_total = dataset_with_margin.loc[MARGIN_IDX, MARGIN_IDX]
    logger.debug("load_results(): %s papers in total", grand_total)

    # all rows/lines but margins
    all_comp_but_margin = pd.Series(idx for idx in dataset_with_margin.index if idx != MARGIN_IDX)
    all_acti_but_margin = pd.Series(idx for idx in dataset_with_margin.columns if idx != MARGIN_IDX)
    # filter out
    comp_margin = dataset_with_margin.loc[all_comp_but_margin, MARGIN_IDX]
    acti_margin = dataset_with_margin.loc[MARGIN_IDX, all_acti_but_margin]

    dataset = dataset_with_margin.loc[all_comp_but_margin, all_acti_but_margin]

    return dataset, comp_margin, acti_margin, grand_total


# %%


def extend_df(src_df: pd.DataFrame, query_mode: str) -> pd.DataFrame:
    """Add extra indexes as last level of rows and columns to store the 2x2 confusion matrix

    Index and columns are multi-level indexes. We duplicate each key to have
    an extra [w/, w/o] index level at the finest level.

    In the end, the orginal KW1 x KW2 matrix is transformed to a KW1 x 2 x KW2 x 2
    each original cell [m] being now a 2x2 confusion submatrix [U, V][X, Y]

    OBSOLETE : if margin are added, a  4 x (KW1 + 1) x (KW2 + 1) is constructed
    """
    logger.debug("extend_df()")

    # out_df = pd.DataFrame().reindex_like(src_df)
    extended_rows = pd.MultiIndex.from_tuples((cls, val, s) for (cls, val) in src_df.index for s in SELECTORS)
    extended_cols = pd.MultiIndex.from_tuples((cls, val, s) for (cls, val) in src_df.columns for s in SELECTORS)

    # https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html
    if query_mode in ("cross", "margins"):
        xs, ys = extended_rows, extended_cols
    elif query_mode == "compounds":
        xs, ys = extended_rows, extended_rows
    elif query_mode == "activities":
        xs, ys = extended_cols, extended_cols
    elif query_mode == "alternatives":
        xs = pd.MultiIndex.from_tuples(
            [(cls, val) for (cls, vals) in src_df.index for val in vals.split(ALT_SEP)]
            + [(cls, val) for (cls, vals) in src_df.columns for val in vals.split(ALT_SEP)]
        )
        ys = pd.Index(["Count"])
    else:
        raise ValueError(f"unknown query mode '{query_mode}' extend results")

    return pd.DataFrame(index=xs, columns=ys).astype("Int32")


# %%
# BUG : CE given_grand_total EST ABSOLUMENT DEGUEULASSE !
def finalize_results(res_df: pd.DataFrame, query_mode: str, *, given_grand_total=438_174) -> pd.DataFrame:
    """Takes a CROSS dataframe with (w/, w/) cells and w/ marginal sums only and fills the remaining ones.

        In other words, it fills the blanks in the following dataframe

                  Σ       w/o      w/     w/o      w/
        Σ    3436.0       C1?  2325.0     C2?  1237.0
        w/o     R1?        Z?      X?      Z?      X?
        w/   3146.0        Y?  2161.0      Y?  1098.0
        w/o     R2?        Z?      X?      Z?      X?
        w/    294.0        Y?   166.0      Y?   141.0

        First it fills the marginal sums R1 and R2, then C1 and C2.
        Then, it fills X cells, then Y cells and Z cells.
        On the example, in the end you have :

                 Σ        w/o      w/     w/o      w/
        Σ    3436.0    1111.0  2325.0  2199.0  1237.0
        w/o   290.0     126.0   164.0   151.0   139.0
        w/   3146.0     985.0  2161.0  2048.0  1098.0
        w/o  3142.0     983.0  2159.0  2046.0  1096.0
        w/    294.0     128.0   166.0   153.0   141.0


    Parameters
    ----------
    src_df : pd.DataFrame
        the source dataframe to be filled, copied to avoid modifications

    Returns
    -------
    pd.DataFrame
        the dataframe filled as described above
    """

    # a copy
    dataset = res_df.copy().astype("float32").fillna(0)

    # sort once and for all, for performance, see
    # https://stackoverflow.com/questions/54307300/what-causes-indexing-past-lexsort-depth-warning-in-pandas
    # https://pandas.pydata.org/docs/reference/api/pandas.Index.is_monotonic_increasing.html
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_index.html
    dataset.sort_index(axis=1, inplace=True, ascending=SORT_ORDER)
    dataset.sort_index(axis=0, inplace=True, ascending=SORT_ORDER)
    # margin identifier
    margin_idx = (CLASS_SYMB, MARGIN_SYMB, SELECTORS[True])
    # the grand total: a.k.a., the number of individuals

    if given_grand_total is not None:
        logger.warn("finalize_results() fixed value for grand_total %s", given_grand_total)
        dataset.loc[margin_idx, margin_idx] = given_grand_total
    grand_total = dataset.loc[margin_idx, margin_idx]
    logger.info("fill_missing_counts() N = %i", grand_total)
    # logger.debug("fill_missing_counts() df =\n%s", dataset)

    # four filters that drops margin sums and keep (w/ or w/o) X (rows or cols)
    w_rows_filter = [(c, a, k) for (c, a, k) in dataset.index if k == SELECTORS[True] and c != CLASS_SYMB]
    wo_rows_filter = [(c, a, k) for (c, a, k) in dataset.index if k == SELECTORS[False] and c != CLASS_SYMB]
    w_cols_filter = [(c, a, k) for (c, a, k) in dataset.columns if k == SELECTORS[True] and c != CLASS_SYMB]
    wo_cols_filter = [(c, a, k) for (c, a, k) in dataset.columns if k == SELECTORS[False] and c != CLASS_SYMB]

    # according to query_mode, we restrict to CxA, CxC or AxA
    if query_mode in ("cross", "margins"):
        xs_w, ys_w = w_rows_filter, w_cols_filter
        xs_wo, ys_wo = wo_rows_filter, wo_cols_filter
    elif query_mode == "compounds":
        xs_w, ys_w = w_rows_filter, w_rows_filter
        xs_wo, ys_wo = wo_rows_filter, wo_rows_filter
    elif query_mode == "activities":
        xs_w, ys_w = w_cols_filter, w_cols_filter
        xs_wo, ys_wo = wo_cols_filter, wo_cols_filter

    # update missing rows margin : w/o  = N - w/
    if query_mode in ["cross", "compounds", "margins"]:
        dataset.T.loc[margin_idx, xs_wo] = grand_total - dataset.T.loc[margin_idx, xs_w].values
    # update missing cols margin : w/o  = N - w/
    if query_mode in ["cross", "activities", "margins"]:
        dataset.loc[margin_idx, ys_wo] = grand_total - dataset.loc[margin_idx, ys_w].values

    # copy margin rows <-> cols and to diagonal
    if query_mode == "compounds":
        dataset.loc[margin_idx] = dataset.T.loc[margin_idx]
    elif query_mode == "activities":
        dataset.T.loc[margin_idx] = dataset.loc[margin_idx]

    # the (w/, w/) base cells from Scopus
    base_values = dataset.loc[xs_w, ys_w].values
    # if not cross mode, symmetrize w/w/ and replicate margin into diagonal
    if query_mode in ["compounds", "activities"]:
        base_values += base_values.T
        base_values += np.diag(dataset.loc[margin_idx, xs_w])
        dataset.loc[xs_w, ys_w] = base_values
        # for idx, val in dataset.loc[margin_idx, xs_w].items():
        #     # logger.debug("%s, %s", idx, val)
        #     dataset.loc[idx, idx] = val

    # logger.debug("x_m =\n%s", dataset.T.loc[margin_idx])
    # logger.debug("y_m =\n%s", dataset.loc[margin_idx])

    logger.debug("base_values=\n%s", base_values)
    logger.debug("dataset =\n%s", dataset)

    if query_mode != "margins":
        # update (w/o, w/) cells = col_margins - (w/, w/)
        dataset.loc[xs_wo, ys_w] = dataset.loc[margin_idx, ys_w].values - base_values
        # update (w/, w/o) cells = row_margins - (w/, w/)
        dataset.loc[xs_w, ys_wo] = dataset.T.loc[margin_idx, xs_w].values.reshape(-1, 1) - base_values
        # update (w/o, w/o) cells = N - (w/, w/) - (w/o, w/) - (w/, w/o)
        dataset.loc[xs_wo, ys_wo] = grand_total - (
            dataset.loc[xs_wo, ys_w].values + dataset.loc[xs_w, ys_wo].values + base_values
        )

    return dataset.astype("int32")


# %%


def build_clause(query: Query) -> str:
    """Build a logical clause of the following form from the given query:

        (c_1 \/ ... \/ c_m)
     /\ (a_1 \/ ... \/ a_n)
     /\ (p_1 /\ ... /\ p_x)
     /\ (!n_1 /\ ... /\ !n_y)

    Where the dataset has m compounds and n activities,
    len(pos_kw) = x and len(neg_kw) = y.

    Classe information are discarded from keywords.
    Keywords that contain alternatives are normalized to conjunctions when
    in a positive position or to disjunctions when in the negative position.

    See tests for more information.
    """

    field_search = "TITLE-ABS-KEY"

    def split_alts(string: str, operator: str = "OR") -> str:
        """transform alternatives in keywords"""
        # base = f" {operator.strip()} ".join(f'KEY("{name}")' for name in string.split(ALT_SEP))
        # return f"({base})"
        base = f" {operator.strip()} ".join(f'"{name}"' for name in string.split(ALT_SEP))
        return f"{field_search}({base})"

    # compounds, activities, pos_kw, neg_kw, _ = query

    # we only keep the last item [-1] of keywords, i.e., we discard their classes in queries
    all_compounds_clause = " OR ".join(split_alts(compound[-1]) for compound in query.kws_1)
    all_ativities_clause = " OR ".join(split_alts(activity[-1]) for activity in query.kws_2)
    positive_clause = " AND ".join(split_alts(kw[-1]) for kw in query.pos_kws)
    negative_clause = " OR ".join(split_alts(kw[-1]) for kw in query.neg_kws)

    clauses = " AND ".join(
        f"({clause})" for clause in [all_compounds_clause, all_ativities_clause, positive_clause] if clause
    )

    if not clauses:
        raise IndexError("at least one positive clause must be non-empty")

    if negative_clause:
        clauses += f" AND NOT ({negative_clause})"

    return clauses


async def broken_search(
    session: ClientSession,
    query: Query,
) -> ResultAPI:
    """Dummy function for tests and error reporting"""
    raise NotImplementedError("Do not call broken_search(), use gen_db() if needed")


async def fake_search(_: ClientSession, query: Query) -> ResultAPI:
    """Fake query tool WITHOUT network, for test purpose"""
    start_time = time.perf_counter()
    clause = build_clause(query)
    logger.debug("fake_search(%s)", query.short())
    logger.debug("               %s", clause)
    results_nb = randint(1, 10000)
    # await asyncio.sleep(randint(1, 1000) / 1000)
    elapsed = time.perf_counter() - start_time
    return results_nb, elapsed


async def httpbin_search(session: ClientSession, query: Query, *, error_rate: int = 0) -> ResultAPI:
    """Fake query tool WITH network on httpbin, for test purpose. Simulates error rate (with http 429)"""
    if randint(1, 100) <= error_rate:
        # N404: OT FOUND, 413: PAYLOAD TOO LARGE, 429  TOO MANY REQUESTS
        http_code = choice((404, 413, 429))
        url = f"http://httpbin.org/status/{http_code}"
    else:
        url = "http://httpbin.org/anything"

    data = {"answer": randint(1, 10000)}
    results_nb: Optional[int] = None
    start_time = time.perf_counter()

    json_query = wrap_scopus(build_clause(query))
    logger.debug("httpbin_search(%s)", query.short())
    logger.debug("%s", json_query)

    try:
        async with session.get(url, params=json_query, data=data, ssl=SSL_CONTEXT) as resp:
            json = await resp.json()
            results_nb = int(json["form"]["answer"])
    except ClientResponseError as err:
        logger.warning("httpbin_search(): ClientResponseError #%i: %s", err.status, err.message)
        logger.warning(err)
        # on HTTP  error, return the "negative" HTTP code
        results_nb = -(err.status)  # pylint: disable=invalid-unary-operand-type
    finally:
        elapsed = time.perf_counter() - start_time
    logger.debug("httpbin_search(%s)=%i in %f sec", query.short(), results_nb, elapsed)
    return results_nb, elapsed


def wrap_scopus(string: str):
    """Wraps a string query into an object to be sent as JSON over Scopus API"""
    if not string:
        raise ValueError("string must be non-empty")
    return {"query": f'DOCTYPE("ar") AND {string}', "count": 1}


async def scopus_search(session: ClientSession, query: Query):
    """Scopus query tool: return the number of article papers having two sets of keywords. Delay is in sec"""
    scopus_url = "https://api.elsevier.com/content/search/scopus"
    results_nb = None
    start_time = time.perf_counter()

    json_query = wrap_scopus(build_clause(query))
    logger.debug("scopus_search(%s)", query.short())
    logger.debug("%s", json_query)

    try:
        async with session.get(scopus_url, params=json_query, headers=API_KEY, ssl=SSL_CONTEXT) as resp:
            logger.debug("X-RateLimit-Remaining=%s", resp.headers.get("X-RateLimit-Remaining", None))
            json = await resp.json()
            results_nb = int(json["search-results"]["opensearch:totalResults"], 10)
    except ClientResponseError as err:
        logger.warning("scopus_search(): ClientResponseError #%i: %s", err.status, err.message)
        logger.warning(err)
    finally:
        elapsed = time.perf_counter() - start_time
    logger.debug("scopus_search(%s)=%s in %f sec", query.short(), results_nb, elapsed)
    return results_nb, elapsed


# query modes : one fake, one fake over network, one true
SEARCH_MODES = {
    "scopus": scopus_search,
    "httpbin": httpbin_search,
    "fake": fake_search,
    "offline": broken_search,
    "broken": broken_search,
}
DEFAULT_SEARCH_MODE = "fake"


QUERY_MODES = ["activities", "compounds", "cross", "margins", "alternatives"]
DEFAULT_QUERY_MODE = "cross"


def number_queries(compounds, activities, mode: str) -> Optional[int]:

    if mode == "cross":
        return len(compounds) * len(activities) + len(compounds) + len(activities) + 1
    if mode == "compounds":
        return len(compounds) * (len(compounds) - 1) // 2 + len(compounds) + 1
    if mode == "activities":
        return len(activities) * (len(activities) - 1) // 2 + len(activities) + 1
    if mode == "margins":
        return len(compounds) + len(activities) + 1
    if mode == "alternatives":
        alt_compunds = [alt for comp in compounds for alt in comp.split(ALT_SEP)]
        alt_activities = [alt for act in activities for alt in act.split(ALT_SEP)]
        return len(alt_compunds) + len(alt_activities)
    return None


def generate_all_queries(data: pd.DataFrame, query_mode: str) -> Iterator[Query]:
    """Generate all queries from a dataset.

    We extract J rows (coumpounds) and I columns (activities) from data.
    The function generates I*J + I + J + 1 = (I + 1) * (J + 1) queries

    Commented code is legacy, when about 4*(I+1)*(J+1) queries where sent

    Parameters
    ----------
    data: pd.DataFrame
        the input data frame, with all keywords
    query_mode: str
        the generation mode in QUERY_MODES

    Yields
    ------
    Iterator[Query]
        yields queries, one for each (w/, w/) cells + two margins (rows and cols) + grand total
    """
    # compounds = list(data.index.get_level_values(1))
    # activities = list(data.columns.get_level_values(1))

    compounds = data.index.to_list()
    activities = data.columns.to_list()
    nb_compounds = len(compounds)
    nb_activities = len(activities)

    logger.info(
        "generate_all_queries will yields I*J + I + J + 1 = %i queries (I = %i, J = %i)",
        nb_compounds * nb_activities + nb_compounds + nb_activities + 1,
        nb_compounds,
        nb_activities,
    )

    # compute all pairs of interest according to query_mode, either
    # compounds x activities, compounds x compounds or activities x activities
    if query_mode == "cross":
        # the main content : |KW1| x |KW2| queries
        # a pair is a compound and an activity
        for pair in product(compounds, activities):
            yield Query([], [], pair, [], (True, True))
    elif query_mode == "compounds":
        # symmetric mode #1: analysis of cooc among compounds
        # pairs of compounds
        for pair in combinations(compounds, 2):
            yield Query([], activities, pair, [], (True, True))
    elif query_mode == "activities":
        # symmetric mode #2: analysis of cooc among activities
        # pairs of activities
        for pair in combinations(activities, 2):
            yield Query(compounds, [], pair, [], (True, True))
    elif query_mode in ("margins", "alternatives"):
        pass
    else:
        raise ValueError(f"unknown query mode '{query_mode}' to generate queries")

    # rows/columns marginal sums (an extra row and an extra column for total)
    # this generates (|KW1| + |KW2| + 1) queries
    # rows margin sums
    if query_mode in ["compounds", "cross", "margins"]:
        for compound in compounds:
            yield Query([], activities, [compound], [], (True, None))
    # cols margin sums
    if query_mode in ["activities", "cross", "margins"]:
        for activity in activities:
            yield Query(compounds, [], [activity], [], (None, True))
    # total margin sum
    if query_mode in ["activities", "compounds", "cross", "margins"]:
        yield Query(compounds, activities, [], [], (None, None))

    if query_mode == "alternatives":
        for compound in ((cls, val) for (cls, vals) in compounds for val in vals.split(ALT_SEP)):
            yield Query([], activities, [compound], [], (ALT_SEP, None))
        for activity in ((cls, val) for (cls, vals) in activities for val in vals.split(ALT_SEP)):
            yield Query(compounds, [], [activity], [], (None, ALT_SEP))


async def consumer(
    session: ClientSession,
    queue: asyncio.Queue,
    results_df: pd.DataFrame,
    task_factory: SearchAPI,
    *,
    worker_delay: float = 1.0,
    consumer_id: Optional[Any] = None,
) -> tuple[int, int]:
    # pylint: disable=too-many-branches
    """A (parallel) consumer that send a query to scopus and then add result to a dataframe.

    Parameters
    ----------
    session : ClientSession
        the "wget" client session to send http queries to, passed to the task_factory function
    queue : asyncio.Queue
        where to read jobs from. Put back queries that return an HTTP error (e.g., 429 or 500)
    results_df : pd.DataFrame
        DataFrame where to store results, passed by reference
    task_factory : SearchAPI
        the function that generate HTTP queries
    worker_delay : float, optional
        delay BEFORE a query, passed to the task_factory function, by default 1.0
    consumer_id : Optional[Any], optional
        an identifier for the task (informative), by default None

    Returns
    -------
    tuple[int, int]
        the total number of queries handled and the number of queries that failed (put back into the queue)

    Raises
    ------
    cancel
        when the task is stopped by the task manager
    """
    jobs_done = 0
    jobs_retried = 0
    jobs_failed = 0
    try:
        # queue must be filled first
        while not queue.empty():
            query = await queue.get()
            nb_results, duration = await task_factory(session, query)
            logger.info("consumer(%s) got %s from job %s after %f", consumer_id, nb_results, query.short(), duration)

            failed_job = nb_results is None or nb_results < 0
            nb_results_orig = nb_results
            if failed_job:
                nb_results = -1

            pos_kws = query.pos_kws
            neg_kws = query.neg_kws
            kind = query.kind
            # ventilate each result into the right cell of results_df
            # (w/, w/) cell
            if kind == (True, True):
                results_df.loc[(*pos_kws[0], SELECTORS[True]), (*pos_kws[1], SELECTORS[True])] = nb_results
            # (w/, w/o) cell
            elif kind == (True, False):
                results_df.loc[(*pos_kws[0], SELECTORS[True]), (*neg_kws[0], SELECTORS[False])] = nb_results
            # (w/o, w/) cell
            elif kind == (False, True):
                results_df.loc[(*neg_kws[0], SELECTORS[False]), (*pos_kws[0], SELECTORS[True])] = nb_results
            # (w/o, w/o) cell
            elif kind == (False, False):
                results_df.loc[(*neg_kws[0], SELECTORS[False]), (*neg_kws[1], SELECTORS[False])] = nb_results
            # (w/, Σ) cell (w/ row margin)
            elif kind == (True, None):
                results_df.loc[(*pos_kws[0], SELECTORS[True]), (CLASS_SYMB, MARGIN_SYMB, SELECTORS[True])] = nb_results
            # (w/o, Σ) cell (w/o row margin)
            elif kind == (False, None):
                results_df.loc[(*neg_kws[0], SELECTORS[False]), (CLASS_SYMB, MARGIN_SYMB, SELECTORS[True])] = nb_results
            # (Σ, w/) cell (w/ col margin)
            elif kind == (None, True):
                results_df.loc[(CLASS_SYMB, MARGIN_SYMB, SELECTORS[True]), (*pos_kws[0], SELECTORS[True])] = nb_results
            # (Σ, w/o) cell (w/o col margin)
            elif kind == (None, False):
                results_df.loc[(CLASS_SYMB, MARGIN_SYMB, SELECTORS[True]), (*neg_kws[0], SELECTORS[False])] = nb_results
            # (Σ, Σ) cell (grand total)
            elif kind == (None, None):
                results_df.loc[
                    (CLASS_SYMB, MARGIN_SYMB, SELECTORS[True]), (CLASS_SYMB, MARGIN_SYMB, SELECTORS[True])
                ] = nb_results
            # count of compounds alternatives
            elif kind == (ALT_SEP, None):
                results_df.loc[pos_kws[0], "Count"] = nb_results
            elif kind == (None, ALT_SEP):
                results_df.loc[pos_kws[0], "Count"] = nb_results

            # that should not happen if all queries are generated properly
            else:
                # raise ValueError(f"{len(pos_kw) = }, {len(neg_kw) = } for {kind = } should not arise")
                logger.error(
                    "consumer(%s): len(pos_kw) = %i, len(neg_kw) = %i should not arise for kind = %s",
                    consumer_id,
                    len(pos_kws),
                    len(neg_kws),
                    kind,
                )

            jobs_done += 1
            queue.task_done()

            # add the same query again in the job queue to retry it if http error is TOO MANY REQUESTS (429)
            # if nb_results is None:
            if failed_job:
                jobs_failed += 1
                if nb_results_orig is not None and nb_results_orig == -429:
                    await queue.put(query)
                    jobs_retried += 1
                    logger.info("consumer(%s) added back %s to the queue from error 429", consumer_id, query.short())
                else:
                    logger.error("consumer(%s) dropped %s from error %s", consumer_id, query.short(), nb_results_orig)
                # raise RuntimeWarning(f"consumer({consumer_id}) broken here ")

            await asyncio.sleep(max(worker_delay - duration, 0))
    except asyncio.CancelledError as cancel:
        logger.error("consumer(%s) received cancel", consumer_id)
        raise cancel
    # except Exception as err:
    #     logger.error("consumer(%s) crashed with exception %s('%s')", consumer_id, type(err).__name__, err)
    # raise err

    logger.info("consumer(%s) ended, done %i jobs, retried %i", consumer_id, jobs_done, jobs_retried)

    # NOTE: results_df contains values: it's passed by reference
    return jobs_done, jobs_retried, jobs_failed


async def observer(queue: asyncio.Queue, frequency: float = 0.5):
    """Observer task that reports the current state of the queue"""
    delay = 1 / frequency
    observations = 0
    try:
        while True:  # not queue.empty():
            print(f">{queue.qsize():>4} jobs in the queue @{datetime.now().strftime('%H-%M-%S.%f')}")
            observations += 1
            await asyncio.sleep(delay)
    except asyncio.CancelledError:
        logger.info("observer() canceled after %i round of observations", observations)


async def spawner(
    src_df: pd.DataFrame,
    *,
    task_factory: SearchAPI,
    parallel_workers: int,
    worker_delay: float,
    query_mode: str,
) -> pd.DataFrame:
    # pylint: disable=too-many-locals
    """Adds tasks into a queue which is emptied in parallel ensuring at most MAX_REQ_BY_SEC requests per second.

    Creates the results DataFrame to store queries results

    Parameters
    ----------
    src_df : pd.DataFrame
        the source dataframe: only rows/cols indexes are used
    task_factory : SearchAPI
        the function that generate HTTP queries, to be passed to consumers
    parallel_workers : int
        the number of parallel consumer that will take jobs from the queue
    worker_delay : float
        delay BEFORE a query, passed to the consumer function
    query_mode:str
        query mode, forwarded to generate_all_queries

    Returns
    -------
    pd.DataFrame
        the results DataFrame, filled by consumer

    Raises
    ------
    RuntimeError
        When the event loop gets crazy
    """
    jobs_queue: asyncio.Queue = asyncio.Queue()
    logger.info("spawner(): task_factory=%s, parallel_workers=%i", task_factory.__name__, parallel_workers)

    # generate all queries according to the selected mode put them into the queue
    all_queries = list(generate_all_queries(src_df, query_mode=query_mode))
    for query in all_queries:
        await jobs_queue.put(query)
        logger.debug("spawner() added query=%s", query.short())

    logger.info("spawner() added %i queries to the queue", len(all_queries))

    observer_task = asyncio.create_task(
        observer(jobs_queue),
        name="observer",
    )
    logger.info("spawner() observer (done=%s) task created", observer_task.done())

    consumer_tasks = []
    # NOTE! ici, dans le cas cross on étend les données d'origine aux 2 dimensions supplémentaires
    result_df = extend_df(src_df, query_mode)
    # BUG : https://stackoverflow.com/questions/48860117/why-performancewarning-when-indexed-lookup-on-sorted-index
    result_df.sort_index(level=[0, 1])
    logger.debug("result_df.index.is_monotonic_increasing %s", result_df.index.is_monotonic_increasing)
    logger.debug(result_df)

    async with ClientSession(raise_for_status=True) as session:

        # on lance tous les exécuteurs de requêtes
        consumer_tasks = [
            asyncio.create_task(
                consumer(session, jobs_queue, result_df, task_factory, worker_delay=worker_delay, consumer_id=f"{i}"),
                name=f"consumer-{i}",
            )
            for i in range(1, parallel_workers + 1)
        ]

        # logger.debug("spawner(): running tasks %s", asyncio.all_tasks())
        logger.info("spawner() %i consumer tasks created", len(consumer_tasks))

        # NOTE : spawner cannot end if all workers are dead!
        # NOTE waiting for BOTH jobs_queue.join() here and queue.empty() in spawner
        # is too much and may wait forever if all workers crashed
        # await jobs_queue.join()
        # logger.debug("spawner() job queue is empty")

        # NOTE : OBSOLETE, do not need to cancel workers, just wait for them
        # stop all consumer stuck waiting job from the queue if any
        # for consumer in consumer_tasks:
        #     consumer.cancel()
        #     logger.debug("jobs_spawner() %s stopped", consumer.get_name())

        finished_tasks = await asyncio.gather(*consumer_tasks, return_exceptions=True)
        logger.info("spawner() nb of jobs/retries by each worker %s", finished_tasks)
        errors = [res for res in finished_tasks if isinstance(res, Exception)]
        done_jobs = [res for res in finished_tasks if not isinstance(res, Exception)]
        logging.debug("spawner() %i errors from workers: %s", len(errors), errors)
        if not done_jobs:
            raise RuntimeError("No async worker ended properly")
        nb_jobs, nb_retries, nb_failed = zip(*done_jobs)
        print(f"Summary: {len(consumer_tasks)} workers ended correctly.")

        print(
            f"{sum(nb_jobs)} jobs with {sum(nb_retries)} retries on error with {sum(nb_failed)} total failed. {len(errors)} workers crashed."
        )

    return result_df


def launcher(
    src_df: pd.DataFrame,
    *,
    task_factory: SearchAPI = SEARCH_MODES[DEFAULT_SEARCH_MODE],
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    worker_delay: float = DEFAULT_WORKER_DELAY,
    query_mode: str = DEFAULT_QUERY_MODE,
) -> pd.DataFrame:
    """Launch the batch of downloads: a simple (non async) wrapper around spawner

    Parameters
    ----------
    src_df : pd.DataFrame
        forwarded to spawner
    task_factory : SearchAPI, optional
        forwarded to spawner, by default SEARCH_MODES[DEFAULT_SEARCH_MODE]
    parallel_workers : int, optional
        forwarded to spawner, by default DEFAULT_PARALLEL_WORKERS
    worker_delay : float, optional
        forwarded to spawner, by default DEFAULT_WORKER_DELAY
    query_mode:str
        query mode, forwarded to generate_all_queries via spawner

    Returns
    -------
    pd.DataFrame
        get results gathered by spawner, fills missing cells and sort the result
    """
    launch_start_time = time.perf_counter()
    logger.info("launcher() launching all async tasks")
    results_df = asyncio.run(
        spawner(
            src_df,
            parallel_workers=parallel_workers,
            task_factory=task_factory,
            worker_delay=worker_delay,
            query_mode=query_mode,
        )
    )

    total_time = time.perf_counter() - launch_start_time
    logger.info("launcher() all jobs done in %fs", total_time)
    # logger.debug("results dataframe %s", results_df)

    if query_mode in (
        "activities",
        "compounds",
        "cross",
        "margins",
    ):
        final_res = finalize_results(results_df, query_mode=query_mode)
    elif query_mode in ("alternatives"):
        final_res = results_df

    return final_res


# %%

if __name__ == "__main__":
    logger.error("Please use biblio_main.py to launch extraction")
    the_filename = Path("../input/pharmaco_chemistry_2_samples.csv")
    the_dataset = load_input(the_filename)
    the_compounds = list(the_dataset.index.get_level_values(1))
    the_activities = list(the_dataset.columns.get_level_values(1))

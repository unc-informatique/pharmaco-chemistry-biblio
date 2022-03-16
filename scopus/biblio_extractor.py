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
from itertools import product
from os import environ
from pathlib import Path
from pprint import pprint
from random import randint, sample
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
    logging.basicConfig()

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
        """shor representation of queries with first two fields omitted"""
        return f"+{self.pos_kws}, -{self.neg_kws}, k={self.kind}"


# type of searches
ResultAPI = tuple[Optional[int], float]
# SearchAPI = Callable[[ClientSession, Query, Optional[Any]], Awaitable[ResultAPI]]
class SearchAPI(Protocol):  # pylint: disable=too-few-public-methods
    """A class to describe callbacks to a web API"""

    __name__: str

    async def __call__(self, session: ClientSession, query: Query, delay: Optional[Any]) -> ResultAPI:
        pass


def load_data(filename: str | Path) -> pd.DataFrame:
    """loads a CSV dataset as a dataframe with two levels keywords"""
    # https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    # row/col dimension 0 is the class, row/col dimension 1 is the keyword
    df: pd.DataFrame = pd.read_csv(filename, index_col=[0, 1], header=[0, 1]).fillna(0)
    logger.debug("load_data(%s): input dataset read", filename)

    def normalize_names(expr: str) -> str:
        """convenience tool for normalizing strings"""
        return ALT_SEP.join(string.strip().lower() for string in expr.split(ALT_SEP))

    # normalize strings
    df.index = pd.MultiIndex.from_tuples([tuple(normalize_names(x) for x in t) for t in df.index])
    df.columns = pd.MultiIndex.from_tuples([tuple(normalize_names(x) for x in t) for t in df.columns])

    logger.info(
        "load_data(%s): %i compounds (with %i classes)", filename, len(df.index.levels[1]), len(df.index.levels[0])
    )
    logger.info(
        "load_data(%s): %i activities (with %i classes)", filename, len(df.columns.levels[1]), len(df.columns.levels[0])
    )

    return df


def extend_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add extra indexes as last level of rows and columns to store the 2x2 confusion matrix

    Index and columns are multi-level indexes. We duplicate each key to have
    an extra [w/, w/o] index level at the finest level.

    In the end, the orginal KW1 x KW2 matrix is transformed to a KW1 x 2 x KW2 x 2
    each original cell [m] being now a 2x2 confusion submatrix [U, V][X, Y]

    OBSOLETE : if margin are added, a  4 x (KW1 + 1) x (KW2 + 1) is constructed
    """
    logger.debug("extend_df()")
    df2 = pd.DataFrame().reindex_like(df)

    # if with_margin:

    # margin_row = pd.DataFrame(index=pd.MultiIndex.from_tuples([(CLASS_SYMB, MARGIN_SYMB)]), columns=df.columns)
    # df2 = pd.concat([df2, margin_row], axis=0)
    # margin_col = pd.DataFrame(index=df.columns, columns=pd.MultiIndex.from_tuples([(CLASS_SYMB, MARGIN_SYMB)]))
    # df2 = pd.concat([df2, margin_col], axis=1)

    # df2 = df2.append(margin_row)
    # df2[(CLASS_SYMB, MARGIN_SYMB)] = None
    # df2[(CLASS_SYMB, MARGIN_SYMB)] = df2[(CLASS_SYMB, MARGIN_SYMB)].astype(int)

    extended_rows = pd.MultiIndex.from_tuples((cls, val, s) for (cls, val) in df2.index for s in SELECTORS)
    extended_cols = pd.MultiIndex.from_tuples((cls, val, s) for (cls, val) in df2.columns for s in SELECTORS)

    # https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html
    extended_df = pd.DataFrame(index=extended_rows, columns=extended_cols).astype("Int64")
    return extended_df


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

    def split_alts(string: str, operator: str = "OR") -> str:
        """transform alternatives in keywords"""
        base = f" {operator.strip()} ".join(f'KEY("{name}")' for name in string.split(ALT_SEP))
        return f"({base})"

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


async def broken_search(session: ClientSession, query: Query, *, delay: float = 0.0) -> ResultAPI:
    """Dummy function for tests and error reporting"""
    raise NotImplementedError("Do not call broken_search(), use gen_db() if needed")


async def fake_search(_: ClientSession, query: Query, *, delay: float = 0.0) -> ResultAPI:
    """Fake query tool WITHOUT network, for test purpose"""
    await asyncio.sleep(delay)
    start_time = time.perf_counter()
    clause = build_clause(query)
    logger.debug("fake_search(%s)", query.short())
    logger.debug("               %s", clause)
    results_nb = randint(1, 10000)
    # await asyncio.sleep(randint(1, 1000) / 1000)
    elapsed = time.perf_counter() - start_time
    return results_nb, elapsed


async def httpbin_search(
    session: ClientSession, query: Query, *, delay: float = 0.0, error_rate: int = 10
) -> ResultAPI:
    """Fake query tool WITH network on httpbin, for test purpose. Simulates error rate (with http 429)"""
    if randint(1, 100) <= error_rate:
        url = "http://httpbin.org/status/429"
    else:
        url = "http://httpbin.org/anything"
    data = {"answer": randint(1, 10000)}
    results_nb = None
    await asyncio.sleep(delay)
    start_time = time.perf_counter()
    logger.debug("httpbin_search(%s)", query.short())
    json_query = wrap_scopus(build_clause(query))

    try:
        async with session.get(url, params=json_query, data=data, ssl=SSL_CONTEXT) as resp:
            json = await resp.json()
            results_nb = int(json["form"]["answer"])
    except ClientResponseError as err:
        logger.warning("scopus_search(): ClientResponseError #%i: %s", err.status, err.message)
    finally:
        elapsed = time.perf_counter() - start_time
    logger.debug("httpbin_search(%s)=%i in %f sec", query.short(), results_nb, elapsed)
    return results_nb, elapsed


def wrap_scopus(string: str):
    """Wraps a string query into an object to be sent as JSON over Scopus API"""
    if not string:
        raise ValueError("string must be non-empty")
    return {"query": f'DOCTYPE("ar") AND {string}', "count": 1}


async def scopus_search(session: ClientSession, query: Query, *, delay=0):
    """Scopus query tool: return the number of article papers having two sets of keywords. Delay is in sec"""
    scopus_url = "https://api.elsevier.com/content/search/scopus"
    results_nb = None

    await asyncio.sleep(delay)
    start_time = time.perf_counter()
    logger.debug("scopus_search(%s)", query.short())
    json_query = wrap_scopus(build_clause(query))
    try:
        async with session.get(scopus_url, params=json_query, headers=API_KEY, ssl=SSL_CONTEXT) as resp:
            logger.debug("X-RateLimit-Remaining=%s", resp.headers.get("X-RateLimit-Remaining", None))
            json = await resp.json()
            results_nb = int(json["search-results"]["opensearch:totalResults"], 10)
    except ClientResponseError as err:
        logger.warning("scopus_search(): ClientResponseError #%i: %s", err.status, err.message)
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


def generate_all_queries(data: pd.DataFrame, *, with_margin: bool = False) -> Iterator[Query]:
    """Generate all queries from a dataset."""
    # compounds = list(data.index.get_level_values(1))
    # activities = list(data.columns.get_level_values(1))

    compounds = data.index.to_list()
    activities = data.columns.to_list()

    # the main content : 4 x |KW1| x |KW2| cells
    for compound in compounds:
        for activity in activities:
            # both the compound and the activity
            yield Query([], [], [compound, activity], [], (True, True))
            # the activity but not this compound (but at least one another in the domain)
            yield Query(compounds, [], [activity], [compound], (False, True))
            # the compound but not this activity (but at least one another in the domain)
            yield Query([], activities, [compound], [activity], (True, False))
            # neither the compound nor the activity (but stil in the domain)
            yield Query(compounds, activities, [], [compound, activity], (False, False))

    # adds extra rows/columns for marginal sums (an extra row and an extra column for total)
    # this should add 4 x (|KW1| + |KW2| + 1) but we exclude 2 + 2 + 3 degenerated combinations which always are 0
    if with_margin:
        # rows margin sums, -2 always 0
        for compound in compounds:
            yield Query([], activities, [compound], [], (True, None))
            yield Query(compounds, activities, [], [compound], (False, None))
        # cols margin sums, -2 always 0
        for activity in activities:
            yield Query(compounds, [], [activity], [], (None, True))
            yield Query(compounds, activities, [], [activity], (None, False))
        # total margin sum, -3 always 0
        yield Query(compounds, activities, [], [], (None, None))


async def consumer(
    session: ClientSession,
    queue: asyncio.Queue,
    results_df: pd.DataFrame,
    task_factory: SearchAPI,
    *,
    worker_delay: float = 1.0,
    consumer_id: Optional[Any] = None,
):
    # pylint: disable=too-many-branches
    """A (parallel) consumer that send a query to scopus and then add result to a dataframe"""
    jobs_done = 0
    jobs_retried = 0
    try:
        # queue must be filled first
        while not queue.empty():
            query = await queue.get()
            nb_results, duration = await task_factory(session, query, delay=0.0)

            logger.info("consumer(%s) got %s from job %s after %f", consumer_id, nb_results, query.short(), duration)
            pos_kws = query.pos_kws
            neg_kws = query.neg_kws
            kind = query.kind
            if kind == (True, True):
                results_df.loc[(*pos_kws[0], SELECTORS[True]), (*pos_kws[1], SELECTORS[True])] = nb_results
            elif kind == (True, False):
                results_df.loc[(*pos_kws[0], SELECTORS[True]), (*neg_kws[0], SELECTORS[False])] = nb_results
            elif kind == (False, True):
                results_df.loc[(*neg_kws[0], SELECTORS[False]), (*pos_kws[0], SELECTORS[True])] = nb_results
            elif kind == (False, False):
                results_df.loc[(*neg_kws[0], SELECTORS[False]), (*neg_kws[1], SELECTORS[False])] = nb_results
            elif kind == (True, None):
                results_df.loc[(*pos_kws[0], SELECTORS[True]), (CLASS_SYMB, MARGIN_SYMB, SELECTORS[True])] = nb_results
            elif kind == (False, None):
                results_df.loc[(*neg_kws[0], SELECTORS[False]), (CLASS_SYMB, MARGIN_SYMB, SELECTORS[True])] = nb_results
            elif kind == (None, True):
                results_df.loc[(CLASS_SYMB, MARGIN_SYMB, SELECTORS[True]), (*pos_kws[0], SELECTORS[True])] = nb_results
            elif kind == (None, False):
                results_df.loc[(CLASS_SYMB, MARGIN_SYMB, SELECTORS[True]), (*neg_kws[0], SELECTORS[False])] = nb_results
            elif kind == (None, None):
                results_df.loc[
                    (CLASS_SYMB, MARGIN_SYMB, SELECTORS[True]), (CLASS_SYMB, MARGIN_SYMB, SELECTORS[True])
                ] = nb_results
            else:
                # raise ValueError(f"{len(pos_kw) = }, {len(neg_kw) = } for {kind = } should not arise")
                logger.error(
                    "consumer(%s): len(pos_kw) = %i, len(neg_kw) = %i should not arise for kind = %s",
                    consumer_id,
                    len(pos_kws),
                    len(neg_kws),
                    kind,
                )
            queue.task_done()
            jobs_done += 1

            # add the same query again in the job queue to retry it
            if nb_results is None:
                await queue.put(query)
                jobs_retried += 1
                logger.info("consumer(%s) added back %s to the queue", consumer_id, query.short())
                # raise RuntimeWarning(f"consumer({consumer_id}) broken here ")

            await asyncio.sleep(max(worker_delay - duration, 0))
    except asyncio.CancelledError as cancel:
        logger.error("consumer(%s) received cancel", consumer_id)
        raise cancel
    # except Exception as err:
    #     logger.error("consumer(%s) crashed with exception %s('%s')", consumer_id, type(err).__name__, err)
    # raise err

    logger.info("consumer(%s) ended, done %i jobs, retried %i", consumer_id, jobs_done, jobs_retried)

    return jobs_done, jobs_retried


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
        logger.info("observer() canceled after %i observations", observations)


async def spawner(
    df: pd.DataFrame,
    *,
    task_factory: SearchAPI,
    with_margin: bool,
    parallel_workers: int,
    worker_delay: float,
    samples: Optional[int],
):
    # pylint: disable=too-many-locals
    """Create tasks in a queue which is emptied in parallele ensuring at most MAX_REQ_BY_SEC requests per second"""
    jobs_queue: asyncio.Queue = asyncio.Queue()
    logger.info("spawner(): task_factory=%s, parallel_workers=%i", task_factory.__name__, parallel_workers)

    # generate all queries put them into the queue
    all_queries = list(generate_all_queries(df, with_margin=with_margin))
    if samples is not None:
        all_queries = sample(all_queries, samples)
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
    result_df = extend_df(df)
    async with ClientSession(raise_for_status=True) as session:

        # on lance tous les exécuteurs de requêtes
        consumer_tasks = [
            asyncio.create_task(
                consumer(session, jobs_queue, result_df, task_factory, worker_delay=worker_delay, consumer_id=f"{i}"),
                name=f"consumer-{i}",
            )
            for i in range(1, parallel_workers + 1)
        ]

        logger.debug("spawner(): running tasks %s", asyncio.all_tasks())
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
        nb_jobs, nb_retries = zip(*done_jobs)
        print(f"Summary: {len(consumer_tasks)} workers ended correctly.")

        print(f"{sum(nb_jobs)} jobs with {sum(nb_retries)} retries on error. {len(errors)} workers crashed.")

    return result_df


def launcher(
    df: pd.DataFrame,
    *,
    task_factory=SEARCH_MODES[DEFAULT_SEARCH_MODE],
    with_margin=False,
    parallel_workers=DEFAULT_PARALLEL_WORKERS,
    worker_delay=DEFAULT_WORKER_DELAY,
    samples=None,
):
    """Launch the batch of downloads: a simple (non async) wrapper around tasks_spawner"""
    launch_start_time = time.perf_counter()
    logger.info("launcher() launching all async tasks")
    results_df = asyncio.run(
        spawner(
            df,
            parallel_workers=parallel_workers,
            task_factory=task_factory,
            worker_delay=worker_delay,
            with_margin=with_margin,
            samples=samples,
        )
    )

    total_time = time.perf_counter() - launch_start_time
    logger.info("launcher() all jobs done in %fs", total_time)

    # order the index for performance, see
    # https://stackoverflow.com/questions/54307300/what-causes-indexing-past-lexsort-depth-warning-in-pandas
    # https://pandas.pydata.org/docs/reference/api/pandas.Index.is_monotonic_increasing.html
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_index.html

    # ascending, ascending, descending
    sort_order = [True, True, False]
    results_df.sort_index(axis=1, inplace=True, ascending=sort_order)
    results_df.sort_index(axis=0, inplace=True, ascending=sort_order)
    return results_df.astype("Int64")


# %%
# for tests only
if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.info("__main__ Scopus API key %s", API_KEY)

    dataset = load_data(SAMPLE_DATA)
    # all_compounds = list(dataset.index.get_level_values(1))
    # all_activities = list(dataset.columns.get_level_values(1))
    all_compounds = list(dataset.index)
    all_activities = list(dataset.columns)
    # logger.debug("__main__ all compounds %s", all_compounds)
    # logger.debug("__main__ all activities %s", all_activities)
    # 383330 / 243964
    # db = gen_db(all_compounds, all_activities, 243964 // 10, 383330 / 243964)
    # db.loc[("alkaloid", "acridine"), ("pharmaco", "cytotoxicity")]

    # NB_KW1 = 2
    # NB_KW2 = 2

    # db = gen_db([("C", f"c_{i+1}") for i in range(NB_KW1)], [("A", f"a_{j+1}") for j in range(NB_KW2)], 10, 2.0)
    # print(db)
    # db.groupby(by=["compound", "activity"]).count()
    # df = pd.DataFrame(db)
    # results = launcher(dataset)
    # print(results)
    # print(results.info())

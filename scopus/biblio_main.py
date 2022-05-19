"""CLI for the bibliographical extractor"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pformat

import biblio_extractor as bex

logging.basicConfig()
logger = logging.getLogger("CHEMOTAXO")

# Output
DEFAULT_OUTPUT_DIR = Path(".")


def get_parser() -> argparse.ArgumentParser:
    """argparse configuration"""
    arg_parser = argparse.ArgumentParser(
        description="Scopus downloader", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        "filename",
        help="csv file to read compounds and activities from.",
    )
    arg_parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="verbosity level. Use -v once for INFO (20) and twice -vv for DEBUG (10).",
    )
    arg_parser.add_argument(
        "--search",
        "-sm",
        action="store",
        default=bex.DEFAULT_SEARCH_MODE,
        help="search mode: 'offline', 'fake', 'httpbin' or 'scopus'.",
    )
    arg_parser.add_argument(
        "--output",
        "-o",
        action="store",
        default=DEFAULT_OUTPUT_DIR,
        help="output directory.",
    )
    arg_parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        action="store",
        default=bex.DEFAULT_PARALLEL_WORKERS,
        help="number of parallel consumers/workers",
    )
    arg_parser.add_argument(
        "--delay",
        "-d",
        type=float,
        action="store",
        default=bex.DEFAULT_WORKER_DELAY,
        help="minimum delay between two consecutive queries from a worker.",
    )
    arg_parser.add_argument(
        "--write",
        "-w",
        action="store_true",
        default=False,
        help="writes results to csv file.",
    )
    arg_parser.add_argument(
        "--mode",
        "-m",
        action="store",
        default=bex.DEFAULT_QUERY_MODE,
        help=f"query mode in {', '.join(bex.QUERY_MODES)}",
    )
    # arg_parser.add_argument(
    #     "--margins",
    #     "-m",
    #     action="store_true",
    #     default=False,
    #     help="returns marginal sums as well.",
    # )
    # arg_parser.add_argument(
    #     "--samples",
    #     "-s",
    #     type=int,
    #     action="store",
    #     default=bex.DEFAULT_SAMPLES,
    #     help="maximum number of queries (random samples).",
    # )
    return arg_parser


def main():
    """entry point"""
    parser = get_parser()
    args = parser.parse_args()

    # https://docs.python.org/3/library/logging.html#levels
    if args.verbose >= 2:
        debug_level = logging.DEBUG
    elif args.verbose == 1:
        debug_level = logging.INFO
    else:
        debug_level = logging.WARNING
    logger.setLevel(debug_level)

    print(f"Scopus downloader started (debug={logger.getEffectiveLevel()})")
    logger.debug(pformat(vars(args)))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output dir is '%s'", output_dir.absolute())
    logger.info("Scopus API key %s", bex.API_KEY)

    filename = Path(args.filename)
    dataset = bex.load_data(filename)
    all_compounds = list(dataset.index.get_level_values(1))
    all_activities = list(dataset.columns.get_level_values(1))
    print(f"Loaded {len(all_compounds)} compounds and {len(all_activities)} activities")
    logger.info("all compounds %s", all_compounds)
    logger.info("all activities %s", all_activities)

    # dataset = load_chemo_activities(args.filename)

    if args.search not in bex.SEARCH_MODES:
        raise ValueError(f"Unknown search mode {args.search}")
    if args.mode not in bex.QUERY_MODES:
        raise ValueError(f"Unknown query mode {args.mode}")

    # if args.search == "offline":
    #     nb_papers = 243964 // (len(all_compounds) * len(all_activities))
    #     results = bex.gen_db(list(dataset.index), list(dataset.columns), nb_papers, 383330 / 243964)
    # else:
    # nb_queries = (
    #     args.samples
    #     if args.samples is not None
    #     else len(all_compounds) * len(all_activities) + len(all_compounds) + len(all_activities) + 1
    # )
    if args.mode == "cross":
        nb_queries = len(all_compounds) * len(all_activities) + len(all_compounds) + len(all_activities) + 1
    elif args.mode == "compounds":
        nb_queries = len(all_compounds) * (len(all_compounds) - 1) // 2 + len(all_compounds) + 1
    elif args.mode == "activities":
        nb_queries = len(all_activities) * (len(all_activities) - 1) // 2 + len(all_activities) + 1

    default_ping = 0.500
    estimated_secs = max(args.delay, default_ping) * nb_queries / args.parallel
    print(
        f"Launching {nb_queries} queries using {args.search} with {args.parallel} parallel workers (w/ min delay {args.delay}), ETA {(datetime.now() + timedelta(seconds=estimated_secs)).strftime('%H:%M:%S')} ({round(estimated_secs)} seconds)"
    )

    results = bex.launcher(
        dataset,
        task_factory=bex.SEARCH_MODES[args.search],
        parallel_workers=args.parallel,
        worker_delay=args.delay,
        query_mode=args.mode,
    )

    print(results)

    if args.write:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = output_dir / f"{filename.stem}_{args.mode}_{now}.csv"
        results.to_csv(output_filename)
        logger.info("results written to %s", output_filename)


if __name__ == "__main__":
    main()

# Bibliographical analysis

Programmes principaux pour l'extraction des données Scopus

- [biblio_extractor.py](biblio_extractor.py) : module d'interrogation de la base Scopus et de manipulation des données
- [biblio_main.py](biblio_main.py) : programme d'extraction : rempli le CSV avec le nombre d'articles

## Extracteur

```raw
usage: biblio_main.py [-h] [--verbose] [--search SEARCH] [--parallel PARALLEL] [--delay DELAY] [--samples SAMPLES] [--write] [--margins] filename

Scopus downloader

positional arguments:
  filename              file to read chemo activities from

options:
  -h, --help            show this help message and exit
  --verbose, -v         verbosity level, default is WARNING (30). Use -v once for INFO (20) and twice -vv for DEBUG (10).
  --search SEARCH, -sm SEARCH
                        search mode: 'offline', 'fake', 'httpbin' or 'scopus' (default 'fake')
  --parallel PARALLEL, -p PARALLEL
                        number of parallel consumers/workers (default 8)
  --delay DELAY, -d DELAY
                        minimum delay between two consecutive queries from a worker (default 1.0)
  --samples SAMPLES, -s SAMPLES
                        maximum number of queries (random samples) (default all queries)
  --write, -w           writes results to csv file (default False)
  --margins, -m         also queries and returns marginal sums
```

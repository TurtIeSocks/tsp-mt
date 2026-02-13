# Benchmark Workflow

This project uses a single benchmark root directory with dedicated subfolders:

- `benchmark/inputs/`: place all benchmark input `.txt` files here.
- `benchmark/outputs/`: benchmark run stdout route outputs.
- `benchmark/logs/`: benchmark run stderr logs.
- `benchmark/results/`: generated summary/report artifacts (`.tsv` and `.md`).

## Input Discovery

`scripts/benchmark-max-chunk-size.sh` automatically discovers inputs from:

- `benchmark/inputs/*.txt` (sorted lexicographically)

If no `.txt` files are present, the script exits with an error.

## Per-Input Short-Circuit

For each input file, the script computes `n` as the number of non-empty lines.

Chunk-size runs short-circuit per input: after the first completed run where
`chunk_size > n`, that input stops and the script moves to the next input.

Example with `n = 1200`:
- runs `900`, `1000`, `1500`
- then stops that input (does not run larger chunk sizes)

## Run Benchmark

```bash
./scripts/benchmark-max-chunk-size.sh
```

By default, each run uses a UTC timestamp (format `YYYYMMDDTHHMMSSZ`) in output file names to prevent overwriting:

- `benchmark/outputs/*-<chunk>-<timestamp>.txt`
- `benchmark/logs/*-<chunk>-<timestamp>.stderr.log`
- `benchmark/results/chunk-size-benchmark-<timestamp>-summary.tsv`
- `benchmark/results/chunk-size-benchmark-<timestamp>-parsed.tsv`
- `benchmark/results/chunk-size-benchmark-<timestamp>-report.md`

## Regenerate Report Only

To regenerate parsed/report files from an existing summary without re-running benchmarks:

```bash
TSP_MT_REPORT_ONLY=1 ./scripts/benchmark-max-chunk-size.sh benchmark/results/chunk-size-benchmark-<timestamp>-summary.tsv
```

## Optional Environment Variables

- `TSP_MT_REPORT_ONLY=1`: skip benchmark execution and only build parsed/report artifacts.
- `TSP_MT_BENCH_TIMESTAMP=<timestamp>`: override the timestamp used in generated file names.

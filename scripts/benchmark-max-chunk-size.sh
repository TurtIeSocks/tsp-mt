#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BENCHMARK_ROOT_DIR="benchmark"
BENCHMARK_INPUTS_DIR="$BENCHMARK_ROOT_DIR/inputs"
BENCHMARK_OUTPUTS_DIR="$BENCHMARK_ROOT_DIR/outputs"
BENCHMARK_LOGS_DIR="$BENCHMARK_ROOT_DIR/logs"
BENCHMARK_RESULTS_DIR="$BENCHMARK_ROOT_DIR/results"
RUN_TIMESTAMP="${TSP_MT_BENCH_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SUMMARY_FILE="${1:-$BENCHMARK_RESULTS_DIR/chunk-size-benchmark-${RUN_TIMESTAMP}-summary.tsv}"
REPORT_ONLY="${TSP_MT_REPORT_ONLY:-0}"

if [[ "$SUMMARY_FILE" == *-summary.tsv ]]; then
  PARSED_FILE="${SUMMARY_FILE%-summary.tsv}-parsed.tsv"
  REPORT_FILE="${SUMMARY_FILE%-summary.tsv}-report.md"
else
  PARSED_FILE="${SUMMARY_FILE%.tsv}-parsed.tsv"
  REPORT_FILE="${SUMMARY_FILE%.tsv}-report.md"
fi

mkdir -p \
  "$BENCHMARK_INPUTS_DIR" \
  "$BENCHMARK_OUTPUTS_DIR" \
  "$BENCHMARK_LOGS_DIR" \
  "$BENCHMARK_RESULTS_DIR"

declare -a CHUNK_SIZES=()
for ((size = 100; size <= 1000; size += 100)); do
  CHUNK_SIZES+=("$size")
done
for ((size = 1500; size <= 10000; size += 500)); do
  CHUNK_SIZES+=("$size")
done

declare -a INPUT_FILES=()

extract_last_matching_line() {
  local pattern="$1"
  local file="$2"

  if command -v rg >/dev/null 2>&1; then
    rg --no-heading "$pattern" "$file" | tail -n1 || true
  else
    grep -F "$pattern" "$file" | tail -n1 || true
  fi
}

count_input_points() {
  local input_file="$1"
  awk 'NF { c++ } END { print c + 0 }' "$input_file"
}

discover_input_files() {
  INPUT_FILES=()
  while IFS= read -r file; do
    INPUT_FILES+=("$file")
  done < <(find "$BENCHMARK_INPUTS_DIR" -maxdepth 1 -type f -name "*.txt" -print | LC_ALL=C sort)

  if [[ "${#INPUT_FILES[@]}" -eq 0 ]]; then
    echo "no .txt input files found in ${BENCHMARK_INPUTS_DIR}/" >&2
    exit 1
  fi
}

generate_markdown_report() {
  local report_date
  local total_runs
  local successful_runs
  local failed_runs
  local failure_note
  local input_count
  local fastest_secs
  local fastest_input
  local fastest_chunk
  local slowest_secs
  local slowest_input
  local slowest_chunk

  perl -F'\t' -lane '
    next if $. == 1;
    my ($in, $chunk, undef, $metrics, $time) = @F;
    my ($n) = $metrics =~ /n=(\d+)/;
    my ($total) = $metrics =~ /total_m=(\d+)/;
    my ($longest) = $metrics =~ /longest_m=(\d+)/;
    my ($avg) = $metrics =~ /avg_m=(\d+)/;
    my ($spikes) = $metrics =~ /spikes=(\d+)/;
    my ($secs) = $time =~ /finished in ([0-9.]+)s/;
    next unless defined $n && defined $total && defined $longest && defined $avg && defined $spikes && defined $secs;
    print join("\t", $in, $chunk, $n, $total, $longest, $avg, $spikes, $secs);
  ' "$SUMMARY_FILE" >"$PARSED_FILE"

  total_runs="$(awk 'END { print NR - 1 }' "$SUMMARY_FILE")"
  successful_runs="$(awk -F '\t' 'NR > 1 && $3 == 0 { c++ } END { print c + 0 }' "$SUMMARY_FILE")"
  failed_runs=$((total_runs - successful_runs))
  if [[ "$failed_runs" -eq 0 ]]; then
    failure_note="No run failures occurred (\`exit_code=0\` for all ${successful_runs} runs)."
  else
    failure_note="Run failures occurred: ${failed_runs}/${total_runs} had non-zero exit codes."
  fi
  report_date="$(date -u +%Y-%m-%d)"
  input_count="$(awk -F '\t' '{ seen[$1] = 1 } END { print length(seen) }' "$PARSED_FILE")"
  read -r fastest_secs fastest_input fastest_chunk <<<"$(awk -F '\t' 'NR == 1 || $8 < min { min = $8; input = $1; chunk = $2 } END { printf "%.2f %s %d", min + 0, input, chunk + 0 }' "$PARSED_FILE")"
  read -r slowest_secs slowest_input slowest_chunk <<<"$(awk -F '\t' 'NR == 1 || $8 > max { max = $8; input = $1; chunk = $2 } END { printf "%.2f %s %d", max + 0, input, chunk + 0 }' "$PARSED_FILE")"

  cat >"$REPORT_FILE" <<EOF
# Max Chunk Size Benchmark Report

Date: $report_date  
Execution mode: serial (single process; no parallel jobs)  
Runs: ${successful_runs}/${total_runs} successful

Source files:
- \`$SUMMARY_FILE\`
- \`$PARSED_FILE\`

## Fastest Runtime Per Input

| Input | n | chunk_size | total_m | longest_m | avg_m | spikes | runtime_s |
|---|---:|---:|---:|---:|---:|---:|---:|
EOF

  MODE="fastest" perl -F'\t' -lane '
    my $mode = $ENV{MODE};
    push @{ $rows{$F[0]} }, [@F];
    END {
      my @order = sort keys %rows;
      for my $in (@order) {
        next unless exists $rows{$in};
        my @sorted = sort { $a->[7] <=> $b->[7] || $a->[3] <=> $b->[3] || $a->[1] <=> $b->[1] } @{ $rows{$in} };
        my $r = $sorted[0];
        printf "| %s | %d | %d | %d | %d | %d | %d | %.2f |\n", $r->[0], $r->[2], $r->[1], $r->[3], $r->[4], $r->[5], $r->[6], $r->[7];
      }
    }
  ' "$PARSED_FILE" >>"$REPORT_FILE"

  cat >>"$REPORT_FILE" <<'EOF'

## Best Metric (Lowest `total_m`) Per Input

| Input | n | chunk_size | total_m | longest_m | avg_m | spikes | runtime_s |
|---|---:|---:|---:|---:|---:|---:|---:|
EOF

  MODE="best_total" perl -F'\t' -lane '
    my $mode = $ENV{MODE};
    push @{ $rows{$F[0]} }, [@F];
    END {
      my @order = sort keys %rows;
      for my $in (@order) {
        next unless exists $rows{$in};
        my @sorted = sort { $a->[3] <=> $b->[3] || $a->[7] <=> $b->[7] || $a->[1] <=> $b->[1] } @{ $rows{$in} };
        my $r = $sorted[0];
        printf "| %s | %d | %d | %d | %d | %d | %d | %.2f |\n", $r->[0], $r->[2], $r->[1], $r->[3], $r->[4], $r->[5], $r->[6], $r->[7];
      }
    }
  ' "$PARSED_FILE" >>"$REPORT_FILE"

  cat >>"$REPORT_FILE" <<'EOF'

## Baseline at `chunk_size=5000` (Current Default)

| Input | n | chunk_size | total_m | longest_m | avg_m | spikes | runtime_s |
|---|---:|---:|---:|---:|---:|---:|---:|
EOF

  MODE="default_5000" perl -F'\t' -lane '
    my $mode = $ENV{MODE};
    push @{ $rows{$F[0]} }, [@F];
    END {
      my @order = sort keys %rows;
      for my $in (@order) {
        next unless exists $rows{$in};
        my ($r) = grep { $_->[1] == 5000 } @{ $rows{$in} };
        next unless defined $r;
        printf "| %s | %d | %d | %d | %d | %d | %d | %.2f |\n", $r->[0], $r->[2], $r->[1], $r->[3], $r->[4], $r->[5], $r->[6], $r->[7];
      }
    }
  ' "$PARSED_FILE" >>"$REPORT_FILE"

  cat >>"$REPORT_FILE" <<EOF

## Performance Notes

- The benchmark was run serially, so reported runtime is per-run elapsed; total wall-clock for the whole sweep is the sum of all runs.
- ${failure_note}
- Inputs discovered from \`${BENCHMARK_INPUTS_DIR}/*.txt\` (sorted): ${input_count}.
- Fastest individual run: \`${fastest_input}\` at \`chunk_size=${fastest_chunk}\` (${fastest_secs}s).
- Slowest individual run: \`${slowest_input}\` at \`chunk_size=${slowest_chunk}\` (${slowest_secs}s).
- Compare the "Fastest Runtime", "Best Metric", and full per-input tables below to choose a default \`max_chunk_size\` tradeoff for your workload.

## All Chunk Sizes By Input

Each table lists every tested \`chunk_size\` with parsed metrics and runtime.

EOF

  perl -F'\t' -lane '
    push @{ $rows{$F[0]} }, [@F];
    END {
      my @order = sort keys %rows;
      for my $in (@order) {
        next unless exists $rows{$in};
        print "### $in";
        print "";
        print "| chunk_size | n | total_m | longest_m | avg_m | spikes | runtime_s |";
        print "|---:|---:|---:|---:|---:|---:|---:|";
        my @sorted = sort { $a->[1] <=> $b->[1] } @{ $rows{$in} };
        for my $r (@sorted) {
          printf "| %d | %d | %d | %d | %d | %d | %.2f |\n", $r->[1], $r->[2], $r->[3], $r->[4], $r->[5], $r->[6], $r->[7];
        }
        print "";
      }
    }
  ' "$PARSED_FILE" >>"$REPORT_FILE"
}

if [[ "$REPORT_ONLY" != "1" ]]; then
  discover_input_files

  echo -e "input_file\tchunk_size\texit_code\tmetrics_line\ttime_line\tstdout_file\tstderr_file" >"$SUMMARY_FILE"

  total_runs=$((${#INPUT_FILES[@]} * ${#CHUNK_SIZES[@]}))
  current_run=0

  for input_file in "${INPUT_FILES[@]}"; do
    input_name="$(basename "$input_file")"
    input_stem="${input_name%.txt}"
    output_stem="${input_stem/input/output}"
    input_n="$(count_input_points "$input_file")"

    for chunk_size in "${CHUNK_SIZES[@]}"; do
      current_run=$((current_run + 1))
      stdout_file="$BENCHMARK_OUTPUTS_DIR/${output_stem}-${chunk_size}-${RUN_TIMESTAMP}.txt"
      stderr_file="$BENCHMARK_LOGS_DIR/${output_stem}-${chunk_size}-${RUN_TIMESTAMP}.stderr.log"

      echo "[run ${current_run}/${total_runs}] input=${input_name} chunk_size=${chunk_size}" >&2

      set +e
      cargo run --release --features fetch-lkh -- \
        --log-level=info \
        --max-chunk-size="${chunk_size}" \
        <"$input_file" \
        >"$stdout_file" \
        2>"$stderr_file"
      exit_code=$?
      set -e

      metrics_line="$(extract_last_matching_line "INFO metrics:" "$stderr_file")"
      time_line="$(extract_last_matching_line "INFO main: finished in" "$stderr_file")"

      if [[ -z "$metrics_line" || -z "$time_line" ]]; then
        metrics_fallback="$(tail -n2 "$stderr_file" | sed -n '1p' || true)"
        time_fallback="$(tail -n2 "$stderr_file" | sed -n '2p' || true)"

        if [[ -z "$metrics_line" && -n "$metrics_fallback" ]]; then
          metrics_line="$metrics_fallback"
        fi
        if [[ -z "$time_line" && -n "$time_fallback" ]]; then
          time_line="$time_fallback"
        fi
      fi

      metrics_line="${metrics_line//$'\t'/ }"
      time_line="${time_line//$'\t'/ }"

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$input_name" \
        "$chunk_size" \
        "$exit_code" \
        "$metrics_line" \
        "$time_line" \
        "$stdout_file" \
        "$stderr_file" \
        >>"$SUMMARY_FILE"

      if (( chunk_size > input_n )); then
        echo "[input complete] input=${input_name} n=${input_n} stop_after_chunk=${chunk_size}" >&2
        break
      fi
    done
  done

  echo "completed ${current_run} runs (max_possible=${total_runs})"
  echo "summary saved to $SUMMARY_FILE"
else
  if [[ ! -f "$SUMMARY_FILE" ]]; then
    echo "summary file not found: $SUMMARY_FILE" >&2
    exit 1
  fi
  echo "report-only mode enabled: skipping benchmark runs"
fi

generate_markdown_report
echo "parsed summary saved to $PARSED_FILE"
echo "markdown report saved to $REPORT_FILE"

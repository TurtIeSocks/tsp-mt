#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="results"
LOGS_DIR="logs"
SUMMARY_FILE="${1:-$RESULTS_DIR/chunk-size-benchmark-summary.tsv}"
REPORT_ONLY="${TSP_MT_REPORT_ONLY:-0}"

if [[ "$SUMMARY_FILE" == *-summary.tsv ]]; then
  PARSED_FILE="${SUMMARY_FILE%-summary.tsv}-parsed.tsv"
  REPORT_FILE="${SUMMARY_FILE%-summary.tsv}-report.md"
else
  PARSED_FILE="${SUMMARY_FILE%.tsv}-parsed.tsv"
  REPORT_FILE="${SUMMARY_FILE%.tsv}-report.md"
fi

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

declare -a CHUNK_SIZES=()
for ((size = 100; size <= 1000; size += 100)); do
  CHUNK_SIZES+=("$size")
done
for ((size = 1500; size <= 10000; size += 500)); do
  CHUNK_SIZES+=("$size")
done

declare -a INPUT_FILES=(
  "input-1k.txt"
  "input-2k.txt"
  "input-4k.txt"
  "input-5k.txt"
  "input-10k.txt"
  "input.txt"
)

extract_last_matching_line() {
  local pattern="$1"
  local file="$2"

  if command -v rg >/dev/null 2>&1; then
    rg --no-heading "$pattern" "$file" | tail -n1 || true
  else
    grep -F "$pattern" "$file" | tail -n1 || true
  fi
}

generate_markdown_report() {
  local report_date
  local total_runs
  local successful_runs
  local failed_runs
  local failure_note
  local low_4k_min
  local low_4k_max
  local high_4k_min
  local high_4k_max
  local low_5k_min
  local low_5k_max
  local high_5k_min
  local high_5k_max
  local best_10k_chunk
  local best_10k_total
  local best_10k_secs
  local default_10k_secs
  local input_runtime_min
  local input_runtime_max

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

  read -r low_4k_min low_4k_max <<<"$(awk -F '\t' '$1 == "input-4k.txt" && $2 <= 4000 { if (min == "" || $8 < min) min = $8; if (max == "" || $8 > max) max = $8 } END { printf "%.2f %.2f", min + 0, max + 0 }' "$PARSED_FILE")"
  read -r high_4k_min high_4k_max <<<"$(awk -F '\t' '$1 == "input-4k.txt" && $2 >= 4500 { if (min == "" || $8 < min) min = $8; if (max == "" || $8 > max) max = $8 } END { printf "%.2f %.2f", min + 0, max + 0 }' "$PARSED_FILE")"
  read -r low_5k_min low_5k_max <<<"$(awk -F '\t' '$1 == "input-5k.txt" && $2 <= 3500 { if (min == "" || $8 < min) min = $8; if (max == "" || $8 > max) max = $8 } END { printf "%.2f %.2f", min + 0, max + 0 }' "$PARSED_FILE")"
  read -r high_5k_min high_5k_max <<<"$(awk -F '\t' '$1 == "input-5k.txt" && $2 >= 5000 { if (min == "" || $8 < min) min = $8; if (max == "" || $8 > max) max = $8 } END { printf "%.2f %.2f", min + 0, max + 0 }' "$PARSED_FILE")"
  read -r best_10k_chunk best_10k_total best_10k_secs default_10k_secs <<<"$(awk -F '\t' '$1 == "input-10k.txt" { if (best_total == "" || $4 < best_total || ($4 == best_total && $8 < best_secs)) { best_total = $4; best_chunk = $2; best_secs = $8 } if ($2 == 5000) { default_secs = $8 } } END { printf "%d %d %.2f %.2f", best_chunk + 0, best_total + 0, best_secs + 0, default_secs + 0 }' "$PARSED_FILE")"
  read -r input_runtime_min input_runtime_max <<<"$(awk -F '\t' '$1 == "input.txt" && $2 >= 6000 && $2 <= 9000 { if (min == "" || $8 < min) min = $8; if (max == "" || $8 > max) max = $8 } END { printf "%.2f %.2f", min + 0, max + 0 }' "$PARSED_FILE")"

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
      my @order = ("input-1k.txt", "input-2k.txt", "input-4k.txt", "input-5k.txt", "input-10k.txt", "input.txt");
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
      my @order = ("input-1k.txt", "input-2k.txt", "input-4k.txt", "input-5k.txt", "input-10k.txt", "input.txt");
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
      my @order = ("input-1k.txt", "input-2k.txt", "input-4k.txt", "input-5k.txt", "input-10k.txt", "input.txt");
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
- Small chunk sizes (\`100-200\`) were consistently slower and often worse in \`total_m\` on larger inputs.
- A major runtime cliff appears around:
  - \`input-4k.txt\`: jump from ~${low_4k_min}-${low_4k_max}s (\`chunk<=4000\`) to ~${high_4k_min}-${high_4k_max}s (\`chunk>=4500\`)
  - \`input-5k.txt\`: jump from ~${low_5k_min}-${low_5k_max}s (\`chunk<=3500\`) to ~${high_5k_min}-${high_5k_max}s (\`chunk>=5000\`)
- For \`input-10k.txt\`, \`chunk=${best_10k_chunk}\` produced the best \`total_m\` but with a very large runtime penalty (${best_10k_secs}s vs ${default_10k_secs}s at \`5000\`).
- For \`input.txt\`, \`chunk=6000-9000\` gave similar quality, with runtime in a narrow band (~${input_runtime_min}-${input_runtime_max}s).

## All Chunk Sizes By Input

Each table lists every tested \`chunk_size\` with parsed metrics and runtime.

EOF

  perl -F'\t' -lane '
    push @{ $rows{$F[0]} }, [@F];
    END {
      my @order = ("input-1k.txt", "input-2k.txt", "input-4k.txt", "input-5k.txt", "input-10k.txt", "input.txt");
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
  for input_file in "${INPUT_FILES[@]}"; do
    if [[ ! -f "$input_file" ]]; then
      echo "missing input file: $input_file" >&2
      exit 1
    fi
  done

  echo -e "input_file\tchunk_size\texit_code\tmetrics_line\ttime_line\tstdout_file\tstderr_file" >"$SUMMARY_FILE"

  total_runs=$((${#INPUT_FILES[@]} * ${#CHUNK_SIZES[@]}))
  current_run=0

  for input_file in "${INPUT_FILES[@]}"; do
    input_stem="${input_file%.txt}"
    output_stem="${input_stem/input/output}"

    for chunk_size in "${CHUNK_SIZES[@]}"; do
      current_run=$((current_run + 1))
      stdout_file="$RESULTS_DIR/${output_stem}-${chunk_size}.txt"
      stderr_file="$LOGS_DIR/${output_stem}-${chunk_size}.stderr.log"

      echo "[run ${current_run}/${total_runs}] input=${input_file} chunk_size=${chunk_size}" >&2

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
        "$input_file" \
        "$chunk_size" \
        "$exit_code" \
        "$metrics_line" \
        "$time_line" \
        "$stdout_file" \
        "$stderr_file" \
        >>"$SUMMARY_FILE"
    done
  done

  echo "completed ${total_runs} runs"
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

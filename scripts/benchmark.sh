#!/usr/bin/env bash
# End-to-end benchmark of the tsp-mt binary on generated geographic inputs.
#
# Generates clustered lat/lng instances of several sizes (unless files already
# exist under benchmark/inputs/), runs the release binary on each, and writes
# a TSV summary of the metrics line to benchmark/results/.
#
# Usage:
#   ./scripts/benchmark.sh [time_limit_seconds]

set -euo pipefail

cd "$(dirname "$0")/.."

TIME_LIMIT="${1:-0}"
STAMP="${TSP_MT_BENCH_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS="benchmark/results/benchmark-${STAMP}.tsv"

mkdir -p benchmark/inputs benchmark/outputs benchmark/logs benchmark/results

generate() {
  local n="$1" file="$2" seed="$3"
  [ -f "$file" ] && return
  awk -v n="$n" -v seed="$seed" 'BEGIN {
    srand(seed);
    clusters = 3 + int(sqrt(n) / 8);
    for (c = 0; c < clusters; c++) { clat[c] = 35 + rand() * 10; clng[c] = -120 + rand() * 15; }
    for (i = 0; i < n; i++) {
      c = int(rand() * clusters);
      printf "%.6f,%.6f\n", clat[c] + (rand() - 0.5) * 0.4, clng[c] + (rand() - 0.5) * 0.4;
    }
  }' > "$file"
}

shopt -s nullglob
inputs=(benchmark/inputs/*.txt)
if [ ${#inputs[@]} -eq 0 ]; then
  for n in 1000 5000 20000 100000; do
    generate "$n" "benchmark/inputs/gen-${n}.txt" "$n"
  done
  inputs=(benchmark/inputs/*.txt)
fi

cargo build --release

printf 'input\tn\tseconds\ttotal_m\tlongest_m\tavg_m\tspikes\n' > "$RESULTS"

for input in "${inputs[@]}"; do
  name="$(basename "$input" .txt)"
  n="$(grep -c . "$input")"
  out="benchmark/outputs/${name}-${STAMP}.txt"
  log="benchmark/logs/${name}-${STAMP}.stderr.log"

  start="$(date +%s.%N)"
  ./target/release/tsp-mt \
    --input "$input" --output "$out" \
    --time-limit "$TIME_LIMIT" --log-level info --log-output "$log"
  end="$(date +%s.%N)"
  secs="$(awk -v a="$start" -v b="$end" 'BEGIN { printf "%.2f", b - a }')"

  metrics="$(grep -o 'metrics: .*' "$log" | tail -1)"
  total="$(sed -n 's/.*total_m=\([0-9.]*\).*/\1/p' <<< "$metrics")"
  longest="$(sed -n 's/.*longest_m=\([0-9.]*\).*/\1/p' <<< "$metrics")"
  avg="$(sed -n 's/.*avg_m=\([0-9.]*\).*/\1/p' <<< "$metrics")"
  spikes="$(sed -n 's/.*spikes=\([0-9]*\).*/\1/p' <<< "$metrics")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$n" "$secs" "$total" "$longest" "$avg" "$spikes" | tee -a "$RESULTS"
done

echo "Results: $RESULTS"

#!/usr/bin/env bash
# Ultimate sweep — comprehensive data for adaptive chunk sizing.
#
# Generates 3 spatial distributions per problem size (3 random seeds
# in the lat/lng generator) and runs both solvers across a 6-wide
# chunk-size sweep. Skips LKH's pathological single-chunk path on
# large n (chunk≈n with n≥10k takes 100s-1300s for LKH per cell).
#
# Output: one big TSV that the data-driven adaptive chunk picker can
# consume to fit a heuristic. Aggregate report shows per-(n, chunk)
# averages across the 3 distributions.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LKH_BIN="${LKH_BIN:-/Users/rin/GitHub/tsp-mt/target/release/tsp-mt}"
NEW_BIN="$ROOT_DIR/target/release/tsp-mt"
RESULTS_DIR="$ROOT_DIR/benchmark/results"
ULTI_INPUTS_DIR="$ROOT_DIR/benchmark/inputs_ultimate"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TSV="$RESULTS_DIR/ultimate-${TIMESTAMP}.tsv"
MD="$RESULTS_DIR/ultimate-${TIMESTAMP}.md"

SIZES=(1000 2000 3000 5000 8000 10000 15000 20000 25000 35000 50000 75000)
DISTRIBUTIONS=(1 2 3)   # 3 random spatial layouts per size
CHUNK_SIZES=(500 1000 2000 3000 5000 7500)

mkdir -p "$RESULTS_DIR" "$ULTI_INPUTS_DIR"

# Generate the spatial distributions once (cached).
python3 - <<PYEOF
import os, random
sizes = $(printf '%s,' "${SIZES[@]}" | sed 's/,$//')
sizes = [int(x) for x in str(sizes).strip("()").split(",")]
distributions = [int(d) for d in "${DISTRIBUTIONS[@]}".split()]
out_dir = "$ULTI_INPUTS_DIR"
for size in sizes:
    radius = 0.3 if size <= 5_000 else 0.5 if size <= 25_000 else 0.8
    for dist in distributions:
        path = os.path.join(out_dir, f"n{size}_d{dist}.txt")
        if os.path.exists(path):
            with open(path) as f:
                if sum(1 for _ in f) == size:
                    continue
        random.seed(size * 100 + dist)
        with open(path, "w") as f:
            for _ in range(size):
                lat = 37.5 + random.uniform(-radius, radius)
                lng = -122.0 + random.uniform(-radius, radius)
                f.write(f"{lat:.6f},{lng:.6f}\n")
        print(f"wrote {os.path.basename(path)}")
PYEOF

echo -e "solver\tn\tdistribution\tchunk_size\ttotal_m\truntime_s" >"$TSV"

run_once() {
  local solver_label="$1" bin="$2" input="$3" chunk="$4" n="$5" dist="$6"
  local stderr=$(mktemp) stdout=$(mktemp)
  local start end runtime total
  start=$(python3 -c 'import time;print(time.time())')
  set +e
  "$bin" --log-level=info --max-chunk-size="$chunk" --input="$input" --output="$stdout" 2>"$stderr"
  set -e
  end=$(python3 -c 'import time;print(time.time())')
  runtime=$(python3 -c "print(f'{$end - $start:.3f}')")
  total=$(grep "INFO metrics:" "$stderr" | tail -n1 | grep -oE 'total_m=[0-9]+' | cut -d= -f2)
  printf '%s\t%d\t%d\t%d\t%s\t%s\n' \
    "$solver_label" "$n" "$dist" "$chunk" "${total:-NA}" "$runtime" >>"$TSV"
  rm -f "$stderr" "$stdout"
}

skip_lkh() {
  local n="$1" chunk="$2"
  # Skip LKH only when it would land in the catastrophic single-chunk
  # branch on a large input (chunk close to n on n ≥ 10k). lin_kernighan
  # always runs.
  if (( chunk >= n && n >= 10000 )); then return 0; fi
  return 1
}

# Pre-count runs
total=0
for n in "${SIZES[@]}"; do
  for d in "${DISTRIBUTIONS[@]}"; do
    for c in "${CHUNK_SIZES[@]}"; do
      total=$((total + 1))                          # lk always
      if ! skip_lkh "$n" "$c"; then total=$((total + 1)); fi
    done
  done
done

current=0
for n in "${SIZES[@]}"; do
  for d in "${DISTRIBUTIONS[@]}"; do
    input="$ULTI_INPUTS_DIR/n${n}_d${d}.txt"
    for c in "${CHUNK_SIZES[@]}"; do
      current=$((current + 1))
      echo "[$current/$total] solver=lin_kernighan n=$n d=$d chunk=$c" >&2
      run_once "lin_kernighan" "$NEW_BIN" "$input" "$c" "$n" "$d"
      if ! skip_lkh "$n" "$c"; then
        current=$((current + 1))
        echo "[$current/$total] solver=lkh n=$n d=$d chunk=$c" >&2
        run_once "lkh" "$LKH_BIN" "$input" "$c" "$n" "$d"
      fi
    done
  done
done

# Aggregate
python3 - <<PYEOF
import csv, statistics, collections
rows = list(csv.DictReader(open("$TSV"), delimiter="\t"))

# (solver, n, chunk) -> list[(total_m, runtime, dist)]
buckets = collections.defaultdict(list)
for r in rows:
    try:
        k = (r["solver"], int(r["n"]), int(r["chunk_size"]))
        buckets[k].append((int(r["total_m"]), float(r["runtime_s"]), int(r["distribution"])))
    except ValueError:
        continue

with open("$MD", "w") as out:
    out.write("# Ultimate Sweep (3 distributions × all chunks × both solvers)\n\n")
    out.write("Mean across 3 spatial distributions. LKH skipped for chunk≥n when n≥10k.\n\n")
    out.write("| n | chunk | lk mean tour | lk stdev | lkh mean tour | lkh stdev | gap | lk mean rt | lkh mean rt |\n")
    out.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    keys = sorted({(k[1], k[2]) for k in buckets})
    for n, chunk in keys:
        lk = buckets.get(("lin_kernighan", n, chunk), [])
        lkh = buckets.get(("lkh", n, chunk), [])
        if not lk: continue
        lk_mean = statistics.mean(t for t, _, _ in lk)
        lk_stdev = statistics.pstdev(t for t, _, _ in lk) if len(lk) > 1 else 0
        lk_rt = statistics.mean(r for _, r, _ in lk)
        if lkh:
            lkh_mean = statistics.mean(t for t, _, _ in lkh)
            lkh_stdev = statistics.pstdev(t for t, _, _ in lkh) if len(lkh) > 1 else 0
            lkh_rt = statistics.mean(r for _, r, _ in lkh)
            gap = (lk_mean - lkh_mean) / lkh_mean * 100.0
            out.write(f"| {n} | {chunk} | {lk_mean:.0f} | {lk_stdev:.0f} | {lkh_mean:.0f} | {lkh_stdev:.0f} | {gap:+.2f}% | {lk_rt:.2f}s | {lkh_rt:.2f}s |\n")
        else:
            out.write(f"| {n} | {chunk} | {lk_mean:.0f} | {lk_stdev:.0f} | (skipped) | - | - | {lk_rt:.2f}s | - |\n")

    out.write("\n## lk best chunk per n (lowest mean tour across distributions)\n\n")
    out.write("| n | best chunk | mean tour | mean rt | tied chunks (within 0.2%) |\n")
    out.write("|---:|---:|---:|---:|---|\n")
    by_n_lk = collections.defaultdict(list)
    for (s, n, c), recs in buckets.items():
        if s != "lin_kernighan": continue
        m = statistics.mean(t for t, _, _ in recs)
        rt = statistics.mean(r for _, r, _ in recs)
        by_n_lk[n].append((c, m, rt))
    for n in sorted(by_n_lk):
        rows_n = by_n_lk[n]
        best = min(rows_n, key=lambda x: x[1])
        threshold = best[1] * 1.002
        tied = [str(c) for c, m, _ in rows_n if m <= threshold and c != best[0]]
        tied_str = ", ".join(tied) if tied else "(only best)"
        out.write(f"| {n} | {best[0]} | {best[1]:.0f} | {best[2]:.2f}s | {tied_str} |\n")
PYEOF

echo "tsv:    $TSV"
echo "report: $MD"

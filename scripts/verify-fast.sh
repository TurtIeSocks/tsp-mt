#!/usr/bin/env bash
# Fast iteration bench — 3 cells covering small/mid/large multi-seed
# regimes. Designed for ~45 second wall-clock so it can be run after
# every algorithm change.
#
# Cells:
#   - n=1k   chunk=1k    small-n noise floor (~5s)
#   - n=5k   chunk=5k    mid-n with measurable variance (~17s)
#   - n=15k  chunk=25k   large-n, sensitive to refinement quality (~21s)
#
# Inputs and chunk sizes match the corresponding rows of
# verify-multi-seed.sh so the LKH baseline (loaded from the cached
# baseline TSV) lines up directly — no LKH re-run needed.
#
# Output TSV header matches verify-multi-seed.sh, so the two scripts'
# results are diffable column-for-column.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEW_BIN="$ROOT_DIR/target/release/tsp-mt"
INPUTS_DIR="$ROOT_DIR/benchmark/inputs"
RESULTS_DIR="$ROOT_DIR/benchmark/results"
BASELINE="$RESULTS_DIR/baseline/sweep-baseline.tsv"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TSV="$RESULTS_DIR/verify-fast-${TIMESTAMP}.tsv"
MD="$RESULTS_DIR/verify-fast-${TIMESTAMP}.md"

CELLS=(
  "input-1k.txt 1000"
  "input-5k.txt 5000"
  "input-15k.txt 25000"
)

echo -e "input\tn\tchunk\tlkh_baseline\tlk_new\tgap_vs_lkh\truntime_lk" >"$TSV"

for cell in "${CELLS[@]}"; do
  read -r input chunk <<< "$cell"
  input_path="$INPUTS_DIR/$input"
  n=$(awk 'NF { c++ } END { print c + 0 }' "$input_path")
  lkh_baseline=$(awk -F'\t' -v i="$input" -v c="$chunk" \
    'NR>1 && $1=="lkh" && $2==i && $4==c {print $6; exit}' "$BASELINE")
  echo "running lk on $input chunk=$chunk (lkh baseline=$lkh_baseline)" >&2
  stderr=$(mktemp); stdout=$(mktemp)
  start=$(python3 -c 'import time;print(time.time())')
  "$NEW_BIN" --log-level=info --max-chunk-size="$chunk" --input="$input_path" --output="$stdout" 2>"$stderr"
  end=$(python3 -c 'import time;print(time.time())')
  runtime=$(python3 -c "print(f'{$end - $start:.2f}')")
  lk_new=$(grep "INFO metrics:" "$stderr" | tail -n1 | grep -oE 'total_m=[0-9]+' | cut -d= -f2)
  gap=$(python3 -c "print(f'{($lk_new-$lkh_baseline)/$lkh_baseline*100:+.2f}%')")
  printf '%s\t%d\t%d\t%s\t%s\t%s\t%s\n' "$input" "$n" "$chunk" "$lkh_baseline" "$lk_new" "$gap" "$runtime" >>"$TSV"
  rm -f "$stderr" "$stdout"
done

echo "" >&2

python3 - <<PYEOF
import csv
rows = list(csv.DictReader(open("$TSV"), delimiter="\t"))
with open("$MD", "w") as out:
    out.write("# Fast verify vs baseline LKH (3 cells, multi-seed)\n\n")
    out.write("| input | n | chunk | LKH baseline | lk new | gap | lk runtime |\n")
    out.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        out.write(f"| {r['input']} | {r['n']} | {r['chunk']} | {r['lkh_baseline']} | {r['lk_new']} | {r['gap_vs_lkh']} | {r['runtime_lk']}s |\n")
PYEOF
cat "$MD"

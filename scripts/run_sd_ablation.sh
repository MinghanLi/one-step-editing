#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="checkpoints/sd-turbo"
PIE_ROOT="datasets/PIE-Bench_v1/"

run_pie_bench() {
  python run_pie_bench.py \
    --model-root "${MODEL_ROOT}" \
    --pie-root "${PIE_ROOT}" \
    "$@"
}

run_pie_bench

for t_delta in 0.0 0.05 0.10 0.20 0.25; do
  run_pie_bench --t-delta "${t_delta}"
done

for t_start in 0.6 0.7 0.8 1.0; do
  run_pie_bench --t-start "${t_start}"
done

for t_end in 0.1 0.2 0.4 0.5; do
  run_pie_bench --t-end "${t_end}"
done

for t_start in 0.6 0.7 0.8 0.9 1.0; do
  run_pie_bench --t-delta 0.0 --t-start "${t_start}"
done

for t_start in 0.6 0.7 0.8 0.9 1.0; do
  run_pie_bench --t-delta 0.0 --t-start "${t_start}" --no-cleanup
done

run_pie_bench --t-delta 0.0 --t-start 0.75
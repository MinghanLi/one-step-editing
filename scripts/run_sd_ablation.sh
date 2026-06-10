# CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.0

CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.05

CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.10

CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.20

CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.25

# CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-start 0.6 

# CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-start 1.0 

# CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-end 0.2

# CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-end 0.1

# CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-end 0.4

# CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-end 0.5


# naive param ablation

# CUDA_VISIBLE_DEVICES=5 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.0 --t-start 0.6

# CUDA_VISIBLE_DEVICES=5 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.0 --t-start 0.7

# CUDA_VISIBLE_DEVICES=5 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.0 --t-start 0.75


# CUDA_VISIBLE_DEVICES=5 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.0 --t-start 0.8

# CUDA_VISIBLE_DEVICES=5 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.0 --t-start 1.0

# CUDA_VISIBLE_DEVICES=5 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.0 --t-start 0.6 --no-cleanup

# CUDA_VISIBLE_DEVICES=5 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.0 --t-start 0.7 --no-cleanup

# CUDA_VISIBLE_DEVICES=5 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.0 --t-start 0.8 --no-cleanup

# CUDA_VISIBLE_DEVICES=5 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.0 --t-start 0.9 --no-cleanup

# CUDA_VISIBLE_DEVICES=5 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.0 --t-start 1.0 --no-cleanup
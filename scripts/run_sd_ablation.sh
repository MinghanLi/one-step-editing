# CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 0.5

# CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 1.0

# CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 2.0

# CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-delta 2.5

CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-start 1.0 

CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-start 0.6 

# CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-end 0.2

# CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-end 0.1

# CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-end 0.4

# CUDA_VISIBLE_DEVICES=6 python run_pie_bench.py --model-root checkpoints/sd-turbo --pie-root datasets/PIE-Bench_v1/ --t-end 0.5
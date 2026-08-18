# SPLADE on Apple Silicon: PyTorch vs MLX

Machine: Apple M4 Max 16c/64GB — macOS-26.5.1-arm64-arm-64bit

Latency = encoder forward + SPLADE pooling, tokenization excluded
(tokenize_ms reported separately in results/*.json). mean over
adaptive iterations after warmup; see protocol block in each JSON.


## splade-cocondenser-ensembledistil

| workload | torch-cpu_fp32 | torch-mps_fp32 | torch-mps_fp16 | mlx-float32 | mlx-bfloat16 | mlx-float32_q8 | mlx-bfloat16_compile | mlx-float32_q4 | best-mlx speedup vs torch-mps_fp16 |
|---|---|---|---|---|---|---|---|---|---|
| query-L32-B1 | 14.92 ms | 7.15 ms | 6.77 ms | 3.96 ms | 3.32 ms | 2.51 ms | 3.12 ms | 2.59 ms | 2.70x |
| query-L32-B8 | 36.23 ms | 19.57 ms | 18.65 ms | 7.69 ms | 7.06 ms | 7.74 ms | 6.88 ms | 7.71 ms | 2.71x |
| query-L32-B32 | 108.80 ms | 60.01 ms | 57.21 ms | 22.64 ms | 20.07 ms | 26.28 ms | 19.42 ms | 25.97 ms | 2.95x |
| doc-L128-B1 | 26.51 ms | 9.11 ms | 8.86 ms | 5.57 ms | 4.69 ms | 4.87 ms | 4.71 ms | 4.74 ms | 1.89x |
| doc-L128-B8 | 110.04 ms | 36.65 ms | 33.55 ms | 23.08 ms | 20.25 ms | 26.57 ms | 19.83 ms | 26.23 ms | 1.69x |
| doc-L128-B32 | 414.98 ms | 130.13 ms | 115.35 ms | 87.08 ms | 74.17 ms | 103.98 ms | 73.12 ms | 100.60 ms | 1.58x |
| doc-L256-B1 | 40.41 ms | 12.42 ms | 11.35 ms | 7.96 ms | 7.28 ms | 8.42 ms | 7.27 ms | 8.23 ms | 1.56x |
| doc-L256-B8 | 213.37 ms | 61.33 ms | 53.80 ms | 45.39 ms | 38.61 ms | 53.21 ms | 38.48 ms | 52.68 ms | 1.40x |
| doc-L256-B32 | 822.96 ms | 230.85 ms | 197.83 ms | 176.96 ms | 147.03 ms | 212.53 ms | 151.65 ms | 211.70 ms | 1.35x |
| doc-L256-B64 | 1600.12 ms | 463.38 ms | 393.32 ms | 358.86 ms | 294.81 ms | 436.60 ms | 294.41 ms | 430.92 ms | 1.34x |

## efficient-splade-V-large-query

| workload | torch-cpu_fp32 | torch-mps_fp32 | torch-mps_fp16 | mlx-float32 | mlx-bfloat16 | mlx-float32_q8 | mlx-bfloat16_compile | mlx-float32_q4 | best-mlx speedup vs torch-mps_fp16 |
|---|---|---|---|---|---|---|---|---|---|
| query-L32-B1 | 8.94 ms | 4.51 ms | 4.65 ms | 2.39 ms | 2.08 ms | 1.57 ms | 2.06 ms | 1.66 ms | 2.96x |
| query-L32-B8 | 23.10 ms | 14.97 ms | 14.94 ms | 4.82 ms | 4.19 ms | 4.82 ms | 4.17 ms | 4.81 ms | 3.59x |
| query-L32-B32 | 68.69 ms | 50.29 ms | 49.41 ms | 14.78 ms | 12.37 ms | 16.65 ms | 12.22 ms | 16.70 ms | 4.04x |

## efficient-splade-V-large-doc

| workload | torch-cpu_fp32 | torch-mps_fp32 | torch-mps_fp16 | mlx-float32 | mlx-bfloat16 | mlx-float32_q8 | mlx-bfloat16_compile | mlx-float32_q4 | best-mlx speedup vs torch-mps_fp16 |
|---|---|---|---|---|---|---|---|---|---|
| doc-L128-B1 | 16.66 ms | 6.56 ms | 6.17 ms | 3.34 ms | 2.81 ms | 3.00 ms | 2.76 ms | 2.93 ms | 2.23x |
| doc-L128-B8 | 76.42 ms | 26.78 ms | 24.42 ms | 14.71 ms | 12.51 ms | 16.90 ms | 12.44 ms | 16.86 ms | 1.96x |
| doc-L128-B32 | 267.09 ms | 95.38 ms | 85.89 ms | 55.19 ms | 46.07 ms | 64.45 ms | 44.68 ms | 64.26 ms | 1.92x |
| doc-L256-B1 | 24.52 ms | 8.28 ms | 7.46 ms | 4.88 ms | 4.27 ms | 5.18 ms | 4.23 ms | 5.17 ms | 1.77x |
| doc-L256-B8 | 138.34 ms | 42.75 ms | 37.56 ms | 28.78 ms | 23.94 ms | 32.96 ms | 23.53 ms | 33.39 ms | 1.60x |
| doc-L256-B32 | 517.95 ms | 158.44 ms | 137.74 ms | 113.73 ms | 92.05 ms | 135.29 ms | 89.24 ms | 136.26 ms | 1.54x |
| doc-L256-B64 | 1032.40 ms | 318.88 ms | 273.49 ms | 227.83 ms | 186.08 ms | 277.69 ms | 179.66 ms | 269.93 ms | 1.52x |

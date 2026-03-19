# Modal A100 Playground

A personal playground for learning and experimenting with CUDA kernels on high-end GPUs (A100, H100) via [Modal](https://modal.com). Each kernel is self-contained with its own test generation, execution, correctness checking, and profiling.

## Project Structure

```
Modal_A100_Playground/
├── common/
│   └── modal_config.py       # notes on shared base image
├── kernels/
│   └── matmul/
│       ├── kernel.cu         # CUDA kernel + host launcher
│       ├── gen_tests.py      # generate test inputs + CPU reference outputs
│       ├── run.py            # Modal: compile kernel, run on GPU, return results
│       └── test.py           # local: compare GPU output vs reference
└── requirements.txt
```

## Setup

```bash
pip install modal
modal setup        # authenticate via browser
```

## Workflow

Each kernel follows the same pattern:

### 1. Generate test data (local, free)
```bash
python kernels/matmul/gen_tests.py --sizes 512,1024,4096
```
Generates random input matrices and CPU reference outputs, saved to `kernels/matmul/testdata/`.

### 2. Run the kernel on an A100
```bash
modal run kernels/matmul/run.py                   # run all available sizes
modal run kernels/matmul/run.py --sizes 1024      # specific size
modal run kernels/matmul/run.py --sizes 512,2048  # multiple sizes (auto-generates missing test data)
```
Compiles `kernel.cu` with `nvcc` inside the Modal container, runs it, and prints runtime + TFLOPS.

### 3. Check correctness (local)
```bash
python kernels/matmul/test.py                     # check all available outputs
python kernels/matmul/test.py --sizes 1024        # specific size
```
Prints absolute error metrics (max, mean, median) comparing GPU output against the CPU reference.

### 4. Profile with Nsight Compute
```bash
modal run kernels/matmul/run.py --profile --size 1024
```
Runs `ncu --set full` inside the container and downloads the `.ncu-rep` file to `kernels/matmul/profiles/`. Open it locally with the Nsight Compute GUI (`ncu-ui`).

## Adding a New Kernel

1. Create a new folder under `kernels/`
2. Add the same 4 files: `kernel.cu`, `gen_tests.py`, `run.py`, `test.py`
3. In `run.py`, define the Modal app and image inline (see matmul for reference)

## Notes

- Test data (`testdata/`) and profile reports (`profiles/`) are gitignored — regenerate locally as needed
- The first Modal run builds and caches the Docker image (~17GB). Subsequent runs reuse the cache and are fast
- Timing includes host-side memory transfers (cudaMalloc, cudaMemcpy) — use `ncu` for isolated kernel timing

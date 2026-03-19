"""
Compile and run the matmul CUDA kernel on Modal.

Usage:
    modal run kernels/matmul/run.py                        # run all sizes
    modal run kernels/matmul/run.py -- --profile           # profile 1024x1024
    modal run kernels/matmul/run.py -- --profile --size 4096
"""

import pathlib
import numpy as np
import modal

app = modal.App("cuda-playground")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel")
    .add_local_file(pathlib.Path(__file__).parent / "kernel.cu", "/root/kernel.cu")
)

profile_image = (
    modal.Image.from_registry("pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel")
    .apt_install("nsight-compute-2024.1.0")
    .add_local_file(pathlib.Path(__file__).parent / "kernel.cu", "/root/kernel.cu")
)


# Small C++ runner that ncu can wrap and profile
RUNNER_SRC = """
#include <cstdlib>

extern "C" void matmul_launch(const float*, const float*, float*, int, int, int);

int main(int argc, char** argv) {
    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);

    float *A = (float*)malloc((size_t)N * N * sizeof(float));
    float *B = (float*)malloc((size_t)N * N * sizeof(float));
    float *C = (float*)malloc((size_t)N * N * sizeof(float));

    for (int i = 0; i < N * N; i++) { A[i] = 1.0f; B[i] = 1.0f; }

    matmul_launch(A, B, C, N, N, N);

    free(A); free(B); free(C);
    return 0;
}
"""


def _compile_kernel():
    import subprocess
    subprocess.run(
        ["nvcc", "-O2", "-shared", "-Xcompiler", "-fPIC",
         "/root/kernel.cu", "-o", "/tmp/kernel.so"],
        check=True,
    )


@app.function(gpu="A100", image=image)
def run_matmul(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    import ctypes
    import subprocess
    import time
    import numpy as np

    _compile_kernel()

    lib = ctypes.CDLL("/tmp/kernel.so")
    lib.matmul_launch.restype = None
    lib.matmul_launch.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]

    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)

    def ptr(arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    # warmup: initializes CUDA context so it doesn't inflate the timed run
    lib.matmul_launch(ptr(A), ptr(B), ptr(C), M, K, N)

    t0 = time.perf_counter()
    lib.matmul_launch(ptr(A), ptr(B), ptr(C), M, K, N)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return C, elapsed_ms


@app.function(gpu="A100", image=profile_image, timeout=300)
def profile_matmul(N: int) -> bytes:
    import subprocess
    import pathlib

    _compile_kernel()

    # compile runner binary that ncu will wrap
    with open("/tmp/runner.cu", "w") as f:
        f.write(RUNNER_SRC)

    subprocess.run(
        ["nvcc", "-O2", "/tmp/runner.cu", "/tmp/kernel.so",
         "-o", "/tmp/runner", "-Xlinker", "-rpath,/tmp"],
        check=True,
    )

    subprocess.run(
        ["ncu", "--set", "full", "--export", "/tmp/report",
         "--force-overwrite", "/tmp/runner", str(N)],
        check=True,
    )

    return pathlib.Path("/tmp/report.ncu-rep").read_bytes()


@app.local_entrypoint()
def main(sizes: str = "", profile: bool = False, size: int = 1024):
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from gen_tests import generate

    testdata = pathlib.Path(__file__).parent / "testdata"

    # resolve which sizes to run
    if sizes:
        size_list = [int(s) for s in sizes.split(",")]
    else:
        size_list = sorted(int(p.stem.split("_")[1]) for p in testdata.glob("A_*.npy"))

    # generate any missing test data
    missing = [N for N in size_list if not (testdata / f"A_{N}.npy").exists()]
    if missing:
        print(f"Generating missing test data for sizes: {missing}")
        generate(missing)

    if profile:
        profiles_dir = pathlib.Path(__file__).parent / "profiles"
        profiles_dir.mkdir(exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = profiles_dir / f"matmul_{size}_{timestamp}.ncu-rep"

        print(f"Profiling {size}x{size} with ncu...")
        rep_bytes = profile_matmul.remote(size)
        out_path.write_bytes(rep_bytes)
        print(f"Saved: {out_path}")
        print(f"Open with: ncu-ui {out_path}")

    else:
        for N in size_list:
            A = np.load(testdata / f"A_{N}.npy")
            B = np.load(testdata / f"B_{N}.npy")

            print(f"[{N}x{N}] running...")
            C, elapsed_ms = run_matmul.remote(A, B)

            ops = 2.0 * N * N * N
            tflops = ops / elapsed_ms / 1e9

            print(f"[{N}x{N}] {elapsed_ms:.2f} ms  |  {tflops:.4f} TFLOPS")
            np.save(testdata / f"output_{N}.npy", C)

        print(f"\nOutputs saved to {testdata}")

"""
Generate test inputs and CPU reference outputs for matmul.
Runs locally -- no GPU needed.

Usage:
    python kernels/matmul/gen_tests.py --sizes 512,1024,4096
"""

import argparse
import pathlib
import numpy as np

TESTDATA = pathlib.Path(__file__).parent / "testdata"


def generate(sizes: list[int]):
    TESTDATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)

    for N in sizes:
        A = rng.standard_normal((N, N)).astype(np.float32)
        B = rng.standard_normal((N, N)).astype(np.float32)
        ref = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)

        np.save(TESTDATA / f"A_{N}.npy", A)
        np.save(TESTDATA / f"B_{N}.npy", B)
        np.save(TESTDATA / f"ref_{N}.npy", ref)

        print(f"[{N}x{N}] generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, required=True)
    args = parser.parse_args()
    generate([int(s) for s in args.sizes.split(",")])

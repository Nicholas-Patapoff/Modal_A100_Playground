"""
Compare kernel outputs against CPU reference.
Run after gen_tests.py and modal run run.py.

Usage:
    python kernels/matmul/test.py                  # test all available outputs
    python kernels/matmul/test.py --sizes 1024     # specific size
    python kernels/matmul/test.py --sizes 512,1024
"""

import argparse
import pathlib
import numpy as np

TESTDATA = pathlib.Path(__file__).parent / "testdata"


def error_metrics(output: np.ndarray, ref: np.ndarray) -> dict:
    diff = np.abs(output.astype(np.float64) - ref.astype(np.float64))
    return {
        "max":    float(diff.max()),
        "mean":   float(diff.mean()),
        "median": float(np.median(diff)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="")
    args = parser.parse_args()

    if args.sizes:
        size_list = [int(s) for s in args.sizes.split(",")]
    else:
        size_list = sorted(int(p.stem.split("_")[1]) for p in TESTDATA.glob("output_*.npy"))

    if not size_list:
        print("No outputs found -- run modal run run.py first")
        return

    for N in size_list:
        output_path = TESTDATA / f"output_{N}.npy"
        ref_path    = TESTDATA / f"ref_{N}.npy"

        if not output_path.exists():
            print(f"[{N}x{N}] missing output -- run modal run run.py first")
            continue
        if not ref_path.exists():
            print(f"[{N}x{N}] missing ref -- run gen_tests.py first")
            continue

        output = np.load(output_path)
        ref    = np.load(ref_path)
        m      = error_metrics(output, ref)

        print(f"[{N}x{N}]  max={m['max']:.4e}  mean={m['mean']:.4e}  median={m['median']:.4e}")


if __name__ == "__main__":
    main()

"""
Sanity check: verify we can reach an A100 on Modal.
Prints nvidia-smi output from inside the container.

Usage:
    modal run kernels/hello_gpu/run.py
"""

import modal

app = modal.App("hello-gpu")

image = modal.Image.from_registry("pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel")


@app.function(gpu="A100", image=image)
def hello_gpu():
    import subprocess
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)


@app.local_entrypoint()
def main():
    hello_gpu.remote()

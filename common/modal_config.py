import modal

app = modal.App("cuda-playground")

cuda_image = (
    modal.Image.from_registry("pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel")
    .pip_install("numpy")
)

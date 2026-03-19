/*
 * Naive matmul: C = A @ B
 * A: (M, K)  B: (K, N)  C: (M, N) -- all float32, row-major
 *
 * Each thread computes one element of C.
 */

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

/*
 * Host launcher -- called via ctypes from Python.
 * Handles device memory allocation, data transfer, kernel launch, and cleanup.
 */
extern "C" void matmul_launch(
    const float* h_A,
    const float* h_B,
    float* h_C,
    int M, int K, int N)
{
    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

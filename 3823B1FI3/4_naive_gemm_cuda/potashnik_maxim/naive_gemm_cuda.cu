#include "naive_gemm_cuda.h"
#include <vector>

/* Optimizations
1. 2D block/grid decomposition
2. Coalesced global memory access
3. Static device memory reuse
4. cudaMemset on device
5. __restrict__ pointers
6. Each thread computes full dot product for one C elemenе (no shared memory)
*/

const int block_size = 32;

__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& A, const std::vector<float>& B, int N) {
    int total_size = N * N;

    static float* d_A = nullptr;
    static float* d_B = nullptr;
    static float* d_C = nullptr;
    static int capacity = 0;

    if (total_size > capacity) {
        if (capacity > 0) {
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
        }
        cudaMalloc(&d_A, total_size * sizeof(float));
        cudaMalloc(&d_B, total_size * sizeof(float));
        cudaMalloc(&d_C, total_size * sizeof(float));
        capacity = total_size;
    }

    cudaMemcpy(d_A, A.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, total_size * sizeof(float));

    dim3 blockDim(block_size, block_size);
    dim3 gridDim((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    std::vector<float> C(total_size);
    cudaMemcpy(C.data(), d_C, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    return C;
}
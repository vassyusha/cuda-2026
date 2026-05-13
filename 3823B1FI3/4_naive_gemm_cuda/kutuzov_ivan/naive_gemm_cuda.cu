#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16

__global__ void gemm_tiled(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int N)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) 
    {
        int a_col = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;

        int b_row = t * TILE_SIZE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    const int total_elements = n * n;
    if (total_elements == 0) return {};

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    cudaMalloc(&d_A, total_elements * sizeof(float));
    cudaMalloc(&d_B, total_elements * sizeof(float));
    cudaMalloc(&d_C, total_elements * sizeof(float));

    cudaMemcpy(d_A, a.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,
              (n + TILE_SIZE - 1) / TILE_SIZE);
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, n);

    std::vector<float> result(total_elements);
    cudaMemcpy(result.data(), d_C, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}
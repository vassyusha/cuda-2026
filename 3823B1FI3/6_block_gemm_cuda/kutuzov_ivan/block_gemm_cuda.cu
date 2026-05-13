#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 32

__global__ void gemm_kernel(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ result,
                            int n)
{
    __shared__ float a_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float b_shared[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    float local_res = 0.0f;

    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) 
    {
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if (row < n && a_col < n)
            a_shared[threadIdx.y][threadIdx.x] = __ldg(a + row * n + a_col);
        else
            a_shared[threadIdx.y][threadIdx.x] = 0.0f;

        int b_row = tile * TILE_SIZE + threadIdx.y;
        if (b_row < n && col < n)
            b_shared[threadIdx.y][threadIdx.x] = __ldg(b + b_row * n + col);
        else
            b_shared[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            local_res += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        result[row * n + col] = local_res;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    const int total_elements = n * n;
    if (total_elements == 0) return {};

    float *a_gpu = nullptr, *b_gpu = nullptr, *result_gpu = nullptr;
    cudaMalloc(&a_gpu, total_elements * sizeof(float));
    cudaMalloc(&b_gpu, total_elements * sizeof(float));
    cudaMalloc(&result_gpu, total_elements * sizeof(float));

    float *h_pinned = nullptr;
    cudaMallocHost(&h_pinned, total_elements * sizeof(float));

    cudaMemcpy(a_gpu, a.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,
              (n + TILE_SIZE - 1) / TILE_SIZE);
    gemm_kernel<<<grid, block>>>(a_gpu, b_gpu, result_gpu, n);

    cudaMemcpy(h_pinned, result_gpu, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> res(h_pinned, h_pinned + total_elements);

    cudaFreeHost(h_pinned);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(result_gpu);

    return res;
}
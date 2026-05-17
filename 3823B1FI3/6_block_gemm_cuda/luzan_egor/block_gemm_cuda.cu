#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

const int kBlock = 16;

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                const int n) {
    __shared__ float tileA[kBlock][kBlock];
    __shared__ float tileB[kBlock][kBlock];

    const int row = blockIdx.y * kBlock + threadIdx.y;
    const int col = blockIdx.x * kBlock + threadIdx.x;

    float acum = 0.0f;

    for (int K = 0; K < n; K += kBlock) {
        tileA[threadIdx.y][threadIdx.x] = (row < n && K + threadIdx.x < n)
                                            ? a[row * n + K + threadIdx.x]
                                            : 0.0f;

        tileB[threadIdx.y][threadIdx.x] = (col < n && K + threadIdx.y < n)
                                            ? b[(K + threadIdx.y) * n + col]
                                            : 0.0f;
        __syncthreads();

        for (int k = 0; k < kBlock; ++k)
            acum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        c[row * n + col] = acum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  int n)
{
    std::vector<float> c(n * n);
    const size_t bytes = n * n * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(kBlock, kBlock);
    dim3 gridDim((n + kBlock - 1) / kBlock,
                 (n + kBlock - 1) / kBlock);

    BlockGemmKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}

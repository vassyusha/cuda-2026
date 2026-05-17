#include <vector>
#include "naive_gemm_cuda.h"

// compute c_ij
__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column

    if (i >= n || j >= n) 
        return;

    float acc = 0.0f;
    for (int k = 0; k < n; k++)
        acc += a[i * n + k] * b[k * n + j];
    c[i * n + j] = acc;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  int n) {
    std::vector<float> c(n * n);
    const size_t bytes = n * n * sizeof(float);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    const int kBlock = 16;
    dim3 blockDim(kBlock, kBlock);
    dim3 gridDim((n + kBlock - 1) / kBlock,
                 (n + kBlock - 1) / kBlock);

    NaiveGemmKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}

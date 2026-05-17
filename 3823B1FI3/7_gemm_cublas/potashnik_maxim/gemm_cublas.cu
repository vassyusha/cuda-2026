#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* Optimizations
1. Static cuBLAS handle
2. Static device memory reuse
3. Pinned host memory
4. cuBLAS column-major trick
5. cublasSgemm
*/

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    int total_size = n * n;

    static float* d_A = nullptr;
    static float* d_B = nullptr;
    static float* d_C = nullptr;
    static float* h_out = nullptr;
    static int capacity = 0;

    static cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);

    if (total_size > capacity) {
        if (capacity > 0) {
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            cudaFreeHost(h_out);
        }
        cudaMalloc(&d_A, total_size * sizeof(float));
        cudaMalloc(&d_B, total_size * sizeof(float));
        cudaMalloc(&d_C, total_size * sizeof(float));
        cudaMallocHost(&h_out, total_size * sizeof(float));
        capacity = total_size;
    }

    cudaMemcpy(d_A, a.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_B, n, d_A, n, &beta, d_C, n);

    cudaMemcpy(h_out, d_C, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    return std::vector<float>(h_out, h_out + total_size);
}